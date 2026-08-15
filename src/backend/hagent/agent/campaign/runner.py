"""
Campaign runner — submit/poll/compare multi start_training jobs.

Respects max_concurrent_jobs. Uses tool_runner.invoke_tool (mockable).
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from hagent.agent.campaign.builder import build_campaign
from hagent.agent.campaign.compare import best_config_payload, compare_campaign
from hagent.agent.campaign.schema import Campaign, CampaignVariant
from hagent.agent.execution.tool_runner import enrich_params, invoke_tool

logger = structlog.get_logger(__name__)


def _extract_job_id(payload: dict) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in ("job_id", "id", "_id"):
        if payload.get(key):
            return str(payload[key])
    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("job_id", "id"):
            if data.get(key):
                return str(data[key])
    return None


def _extract_status(payload: dict) -> str:
    if not isinstance(payload, dict):
        return "unknown"
    st = payload.get("status")
    if st is None and isinstance(payload.get("data"), dict):
        st = payload["data"].get("status")
    # Numeric codes sometimes used in HAutoML
    if st in (0, "0"):
        return "running"
    if st in (1, "1"):
        return "completed"
    if st in (-1, "-1"):
        return "failed"
    return str(st or "unknown").lower()


def _extract_metrics(payload: dict) -> dict[str, float]:
    metrics = payload.get("metrics") or {}
    if not metrics and isinstance(payload.get("data"), dict):
        metrics = payload["data"].get("metrics") or {}
    out: dict[str, float] = {}
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
    return out


async def _submit_variant(
    variant: CampaignVariant,
    *,
    campaign_id: str,
    user_id: str | None,
    user_token: str | None,
    goal: dict,
    world_model: dict | None,
) -> CampaignVariant:
    params = enrich_params(
        "start_training",
        dict(variant.params),
        user_id=user_id,
        user_token=user_token,
        goal=goal,
        world_model=world_model,
        action_id=(
            f"campaign:{campaign_id}:variant:{variant.variant_id}"
            if campaign_id and variant.variant_id
            else None
        ),
    )
    variant.params = {
        key: value
        for key, value in params.items()
        if key not in {"token", "idempotency_key"}
    }
    # Required fields guard
    if (
        not params.get("dataset_id")
        or not params.get("target_column")
        or not params.get("idempotency_key")
    ):
        variant.status = "failed"
        variant.error = (
            "missing dataset_id, target_column or trusted action identity "
            "for campaign variant"
        )
        return variant

    payload = await invoke_tool("start_training", params)
    if isinstance(payload, dict) and payload.get("error"):
        variant.status = "failed"
        variant.error = str(payload["error"])
        return variant

    job_id = _extract_job_id(payload if isinstance(payload, dict) else {})
    if not job_id:
        # Some backends return success without id — treat as submitted synthetic
        variant.status = "failed"
        variant.error = "start_training returned no job_id"
        return variant

    variant.job_id = job_id
    variant.status = "submitted"
    return variant


async def _poll_variant(
    variant: CampaignVariant,
    *,
    user_token: str | None,
) -> CampaignVariant:
    if not variant.job_id:
        return variant
    payload = await invoke_tool(
        "get_job_info",
        {"job_id": variant.job_id, "token": user_token}
        if user_token
        else {"job_id": variant.job_id},
    )
    if isinstance(payload, dict) and payload.get("error"):
        # Keep monitoring on transient errors
        variant.error = str(payload["error"])
        return variant

    payload = payload if isinstance(payload, dict) else {}
    st = _extract_status(payload)
    if st in ("completed", "done", "success"):
        variant.status = "completed"
        variant.metrics = _extract_metrics(payload)
        variant.best_model = payload.get("best_model") or (
            (payload.get("data") or {}).get("best_model")
            if isinstance(payload.get("data"), dict)
            else None
        )
        score = payload.get("best_score")
        if score is None and isinstance(payload.get("data"), dict):
            score = payload["data"].get("best_score")
        if score is None and variant.metrics:
            score = max(variant.metrics.values())
        try:
            variant.best_score = float(score) if score is not None else None
        except (TypeError, ValueError):
            variant.best_score = None
        variant.error = None
    elif st in ("failed", "error", "cancelled", "canceled"):
        variant.status = "failed"
        variant.error = str(payload.get("error") or st)
    else:
        variant.status = "running"
    return variant


async def write_warm_start_memory(
    user_id: str | None,
    best_config: dict,
    *,
    fact_store: Any | None = None,
) -> None:
    if not user_id or not best_config:
        return
    try:
        from hagent.agent.memory import Fact, create_fact_store

        store = fact_store or create_fact_store()
        ptype = str(best_config.get("problem_type") or "unknown")
        key = f"warm_start_{ptype}"
        fact = Fact(
            key=key,
            content=json.dumps(best_config, ensure_ascii=False, default=str),
            category="model",
            source="campaign",
            confidence=0.9,
        )
        await store.save(user_id, fact)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Warm-start memory write failed: %s", exc)


def _extension_enabled() -> bool:
    try:
        from hagent.bridge.config import get_campaign_config

        return bool(
            (get_campaign_config().get("surprise_extension") or {}).get(
                "enabled", False
            )
        )
    except Exception:  # noqa: BLE001
        return False


def _maybe_extend_on_surprise(
    campaign: Campaign,
    *,
    goal: dict,
    wm_snap: dict,
    events: list,
    outcome_model: Any,
    user_id: str | None,
) -> bool:
    """
    Nếu có variant completed với outcome surprise HIGH và còn quota vòng mở
    rộng → thêm variant khám phá, trả True (caller quay lại submitting).

    Tính lại surprise tại đây thay vì gom event qua các tick — event của tick
    trước không còn trong `events` của tick này, còn model thì đã memoize.
    """
    try:
        from hagent.bridge.config import get_campaign_config, get_world_model_config

        ext_cfg = dict(get_campaign_config().get("surprise_extension") or {})
        max_rounds = int(ext_cfg.get("max_rounds", 1))
        if campaign.extension_rounds >= max_rounds:
            return False
        n_extra = max(1, int(ext_cfg.get("n_extra", 2)))
        surprise_cfg = dict((get_world_model_config() or {}).get("surprise") or {})
        if not surprise_cfg.get("outcome_enabled", True):
            return False

        from hagent.agent.campaign.wm_hooks import campaign_outcome_surprise

        model_arg = None if isinstance(outcome_model, str) else outcome_model

        trigger = None
        for variant in campaign.variants:
            if variant.status != "completed" or variant.best_score is None:
                continue
            ds_id = (variant.params or {}).get("dataset_id")
            meta = (wm_snap.get("datasets") or {}).get(ds_id)
            outcome = campaign_outcome_surprise(
                variant=variant,
                dataset_meta=meta,
                outcome_model=model_arg,
                surprise_config=surprise_cfg,
            )
            if outcome and outcome.get("level") == "high":
                trigger = (variant, outcome, meta)
                break
        if trigger is None:
            return False

        variant, outcome, meta = trigger
        from hagent.agent.campaign.builder import propose_extension_variants

        new_variants = propose_extension_variants(
            campaign,
            goal,
            user_id=user_id,
            dataset_meta=meta,
            outcome_model=outcome_model,
            n_extra=n_extra,
            exploration_weight=float(ext_cfg.get("exploration_weight", 0.5)),
        )
        if not new_variants:
            return False

        campaign.variants.extend(new_variants)
        campaign.extension_rounds += 1
        events.append(
            {
                "type": "campaign_extended",
                "round": campaign.extension_rounds,
                "trigger_variant_id": variant.variant_id,
                "trigger_zscore": outcome.get("zscore"),
                "n_added": len(new_variants),
            }
        )
        logger.info(
            "Campaign %s mở rộng vòng %d: +%d variant (zscore=%.2f)",
            campaign.campaign_id,
            campaign.extension_rounds,
            len(new_variants),
            float(outcome.get("zscore") or 0.0),
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("surprise extension skipped: %s", exc)
        return False


async def campaign_step(
    campaign: Campaign,
    *,
    user_id: str | None = None,
    user_token: str | None = None,
    world_model: dict | None = None,
    fact_store: Any | None = None,
    wm_service: Any | None = None,
    surprise_events: list | None = None,
    outcome_model: Any = "auto",
) -> Campaign:
    """
    One graph tick of the campaign:
    - submit up to free concurrency slots
    - poll in-flight jobs
    - when all terminal → compare + mark done

    Optional wm_service records LeWM surprise on submit/poll.
    outcome_model: "auto" → model mặc định từ config cho outcome surprise;
    None → tắt hẳn; object → dùng model được truyền (benchmark dùng model
    train online tại đây).
    """
    goal = campaign.goal or {}
    wm_snap = dict(world_model or {"user_id": user_id or ""})
    events = surprise_events if surprise_events is not None else []

    # Submit phase
    in_flight = campaign.in_flight()
    free = max(0, campaign.max_concurrent - len(in_flight))
    if free > 0:
        for variant in list(campaign.pending_submit())[:free]:
            before = dict(wm_snap)
            await _submit_variant(
                variant,
                campaign_id=campaign.campaign_id,
                user_id=user_id,
                user_token=user_token,
                goal=goal,
                world_model=wm_snap,
            )
            if variant.job_id:
                campaign.spent_budget += 1
                jobs = dict(wm_snap.get("jobs") or {})
                jobs[variant.job_id] = variant.to_submission_job_entry()
                wm_snap["jobs"] = jobs
            try:
                from hagent.agent.campaign.wm_hooks import campaign_wm_step

                surprise, wm_snap = await campaign_wm_step(
                    wm_service=wm_service,
                    world_model=before,
                    user_id=user_id,
                    action_type="start_training",
                    params=dict(variant.params or {}),
                    goal=goal,
                    next_world_model=wm_snap,
                )
                if surprise:
                    events.append(
                        {
                            "type": "campaign_surprise",
                            "action": "start_training",
                            "variant_id": variant.variant_id,
                            "surprise": surprise,
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                logger.debug("campaign submit WM step: %s", exc)
        campaign.status = (
            "monitoring"
            if campaign.in_flight() or campaign.pending_submit()
            else campaign.status
        )

    # Poll phase
    for variant in list(campaign.in_flight()):
        before = dict(wm_snap)
        prev_status = variant.status
        await _poll_variant(variant, user_token=user_token)
        if variant.job_id:
            jobs = dict(wm_snap.get("jobs") or {})
            jobs[variant.job_id] = variant.to_job_entry()
            wm_snap["jobs"] = jobs
        try:
            from hagent.agent.campaign.wm_hooks import campaign_wm_step

            surprise, wm_snap = await campaign_wm_step(
                wm_service=wm_service,
                world_model=before,
                user_id=user_id,
                action_type="get_job_info",
                params={"job_id": variant.job_id, "status_hint": variant.status},
                goal=goal,
                next_world_model=wm_snap,
            )
            if surprise:
                events.append(
                    {
                        "type": "campaign_surprise",
                        "action": "get_job_info",
                        "variant_id": variant.variant_id,
                        "job_id": variant.job_id,
                        "status": variant.status,
                        "surprise": surprise,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("campaign poll WM step: %s", exc)

        # Outcome-space surprise — chỉ đúng lúc variant chuyển sang completed
        if (
            prev_status != "completed"
            and variant.status == "completed"
            and outcome_model is not None
        ):
            try:
                from hagent.agent.campaign.wm_hooks import campaign_outcome_surprise
                from hagent.bridge.config import get_world_model_config

                surprise_cfg = dict(
                    (get_world_model_config() or {}).get("surprise") or {}
                )
                if surprise_cfg.get("outcome_enabled", True):
                    ds_id = (variant.params or {}).get("dataset_id")
                    meta = (wm_snap.get("datasets") or {}).get(ds_id)
                    outcome = campaign_outcome_surprise(
                        variant=variant,
                        dataset_meta=meta,
                        outcome_model=(
                            None if isinstance(outcome_model, str) else outcome_model
                        ),
                        surprise_config=surprise_cfg,
                    )
                    if outcome:
                        events.append(
                            {
                                "type": "campaign_outcome_surprise",
                                "variant_id": variant.variant_id,
                                "job_id": variant.job_id,
                                "outcome": outcome,
                            }
                        )
            except Exception as exc:  # noqa: BLE001
                logger.debug("campaign outcome surprise: %s", exc)

    # Attach latest WM for callers
    campaign._world_model_snapshot = wm_snap  # type: ignore[attr-defined]
    campaign._surprise_events = events  # type: ignore[attr-defined]

    # Early stopping check: nếu các variants đã hoàn thành hội tụ score
    try:
        from hagent.bridge.config import get_campaign_config

        early_cfg = dict((get_campaign_config() or {}).get("early_stopping") or {})
    except Exception:  # noqa: BLE001
        early_cfg = {"enabled": True, "convergence_threshold": 0.005, "patience": 2}

    if early_cfg.get("enabled", True):
        threshold = float(early_cfg.get("convergence_threshold", 0.005))
        patience = int(early_cfg.get("patience", 2))
        metric = (campaign.goal or {}).get("metric", "")
        higher_is_better = str(metric).lower() not in {
            "mae",
            "mse",
            "rmse",
            "rmsle",
            "loss",
        }
        if check_early_stopping(
            campaign,
            convergence_threshold=threshold,
            patience=patience,
            higher_is_better=higher_is_better,
        ):
            for v in campaign.pending_submit():
                v.status = "failed"
                v.error = "Early stopped: score converged"
            campaign.early_stopped = True  # type: ignore[attr-defined]
            logger.info(
                "Campaign early stopped due to score convergence",
                campaign_id=campaign.campaign_id,
            )

    unfinished = campaign.unfinished()
    still_pending_submit = campaign.pending_submit()
    if unfinished or still_pending_submit:
        # If only pending and no capacity was free, keep monitoring
        campaign.status = "monitoring"
        return campaign

    # All terminal → trước khi so sánh, cân nhắc VÒNG MỞ RỘNG theo outcome
    # surprise (cơ chế surprise-driven replanning; gate config, mặc định tắt)
    if outcome_model is not None and _extension_enabled():
        extended = _maybe_extend_on_surprise(
            campaign,
            goal=goal,
            wm_snap=wm_snap,
            events=events,
            outcome_model=outcome_model,
            user_id=user_id,
        )
        if extended:
            campaign.status = "submitting"
            return campaign

    campaign.status = "comparing"
    best, table = compare_campaign(campaign)
    campaign.comparison = table
    if best:
        campaign.best_variant_id = best.variant_id
        payload = best_config_payload(best, goal)
        await write_warm_start_memory(user_id, payload, fact_store=fact_store)
        campaign.status = "done"
    else:
        campaign.status = "failed"
    return campaign


def check_early_stopping(
    campaign: Campaign,
    *,
    convergence_threshold: float = 0.005,
    patience: int = 2,
    higher_is_better: bool = True,
) -> bool:
    """Kiểm tra điều kiện early stopping khi các kết quả variants hội tụ.

    Dừng sớm khi độ chênh lệch điểm số (improvement) giữa các lần hoàn thành
    gần nhất < convergence_threshold liên tục trong `patience` bước.
    """
    completed = [
        v
        for v in campaign.variants
        if v.status == "completed" and v.best_score is not None
    ]
    if len(completed) < patience + 1:
        return False

    scores = [float(v.best_score) for v in completed if v.best_score is not None]
    if len(scores) < patience + 1:
        return False

    running_best: list[float] = []
    curr_best = -float("inf") if higher_is_better else float("inf")
    for s in scores:
        if higher_is_better:
            curr_best = max(curr_best, s)
        else:
            curr_best = min(curr_best, s)
        running_best.append(curr_best)

    recent = running_best[-(patience + 1) :]
    improvements = [abs(recent[i] - recent[i - 1]) for i in range(1, len(recent))]
    return all(imp < convergence_threshold for imp in improvements)


async def ensure_campaign(
    state: dict,
    *,
    fact_store: Any | None = None,
    config: dict | None = None,
) -> Campaign:
    """Load campaign from state or build a new one."""
    raw = state.get("campaign")
    if isinstance(raw, dict) and raw.get("variants"):
        return Campaign.from_dict(raw)

    goal = state.get("goal") if isinstance(state.get("goal"), dict) else {}
    if not goal:
        goal = (
            state.get("user_requirements")
            if isinstance(state.get("user_requirements"), dict)
            else {}
        )

    return await build_campaign(
        goal,
        user_id=state.get("user_id"),
        world_model=state.get("world_model"),
        fact_store=fact_store,
        config=config,
    )


# Alias
run_campaign_tick = campaign_step
