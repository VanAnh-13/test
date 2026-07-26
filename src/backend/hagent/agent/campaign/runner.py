"""
Campaign runner — submit/poll/compare multi start_training jobs.

Respects max_concurrent_jobs. Uses tool_runner.invoke_tool (mockable).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from hagent.agent.campaign.builder import build_campaign
from hagent.agent.campaign.compare import best_config_payload, compare_campaign
from hagent.agent.campaign.schema import Campaign, CampaignVariant
from hagent.agent.execution.tool_runner import enrich_params, invoke_tool

logger = logging.getLogger(__name__)


def _extract_job_id(payload: dict) -> Optional[str]:
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


def _extract_metrics(payload: dict) -> Dict[str, float]:
    metrics = payload.get("metrics") or {}
    if not metrics and isinstance(payload.get("data"), dict):
        metrics = payload["data"].get("metrics") or {}
    out: Dict[str, float] = {}
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
    )
    # Required fields guard
    if not params.get("dataset_id") or not params.get("target_column"):
        variant.status = "failed"
        variant.error = "missing dataset_id or target_column for campaign variant"
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
    variant.params = params
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
        {"job_id": variant.job_id, "token": user_token} if user_token else {"job_id": variant.job_id},
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
    except Exception as exc:
        logger.debug("Warm-start memory write failed: %s", exc)


async def campaign_step(
    campaign: Campaign,
    *,
    user_id: str | None,
    user_token: str | None,
    world_model: dict | None,
    fact_store: Any | None = None,
    wm_service: Any | None = None,
    surprise_events: list | None = None,
) -> Campaign:
    """
    One graph tick of the campaign:
    - submit up to free concurrency slots
    - poll in-flight jobs
    - when all terminal → compare + mark done

    Optional wm_service records LeWM surprise on submit/poll.
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
                user_id=user_id,
                user_token=user_token,
                goal=goal,
                world_model=wm_snap,
            )
            if variant.job_id:
                jobs = dict(wm_snap.get("jobs") or {})
                jobs[variant.job_id] = {
                    "id": variant.job_id,
                    "status": variant.status,
                    "config": variant.params,
                    "dataset_id": variant.params.get("dataset_id"),
                }
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
            except Exception as exc:
                logger.debug("campaign submit WM step: %s", exc)
        campaign.status = "monitoring" if campaign.in_flight() or campaign.pending_submit() else campaign.status

    # Poll phase
    for variant in list(campaign.in_flight()):
        before = dict(wm_snap)
        prev_status = variant.status
        await _poll_variant(variant, user_token=user_token)
        if variant.job_id:
            jobs = dict(wm_snap.get("jobs") or {})
            jobs[variant.job_id] = {
                "id": variant.job_id,
                "status": variant.status,
                "best_model": variant.best_model,
                "best_score": variant.best_score,
                "metrics": variant.metrics,
                "config": variant.params,
                "dataset_id": (variant.params or {}).get("dataset_id"),
            }
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
        except Exception as exc:
            logger.debug("campaign poll WM step: %s", exc)

        # Outcome-space surprise — chỉ đúng lúc variant chuyển sang completed
        if prev_status != "completed" and variant.status == "completed":
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
            except Exception as exc:
                logger.debug("campaign outcome surprise: %s", exc)

    # Attach latest WM for callers
    campaign._world_model_snapshot = wm_snap  # type: ignore[attr-defined]
    campaign._surprise_events = events  # type: ignore[attr-defined]

    unfinished = campaign.unfinished()
    still_pending_submit = campaign.pending_submit()
    if unfinished or still_pending_submit:
        # If only pending and no capacity was free, keep monitoring
        campaign.status = "monitoring"
        return campaign

    # All terminal → compare
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
        goal = state.get("user_requirements") if isinstance(state.get("user_requirements"), dict) else {}

    return await build_campaign(
        goal,
        user_id=state.get("user_id"),
        world_model=state.get("world_model"),
        fact_store=fact_store,
        config=config,
    )
