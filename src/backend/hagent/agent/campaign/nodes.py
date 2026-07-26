"""LangGraph nodes for multi-candidate campaigns."""

from __future__ import annotations

import logging
from typing import Any

try:
    from langchain_core.messages import AIMessage
except ImportError:  # pragma: no cover
    class AIMessage:  # type: ignore[no-redef]
        def __init__(self, content: str = "", **kwargs):
            self.content = content
            self.type = "ai"

from hagent.agent.campaign.runner import campaign_step, ensure_campaign
from hagent.agent.state import AutoMLState

logger = logging.getLogger(__name__)


def _append_event(state: AutoMLState, event: dict) -> list:
    events = list(state.get("execution_events") or [])
    events.append(event)
    return events


def _max_monitor_ticks() -> int:
    try:
        from hagent.bridge.config import get_campaign_config

        return int(get_campaign_config().get("max_monitor_ticks", 50))
    except Exception:
        return 50


def _select_surprise(surprise_buf: list) -> dict | None:
    """
    Chọn surprise để promote vào state["surprise"].

    Ưu tiên outcome surprise (payload nằm dưới key "outcome" — bug cũ chỉ đọc
    key "surprise" nên event outcome bị vứt); lấy zscore cao nhất. Fallback:
    latent surprise cuối cùng như hành vi cũ.
    """
    outcomes = [
        e.get("outcome")
        for e in surprise_buf
        if e.get("type") == "campaign_outcome_surprise" and e.get("outcome")
    ]
    if outcomes:
        picked = max(outcomes, key=lambda o: float(o.get("zscore") or 0.0))
        return {"kind": "outcome", **picked}
    for e in reversed(surprise_buf):
        s = e.get("surprise")
        if s:
            return s
    return None


async def campaign_node(state: AutoMLState) -> dict:
    """
    Multi-candidate job campaign tick.

    Builds campaign on first entry, then submit/poll until done.
    """
    events = list(state.get("execution_events") or [])
    cost = dict(state.get("cost_metrics") or {})
    ticks = int(state.get("campaign_tick") or 0) + 1

    campaign = await ensure_campaign(state)
    prev_status = campaign.status
    surprise_buf: list = []

    if prev_status == "building":
        events.append(
            {
                "type": "campaign_built",
                "campaign_id": campaign.campaign_id,
                "n_variants": len(campaign.variants),
                "warm_start_used": campaign.warm_start_used,
                "variants": [
                    {
                        "id": v.variant_id,
                        "label": v.label,
                        "source": v.source,
                        "search_algorithm": v.params.get("search_algorithm"),
                        "time_limit": v.params.get("time_limit"),
                    }
                    for v in campaign.variants
                ],
            }
        )
        campaign.status = "submitting"

    campaign = await campaign_step(
        campaign,
        user_id=state.get("user_id"),
        user_token=state.get("user_token"),
        world_model=state.get("world_model"),
        wm_service=state.get("_wm_service"),
        surprise_events=surprise_buf,
    )
    events.extend(surprise_buf)
    wm_from_camp = getattr(campaign, "_world_model_snapshot", None)

    # Safety: avoid infinite graph loops when jobs never finish
    max_ticks = _max_monitor_ticks()
    if campaign.status == "monitoring" and ticks >= max_ticks:
        for v in campaign.variants:
            if v.status in ("pending", "submitted", "running"):
                v.status = "failed"
                v.error = v.error or f"campaign monitor timeout after {max_ticks} ticks"
        more_surp: list = []
        campaign = await campaign_step(
            campaign,
            user_id=state.get("user_id"),
            user_token=state.get("user_token"),
            world_model=state.get("world_model"),
            wm_service=state.get("_wm_service"),
            surprise_events=more_surp,
        )
        events.extend(more_surp)
        wm_from_camp = getattr(campaign, "_world_model_snapshot", None) or wm_from_camp
        events.append(
            {
                "type": "campaign_timeout",
                "campaign_id": campaign.campaign_id,
                "ticks": ticks,
                "max_ticks": max_ticks,
            }
        )

    cost["campaign_variants"] = len(campaign.variants)
    cost["campaign_submitted"] = sum(
        1 for v in campaign.variants if v.job_id
    )
    cost["campaign_completed"] = sum(
        1 for v in campaign.variants if v.status == "completed"
    )
    cost["campaign_failed"] = sum(
        1 for v in campaign.variants if v.status == "failed"
    )

    events.append(
        {
            "type": "campaign_tick",
            "campaign_id": campaign.campaign_id,
            "status": campaign.status,
            "variants": [
                {
                    "id": v.variant_id,
                    "label": v.label,
                    "status": v.status,
                    "job_id": v.job_id,
                    "best_score": v.best_score,
                }
                for v in campaign.variants
            ],
        }
    )

    update: dict[str, Any] = {
        "campaign": campaign.to_dict(),
        "campaign_status": campaign.status,
        "campaign_tick": ticks,
        "execution_events": events,
        "cost_metrics": cost,
        "current_phase": "train" if campaign.status != "done" else "evaluate",
    }
    if surprise_buf:
        picked = _select_surprise(surprise_buf)
        if picked:
            update["surprise"] = picked

    if campaign.status == "done":
        best = next(
            (v for v in campaign.variants if v.variant_id == campaign.best_variant_id),
            None,
        )
        comparison = campaign.comparison
        update["evaluation"] = {
            "job_ids": [v.job_id for v in campaign.variants if v.job_id],
            "comparison_table": comparison,
            "best_job_id": best.job_id if best else None,
            "recommendation": best.best_model if best else None,
        }
        update["plan_status"] = "done"
        score_txt = (
            f"score={best.best_score}, model={best.best_model}"
            if best
            else "no winner"
        )
        msg = (
            f"Campaign {campaign.campaign_id[:8]} xong: "
            f"{cost.get('campaign_completed', 0)}/{len(campaign.variants)} completed. "
            f"Best: {score_txt}."
        )
        events.append(
            {
                "type": "campaign_done",
                "campaign_id": campaign.campaign_id,
                "best_variant_id": campaign.best_variant_id,
                "comparison": comparison,
            }
        )
        update["execution_events"] = events
        update["messages"] = [AIMessage(content=msg)]
        # Sync jobs into world_model lightly
        wm = dict(
            wm_from_camp
            or state.get("world_model")
            or {"user_id": state.get("user_id")}
        )
        jobs = dict(wm.get("jobs") or {})
        for v in campaign.variants:
            if v.job_id:
                jobs[v.job_id] = {
                    "id": v.job_id,
                    "status": v.status,
                    "best_model": v.best_model,
                    "best_score": v.best_score,
                    "metrics": v.metrics,
                    "config": v.params,
                    "dataset_id": v.params.get("dataset_id"),
                }
        wm["jobs"] = jobs
        update["world_model"] = wm
    elif wm_from_camp is not None:
        update["world_model"] = wm_from_camp

    elif campaign.status == "failed":
        update["plan_status"] = "failed"
        update["messages"] = [
            AIMessage(content=f"Campaign thất bại: không có variant thành công.")
        ]
        events.append(
            {
                "type": "campaign_failed",
                "campaign_id": campaign.campaign_id,
                "variants": [v.to_dict() for v in campaign.variants],
            }
        )
        update["execution_events"] = events
    else:
        n_run = sum(1 for v in campaign.variants if v.status in ("submitted", "running"))
        n_pend = sum(1 for v in campaign.variants if v.status == "pending")
        update["messages"] = [
            AIMessage(
                content=(
                    f"Campaign monitoring: running={n_run}, pending={n_pend}, "
                    f"done={cost.get('campaign_completed', 0)}, "
                    f"failed={cost.get('campaign_failed', 0)}."
                )
            )
        ]

    return update


def campaign_route(state: AutoMLState) -> str:
    status = state.get("campaign_status") or (
        (state.get("campaign") or {}).get("status")
        if isinstance(state.get("campaign"), dict)
        else None
    )
    if status in ("done", "failed"):
        return "synthesize"
    if status in (
        "building",
        "submitting",
        "monitoring",
        "comparing",
        None,
    ):
        return "campaign"
    return "synthesize"
