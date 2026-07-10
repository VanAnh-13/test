"""Adapt latent PlanResult → graph-friendly structures + state patches."""

from __future__ import annotations

from typing import Any, Dict, List

from hagent.world.schema import PlanResult, utc_now


def plan_result_to_entry(plan: PlanResult, *, status: str = "draft") -> Dict[str, Any]:
    return {
        "plan_id": plan.plan_id,
        "title": plan.title,
        "status": status,
        "steps": [s.to_dict() for s in plan.steps],
        "score_estimate": plan.score_estimate,
        "cost": plan.cost,
        "meta": dict(plan.meta),
        "created_at": utc_now().isoformat(),
        "updated_at": utc_now().isoformat(),
    }


def plan_results_to_state_update(
    plans: List[PlanResult],
    *,
    select_best: bool = True,
) -> Dict[str, Any]:
    """Build AutoMLState / WorldState patch from CEM-lite outputs."""
    entries = {p.plan_id: plan_result_to_entry(p, status="draft") for p in plans}
    # Store JSON-serializable plan dicts only (LangGraph state friendly)
    update: Dict[str, Any] = {
        "plans": [p.to_dict() for p in plans],
        "plan_entries": entries,
    }
    if select_best and plans:
        best = plans[0]
        update["selected_plan"] = best.to_dict()
        update["active_plan_id"] = best.plan_id
        entries[best.plan_id]["status"] = "selected"
    return update


def selected_plan_actions(selected_plan: Dict[str, Any] | PlanResult | None) -> List[Dict[str, Any]]:
    """Flatten plan steps to action dicts for executor."""
    if selected_plan is None:
        return []
    if isinstance(selected_plan, PlanResult):
        return [s.to_dict() for s in selected_plan.steps]
    steps = selected_plan.get("steps") or []
    return list(steps)
