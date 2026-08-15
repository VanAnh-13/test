"""Chuyển PlanResult latent thành cấu trúc phù hợp với graph và các patch state."""

from __future__ import annotations

from typing import Any

from hagent.core.types import Plan, PlanAction, PlanStatus
from hagent.core.types import PlanStep as TypedPlanStep
from hagent.world.schema import AutoMLAction, PlanResult, utc_now
from hagent.world.schema import PlanStep as SchemaPlanStep


def plan_result_to_entry(plan: PlanResult, *, status: str = "draft") -> dict[str, Any]:
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


def plan_result_to_typed_plan(
    plan: PlanResult,
    *,
    status: PlanStatus = PlanStatus.READY,
) -> Plan:
    """Chuyển PlanResult của World Model thành Plan Pydantic có kiểu."""
    typed_steps: list[TypedPlanStep] = []
    for s in plan.steps:
        action_dict = (
            s.action.to_dict() if hasattr(s.action, "to_dict") else dict(s.action)
        )
        typed_steps.append(
            TypedPlanStep(
                action=PlanAction(
                    type=str(action_dict.get("type", "")),
                    params=dict(action_dict.get("params") or {}),
                ),
                agent=s.agent,
            )
        )
    return Plan(
        plan_id=plan.plan_id,
        steps=typed_steps,
        title=plan.title,
        cost=plan.cost,
        score_estimate=plan.score_estimate,
        status=status,
        meta=dict(plan.meta),
    )


def typed_plan_to_result(plan: Plan) -> PlanResult:
    """Chuyển Plan Pydantic có kiểu thành PlanResult của World Model."""
    schema_steps: list[SchemaPlanStep] = []
    for s in plan.steps:
        schema_steps.append(
            SchemaPlanStep(
                action=AutoMLAction(
                    type=s.get_action_type(),
                    params=s.get_action_params(),
                ),
                agent=s.agent,
            )
        )
    return PlanResult(
        plan_id=plan.plan_id,
        steps=schema_steps,
        cost=plan.cost,
        score_estimate=plan.score_estimate,
        title=plan.title,
        meta=dict(plan.meta),
    )


def plan_results_to_state_update(
    plans: list[PlanResult],
    *,
    select_best: bool = True,
) -> dict[str, Any]:
    """Tạo patch AutoMLState và WorldState từ kết quả CEM-lite."""
    entries = {p.plan_id: plan_result_to_entry(p, status="draft") for p in plans}
    # Chỉ lưu dict plan có thể serialize thành JSON để phù hợp với state LangGraph.
    update: dict[str, Any] = {
        "plans": [p.to_dict() for p in plans],
        "plan_entries": entries,
    }
    if select_best and plans:
        best = plans[0]
        update["selected_plan"] = best.to_dict()
        update["active_plan_id"] = best.plan_id
        entries[best.plan_id]["status"] = "selected"
    return update


def selected_plan_actions(
    selected_plan: dict[str, Any] | PlanResult | Plan | None,
) -> list[dict[str, Any]]:
    """Làm phẳng các bước plan thành dict action cho bộ thực thi."""
    if selected_plan is None:
        return []
    if isinstance(selected_plan, Plan):
        return [s.model_dump() for s in selected_plan.steps]
    if isinstance(selected_plan, PlanResult):
        return [s.to_dict() for s in selected_plan.steps]
    steps = selected_plan.get("steps") or []
    return list(steps)
