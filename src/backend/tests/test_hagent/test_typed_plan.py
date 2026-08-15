"""
Unit tests for typed Plan and PlanStep Pydantic models (REFAC-015).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hagent.agent.execution.plan_executor import _action_from_step, _steps_from_plan
from hagent.agent.planning.plan_adapter import (
    plan_result_to_typed_plan,
    selected_plan_actions,
    typed_plan_to_result,
)
from hagent.core.types import Plan, PlanAction, PlanStatus, PlanStep
from hagent.world.schema import AutoMLAction, PlanResult
from hagent.world.schema import PlanStep as SchemaPlanStep


def test_plan_pydantic_validation() -> None:
    """Kiểm tra validation cấp Pydantic cho Plan và PlanStep."""
    # 1. Hợp lệ
    action = PlanAction(type="start_training", params={"dataset_id": "ds1"})
    step = PlanStep(action=action, agent="training_monitor")
    plan = Plan(
        plan_id="plan_123",
        steps=[step],
        title="Training Plan",
        cost=1.5,
        status=PlanStatus.READY,
    )
    assert plan.plan_id == "plan_123"
    assert len(plan.steps) == 1
    assert step.get_action_type() == "start_training"
    assert step.get_action_params() == {"dataset_id": "ds1"}

    # 2. Plan với plan_id rỗng -> ném ValidationError
    with pytest.raises(ValidationError):
        Plan(plan_id="", steps=[step])

    # 3. PlanAction với type rỗng -> ném ValidationError
    with pytest.raises(ValidationError):
        PlanAction(type="")

    # 4. PlanStep với action dict thiếu type -> ném ValidationError
    with pytest.raises(ValidationError):
        PlanStep(action={"params": {}})


def test_plan_adapter_conversions() -> None:
    """Chuyển đổi hai chiều giữa Schema PlanResult và Pydantic Plan."""
    schema_step = SchemaPlanStep(
        action=AutoMLAction(type="get_dataset_info", params={"dataset_id": "d1"}),
        agent="data_analyst",
    )
    res = PlanResult(
        plan_id="plan_xyz",
        steps=[schema_step],
        cost=0.5,
        score_estimate=0.88,
        title="Analysis",
        meta={"source": "cem_lite"},
    )

    # 1. PlanResult -> typed Plan
    typed_p = plan_result_to_typed_plan(res, status=PlanStatus.EXECUTING)
    assert isinstance(typed_p, Plan)
    assert typed_p.plan_id == "plan_xyz"
    assert typed_p.status == PlanStatus.EXECUTING
    assert len(typed_p.steps) == 1
    assert typed_p.steps[0].get_action_type() == "get_dataset_info"
    assert typed_p.steps[0].get_action_params() == {"dataset_id": "d1"}

    # 2. typed Plan -> PlanResult
    res_back = typed_plan_to_result(typed_p)
    assert isinstance(res_back, PlanResult)
    assert res_back.plan_id == "plan_xyz"
    assert len(res_back.steps) == 1
    assert res_back.steps[0].action.type == "get_dataset_info"

    # 3. selected_plan_actions
    actions = selected_plan_actions(typed_p)
    assert len(actions) == 1


def test_plan_executor_supports_typed_plan() -> None:
    """plan_executor trích xuất steps và actions từ Typed Plan một cách mượt mà."""
    action = PlanAction(type="preview_data", params={"dataset_id": "ds_preview"})
    step = PlanStep(action=action, agent="data_analyst")
    plan = Plan(plan_id="plan_exec", steps=[step])

    # _steps_from_plan
    steps = _steps_from_plan(plan)
    assert len(steps) == 1
    assert steps[0] == step

    # _action_from_step
    act = _action_from_step(step)
    assert isinstance(act, AutoMLAction)
    assert act.type == "preview_data"
    assert act.params == {"dataset_id": "ds_preview"}
