"""
Unit tests for Plan Validation Pipeline (REFAC-020).
"""

from __future__ import annotations

import pytest

from hagent.agent.planning.validator import PlanValidator
from hagent.core.errors import PlanningError
from hagent.core.types import Plan, PlanAction, PlanStep
from hagent.world.schema import AutoMLObservation, DatasetEntry


def test_valid_plan_passes_validation() -> None:
    """Kế hoạch hợp lệ với các tool chuẩn được thông qua."""
    validator = PlanValidator()
    plan = Plan(
        plan_id="plan_valid_01",
        title="Valid Test Plan",
        cost=1.0,
        steps=[
            PlanStep(
                step_id="step_1",
                action=PlanAction(
                    type="list_datasets",
                    params={"user_id": "test_user"},
                ),
            ),
            PlanStep(
                step_id="step_2",
                action=PlanAction(
                    type="get_dataset_info",
                    params={"dataset_id": "ds_1"},
                ),
            ),
        ],
    )

    obs = AutoMLObservation(
        user_id="test_user",
        datasets={
            "ds_1": DatasetEntry(
                id="ds_1", name="Iris", features=["f1", "f2"], target="target"
            )
        },
    )

    result = validator.validate(plan, observation=obs, raise_on_error=False)
    assert result.ok is True
    assert len(result.reasons) == 0


def test_empty_plan_fails_validation() -> None:
    """Kế hoạch rỗng không chứa bước nào bị từ chối và raise PlanningError."""
    validator = PlanValidator()
    plan = Plan(plan_id="plan_empty", title="Empty Plan", cost=0.0, steps=[])

    result = validator.validate(plan, raise_on_error=False)
    assert result.ok is False
    assert any("at least one step" in r for r in result.reasons)

    with pytest.raises(PlanningError) as exc_info:
        validator.validate(plan, raise_on_error=True)
    assert "at least one step" in str(exc_info.value)


def test_unregistered_tool_fails_validation() -> None:
    """Kế hoạch chứa tool không được đăng ký trong registry bị từ chối."""
    validator = PlanValidator()
    plan = {
        "plan_id": "plan_bad_tool",
        "title": "Bad Tool Plan",
        "steps": [
            {
                "step_id": "step_1",
                "action": {
                    "type": "malicious_unregistered_tool",
                    "params": {"cmd": "rm -rf /"},
                },
            }
        ],
    }

    result = validator.validate(plan, raise_on_error=False)
    assert result.ok is False
    assert any("unregistered tool" in r for r in result.reasons)

    with pytest.raises(PlanningError):
        validator.validate(plan, raise_on_error=True)


def test_constraint_violation_fails_validation() -> None:
    """Kế hoạch vi phạm ràng buộc (time_limit âm, target không có trong dataset) bị bắt lỗi."""
    validator = PlanValidator()
    obs = AutoMLObservation(
        user_id="test_user",
        datasets={
            "ds_test": DatasetEntry(
                id="ds_test", name="Data", features=["feat1", "feat2"], target="label"
            )
        },
    )

    plan = Plan(
        plan_id="plan_bad_constraints",
        title="Bad Constraints Plan",
        cost=2.0,
        steps=[
            PlanStep(
                step_id="step_1",
                action=PlanAction(
                    type="start_training",
                    params={
                        "dataset_id": "ds_test",
                        "target_column": "non_existent_column",
                        "problem_type": "classification",
                        "time_limit": -100,  # Negative time limit
                    },
                ),
            )
        ],
    )

    result = validator.validate(plan, observation=obs, raise_on_error=False)
    assert result.ok is False
    assert any("time_limit must be positive" in r for r in result.reasons)
    assert any("not in dataset features" in r for r in result.reasons)

    with pytest.raises(PlanningError):
        validator.validate(plan, observation=obs, raise_on_error=True)
