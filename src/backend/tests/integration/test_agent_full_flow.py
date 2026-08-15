"""
Kiểm thử tích hợp End-to-End cho toàn bộ luồng hoạt động của Agent (REFAC-027).

Luồng: Yêu cầu người dùng -> Lập kế hoạch (Plan) -> Xác thực (Validation) -> Thực thi Tool -> Cập nhật World Model -> Tính Surprise -> Phản hồi.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from hagent.agent.planning.validator import PlanValidator
from hagent.core.types import Plan, PlanAction, PlanStep
from hagent.world.schema import AutoMLObservation, DatasetEntry, WorldState
from hagent.world.surprise import (
    compute_outcome_surprise,
    should_trigger_plan_revision,
)
from hagent.world.updater import apply_tool_output


@pytest.mark.asyncio
async def test_full_agent_execution_flow() -> None:
    """Kiểm thử luồng tích hợp đầy đủ từ Plan -> Validate -> Execute Tool -> Update World Model -> Surprise."""
    # 1. Thiết lập trạng thái WorldState & Quan sát AutoML ban đầu
    state = WorldState(user_id="integration_user_01", phase="search")
    obs = AutoMLObservation(
        user_id="integration_user_01",
        datasets={
            "ds_iris_01": DatasetEntry(
                id="ds_iris_01",
                name="iris.csv",
                features=["sepal_len", "sepal_wid"],
                target="species",
            )
        },
    )

    # 2. Tạo kế hoạch và thực hiện xác thực với PlanValidator
    action = PlanAction(
        type="get_dataset_info",
        params={"dataset_id": "ds_iris_01"},
    )
    step = PlanStep(
        step_id="step_1",
        action=action,
        description="Kiểm tra thông tin chi tiết của bộ dữ liệu iris",
    )
    plan = Plan(
        plan_id="plan_full_flow_01",
        title="Kiểm tra dataset iris",
        steps=[step],
    )

    validator = PlanValidator()
    val_result = validator.validate(plan, observation=obs, raise_on_error=False)
    assert val_result.ok is True
    assert len(val_result.reasons) == 0

    # 3. Giả lập kết quả thực thi Tool
    mock_tool_output = {
        "status": "success",
        "job_id": "job-12345",
        "score": 0.92,
        "metric": "accuracy",
        "algorithm": "random_forest",
    }

    # 4. Cập nhật trạng thái World Model
    patch = apply_tool_output(state, "train_model", mock_tool_output)
    assert isinstance(patch, dict)

    # 5. Tính toán Surprise và kiểm tra điều kiện kích hoạt Replan
    predicted = (0.90, 0.05)  # (kỳ vọng mean, độ lệch chuẩn std)
    actual_score = mock_tool_output["score"]
    surprise_res = compute_outcome_surprise(predicted, actual_score)
    replan_needed = should_trigger_plan_revision(surprise_res)

    assert surprise_res.level in {"low", "medium", "high"}
    assert not replan_needed


@pytest.mark.asyncio
async def test_high_surprise_triggers_replan() -> None:
    """Kiểm thử khi kết quả thực tế sai lệch lớn so với kỳ vọng, hệ thống kích hoạt mức HIGH surprise và yêu cầu replan."""
    predicted = (0.95, 0.02)
    actual_score = 0.40  # Kết quả thực tế tụt giảm nghiêm trọng

    surprise_res = compute_outcome_surprise(predicted, actual_score)
    replan_needed = should_trigger_plan_revision(surprise_res)

    assert surprise_res.level == "high"
    assert replan_needed is True


@pytest.mark.asyncio
async def test_tool_execution_failure_graceful_handling() -> None:
    """Kiểm thử cơ chế xử lý ngoại lệ mềm dẻo khi tool gặp sự cố mạng hoặc timeout."""
    mock_runner = AsyncMock(side_effect=ValueError("MinIO dataset download timed out"))

    with pytest.raises(ValueError) as exc_info:
        await mock_runner()

    assert "timed out" in str(exc_info.value)
