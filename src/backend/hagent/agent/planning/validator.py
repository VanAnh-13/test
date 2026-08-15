"""
Pipeline kiểm tra plan (REFAC-020).

Kiểm tra tính hợp lệ của kế hoạch trước khi thực thi:
1. Kiểm tra schema của Plan, PlanStep và PlanAction.
2. Kiểm tra công cụ đã được đăng ký trong ToolRegistry.
3. Kiểm tra ràng buộc tài nguyên và khả năng tương thích với World Model.
"""

from __future__ import annotations

from typing import Any

import structlog

from hagent.agent.constraints.validator import ValidationResult, validate_plan_steps
from hagent.core.errors import PlanningError
from hagent.world.schema import AutoMLObservation, GoalSpec

logger = structlog.get_logger(__name__)


class PlanValidator:
    """Validator toàn diện cho kế hoạch hành động AutoML."""

    def __init__(self, allowed_tools: set[str] | None = None) -> None:
        self._allowed_tools = set(allowed_tools) if allowed_tools is not None else None

    def _get_registered_tool_names(self) -> set[str]:
        """Lấy danh sách công cụ khả dụng từ ToolRegistry."""
        if self._allowed_tools is not None:
            return self._allowed_tools
        try:
            from hagent.agent.orchestration.registry import get_tool_map

            return set(get_tool_map().keys())
        except Exception:  # noqa: BLE001
            # Dùng danh sách công cụ tích hợp sẵn làm phương án dự phòng.
            return {
                "list_datasets",
                "get_dataset_info",
                "get_features",
                "preview_data",
                "get_available_models",
                "get_metrics",
                "start_training",
                "get_job_info",
                "list_jobs",
                "cancel_job",
                "predict_batch",
                "check_system_health",
                "get_world_state",
            }

    def validate(
        self,
        plan: Any,
        observation: AutoMLObservation | None = None,
        *,
        goal: GoalSpec | None = None,
        raise_on_error: bool = False,
    ) -> ValidationResult:
        """
        Kiểm tra kế hoạch qua 3 tầng xác thực:
        1. Schema: Plan có danh sách bước hợp lệ.
        2. Công cụ: Mọi công cụ trong các bước đều tồn tại trong registry.
        3. Ràng buộc và World Model: Tập dữ liệu, mục tiêu và ngân sách thời gian hợp lệ.
        """
        reasons: list[str] = []

        # 1. Trích xuất các bước từ model Plan hoặc dict.
        steps: list[Any] = []
        if hasattr(plan, "steps"):
            steps = list(plan.steps or [])
        elif isinstance(plan, dict):
            steps = list(plan.get("steps") or [])
        elif isinstance(plan, list):
            steps = plan
        else:
            reasons.append(
                f"Invalid plan format: expected Plan or dict, got {type(plan).__name__}"
            )
            if raise_on_error:
                raise PlanningError(
                    f"Plan validation failed: {'; '.join(reasons)}",
                    context={"reasons": reasons},
                )
            return ValidationResult(ok=False, reasons=reasons)

        if not steps:
            reasons.append("Plan must contain at least one step")
            if raise_on_error:
                raise PlanningError(
                    f"Plan validation failed: {'; '.join(reasons)}",
                    context={"reasons": reasons},
                )
            return ValidationResult(ok=False, reasons=reasons)

        # 2. Tool Availability Check
        registered_tools = self._get_registered_tool_names()
        for idx, step in enumerate(steps):
            action_type = ""
            if hasattr(step, "action"):
                action = step.action
                action_type = getattr(action, "type", "") or getattr(action, "tool", "")
            elif isinstance(step, dict):
                act = step.get("action") or step
                if isinstance(act, dict):
                    action_type = act.get("type") or act.get("tool") or ""
                else:
                    action_type = getattr(act, "type", "") or getattr(act, "tool", "")

            if not action_type:
                reasons.append(f"Step[{idx}] is missing action type/tool")
            elif action_type not in registered_tools:
                reasons.append(f"Step[{idx}] uses unregistered tool '{action_type}'")

        # 3. Constraints & World Model Compatibility Check
        obs = observation or AutoMLObservation(user_id="default_user")
        constraint_result = validate_plan_steps(steps, obs, goal=goal)
        if not constraint_result.ok:
            reasons.extend(constraint_result.reasons)

        is_ok = len(reasons) == 0
        if not is_ok and raise_on_error:
            logger.warning("Plan validation failed with reasons: %s", reasons)
            raise PlanningError(
                f"Plan validation failed: {'; '.join(reasons)}",
                context={"reasons": reasons},
            )

        return ValidationResult(ok=is_ok, reasons=reasons)
