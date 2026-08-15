"""
Node sửa hoặc lập lại plan khi kiểm tra thất bại, công cụ lỗi hay surprise cao.
"""

from __future__ import annotations

import structlog

try:
    from langchain_core.messages import AIMessage
except ImportError:  # pragma: no cover

    class AIMessage:  # type: ignore[no-redef]
        def __init__(self, content: str = "", **kwargs):
            self.content = content
            self.type = "ai"


from hagent.agent.orchestration import AutoMLState
from hagent.agent.planning.plan_adapter import plan_results_to_state_update

logger = structlog.get_logger(__name__)


def _max_revisions() -> int:
    try:
        from hagent.bridge.config import get_planning_config

        return int(get_planning_config().get("max_revisions", 2))
    except Exception as exc:  # noqa: BLE001 - cấu hình tùy chọn giữ giá trị dự phòng
        logger.debug(
            "Không đọc được cấu hình max_revisions", error_type=type(exc).__name__
        )
        return 2


def _patch_plan_for_error(
    plan: dict,
    error: str,
    goal: dict | None,
) -> dict | None:
    """
    Các patch xác định không dùng LLM:
    - thiếu tập dữ liệu thì thêm list_datasets hoặc get_dataset_info vào đầu;
    - mục tiêu sai thì bỏ target của start_training và thêm get_features vào đầu;
    - thiếu thông tin job thì thêm list_jobs vào đầu.
    """
    if not plan:
        return None
    steps = list(plan.get("steps") or [])
    err = (error or "").lower()
    new_steps = list(steps)

    def _has(action_type: str) -> bool:
        for s in new_steps:
            act = (s.get("action") if isinstance(s, dict) else None) or {}
            if isinstance(act, dict) and act.get("type") == action_type:
                return True
            if isinstance(s, dict) and s.get("type") == action_type:
                return True
        return False

    def _prepend(action_type: str, params: dict | None = None) -> None:
        new_steps.insert(
            0,
            {
                "action": {"type": action_type, "params": dict(params or {})},
                "agent": None,
            },
        )

    if "dataset_id" in err or "not in world model" in err:
        if not _has("list_datasets"):
            _prepend("list_datasets")
        if not _has("get_dataset_info"):
            ds = (goal or {}).get("dataset_id")
            _prepend("get_dataset_info", {"dataset_id": ds} if ds else {})

    if ("target_column" in err or "features" in err) and not _has("get_features"):
        ds = (goal or {}).get("dataset_id")
        _prepend("get_features", {"dataset_id": ds} if ds else {})

    if "job_id" in err and not _has("list_jobs"):
        _prepend("list_jobs")

    # Sắp xếp mềm để bảo đảm phân tích trước khi huấn luyện.
    if (
        "high surprise" in err
        and _has("start_training")
        and not _has("get_dataset_info")
    ):
        _prepend("get_dataset_info")

    if new_steps == steps:
        return None

    revised = dict(plan)
    revised["steps"] = new_steps
    revised["title"] = (plan.get("title") or "plan") + " [revised]"
    revised["status"] = "revised"
    return revised


async def reviser_node(state: AutoMLState) -> dict:
    """Sửa plan hoặc dừng khi đã dùng hết ngân sách sửa đổi."""
    max_rev = _max_revisions()
    count = int(state.get("revision_count") or 0) + 1
    error = state.get("last_step_error") or "unknown error"
    goal = state.get("goal") if isinstance(state.get("goal"), dict) else {}
    plan = (
        state.get("selected_plan")
        if isinstance(state.get("selected_plan"), dict)
        else {}
    )
    events = list(state.get("execution_events") or [])
    cost = dict(state.get("cost_metrics") or {})
    cost["revisions"] = count

    events.append(
        {
            "type": "revise_start",
            "revision": count,
            "max_revisions": max_rev,
            "error": error,
        }
    )

    if count > max_rev:
        msg = (
            f"Đã hết ngân sách revise ({max_rev}). "
            f"Lỗi cuối: {error}. Dừng thực thi plan."
        )
        events.append({"type": "revise_abort", "revision": count, "error": error})
        return {
            "messages": [AIMessage(content=msg)],
            "revision_count": count,
            "plan_status": "failed",
            "current_phase": "respond",
            "execution_events": events,
            "cost_metrics": cost,
            "plan_verification": {
                "pass": False,
                "ok": False,
                "reasons": [error, f"max_revisions={max_rev}"],
            },
        }

    # 1. Áp dụng patch xác định.
    patched = _patch_plan_for_error(plan, error, goal)

    # 2. Nếu patch chưa đủ, chạy lại CEM-lite với gợi ý lỗi trong mô tả goal.
    if patched is None:
        try:
            from hagent.world.service import WorldModelService

            wm = state.get("_wm_service") or WorldModelService.from_config()
            snap = state.get("world_model") or {"user_id": state.get("user_id") or ""}
            g = dict(goal or {})
            g["description"] = (
                str(g.get("description") or "") + f" | previous_failure: {error}"
            )
            # Buộc goal không đi theo nhánh phản hồi trực tiếp.
            if g.get("goal_type") in (None, "respond"):
                g["goal_type"] = "train"
            obs = wm.observation_from_snapshot(
                snap, user_id=state.get("user_id"), goal=g
            )
            plans = wm.plan(obs, g)
            if plans:
                update = plan_results_to_state_update(plans, select_best=True)
                events.append(
                    {
                        "type": "revise_replan",
                        "revision": count,
                        "new_plan_id": update.get("active_plan_id"),
                        "title": (update.get("selected_plan") or {}).get("title"),
                    }
                )
                return {
                    "messages": [
                        AIMessage(
                            content=(
                                f"Revise #{count}/{max_rev}: lập plan mới sau lỗi "
                                f"`{error}`."
                            )
                        )
                    ],
                    "revision_count": count,
                    "plan_status": "ready",
                    "plan_step_index": 0,
                    "last_step_error": None,
                    "goal": g,
                    "execution_events": events,
                    "cost_metrics": cost,
                    **{k: v for k, v in update.items() if k != "plan_entries"},
                }
        except Exception as exc:  # noqa: BLE001 - lỗi lập lại plan chuyển sang patch cũ
            logger.warning("CEM replan failed: %s", exc)
            patched = plan  # tiếp tục bằng plan hiện tại

    if patched is None:
        patched = plan

    events.append(
        {
            "type": "revise_patch",
            "revision": count,
            "steps": [
                (s.get("action") or {}).get("type") if isinstance(s, dict) else None
                for s in (patched.get("steps") or [])
            ],
        }
    )

    return {
        "messages": [
            AIMessage(
                content=(f"Revise #{count}/{max_rev}: đã chỉnh plan sau lỗi `{error}`.")
            )
        ],
        "selected_plan": patched,
        "revision_count": count,
        "plan_status": "ready",
        "plan_step_index": 0,
        "last_step_error": None,
        "execution_events": events,
        "cost_metrics": cost,
        "current_phase": "execute",
    }


def reviser_route(state: AutoMLState) -> str:
    status = state.get("plan_status")
    if status == "failed":
        return "synthesize"
    if status in ("ready", "executing"):
        return "plan_executor"
    return "synthesize"
