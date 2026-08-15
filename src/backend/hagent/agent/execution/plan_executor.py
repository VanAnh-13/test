"""
Node thực thi lần lượt từng bước của selected_plan theo kiểu MPC.

Luồng mỗi lượt:
  kiểm tra → công cụ → cập nhật World Model → surprise → tiếp tục | sửa | hoàn tất
"""

from __future__ import annotations

import json
from typing import Any

import structlog

try:
    from langchain_core.messages import AIMessage, ToolMessage
except ImportError:  # pragma: no cover — tests without langchain

    class AIMessage:  # type: ignore[no-redef]
        def __init__(self, content: str = "", **kwargs):
            self.content = content
            self.type = "ai"

    class ToolMessage:  # type: ignore[no-redef]
        def __init__(self, content: str = "", name: str = "", tool_call_id: str = ""):
            self.content = content
            self.name = name
            self.tool_call_id = tool_call_id
            self.type = "tool"


from hagent.agent.constraints import validate_action
from hagent.agent.execution.tool_runner import enrich_params, invoke_tool
from hagent.agent.orchestration import AutoMLState
from hagent.world.schema import AutoMLAction, SurpriseResult, WorldState
from hagent.world.surprise import (
    compute_aggregate_plan_surprise,
    should_trigger_plan_revision,
)
from hagent.world.updater import apply_tool_output

logger = structlog.get_logger(__name__)


def _revise_on_high_surprise(action_type: str) -> bool:
    """Kiểm tra surprise cao sau action này có cần kích hoạt bộ sửa plan hay không."""
    try:
        from hagent.bridge.config import get_planning_config

        cfg = get_planning_config()
    except Exception:  # noqa: BLE001
        cfg = {}
    if not cfg.get("revise_on_high_surprise", True):
        return False
    ignore = cfg.get("surprise_ignore_actions") or [
        "start_training",
        "list_datasets",
        "list_jobs",
        "get_world_state",
    ]
    return action_type not in set(ignore)


def _steps_from_plan(plan: Any) -> list[Any]:
    if not plan:
        return []
    if hasattr(plan, "steps"):
        return list(plan.steps or [])
    if isinstance(plan, dict):
        return list(plan.get("steps") or [])
    return []


def _action_from_step(step: Any) -> AutoMLAction:
    if hasattr(step, "get_action_type") and hasattr(step, "get_action_params"):
        return AutoMLAction(
            type=step.get_action_type(),
            params=step.get_action_params(),
        )
    if hasattr(step, "action"):
        act = step.action
        if hasattr(act, "type"):
            return AutoMLAction(
                type=str(act.type),
                params=dict(getattr(act, "params", {}) or {}),
            )
        if isinstance(act, dict):
            return AutoMLAction(
                type=str(act.get("type") or ""),
                params=dict(act.get("params") or {}),
            )
    if isinstance(step, dict):
        act = step.get("action")
        if isinstance(act, dict):
            return AutoMLAction(
                type=str(act.get("type") or ""),
                params=dict(act.get("params") or {}),
            )
        if step.get("type"):
            return AutoMLAction(
                type=str(step.get("type")),
                params=dict(step.get("params") or {}),
            )
    return AutoMLAction(type="", params={})


def _observation_from_state(state: AutoMLState):
    from hagent.world.service import WorldModelService

    wm_service = state.get("_wm_service")
    if wm_service is None:
        wm_service = WorldModelService.from_config()
    snapshot = state.get("world_model") or {"user_id": state.get("user_id") or ""}
    goal = state.get("goal")
    return wm_service.observation_from_snapshot(
        snapshot, user_id=state.get("user_id"), goal=goal
    ), wm_service


def _merge_world_patch(state: AutoMLState, patch: dict[str, Any]) -> dict:
    snap = dict(state.get("world_model") or {"user_id": state.get("user_id") or ""})
    for k, v in patch.items():
        snap[k] = v
    return snap


def _append_event(state: AutoMLState, event: dict) -> list:
    events = list(state.get("execution_events") or [])
    events.append(event)
    return events


async def plan_executor_node(state: AutoMLState) -> dict:
    """
    Thực thi một bước plan. Graph lặp đến khi hoàn tất, cần sửa hoặc thất bại.
    """
    plan = state.get("selected_plan")
    steps = _steps_from_plan(plan)
    idx = int(state.get("plan_step_index") or 0)
    revision_count = int(state.get("revision_count") or 0)
    log = list(state.get("execution_log") or [])
    cost = dict(state.get("cost_metrics") or {})
    cost["steps_executed"] = int(cost.get("steps_executed") or 0)

    # Không có plan thì hoàn tất.
    if not steps:
        msg = AIMessage(content="Không có plan steps để thực thi.")
        return {
            "messages": [msg],
            "plan_status": "done",
            "current_phase": "respond",
            "execution_events": _append_event(
                state, {"type": "plan_empty", "status": "done"}
            ),
        }

    if idx >= len(steps):
        msg = AIMessage(
            content=f"Đã hoàn thành plan ({len(steps)} bước). revision={revision_count}."
        )
        return {
            "messages": [msg],
            "plan_status": "done",
            "current_phase": "respond",
            "plan_step_index": idx,
            "cost_metrics": cost,
            "execution_events": _append_event(
                state,
                {
                    "type": "plan_done",
                    "steps": len(steps),
                    "revision_count": revision_count,
                },
            ),
        }

    step = steps[idx]
    action = _action_from_step(step if isinstance(step, dict) else {})
    if not action.type:
        return {
            "messages": [AIMessage(content=f"Step {idx} thiếu action type.")],
            "plan_status": "need_revise",
            "last_step_error": f"step[{idx}] missing action type",
            "execution_events": _append_event(
                state, {"type": "step_invalid", "index": idx}
            ),
        }

    obs, wm_service = _observation_from_state(state)
    goal = state.get("goal") or obs.goal

    params = enrich_params(
        action.type,
        action.params,
        user_id=state.get("user_id"),
        user_token=state.get("user_token"),
        goal=goal if isinstance(goal, dict) else None,
        world_model=state.get("world_model"),
        action_id=(
            f"plan:{plan['plan_id']}:step:{idx}"
            if isinstance(plan, dict) and plan.get("plan_id")
            else None
        ),
    )
    if action.type == "start_training" and not params.get("idempotency_key"):
        return {
            "messages": [
                AIMessage(content="Training step thiếu action identity đáng tin cậy.")
            ],
            "plan_status": "need_revise",
            "last_step_error": "training action identity required",
            "execution_events": _append_event(
                state,
                {"type": "step_invalid", "index": idx},
            ),
        }
    persisted_params = {
        key: value
        for key, value in params.items()
        if key not in {"token", "idempotency_key"}
    }
    action = AutoMLAction(type=action.type, params=persisted_params)

    # Hard validate
    validation = validate_action(
        action, obs, goal=goal if isinstance(goal, dict) else None
    )
    events = _append_event(
        state,
        {
            "type": "step_start",
            "index": idx,
            "action": action.type,
            "params": persisted_params,
        },
    )

    if not validation.ok:
        logger.info("Step %s validate fail: %s", idx, validation.reasons)
        log.append(
            {
                "index": idx,
                "action": action.type,
                "status": "validate_fail",
                "reasons": validation.reasons,
            }
        )
        return {
            "messages": [
                AIMessage(
                    content=(
                        f"Bước {idx + 1}/{len(steps)} ({action.type}) "
                        f"không hợp lệ: {'; '.join(validation.reasons)}"
                    )
                )
            ],
            "plan_status": "need_revise",
            "last_step_error": "; ".join(validation.reasons),
            "plan_verification": validation.to_dict(),
            "execution_log": log,
            "execution_events": events
            + [{"type": "validate_fail", "index": idx, "reasons": validation.reasons}],
            "current_phase": "execute",
        }

    # Invoke tool
    payload = await invoke_tool(action.type, params)
    # Prefer JSON string content for consistency
    tool_msg = ToolMessage(
        content=json.dumps(payload, ensure_ascii=False, default=str),
        name=action.type,
        tool_call_id=f"plan-step-{idx}",
    )

    has_error = isinstance(payload, dict) and bool(payload.get("error"))
    cost["steps_executed"] = int(cost.get("steps_executed") or 0) + 1
    cost["tools_called"] = int(cost.get("tools_called") or 0) + 1

    # Cập nhật snapshot World Model từ kết quả công cụ.
    snap = state.get("world_model") or {"user_id": state.get("user_id") or ""}
    ws = WorldState.from_execution_snapshot(
        snap,
        user_id=state.get("user_id"),
    )
    new_snap = dict(snap)
    if isinstance(payload, dict) and not has_error:
        patch = apply_tool_output(ws, action.type, payload)
        for k, v in patch.items():
            if hasattr(ws, k):
                setattr(ws, k, v)
        new_snap = ws.to_dict()

    # Surprise via WM
    surprise_obj: SurpriseResult | None = None
    surprise_dict = None
    surprise_level = "low"
    try:
        next_obs = wm_service.observation_from_snapshot(
            new_snap,
            user_id=state.get("user_id"),
            goal=goal if isinstance(goal, dict) else None,
        )
        _, _, _, surprise_obj = await wm_service.update(obs, action, next_obs)
        surprise_dict = surprise_obj.to_dict()
        surprise_level = surprise_obj.level
        new_snap["last_surprise"] = surprise_dict
    except Exception as exc:  # noqa: BLE001
        logger.debug("Surprise update skipped: %s", exc)

    # Multi-scale aggregate plan surprise
    step_surprises = [
        e["surprise"]
        for e in events
        if isinstance(e, dict) and e.get("type") == "step_end" and e.get("surprise")
    ]
    if surprise_dict:
        step_surprises.append(surprise_dict)

    surprise_cfg = (
        wm_service.surprise_config if hasattr(wm_service, "surprise_config") else None
    )
    plan_surprise = compute_aggregate_plan_surprise(step_surprises, config=surprise_cfg)
    new_snap["last_plan_surprise"] = plan_surprise.to_dict()

    log.append(
        {
            "index": idx,
            "action": action.type,
            "status": "error" if has_error else "ok",
            "error": payload.get("error") if has_error else None,
            "surprise": surprise_level,
            "plan_surprise": plan_surprise.level,
        }
    )
    events.append(
        {
            "type": "step_end",
            "index": idx,
            "action": action.type,
            "ok": not has_error,
            "surprise": surprise_dict,
            "plan_surprise": plan_surprise.to_dict(),
            "error": payload.get("error") if has_error else None,
        }
    )

    ai_summary = (
        f"Bước {idx + 1}/{len(steps)}: {action.type} "
        + ("❌ " + str(payload.get("error")) if has_error else "✅")
        + (f" | surprise={surprise_level}" if surprise_dict else "")
    )

    if has_error:
        return {
            "messages": [AIMessage(content=ai_summary), tool_msg],
            "plan_status": "need_revise",
            "last_step_error": str(payload.get("error")),
            "world_model": new_snap,
            "surprise": surprise_dict,
            "execution_log": log,
            "execution_events": events,
            "cost_metrics": cost,
            "current_phase": "execute",
            "plan_step_index": idx,
        }

    # Surprise cao theo bước hoặc toàn plan có thể kích hoạt sửa plan.
    needs_revision = should_trigger_plan_revision(
        surprise_obj, plan_surprise, config=surprise_cfg
    )
    if needs_revision and _revise_on_high_surprise(action.type):
        reason = (
            f"high surprise after {action.type}"
            if surprise_level == "high"
            else f"high cumulative plan surprise ({plan_surprise.value:.2f})"
        )
        return {
            "messages": [AIMessage(content=ai_summary + " (surprise cao → revise)")],
            "plan_status": "need_revise",
            "last_step_error": reason,
            "world_model": new_snap,
            "surprise": surprise_dict,
            "plan_surprise": plan_surprise.to_dict(),
            "execution_log": log,
            "execution_events": events,
            "cost_metrics": cost,
            "current_phase": "execute",
            "plan_step_index": idx + 1,  # advance past surprising step
            "latent": None,
        }

    # Thành công thì chuyển sang bước tiếp theo.
    next_idx = idx + 1
    done = next_idx >= len(steps)
    return {
        "messages": [AIMessage(content=ai_summary), tool_msg],
        "plan_status": "done" if done else "executing",
        "plan_step_index": next_idx,
        "world_model": new_snap,
        "surprise": surprise_dict,
        "execution_log": log,
        "execution_events": events,
        "cost_metrics": cost,
        "current_phase": "respond" if done else "execute",
        "last_step_error": None,
    }


def plan_executor_route(state: AutoMLState) -> str:
    """Sau khi thực thi: tiếp tục, sửa plan hoặc tổng hợp."""
    status = state.get("plan_status") or ""
    if status == "need_revise":
        return "reviser"
    if status in ("done", "failed", "aborted"):
        return "synthesize"
    if status in ("executing", "ready"):
        return "plan_executor"
    # Fallback
    steps = _steps_from_plan(state.get("selected_plan"))
    idx = int(state.get("plan_step_index") or 0)
    if steps and idx < len(steps):
        return "plan_executor"
    return "synthesize"
