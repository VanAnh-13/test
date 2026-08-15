"""
Node điều khiển hierarchy đang hoạt động.

Mỗi tick của graph:
  1. bảo đảm hierarchy tồn tại và áp dụng bỏ qua thông minh;
  2. chạy hoặc tiếp tục leaf hiện tại (phân tích, campaign hoặc đánh giá);
  3. chuyển sang leaf tiếp theo khi leaf hiện tại hoàn tất;
  4. lặp đến khi hierarchy hoàn tất rồi tổng hợp kết quả.
"""

from __future__ import annotations

from typing import Any

import structlog

try:
    from langchain_core.messages import AIMessage
except ImportError:  # pragma: no cover

    class AIMessage:  # type: ignore[no-redef]
        def __init__(self, content: str = "", **kwargs):
            self.content = content
            self.type = "ai"


from hagent.agent.campaign.runner import campaign_step, ensure_campaign
from hagent.agent.campaign.settings import max_monitor_ticks
from hagent.agent.execution.tool_runner import enrich_params, invoke_tool
from hagent.agent.orchestration import AutoMLState
from hagent.agent.planning.hierarchy import (
    apply_smart_skips,
    ensure_hierarchy,
    subgoal_as_goal,
)
from hagent.world.schema import WorldState
from hagent.world.updater import apply_tool_output

logger = structlog.get_logger(__name__)


def _append_events(state: AutoMLState, *events: dict) -> list:
    out = list(state.get("execution_events") or [])
    out.extend(events)
    return out


def _merge_tool_into_wm(state: AutoMLState, tool_name: str, payload: dict) -> dict:
    snap = dict(state.get("world_model") or {"user_id": state.get("user_id") or ""})
    if not isinstance(payload, dict) or payload.get("error"):
        return snap
    ws = WorldState.from_execution_snapshot(
        snap,
        user_id=state.get("user_id"),
    )
    patch = apply_tool_output(ws, tool_name, payload)
    for k, v in patch.items():
        if hasattr(ws, k):
            setattr(ws, k, v)
    return ws.to_dict()


async def _wm_leaf_update(
    state: AutoMLState,
    *,
    before_wm: dict,
    after_wm: dict,
    action_type: str,
    params: dict | None = None,
    goal: dict | None = None,
) -> tuple[dict | None, dict]:
    """Tính surprise LeWM tùy chọn sau khi công cụ của hierarchy leaf chạy."""
    try:
        from hagent.agent.campaign.wm_hooks import campaign_wm_step

        surprise, next_wm = await campaign_wm_step(
            wm_service=state.get("_wm_service"),
            world_model=before_wm,
            user_id=state.get("user_id"),
            action_type=action_type,
            params=params or {},
            goal=goal,
            next_world_model=after_wm,
        )
        return surprise, next_wm or after_wm
    except Exception as exc:  # noqa: BLE001 - boundary World Model tùy chọn không được chặn luồng chính
        logger.debug("hierarchy WM update: %s", exc)
        return None, after_wm


async def _run_analyze_leaf(state: AutoMLState, leaf_goal: dict) -> dict[str, Any]:
    user_id = state.get("user_id")
    token = state.get("user_token")
    wm = state.get("world_model")
    tools_used = []
    new_wm = dict(wm or {})
    last_surprise = None

    for action_type in ("get_dataset_info", "get_features"):
        before = dict(new_wm)
        params = enrich_params(
            action_type,
            {},
            user_id=user_id,
            user_token=token,
            goal=leaf_goal,
            world_model=new_wm,
        )
        payload = await invoke_tool(action_type, params)
        tools_used.append(action_type)
        if isinstance(payload, dict):
            new_wm = _merge_tool_into_wm(
                {**state, "world_model": new_wm}, action_type, payload
            )
            last_surprise, new_wm = await _wm_leaf_update(
                state,
                before_wm=before,
                after_wm=new_wm,
                action_type=action_type,
                params=params,
                goal=leaf_goal,
            )

    out = {
        "done": True,
        "world_model": new_wm,
        "summary": f"analyze via {', '.join(tools_used)}",
        "tools": tools_used,
    }
    if last_surprise:
        out["surprise"] = last_surprise
    return out


async def _run_select_leaf(state: AutoMLState, leaf_goal: dict) -> dict[str, Any]:
    ptype = leaf_goal.get("problem_type") or "classification"
    payload = await invoke_tool("get_available_models", {"problem_type": ptype})
    metrics = await invoke_tool("get_metrics", {"problem_type": ptype})
    return {
        "done": True,
        "summary": f"select models/metrics for {ptype}",
        "tools": ["get_available_models", "get_metrics"],
        "select_payload": {
            "models": payload if isinstance(payload, dict) else {},
            "metrics": metrics if isinstance(metrics, dict) else {},
        },
    }


async def _run_monitor_leaf(state: AutoMLState, leaf_goal: dict) -> dict[str, Any]:
    user_id = state.get("user_id")
    params = enrich_params(
        "list_jobs",
        {},
        user_id=user_id,
        user_token=state.get("user_token"),
        goal=leaf_goal,
        world_model=state.get("world_model"),
    )
    payload = await invoke_tool("list_jobs", params)
    new_wm = state.get("world_model")
    if isinstance(payload, dict):
        new_wm = _merge_tool_into_wm(state, "list_jobs", payload)
    return {
        "done": True,
        "world_model": new_wm,
        "summary": "listed jobs",
        "tools": ["list_jobs"],
    }


async def _run_evaluate_leaf(state: AutoMLState, leaf_goal: dict) -> dict[str, Any]:
    # Ưu tiên kết quả đánh giá campaign đã có.
    evaluation = state.get("evaluation")
    if evaluation and evaluation.get("best_job_id"):
        return {
            "done": True,
            "summary": f"best_job={evaluation.get('best_job_id')}",
            "evaluation": evaluation,
            "tools": [],
        }

    # Tạo bảng so sánh gọn từ các job trong World Model.
    jobs = (state.get("world_model") or {}).get("jobs") or {}
    rows = []
    for jid, j in jobs.items():
        if str(j.get("status", "")).lower() not in ("completed", "done", "success"):
            continue
        rows.append(
            {
                "job_id": jid,
                "best_model": j.get("best_model"),
                "best_score": j.get("best_score"),
                "status": j.get("status"),
            }
        )
    rows.sort(
        key=lambda r: (
            float(r["best_score"]) if r.get("best_score") is not None else -1.0
        ),
        reverse=True,
    )
    evaluation = {
        "job_ids": [r["job_id"] for r in rows],
        "comparison_table": rows,
        "best_job_id": rows[0]["job_id"] if rows else None,
        "recommendation": rows[0].get("best_model") if rows else None,
    }
    return {
        "done": True,
        "summary": f"evaluated {len(rows)} jobs",
        "evaluation": evaluation,
        "tools": [],
    }


def _train_active(state: AutoMLState) -> bool:
    """Kiểm tra campaign của leaf huấn luyện có đang chạy hay không, kể cả khóa cũ."""
    return bool(
        state.get("hierarchy_train_active") or state.get("_hierarchy_train_active")
    )


def _max_hierarchy_ticks() -> int:
    try:
        from hagent.bridge.config import get_hierarchy_config

        return int(get_hierarchy_config().get("max_ticks", 40))
    except Exception:  # noqa: BLE001 - boundary cấu hình giữ giá trị dự phòng cũ
        return 40


async def _run_train_leaf(state: AutoMLState, leaf_goal: dict) -> dict[str, Any]:
    """Điều khiển các tick campaign đến khi leaf huấn luyện hoàn tất."""
    # Tách riêng mục tiêu của leaf cho campaign.
    leaf_state = {
        **state,
        "goal": leaf_goal,
        # Đặt lại campaign đã xong khi bắt đầu một leaf huấn luyện mới.
    }
    cstatus = state.get("campaign_status")
    # Nếu campaign của root trước đã xong thì khởi tạo lại cho leaf này.
    if cstatus in ("done", "failed") and not _train_active(state):
        leaf_state = {**leaf_state, "campaign": None, "campaign_status": None}

    # Giới hạn an toàn trong leaf huấn luyện; nhánh hierarchy cần giới hạn riêng.
    train_ticks = int(state.get("campaign_tick") or 0) + 1
    surp_events: list = []
    campaign = await ensure_campaign(leaf_state)
    campaign = await campaign_step(
        campaign,
        user_id=state.get("user_id"),
        user_token=state.get("user_token"),
        world_model=state.get("world_model"),
        wm_service=state.get("_wm_service"),
        surprise_events=surp_events,
    )

    max_monitor = max_monitor_ticks()

    if campaign.status == "monitoring" and train_ticks >= max_monitor:
        for v in campaign.variants:
            if v.status in ("pending", "submitted", "running"):
                v.status = "failed"
                v.error = (
                    v.error or f"hierarchy train timeout after {max_monitor} ticks"
                )
        more: list = []
        campaign = await campaign_step(
            campaign,
            user_id=state.get("user_id"),
            user_token=state.get("user_token"),
            world_model=state.get("world_model"),
            wm_service=state.get("_wm_service"),
            surprise_events=more,
        )
        surp_events.extend(more)

    train_active = campaign.status not in ("done", "failed")
    update: dict[str, Any] = {
        "campaign": campaign.to_dict(),
        "campaign_status": campaign.status,
        "campaign_tick": train_ticks,
        "hierarchy_train_active": train_active,
        "_hierarchy_train_active": train_active,  # giữ tương thích với các bên đọc khóa cũ
        "surprise_events": surp_events,
    }
    if surp_events:
        from hagent.agent.campaign.nodes import _select_surprise

        picked = _select_surprise(surp_events)
        if picked:
            update["surprise"] = picked

    if campaign.status == "done":
        from hagent.agent.campaign.compare import compare_campaign

        best, table = compare_campaign(campaign)
        evaluation = {
            "job_ids": [v.job_id for v in campaign.variants if v.job_id],
            "comparison_table": table,
            "best_job_id": best.job_id if best else None,
            "recommendation": best.best_model if best else None,
        }
        # Đồng bộ các job vào World Model.
        wm = dict(
            getattr(campaign, "_world_model_snapshot", None)
            or state.get("world_model")
            or {"user_id": state.get("user_id")}
        )
        jobs = dict(wm.get("jobs") or {})
        for v in campaign.variants:
            if v.job_id:
                jobs[v.job_id] = v.to_job_entry()
        wm["jobs"] = jobs
        update.update(
            {
                "done": True,
                "world_model": wm,
                "evaluation": evaluation,
                "summary": (
                    f"campaign done best={evaluation.get('best_job_id')} "
                    f"score={best.best_score if best else None}"
                ),
                "tools": ["start_training", "get_job_info"],
                "hierarchy_train_active": False,
                "_hierarchy_train_active": False,
            }
        )
    elif campaign.status == "failed":
        update.update(
            {
                "done": True,
                "failed": True,
                "summary": "campaign failed",
                "hierarchy_train_active": False,
                "_hierarchy_train_active": False,
            }
        )
    else:
        update.update(
            {
                "done": False,
                "summary": f"campaign {campaign.status}",
            }
        )
    return update


async def hierarchy_node(state: AutoMLState) -> dict:
    """Node LangGraph điều khiển hierarchy thích ứng."""
    events = list(state.get("execution_events") or [])
    cost = dict(state.get("cost_metrics") or {})
    ticks = int(cost.get("hierarchy_ticks") or 0) + 1
    cost["hierarchy_ticks"] = ticks

    # Giới hạn an toàn toàn cục để tránh vòng lặp đến giới hạn đệ quy của LangGraph.
    max_ticks = _max_hierarchy_ticks()
    if ticks > max_ticks:
        events.append(
            {
                "type": "hierarchy_timeout",
                "ticks": ticks,
                "max_ticks": max_ticks,
            }
        )
        return {
            "messages": [
                AIMessage(
                    content=(
                        f"Hierarchy timeout after {ticks} ticks "
                        f"(max={max_ticks}). Stopping."
                    )
                )
            ],
            "hierarchy": state.get("hierarchy"),
            "hierarchy_status": "failed",
            "plan_status": "failed",
            "execution_events": events,
            "cost_metrics": cost,
            "current_phase": "respond",
            "hierarchy_train_active": False,
        }

    hier = ensure_hierarchy(state)
    # Áp dụng lại điều kiện bỏ qua theo World Model và đánh giá mới nhất ở mỗi tick.
    skip_events = apply_smart_skips(
        hier,
        world_model=state.get("world_model"),
        evaluation=state.get("evaluation"),
        campaign_status=state.get("campaign_status"),
    )
    events.extend(skip_events)

    if hier.is_complete():
        prog = hier.progress()
        msg = (
            f"Hierarchy xong: done={prog.get('done', 0)}, "
            f"skipped={prog.get('skipped', 0)}, failed={prog.get('failed', 0)}."
        )
        events.append(
            {
                "type": "hierarchy_done",
                "hierarchy_id": hier.hierarchy_id,
                "progress": prog,
            }
        )
        return {
            "messages": [AIMessage(content=msg)],
            "hierarchy": hier.to_dict(),
            "hierarchy_status": "done",
            "plan_status": "done",
            "execution_events": events,
            "cost_metrics": cost,
            "current_phase": "respond",
            "goal": hier.root_goal,
        }

    cur = hier.current()
    assert cur is not None
    leaf_goal = subgoal_as_goal(hier)
    events.append(
        {
            "type": "subgoal_start",
            "goal_type": cur.goal_type,
            "index": hier.current_index,
            "subgoal_id": cur.subgoal_id,
            "description": cur.description,
        }
    )

    gtype = cur.goal_type.lower()
    leaf_out: dict[str, Any]
    if gtype == "analyze":
        leaf_out = await _run_analyze_leaf(state, leaf_goal)
    elif gtype == "select":
        leaf_out = await _run_select_leaf(state, leaf_goal)
    elif gtype == "monitor":
        leaf_out = await _run_monitor_leaf(state, leaf_goal)
    elif gtype == "evaluate":
        leaf_out = await _run_evaluate_leaf(state, leaf_goal)
    elif gtype == "train":
        leaf_out = await _run_train_leaf(state, leaf_goal)
    else:
        # Bỏ qua loại leaf không được hỗ trợ.
        leaf_out = {
            "done": True,
            "summary": f"unsupported leaf {gtype}, skipped",
            "tools": [],
        }
        cur.status = "skipped"
        cur.skip_reason = f"unsupported leaf type {gtype}"

    update: dict[str, Any] = {
        "hierarchy": hier.to_dict(),
        "hierarchy_status": "running",
        "goal": leaf_goal,  # công khai leaf đang hoạt động
        "execution_events": events,
        "cost_metrics": cost,
        "current_phase": gtype if gtype != "train" else "train",
    }

    if leaf_out.get("world_model") is not None:
        update["world_model"] = leaf_out["world_model"]
    if leaf_out.get("evaluation") is not None:
        update["evaluation"] = leaf_out["evaluation"]
    if "campaign" in leaf_out:
        update["campaign"] = leaf_out["campaign"]
    if "campaign_status" in leaf_out:
        update["campaign_status"] = leaf_out["campaign_status"]
    if "_hierarchy_train_active" in leaf_out:
        update["_hierarchy_train_active"] = leaf_out["_hierarchy_train_active"]
    if "hierarchy_train_active" in leaf_out:
        update["hierarchy_train_active"] = leaf_out["hierarchy_train_active"]
    if "campaign_tick" in leaf_out:
        update["campaign_tick"] = leaf_out["campaign_tick"]
    if leaf_out.get("surprise") is not None:
        update["surprise"] = leaf_out["surprise"]
    if leaf_out.get("surprise_events"):
        events.extend(leaf_out["surprise_events"])
        update["execution_events"] = events

    tools = leaf_out.get("tools") or []
    cost["tools_called"] = int(cost.get("tools_called") or 0) + len(tools)
    update["cost_metrics"] = cost

    if leaf_out.get("done"):
        if leaf_out.get("failed"):
            hier.advance(status="failed", summary=leaf_out.get("summary"))
            # Tiếp tục hierarchy để có thể đánh giá, trừ khi cấu hình yêu cầu dừng.
            abort = False
            try:
                from hagent.bridge.config import get_hierarchy_config

                abort = bool(get_hierarchy_config().get("abort_on_leaf_fail", False))
            except Exception as exc:  # noqa: BLE001 - cấu hình này là tùy chọn
                logger.debug("hierarchy abort config unavailable: %s", exc)
            events.append(
                {
                    "type": "subgoal_failed",
                    "goal_type": gtype,
                    "summary": leaf_out.get("summary"),
                }
            )
            if abort:
                update.update(
                    {
                        "messages": [
                            AIMessage(content=f"Hierarchy dừng: leaf `{gtype}` failed.")
                        ],
                        "hierarchy": hier.to_dict(),
                        "hierarchy_status": "failed",
                        "execution_events": events,
                    }
                )
                return update
        else:
            hier.advance(status="done", summary=leaf_out.get("summary"))
            events.append(
                {
                    "type": "subgoal_done",
                    "goal_type": gtype,
                    "summary": leaf_out.get("summary"),
                    "index": hier.current_index - 1,
                }
            )

        # Bỏ qua thông minh các leaf còn lại sau khi World Model được cập nhật.
        more_skips = apply_smart_skips(
            hier,
            world_model=update.get("world_model") or state.get("world_model"),
            evaluation=update.get("evaluation") or state.get("evaluation"),
            campaign_status=update.get("campaign_status"),
        )
        events.extend(more_skips)
        update["hierarchy"] = hier.to_dict()
        update["execution_events"] = events

        if hier.is_complete():
            prog = hier.progress()
            events.append(
                {
                    "type": "hierarchy_done",
                    "hierarchy_id": hier.hierarchy_id,
                    "progress": prog,
                }
            )
            update["execution_events"] = events
            update.update(
                {
                    "messages": [
                        AIMessage(
                            content=(
                                f"Hierarchy hoàn tất (done={prog.get('done')}, "
                                f"skipped={prog.get('skipped')}). "
                                f"Last: {leaf_out.get('summary')}"
                            )
                        )
                    ],
                    "hierarchy_status": "done",
                    "plan_status": "done",
                    "goal": hier.root_goal,
                    "current_phase": "respond",
                }
            )
        else:
            nxt = hier.current()
            update["messages"] = [
                AIMessage(
                    content=(
                        f"Subgoal `{gtype}` xong → next `{nxt.goal_type if nxt else '?'}` "
                        f"({leaf_out.get('summary')})"
                    )
                )
            ]
            update["hierarchy_status"] = "running"
    else:
        # Leaf vẫn đang chạy trong giai đoạn theo dõi campaign.
        update["messages"] = [
            AIMessage(content=f"Hierarchy leaf `{gtype}`: {leaf_out.get('summary')}")
        ]
        update["hierarchy_status"] = "running"
        # Lưu hierarchy nhưng chưa chuyển sang leaf tiếp theo.
        update["hierarchy"] = hier.to_dict()
        update["execution_events"] = events

    return update


def hierarchy_route(state: AutoMLState) -> str:
    status = state.get("hierarchy_status")
    if status in ("done", "failed"):
        return "synthesize"

    # Nếu trạng thái bị mất, dùng tiến độ hierarchy làm phương án dự phòng.
    hier = state.get("hierarchy")
    if isinstance(hier, dict):
        try:
            idx = int(hier.get("current_index") or 0)
            n = len(hier.get("subgoals") or [])
            if n and idx >= n:
                return "synthesize"
        except (TypeError, ValueError):
            logger.debug("invalid hierarchy progress payload", hierarchy=hier)

    if status == "running" or hier:
        return "hierarchy"
    return "synthesize"
