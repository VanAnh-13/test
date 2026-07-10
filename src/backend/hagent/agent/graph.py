"""
DeerFlow-AutoML — LangGraph StateGraph Definition (Phase 2, SOLID).

Dynamic multi-agent graph — đọc YAML config qua AgentRegistry.
KHÔNG hardcode bất kỳ tên agent, module path, hay tool name nào.

Thêm agent mới: chỉ cần thêm entry trong hagent.yaml → graph tự build.

SOLID:
  S — Graph chỉ làm 1 việc: build topology từ registry
  O — Mở rộng qua YAML config, không sửa Python
  D — Inject registry, không import sub-agents trực tiếp
"""

from __future__ import annotations

import json
import logging
from typing import Any

try:
    from langchain_core.messages import ToolMessage
    from langgraph.graph import END, StateGraph
    from langgraph.prebuilt import ToolNode
except ImportError:  # pragma: no cover
    ToolMessage = object  # type: ignore[misc, assignment]
    END = "__end__"
    StateGraph = None  # type: ignore[assignment]
    ToolNode = None  # type: ignore[assignment]

from hagent.agent.campaign.nodes import campaign_node, campaign_route
from hagent.agent.execution.hierarchy_node import hierarchy_node, hierarchy_route
from hagent.agent.execution.plan_executor import plan_executor_node, plan_executor_route
from hagent.agent.execution.reviser import reviser_node, reviser_route
from hagent.agent.state import AutoMLState

# Heavy imports deferred inside build/run when langgraph present

logger = logging.getLogger(__name__)


def _planning_execute_enabled() -> bool:
    try:
        from hagent.bridge.config import get_planning_config

        cfg = get_planning_config()
        return bool(cfg.get("enabled", True)) and bool(
            cfg.get("execute_plans", True)
        )
    except Exception:
        return True


def _campaign_enabled() -> bool:
    try:
        from hagent.bridge.config import get_campaign_config

        return bool(get_campaign_config().get("enabled", True))
    except Exception:
        return True


def _campaign_goal_types() -> set[str]:
    try:
        from hagent.bridge.config import get_campaign_config

        types = get_campaign_config().get("prefer_for_goal_types") or ["train"]
        return {str(t).lower() for t in types}
    except Exception:
        return {"train"}


def _hierarchy_live_enabled() -> bool:
    try:
        from hagent.bridge.config import get_hierarchy_config

        cfg = get_hierarchy_config()
        return bool(cfg.get("enabled", True)) and bool(cfg.get("live_controller", True))
    except Exception:
        return True


def _should_run_hierarchy(state: AutoMLState) -> bool:
    """Live adaptive hierarchy for multi-step roots (e.g. train)."""
    if not _hierarchy_live_enabled():
        return False
    hstatus = state.get("hierarchy_status")
    if hstatus == "running":
        return True
    if hstatus in ("done", "failed"):
        return False
    if isinstance(state.get("hierarchy"), dict) and state["hierarchy"].get("subgoals"):
        # Mid-flight hierarchy
        if hstatus != "done":
            idx = int(state["hierarchy"].get("current_index") or 0)
            n = len(state["hierarchy"].get("subgoals") or [])
            if idx < n:
                return True

    goal = state.get("goal") if isinstance(state.get("goal"), dict) else {}
    gtype = str(goal.get("goal_type") or "").lower()
    # Multi-step roots only
    if gtype not in ("train", "evaluate"):
        return False
    if gtype == "train" and not (
        goal.get("dataset_id") and goal.get("target_column")
    ):
        wm = state.get("world_model") or {}
        if not (wm.get("active_dataset_id") or goal.get("dataset_id")):
            return False
    return True


def _should_run_campaign(state: AutoMLState) -> bool:
    """Phase 6: multi-job campaign — used when hierarchy is off or for direct train."""
    if not _campaign_enabled():
        return False
    # Hierarchy owns train when live controller on
    if _should_run_hierarchy(state):
        return False
    # Resume in-progress campaign only if not under hierarchy
    cstatus = state.get("campaign_status")
    if state.get("_hierarchy_train_active"):
        return False
    if cstatus in ("building", "submitting", "monitoring", "comparing"):
        return True
    if isinstance(state.get("campaign"), dict):
        st = state["campaign"].get("status")
        if st in ("building", "submitting", "monitoring", "comparing"):
            return True

    goal = state.get("goal") if isinstance(state.get("goal"), dict) else {}
    gtype = str(goal.get("goal_type") or "").lower()
    if gtype not in _campaign_goal_types():
        return False
    if not goal.get("dataset_id") or not goal.get("target_column"):
        wm = state.get("world_model") or {}
        if not (wm.get("active_dataset_id") or goal.get("dataset_id")):
            return False
    if cstatus in ("done", "failed"):
        return False
    return True


def _should_run_plan_executor(state: AutoMLState) -> bool:
    """True when a selected plan should be executed by plan_executor loop."""
    if not _planning_execute_enabled():
        return False
    if _should_run_hierarchy(state):
        return False
    if _should_run_campaign(state):
        return False
    plan = state.get("selected_plan")
    if not isinstance(plan, dict) or not plan.get("steps"):
        return False
    status = state.get("plan_status")
    if status in ("done", "failed", "aborted"):
        return False
    goal = state.get("goal") if isinstance(state.get("goal"), dict) else {}
    gtype = str(goal.get("goal_type") or plan.get("meta", {}).get("goal_type") or "")
    if gtype == "respond":
        return False
    return True


# ── Routing functions ────────────────────────────────────


def coordinator_route(state: AutoMLState) -> str:
    """
    Coordinator quyết định route tới đâu.

    Ưu tiên:
    1. tool_calls → coordinator_tools
    2. hierarchy (adaptive multi-subgoal) → hierarchy
    3. campaign (Phase 6 multi-job) → campaign
    4. selected_plan executable → plan_executor
    5. next_agent hợp lệ → sub-agent
    6. END
    """
    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "coordinator_tools"

    if _should_run_hierarchy(state):
        logger.info("Coordinator routing → hierarchy")
        return "hierarchy"

    if _should_run_campaign(state):
        logger.info("Coordinator routing → campaign")
        return "campaign"

    if _should_run_plan_executor(state):
        logger.info("Coordinator routing → plan_executor")
        return "plan_executor"

    next_agent = state.get("next_agent")
    try:
        from hagent.agent.registry import get_agent_registry

        registry = get_agent_registry()
        if next_agent and registry.is_valid_agent(next_agent):
            logger.info("Coordinator routing → %s", next_agent)
            return next_agent
    except Exception:
        if next_agent:
            return next_agent

    return "end"


def subagent_route(state: AutoMLState) -> str:
    """
    Sau khi sub-agent chạy:
    - Nếu cần tools → "sub_tools"
    - Nếu không → "synthesize"
    """
    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "sub_tools"

    return "synthesize"


def after_sub_tools(state: AutoMLState) -> str:
    """Sau khi tool chạy xong, quay lại sub-agent hiện tại."""
    from hagent.agent.registry import get_agent_registry

    next_agent = state.get("next_agent")
    registry = get_agent_registry()
    if next_agent and registry.is_valid_agent(next_agent):
        return next_agent
    return "synthesize"


def should_continue(state: AutoMLState | dict) -> str:
    """
    Backward-compatible helper used by older unit tests.

    Legacy single-agent style:
      - tool_calls present → \"tools\"
      - otherwise → \"end\"
    """
    messages = state.get("messages") or []
    if not messages:
        return "end"
    last = messages[-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


# ── Synthesizer node ─────────────────────────────────────


async def synthesizer_node(state: AutoMLState) -> dict:
    """Tổng hợp kết quả execution/sub-agent. Reset next_agent."""
    log = state.get("execution_log") or []
    status = state.get("plan_status")
    parts = []
    if status:
        parts.append(f"Plan status: {status}.")
    if log:
        ok = sum(1 for x in log if x.get("status") == "ok")
        parts.append(f"Executed {len(log)} steps ({ok} ok).")
    if state.get("last_step_error") and status == "failed":
        parts.append(f"Last error: {state['last_step_error']}.")
    summary = " ".join(parts) if parts else ""
    update: dict[str, Any] = {
        "next_agent": None,
        "current_phase": state.get("current_phase") or "respond",
    }
    if summary:
        from langchain_core.messages import AIMessage

        update["messages"] = [AIMessage(content=summary)]
    # Phase-5: fold execution_events into a terminal event
    events = list(state.get("execution_events") or [])
    events.append(
        {
            "type": "synthesize",
            "plan_status": status,
            "cost_metrics": state.get("cost_metrics"),
        }
    )
    update["execution_events"] = events
    return update


# ── Dynamic graph builder ───────────────────────────────


def build_automl_graph() -> StateGraph:
    """
    Xây dựng LangGraph StateGraph từ AgentRegistry.

    Đọc agent.subagents từ YAML → dynamic add nodes + edges.
    KHÔNG hardcode tên agent nào.

    Flow:
        User → coordinator → plan_executor ⇄ reviser → synthesize → END
                         ↘ sub-agent ↔ tools → synthesize → END
    """
    if StateGraph is None or ToolNode is None:
        raise ImportError(
            "langgraph/langchain_core required to build AutoML graph"
        )

    from hagent.agent.coordinator import coordinator_node
    from hagent.agent.registry import get_agent_registry

    registry = get_agent_registry()
    agent_names = registry.agent_names()
    all_tools = registry.get_all_tools()

    if not all_tools:
        # Fallback: import ALL_TOOLS nếu registry chưa có tools
        from hagent.agent.tools.automl_tools import ALL_TOOLS
        all_tools = ALL_TOOLS

    # Tool nodes
    tool_node_all = ToolNode(all_tools)

    graph = StateGraph(AutoMLState)

    # ── Fixed nodes ──────────────────────────────────────
    graph.add_node("coordinator", coordinator_node)
    graph.add_node("coordinator_tools", tool_node_all)
    graph.add_node("sub_tools", tool_node_all)
    graph.add_node("plan_executor", plan_executor_node)
    graph.add_node("reviser", reviser_node)
    graph.add_node("campaign", campaign_node)
    graph.add_node("hierarchy", hierarchy_node)
    graph.add_node("synthesize", synthesizer_node)

    # ── Dynamic sub-agent nodes (từ registry) ────────────
    node_functions = registry.get_node_functions()
    for name, node_fn in node_functions.items():
        graph.add_node(name, node_fn)
        logger.debug("Graph node added: %s", name)

    # ── Entry ────────────────────────────────────────────
    graph.set_entry_point("coordinator")

    # ── Coordinator routing (dynamic) ────────────────────
    route_map: dict[str, str] = {
        "coordinator_tools": "coordinator_tools",
        "plan_executor": "plan_executor",
        "campaign": "campaign",
        "hierarchy": "hierarchy",
        "end": END,
    }
    for name in agent_names:
        route_map[name] = name

    graph.add_conditional_edges("coordinator", coordinator_route, route_map)
    graph.add_edge("coordinator_tools", "coordinator")

    # ── Plan executor loop ───────────────────────────────
    graph.add_conditional_edges(
        "plan_executor",
        plan_executor_route,
        {
            "plan_executor": "plan_executor",
            "reviser": "reviser",
            "synthesize": "synthesize",
        },
    )
    graph.add_conditional_edges(
        "reviser",
        reviser_route,
        {
            "plan_executor": "plan_executor",
            "synthesize": "synthesize",
        },
    )

    # ── Phase 6 campaign loop ────────────────────────────
    graph.add_conditional_edges(
        "campaign",
        campaign_route,
        {
            "campaign": "campaign",
            "synthesize": "synthesize",
        },
    )

    # ── Adaptive hierarchy loop ──────────────────────────
    graph.add_conditional_edges(
        "hierarchy",
        hierarchy_route,
        {
            "hierarchy": "hierarchy",
            "synthesize": "synthesize",
        },
    )

    # ── Sub-agent routing (dynamic) ──────────────────────
    for name in agent_names:
        graph.add_conditional_edges(
            name,
            subagent_route,
            {"sub_tools": "sub_tools", "synthesize": "synthesize"},
        )

    # ── Sub tools → quay lại sub-agent (dynamic) ────────
    after_tools_map: dict[str, str] = {"synthesize": "synthesize"}
    for name in agent_names:
        after_tools_map[name] = name

    graph.add_conditional_edges("sub_tools", after_sub_tools, after_tools_map)

    # ── Synthesize → END ─────────────────────────────────
    graph.add_edge("synthesize", END)

    logger.info(
        "Graph built: %d sub-agents [%s] + plan_executor/reviser",
        len(agent_names),
        ", ".join(sorted(agent_names)),
    )
    return graph


# ── Compiled graph singleton ─────────────────────────────

_compiled_graph = None


def get_automl_graph():
    """Trả về compiled graph (singleton)."""
    global _compiled_graph
    if _compiled_graph is None:
        graph = build_automl_graph()
        _compiled_graph = graph.compile()
        logger.info("DeerFlow-AutoML graph compiled ✓")
    return _compiled_graph


def reset_graph():
    """Reset compiled graph — dùng khi config thay đổi hoặc testing."""
    global _compiled_graph
    _compiled_graph = None


# ── Convenience runner ───────────────────────────────────


async def run_agent(
    message: str,
    *,
    user_id: str | None = None,
    user_token: str | None = None,
    world_model: dict[str, Any] | None = None,
    memory_context: str | None = None,
) -> dict[str, Any]:
    """Chạy multi-agent graph với middleware pipeline."""
    from langchain_core.messages import HumanMessage
    from hagent.agent.middlewares import create_default_chain

    graph = get_automl_graph()
    middleware = create_default_chain()

    initial_state: AutoMLState = {
        "messages": [HumanMessage(content=message)],
        "world_model": world_model,
        "memory_context": memory_context,
        "user_id": user_id,
        "user_token": user_token,
        "next_agent": None,
        "current_phase": None,
        "plan_step_index": 0,
        "revision_count": 0,
        "execution_log": [],
        "execution_events": [],
        "cost_metrics": {},
    }

    import os
    import time

    if user_token:
        os.environ["USER_TOKEN"] = user_token
    if user_id:
        os.environ["USER_ID"] = user_id

    # Middleware pre-process
    t0 = time.time()
    initial_state = await middleware.run_pre(initial_state)

    final_state = await graph.ainvoke(initial_state)

    messages = final_state["messages"]
    last_ai_message = None
    tool_outputs = []

    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            tool_outputs.append({
                "tool_name": msg.name,
                "payload": _safe_json_parse(msg.content),
            })
        elif hasattr(msg, "content") and msg.content and not last_ai_message:
            if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                last_ai_message = msg

    response_text = last_ai_message.content if last_ai_message else "Không có phản hồi."

    cost = dict(final_state.get("cost_metrics") or {})
    cost["elapsed_seconds"] = round(time.time() - t0, 3)

    result = {
        "response": response_text,
        "tool_outputs": list(reversed(tool_outputs)),
        "sources": [],
        "provider": "deerflow-automl",
        "model": "multi-agent",
        "route": final_state.get("current_phase", "direct"),
        "plan_status": final_state.get("plan_status"),
        "selected_plan": final_state.get("selected_plan"),
        "surprise": final_state.get("surprise"),
        "cost_metrics": cost,
        "execution_events": final_state.get("execution_events") or [],
        "execution_log": final_state.get("execution_log") or [],
        "revision_count": final_state.get("revision_count") or 0,
        "world_model": final_state.get("world_model"),
        "campaign": final_state.get("campaign"),
        "campaign_status": final_state.get("campaign_status"),
        "evaluation": final_state.get("evaluation"),
        "hierarchy": final_state.get("hierarchy"),
        "hierarchy_status": final_state.get("hierarchy_status"),
    }

    # Middleware post-process
    result = await middleware.run_post(initial_state, result)

    return result


def _safe_json_parse(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


# ── Streaming runner ─────────────────────────────────────


async def stream_agent(
    message: str,
    *,
    user_id: str | None = None,
    user_token: str | None = None,
    world_model: dict[str, Any] | None = None,
    memory_context: str | None = None,
):
    """Stream multi-agent graph events qua async generator (Phase 5 enriched)."""
    from langchain_core.messages import HumanMessage

    graph = get_automl_graph()
    registry = get_agent_registry()
    valid_names = set(registry.agent_names()) | {
        "plan_executor",
        "reviser",
        "campaign",
        "hierarchy",
        "coordinator",
        "synthesize",
    }

    initial_state: AutoMLState = {
        "messages": [HumanMessage(content=message)],
        "world_model": world_model,
        "memory_context": memory_context,
        "user_id": user_id,
        "user_token": user_token,
        "next_agent": None,
        "current_phase": None,
        "plan_step_index": 0,
        "revision_count": 0,
        "execution_log": [],
        "execution_events": [],
        "cost_metrics": {},
    }

    import os
    import time

    if user_token:
        os.environ["USER_TOKEN"] = user_token
    if user_id:
        os.environ["USER_ID"] = user_id

    # Middleware so WM service is available
    try:
        from hagent.agent.middlewares import create_default_chain

        middleware = create_default_chain()
        initial_state = await middleware.run_pre(initial_state)
    except Exception:
        middleware = None

    t0 = time.time()
    final_content = ""
    current_route = None
    last_events_len = 0
    final_state_snapshot: dict[str, Any] = {}

    async for event in graph.astream_events(initial_state, version="v2"):
        kind = event["event"]

        if kind == "on_chain_start":
            node_name = event.get("name", "")
            if node_name in valid_names:
                current_route = node_name
                yield {"type": "route", "agent": node_name}
                if node_name == "plan_executor":
                    yield {"type": "phase", "phase": "execute"}
                elif node_name == "reviser":
                    yield {"type": "phase", "phase": "revise"}
                elif node_name == "campaign":
                    yield {"type": "phase", "phase": "campaign"}
                elif node_name == "hierarchy":
                    yield {"type": "phase", "phase": "hierarchy"}

        elif kind == "on_chain_end":
            # Emit structured execution_events deltas from node output
            data = event.get("data") or {}
            output = data.get("output")
            if isinstance(output, dict):
                final_state_snapshot.update(
                    {
                        k: output[k]
                        for k in (
                            "plan_status",
                            "selected_plan",
                            "surprise",
                            "cost_metrics",
                            "execution_events",
                            "revision_count",
                            "campaign",
                            "campaign_status",
                            "evaluation",
                            "hierarchy",
                            "hierarchy_status",
                        )
                        if k in output
                    }
                )
                events = output.get("execution_events") or []
                if isinstance(events, list) and len(events) > last_events_len:
                    for ev in events[last_events_len:]:
                        yield {"type": "plan_event", "event": ev}
                        if isinstance(ev, dict) and ev.get("type") == "step_end":
                            if ev.get("surprise"):
                                yield {
                                    "type": "surprise",
                                    "surprise": ev["surprise"],
                                    "index": ev.get("index"),
                                    "action": ev.get("action"),
                                }
                    last_events_len = len(events)
                if output.get("selected_plan") and event.get("name") == "coordinator":
                    sp = output["selected_plan"]
                    yield {
                        "type": "plan",
                        "plan_id": sp.get("plan_id"),
                        "title": sp.get("title"),
                        "steps": [
                            (s.get("action") or {}).get("type")
                            if isinstance(s, dict)
                            else None
                            for s in (sp.get("steps") or [])
                        ],
                        "cost": sp.get("cost"),
                    }

        elif kind == "on_chat_model_stream":
            chunk = event["data"].get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                final_content += chunk.content
                yield {"type": "token", "content": chunk.content}

        elif kind == "on_tool_start":
            yield {
                "type": "tool_call",
                "tool": event.get("name", ""),
                "args": event.get("data", {}).get("input", {}),
            }

        elif kind == "on_tool_end":
            yield {
                "type": "tool_result",
                "tool": event.get("name", ""),
                "output": _safe_json_parse(
                    event.get("data", {}).get("output", "")
                ),
            }

    cost = dict(final_state_snapshot.get("cost_metrics") or {})
    cost["elapsed_seconds"] = round(time.time() - t0, 3)

    yield {
        "type": "done",
        "response": final_content,
        "route": current_route or "direct",
        "plan_status": final_state_snapshot.get("plan_status"),
        "cost_metrics": cost,
        "revision_count": final_state_snapshot.get("revision_count") or 0,
        "surprise": final_state_snapshot.get("surprise"),
    }
