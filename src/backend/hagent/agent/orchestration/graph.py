"""
Định nghĩa LangGraph StateGraph của HAgent cho giai đoạn 2 theo SOLID.

Graph đa agent động đọc cấu hình YAML qua AgentRegistry.

Để thêm agent mới, chỉ cần thêm mục trong hagent.yaml; graph sẽ tự tạo.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Hashable
from typing import Any, cast

import structlog

from hagent.core.types import AgentState, RouteType

try:
    from langchain_core.messages import AIMessage, ToolMessage
    from langgraph.graph import END, StateGraph
    from langgraph.prebuilt import ToolNode
except ImportError:  # pragma: no cover
    # Fallback chỉ giữ module import được khi dependency graph không được cài.
    AIMessage = object  # type: ignore[misc, assignment]
    ToolMessage = object  # type: ignore[misc, assignment]
    END = "__end__"
    StateGraph = None  # type: ignore[misc, assignment]
    ToolNode = None  # type: ignore[misc, assignment]

from hagent.agent.campaign.nodes import campaign_node, campaign_route
from hagent.agent.execution.hierarchy_node import hierarchy_node, hierarchy_route
from hagent.agent.execution.plan_executor import plan_executor_node, plan_executor_route
from hagent.agent.execution.reviser import reviser_node, reviser_route
from hagent.agent.runtime.context import GraphRequestContext, bind_request_context

# Hoãn các import nặng đến lúc tạo hoặc chạy khi có LangGraph.

logger = structlog.get_logger(__name__)


def _as_agent_state_node(
    node: Callable[[Any], Awaitable[Any]],
) -> Callable[[AgentState], Awaitable[AgentState]]:
    """Chuẩn hóa node cũ tại biên để LangGraph chỉ thấy AgentState."""

    async def typed_node(state: AgentState) -> AgentState:
        result = await node(state)
        if not isinstance(result, dict):
            raise TypeError("Agent node must return a state update")
        return cast(AgentState, result)

    typed_node.__name__ = getattr(node, "__name__", "typed_node")
    typed_node.__doc__ = getattr(node, "__doc__", None)
    return typed_node


def _as_agent_state_route(
    route: Callable[[Any], str],
) -> Callable[[AgentState], str]:
    """Chuẩn hóa hàm định tuyến cũ để không đăng ký lại schema AutoMLState."""

    def typed_route(state: AgentState) -> str:
        return route(state)

    typed_route.__name__ = getattr(route, "__name__", "typed_route")
    typed_route.__doc__ = getattr(route, "__doc__", None)
    return typed_route


def _bind_context_node(
    node: Callable[..., Awaitable[Any]],
    *,
    include_credential: bool = True,
) -> Any:
    """Cô lập phép ép kiểu cần thiết giữa node TypedDict cũ và adapter context."""
    legacy_node = cast(Callable[[dict[str, Any]], Awaitable[Any]], node)
    return bind_request_context(
        legacy_node,
        include_credential=include_credential,
    )


async def _inject_request_scope_into_tool_call(
    request: Any,
    execute: Callable[[Any], Awaitable[Any]],
) -> Any:
    """Chỉ inject authority từ runtime context khi gọi tool."""
    tool = request.tool
    tool_schema = getattr(tool, "args", {}) if tool is not None else {}
    call = dict(request.tool_call)
    args = dict(call.get("args") or {})
    args.pop("token", None)
    args.pop("user_id", None)
    args.pop("idempotency_key", None)

    if call.get("name") == "start_training":
        return _trusted_training_action_required_tool_message(call)

    runtime = getattr(request, "runtime", None)
    context = getattr(runtime, "context", None)
    requires_context = "token" in tool_schema or "user_id" in tool_schema
    if requires_context and not isinstance(context, GraphRequestContext):
        return _auth_scope_required_tool_message(call)
    typed_context = cast(GraphRequestContext, context)

    if "token" in tool_schema:
        if typed_context.credential is None:
            return _auth_scope_required_tool_message(call)
        args["token"] = typed_context.credential

    if "user_id" in tool_schema:
        args["user_id"] = typed_context.principal_id

    scoped_call = {**call, "args": args}
    return await execute(request.override(tool_call=scoped_call))


def _auth_scope_required_tool_message(call: dict[str, Any]) -> Any:
    """Tạo lỗi ổn định mà không phản chiếu credential từ input."""
    return ToolMessage(
        content=json.dumps(
            {
                "error": {
                    "code": "AUTH_SCOPE_REQUIRED",
                    "message": "Authenticated request scope is required",
                }
            },
            ensure_ascii=False,
        ),
        name=str(call.get("name") or ""),
        tool_call_id=str(call.get("id") or ""),
        status="error",
    )


def _trusted_training_action_required_tool_message(call: dict[str, Any]) -> Any:
    """Từ chối mutation trực tiếp không có action identity từ runtime."""

    return ToolMessage(
        content=json.dumps(
            {
                "error": {
                    "code": "TRAINING_TRUSTED_ACTION_REQUIRED",
                    "message": "Training requires an approved runtime action",
                }
            },
            ensure_ascii=False,
        ),
        name=str(call.get("name") or ""),
        tool_call_id=str(call.get("id") or ""),
        status="error",
    )


def _graph_stream_accepts_context(graph: Any) -> bool:
    """Giữ tương thích với đối tượng giả cũ mà không hạ cấp graph production."""
    parameters = inspect.signature(graph.astream_events).parameters.values()
    return any(
        parameter.name == "context" or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _planning_execute_enabled() -> bool:
    try:
        from hagent.bridge.config import get_planning_config

        cfg = get_planning_config()
        return bool(cfg.get("enabled", True)) and bool(cfg.get("execute_plans", True))
    except Exception:  # noqa: BLE001
        return True


def _campaign_enabled() -> bool:
    try:
        from hagent.bridge.config import get_campaign_config

        return bool(get_campaign_config().get("enabled", True))
    except Exception:  # noqa: BLE001
        return True


def _campaign_goal_types() -> set[str]:
    try:
        from hagent.bridge.config import get_campaign_config

        types = get_campaign_config().get("prefer_for_goal_types") or ["train"]
        return {str(t).lower() for t in types}
    except Exception:  # noqa: BLE001
        return {"train"}


def _hierarchy_live_enabled() -> bool:
    try:
        from hagent.bridge.config import get_hierarchy_config

        cfg = get_hierarchy_config()
        return bool(cfg.get("enabled", True)) and bool(cfg.get("live_controller", True))
    except Exception:  # noqa: BLE001
        return True


def _should_run_hierarchy(state: AgentState) -> bool:
    """Chạy hierarchy thích ứng cho mục tiêu gốc nhiều bước, chẳng hạn huấn luyện."""
    if not _hierarchy_live_enabled():
        return False
    hstatus = state.get("hierarchy_status")
    if hstatus == "running":
        return True
    if hstatus in ("done", "failed"):
        return False
    hierarchy = state.get("hierarchy")
    if isinstance(hierarchy, dict) and hierarchy.get("subgoals") and hstatus != "done":
        idx = int(hierarchy.get("current_index") or 0)
        n = len(hierarchy.get("subgoals") or [])
        if idx < n:
            return True

    goal_value = state.get("goal")
    goal = goal_value if isinstance(goal_value, dict) else {}
    gtype = str(goal.get("goal_type") or "").lower()
    # Chỉ áp dụng cho mục tiêu gốc nhiều bước.
    if gtype not in (RouteType.TRAIN.value, RouteType.EVALUATE.value):
        return False
    if gtype == RouteType.TRAIN.value and not (
        goal.get("dataset_id") and goal.get("target_column")
    ):
        wm = state.get("world_model") or {}
        if not (wm.get("active_dataset_id") or goal.get("dataset_id")):
            return False
    return True


def _should_run_campaign(state: AgentState) -> bool:
    """Campaign nhiều job ở giai đoạn 6 khi tắt hierarchy hoặc huấn luyện trực tiếp."""
    if not _campaign_enabled():
        return False
    # Hierarchy sở hữu luồng huấn luyện khi bộ điều khiển trực tiếp đang bật.
    if _should_run_hierarchy(state):
        return False
    # Chỉ tiếp tục campaign đang chạy khi không nằm dưới hierarchy.
    cstatus = state.get("campaign_status")
    if state.get("_hierarchy_train_active"):
        return False
    if cstatus in ("building", "submitting", "monitoring", "comparing"):
        return True
    campaign = state.get("campaign")
    if isinstance(campaign, dict):
        st = campaign.get("status")
        if st in ("building", "submitting", "monitoring", "comparing"):
            return True

    goal_value = state.get("goal")
    goal = goal_value if isinstance(goal_value, dict) else {}
    gtype = str(goal.get("goal_type") or "").lower()
    if gtype not in _campaign_goal_types():
        return False
    if not goal.get("dataset_id") or not goal.get("target_column"):
        wm = state.get("world_model") or {}
        if not (wm.get("active_dataset_id") or goal.get("dataset_id")):
            return False
    return cstatus not in ("done", "failed")


def _should_run_plan_executor(state: AgentState) -> bool:
    """Trả về True khi plan đã chọn cần được vòng lặp plan_executor thực thi."""
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
    goal_value = state.get("goal")
    goal = goal_value if isinstance(goal_value, dict) else {}
    gtype = str(goal.get("goal_type") or plan.get("meta", {}).get("goal_type") or "")
    return gtype != RouteType.RESPOND.value


# ── Routing functions ────────────────────────────────────


def coordinator_route(state: AgentState) -> str:
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
        return RouteType.COORDINATOR_TOOLS.value

    if _should_run_hierarchy(state):
        logger.info("Coordinator routing → hierarchy")
        return RouteType.HIERARCHY.value

    if _should_run_campaign(state):
        logger.info("Coordinator routing → campaign")
        return RouteType.CAMPAIGN.value

    if _should_run_plan_executor(state):
        logger.info("Coordinator routing → plan_executor")
        return RouteType.PLAN_EXECUTOR.value

    next_agent = state.get("next_agent")
    try:
        from hagent.agent.orchestration import registry as registry_module

        registry = registry_module.get_agent_registry()
        if next_agent and registry.is_valid_agent(next_agent):
            logger.info("Coordinator routing → %s", next_agent)
            return next_agent
    except Exception:  # noqa: BLE001
        if next_agent:
            return next_agent

    return RouteType.END.value


def subagent_route(state: AgentState) -> str:
    """
    Sau khi sub-agent chạy:
    - Nếu cần tools → "sub_tools"
    - Nếu không → "synthesize"
    """
    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return RouteType.SUB_TOOLS.value

    return RouteType.SYNTHESIZE.value


def after_sub_tools(state: AgentState) -> str:
    """Sau khi tool chạy xong, quay lại sub-agent hiện tại."""
    from hagent.agent.orchestration import registry as registry_module

    next_agent = state.get("next_agent")
    registry = registry_module.get_agent_registry()
    if next_agent and registry.is_valid_agent(next_agent):
        return next_agent
    return RouteType.SYNTHESIZE.value


def should_continue(state: AgentState) -> str:
    """
    Hàm hỗ trợ tương thích ngược cho các unit test cũ.

    Kiểu single-agent cũ:
      - Có tool_calls thì trả "tools".
      - Trường hợp khác trả "end".
    """
    messages = state.get("messages") or []
    if not messages:
        return RouteType.END.value
    last = messages[-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return RouteType.TOOLS.value
    return RouteType.END.value


# ── Synthesizer node ─────────────────────────────────────


async def synthesizer_node(state: AgentState) -> AgentState:
    """Tổng hợp kết quả thực thi hoặc sub-agent rồi đặt lại next_agent."""
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
    update: AgentState = {
        "next_agent": None,
        "current_phase": state.get("current_phase") or "respond",
    }
    if summary:
        from langchain_core.messages import AIMessage

        update["messages"] = [AIMessage(content=summary)]
    # Giai đoạn 5: gộp execution_events vào một sự kiện kết thúc.
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


def build_automl_graph() -> StateGraph[
    AgentState, GraphRequestContext, AgentState, AgentState
]:
    """
    Xây dựng LangGraph StateGraph từ AgentRegistry.

    Đọc agent.subagents từ YAML rồi thêm node và cạnh động.
    Không mã hóa cứng tên agent nào.

    Luồng:
        User → coordinator → plan_executor ⇄ reviser → synthesize → END
                         ↘ sub-agent ↔ tools → synthesize → END
    """
    if StateGraph is None or ToolNode is None:
        raise ImportError("langgraph/langchain_core required to build AutoML graph")

    from hagent.agent.orchestration import registry as registry_module
    from hagent.agent.orchestration.coordinator import coordinator_node

    registry = registry_module.get_agent_registry()
    agent_names = registry.agent_names()
    all_tools = registry.get_all_tools()

    if not all_tools:
        # Nhập ALL_TOOLS làm phương án dự phòng nếu registry chưa có công cụ.
        from hagent.agent.tools.automl_tools import ALL_TOOLS

        all_tools = ALL_TOOLS

    # Tool nodes
    tool_node_all = ToolNode(
        all_tools,
        awrap_tool_call=_inject_request_scope_into_tool_call,
    )

    graph = StateGraph(AgentState, context_schema=GraphRequestContext)

    # ── Fixed nodes ──────────────────────────────────────
    graph.add_node(
        "coordinator",
        _bind_context_node(coordinator_node, include_credential=False),
    )
    graph.add_node("coordinator_tools", tool_node_all)
    graph.add_node("sub_tools", tool_node_all)
    graph.add_node("plan_executor", _bind_context_node(plan_executor_node))
    graph.add_node(
        "reviser",
        _bind_context_node(reviser_node, include_credential=False),
    )
    graph.add_node("campaign", _bind_context_node(campaign_node))
    graph.add_node("hierarchy", _bind_context_node(hierarchy_node))
    graph.add_node("synthesize", synthesizer_node)

    # ── Dynamic sub-agent nodes (từ registry) ────────────
    node_functions = registry.get_node_functions()
    for name, node_fn in node_functions.items():
        graph.add_node(name, cast(Any, _as_agent_state_node(node_fn)))
        logger.debug("Graph node added: %s", name)

    # ── Entry ────────────────────────────────────────────
    graph.set_entry_point("coordinator")

    # ── Coordinator routing (dynamic) ────────────────────
    route_map: dict[Hashable, str] = {
        RouteType.COORDINATOR_TOOLS.value: RouteType.COORDINATOR_TOOLS.value,
        RouteType.PLAN_EXECUTOR.value: RouteType.PLAN_EXECUTOR.value,
        RouteType.CAMPAIGN.value: RouteType.CAMPAIGN.value,
        RouteType.HIERARCHY.value: RouteType.HIERARCHY.value,
        RouteType.END.value: END,
    }
    for name in agent_names:
        route_map[name] = name

    graph.add_conditional_edges("coordinator", coordinator_route, route_map)
    graph.add_edge("coordinator_tools", "coordinator")

    # ── Plan executor loop ───────────────────────────────
    graph.add_conditional_edges(
        "plan_executor",
        _as_agent_state_route(plan_executor_route),
        {
            "plan_executor": "plan_executor",
            "reviser": "reviser",
            "synthesize": "synthesize",
        },
    )
    graph.add_conditional_edges(
        "reviser",
        _as_agent_state_route(reviser_route),
        {
            "plan_executor": "plan_executor",
            "synthesize": "synthesize",
        },
    )

    # ── Phase 6 campaign loop ────────────────────────────
    graph.add_conditional_edges(
        "campaign",
        _as_agent_state_route(campaign_route),
        {
            "campaign": "campaign",
            "synthesize": "synthesize",
        },
    )

    # ── Adaptive hierarchy loop ──────────────────────────
    graph.add_conditional_edges(
        "hierarchy",
        _as_agent_state_route(hierarchy_route),
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
    after_tools_map: dict[Hashable, str] = {"synthesize": "synthesize"}
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

_compiled_graph: Any | None = None


def get_automl_graph() -> Any:
    """Trả về graph đã biên dịch theo singleton."""
    global _compiled_graph
    if _compiled_graph is None:
        graph = build_automl_graph()
        _compiled_graph = graph.compile()
        logger.info("HAgent graph compiled ✓")
    return _compiled_graph


def reset_graph() -> None:
    """Đặt lại graph đã biên dịch khi cấu hình thay đổi hoặc khi kiểm thử."""
    global _compiled_graph
    _compiled_graph = None


# ── Convenience runner ───────────────────────────────────


def _build_initial_messages(
    message: str,
    history: list[dict[str, str]] | None = None,
) -> list[Any]:
    from langchain_core.messages import AIMessage, HumanMessage

    messages = []
    for item in list(history or [])[-20:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        message_type = HumanMessage if role == "user" else AIMessage
        messages.append(message_type(content=content))
    messages.append(HumanMessage(content=message))
    return messages


async def run_agent(
    message: str,
    *,
    user_id: str | None = None,
    user_token: str | None = None,
    history: list[dict[str, str]] | None = None,
    world_model: dict[str, Any] | None = None,
    memory_context: str | None = None,
    mongo_client: Any | None = None,
    db_name: str | None = None,
    world_store: Any | None = None,
    wm_service: Any | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Facade kết quả tương thích trên một đường event AgentRuntime duy nhất."""
    if model_name:
        from hagent.agent.llm import require_model_config

        require_model_config(model_name)

    from hagent.agent.runtime import (
        build_start_turn,
        collect_runtime_result,
        get_agent_runtime,
    )

    command, scope = build_start_turn(
        message,
        user_id=user_id,
        user_token=user_token,
        history=history,
        world_model=world_model,
        memory_context=memory_context,
        mongo_client=mongo_client,
        db_name=db_name,
        world_store=world_store,
        wm_service=wm_service,
        model_name=model_name,
    )
    result = await collect_runtime_result(
        get_agent_runtime(),
        command,
        scope=scope,
    )
    legacy_result = dict(result)
    legacy_result["response"] = legacy_result.pop(
        "message",
        legacy_result.get("response", ""),
    )
    return legacy_result


def _safe_json_parse(text: Any) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


# ── Streaming runner ─────────────────────────────────────


def _stream_response_from_state(
    final_state: dict[str, Any],
    *,
    model_name: str | None,
    route: str | None,
    cost_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Chỉ tạo kết quả agent công khai từ state cuối cùng ở root graph."""
    messages = list(final_state.get("messages") or [])
    last_ai_message = None
    tool_outputs = []
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            tool_outputs.append(
                {
                    "tool_name": msg.name,
                    "payload": _safe_json_parse(msg.content),
                }
            )
        elif (
            isinstance(msg, AIMessage)
            and msg.content
            and last_ai_message is None
            and not (hasattr(msg, "tool_calls") and msg.tool_calls)
        ):
            last_ai_message = msg

    plan_status = final_state.get("plan_status")
    selected_plan = final_state.get("selected_plan")
    planning = None
    if plan_status is not None or selected_plan is not None:
        planning = {"status": plan_status, "selected_plan": selected_plan}

    return {
        "response": last_ai_message.content if last_ai_message else "",
        "sources": [],
        "suggestions": [],
        "tool_outputs": list(reversed(tool_outputs)),
        "provider": "hagent",
        "model": model_name or "multi-agent",
        "route": final_state.get("current_phase") or route or "direct",
        "plan_status": plan_status,
        "selected_plan": selected_plan,
        "planning": planning,
        "surprise": final_state.get("surprise"),
        "cost_metrics": cost_metrics,
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


async def _stream_legacy_graph_events(
    message: str,
    *,
    user_id: str | None = None,
    user_token: str | None = None,
    history: list[dict[str, str]] | None = None,
    world_model: dict[str, Any] | None = None,
    memory_context: str | None = None,
    mongo_client: Any | None = None,
    db_name: str | None = None,
    world_store: Any | None = None,
    wm_service: Any | None = None,
    model_name: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Chạy graph cũ và chỉ phát các sự kiện truyền tải của nó một lần."""
    from hagent.agent.llm import config as llm_config
    from hagent.agent.middlewares.usage_tracker import (
        create_usage_tracker,
        reset_current_tracker,
        set_current_tracker,
    )
    from hagent.agent.orchestration import registry as registry_module

    if model_name:
        llm_config.require_model_config(model_name)
    model_token = llm_config.set_current_model_name(model_name)
    usage_token = None
    usage_tracker = None

    try:
        usage_tracker = create_usage_tracker()
        usage_token = set_current_tracker(usage_tracker)

        graph = get_automl_graph()
        registry = registry_module.get_agent_registry()
        valid_names = set(registry.agent_names()) | {
            "plan_executor",
            "reviser",
            "campaign",
            "hierarchy",
            "coordinator",
            "synthesize",
        }

        if wm_service is None or world_store is None:
            try:
                from hagent.world.runtime import build_wm_runtime

                runtime_wm, runtime_store = build_wm_runtime(
                    mongo_client=mongo_client, db_name=db_name
                )
                if wm_service is None:
                    wm_service = runtime_wm
                if world_store is None:
                    world_store = runtime_store
            except Exception as exc:  # noqa: BLE001
                logger.debug("WM runtime build skipped (stream): %s", exc)

        initial_state: AgentState = {
            "messages": _build_initial_messages(message, history),
            "world_model": world_model,
            "memory_context": memory_context,
            "user_id": user_id,
            "next_agent": None,
            "current_phase": None,
            "plan_step_index": 0,
            "revision_count": 0,
            "execution_log": [],
            "execution_events": [],
            "cost_metrics": {},
        }

        import time

        middleware = None
        middleware_wm_service = wm_service
        middleware_world_store = world_store
        try:
            from hagent.agent.middlewares import create_default_chain

            middleware = create_default_chain()
            middleware_state = dict(initial_state)
            if wm_service is not None:
                middleware_state["_wm_service"] = wm_service
            if world_store is not None:
                middleware_state["_world_store"] = world_store
            processed_state = await middleware.run_pre(middleware_state)
            middleware_wm_service = wm_service or processed_state.get("_wm_service")
            middleware_world_store = world_store or processed_state.get("_world_store")
            initial_state = cast(AgentState, dict(processed_state))
        except Exception as exc:  # noqa: BLE001
            logger.debug("stream middleware pre-process skipped: %s", exc)
            middleware = None

        initial_state.pop("_world_store", None)
        initial_state.pop("_wm_service", None)
        initial_state.pop("user_token", None)
        request_context = GraphRequestContext(
            principal_id=str(user_id) if user_id else "anonymous",
            credential=user_token or None,
            services={
                "mongo_client": mongo_client,
                "db_name": db_name,
                "world_store": middleware_world_store,
                "wm_service": middleware_wm_service,
            },
        )

        started_at = time.time()
        current_route = None
        last_events_len = 0
        final_state_snapshot: dict[str, Any] = {}
        root_final_state: dict[str, Any] = {}

        stream_kwargs: dict[str, Any] = {"version": "v2"}
        if _graph_stream_accepts_context(graph):
            stream_kwargs["context"] = request_context

        async for event in graph.astream_events(initial_state, **stream_kwargs):
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
                data = event.get("data") or {}
                output = data.get("output")
                if isinstance(output, dict):
                    final_state_snapshot.update(output)
                    if not event.get("parent_ids"):
                        root_final_state = dict(output)

                    events = output.get("execution_events") or []
                    if isinstance(events, list) and len(events) > last_events_len:
                        for execution_event in events[last_events_len:]:
                            yield {"type": "plan_event", "event": execution_event}
                            if (
                                isinstance(execution_event, dict)
                                and execution_event.get("type") == "step_end"
                                and execution_event.get("surprise")
                            ):
                                yield {
                                    "type": "surprise",
                                    "surprise": execution_event["surprise"],
                                    "index": execution_event.get("index"),
                                    "action": execution_event.get("action"),
                                }
                        last_events_len = len(events)

                    selected_plan = output.get("selected_plan")
                    if (
                        isinstance(selected_plan, dict)
                        and event.get("name") == "coordinator"
                    ):
                        yield {
                            "type": "plan",
                            "plan_id": selected_plan.get("plan_id"),
                            "title": selected_plan.get("title"),
                            "steps": [
                                (step.get("action") or {}).get("type")
                                if isinstance(step, dict)
                                else None
                                for step in (selected_plan.get("steps") or [])
                            ],
                            "cost": selected_plan.get("cost"),
                        }

            elif kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield {"type": "token", "content": chunk.content}

            elif kind == "on_tool_start":
                yield {
                    "type": "tool_call",
                    "tool": event.get("name", ""),
                    "args": event.get("data", {}).get("input", {}),
                }

            elif kind == "on_tool_end":
                raw_output = event.get("data", {}).get("output", "")
                if hasattr(raw_output, "content"):
                    raw_output = raw_output.content
                yield {
                    "type": "tool_result",
                    "tool": event.get("name", ""),
                    "output": _safe_json_parse(raw_output),
                }

        final_state = dict(final_state_snapshot)
        final_state.update(root_final_state)
        cost = dict(final_state.get("cost_metrics") or {})
        cost["elapsed_seconds"] = round(time.time() - started_at, 3)
        if usage_tracker is not None:
            cost.update(usage_tracker.summary())

        result = _stream_response_from_state(
            final_state,
            model_name=model_name,
            route=current_route,
            cost_metrics=cost,
        )
        if middleware is not None:
            middleware_state = dict(initial_state)
            if middleware_wm_service is not None:
                middleware_state["_wm_service"] = middleware_wm_service
            if middleware_world_store is not None:
                middleware_state["_world_store"] = middleware_world_store
            result = await middleware.run_post(middleware_state, result)

        public_result = dict(result)
        response_text = public_result.pop("response", public_result.get("message", ""))
        public_result["message"] = response_text
        yield {"type": "done", "response": public_result}
    finally:
        if usage_token is not None:
            reset_current_tracker(usage_token)
        llm_config.reset_current_model_name(model_token)


async def stream_agent(
    message: str,
    *,
    user_id: str | None = None,
    user_token: str | None = None,
    history: list[dict[str, str]] | None = None,
    world_model: dict[str, Any] | None = None,
    memory_context: str | None = None,
    mongo_client: Any | None = None,
    db_name: str | None = None,
    world_store: Any | None = None,
    wm_service: Any | None = None,
    model_name: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Facade stream tương thích trên một đường event AgentRuntime duy nhất."""
    from hagent.agent.runtime import (
        build_start_turn,
        get_agent_runtime,
        stream_legacy_events,
    )

    command, scope = build_start_turn(
        message,
        user_id=user_id,
        user_token=user_token,
        history=history,
        world_model=world_model,
        memory_context=memory_context,
        mongo_client=mongo_client,
        db_name=db_name,
        world_store=world_store,
        wm_service=wm_service,
        model_name=model_name,
    )
    async for event in stream_legacy_events(
        get_agent_runtime(),
        command,
        scope=scope,
    ):
        yield event
