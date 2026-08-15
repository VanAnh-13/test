"""
Agent điều phối chính của HAgent.

Agent điều phối quyết định định tuyến hoặc trả lời trực tiếp.
"""

from __future__ import annotations

import re
from typing import Any

import structlog
from langchain_core.messages import AIMessage, SystemMessage

from hagent.agent.orchestration.state import AutoMLState

logger = structlog.get_logger(__name__)


# ── World Model formatting ───────────────────────────────


def _format_world_model_summary(world_model: dict[str, Any] | None) -> str:
    """Định dạng snapshot World Model cho system prompt qua world.query."""
    if not world_model:
        return "Chưa có dữ liệu World Model."
    try:
        from hagent.world.query import format_for_prompt

        return format_for_prompt(world_model)
    except Exception:  # noqa: BLE001
        datasets = world_model.get("datasets", {})
        jobs = world_model.get("jobs", {})
        lines = []
        if datasets:
            ds_names = [
                f"- {did}: {d.get('name', '?')}"
                for did, d in list(datasets.items())[:10]
            ]
            lines.append(f"**Datasets ({len(datasets)}):**\n" + "\n".join(ds_names))
        else:
            lines.append("**Datasets:** Chưa có")
        if jobs:
            job_summaries = [
                f"- {jid}: status={j.get('status', '?')}"
                for jid, j in list(jobs.items())[:10]
            ]
            lines.append(f"**Jobs ({len(jobs)}):**\n" + "\n".join(job_summaries))
        else:
            lines.append("**Jobs:** Chưa có")
        return "\n".join(lines)


def _attach_latent_plan(
    state: AutoMLState,
    user_message: str,
) -> dict[str, Any]:
    """
    Phân tích goal và tạo plan CEM-lite dựa trên World Model.
    Chỉ trả về các trường state, không có message, và bỏ qua truy vấn đơn giản.
    """
    try:
        from hagent.agent.constraints import validate_plan_steps
        from hagent.agent.planning.goal_parser import is_simple_query, parse_goal
        from hagent.agent.planning.plan_adapter import plan_results_to_state_update
        from hagent.bridge.config import get_planning_config
        from hagent.world.service import WorldModelService
    except Exception as exc:  # noqa: BLE001
        logger.debug("Planning imports failed: %s", exc)
        return {}

    planning_cfg = get_planning_config()
    if not planning_cfg.get("enabled", True):
        return {}
    if planning_cfg.get("skip_planner_for_simple_queries", True) and is_simple_query(
        user_message,
        planning_cfg.get("simple_query_keywords"),
    ):
        return {}

    wm_service = state.get("_wm_service")  # type: ignore[assignment]
    if wm_service is None:
        wm_service = WorldModelService.from_config()

    snapshot = state.get("world_model") or {"user_id": state.get("user_id") or ""}
    known_ids = list((snapshot.get("datasets") or {}).keys())
    goal = parse_goal(user_message, known_dataset_ids=known_ids)

    # Chỉ chạy bộ lập kế hoạch latent cho các goal không phải phản hồi trực tiếp.
    if goal.get("goal_type") in (None, "respond"):
        return {"goal": goal}

    obs = wm_service.observation_from_snapshot(
        snapshot, user_id=state.get("user_id"), goal=goal
    )
    plans = wm_service.plan(obs, goal)
    update = plan_results_to_state_update(plans, select_best=True)
    update["goal"] = goal
    update["user_requirements"] = goal

    selected = update.get("selected_plan")
    if selected:
        steps = selected.get("steps") or []
        verification = validate_plan_steps(steps, obs, goal=goal)
        update["plan_verification"] = verification.to_dict()
        z = wm_service.encode(obs)
        update["latent"] = z.to_dict()
        update["cost_metrics"] = {
            "plans_generated": len(plans),
            "best_cost": plans[0].cost if plans else None,
        }
        # Ưu tiên agent của bước plan đầu tiên nếu hợp lệ.
        if steps and not state.get("next_agent"):
            first = steps[0]
            agent = None
            if isinstance(first, dict):
                agent = first.get("agent")
            if agent:
                update["_suggested_agent"] = agent

    return update


# ── Routing logic (config-driven) ────────────────────────


def _get_valid_agents() -> set[str]:
    """Lấy danh sách tên agent từ AgentRegistry đọc từ YAML."""
    from hagent.agent.orchestration import get_agent_registry

    return get_agent_registry().agent_names()


def keyword_route(message: str) -> str | None:
    """
    Định tuyến nhanh theo từ khóa đọc từ hagent.yaml.
    Tên agent cũng lấy từ YAML, không mã hóa cứng.
    """
    from hagent.bridge.config import get_routing_config

    routing = get_routing_config()
    if not routing:
        return None

    valid_agents = _get_valid_agents()
    lower = message.lower()
    scores: dict[str, int] = {}

    for agent_name, agent_cfg in routing.items():
        # Chỉ route tới agents đã đăng ký trong registry
        if agent_name not in valid_agents:
            continue
        keywords = (
            agent_cfg if isinstance(agent_cfg, list) else agent_cfg.get("keywords", [])
        )
        score = sum(1 for kw in keywords if kw in lower)
        if score > 0:
            scores[agent_name] = score

    if not scores:
        return None
    return max(scores, key=scores.get)


def parse_coordinator_response(response_text: str) -> tuple[str | None, str]:
    """
    Phân tích response của agent điều phối để lấy quyết định định tuyến.
    Định dạng: [ROUTE:agent_name] kèm nội dung lý do.
    Kiểm tra agent_name theo registry.
    """
    match = re.match(r"\[ROUTE:(\w+)\]\s*(.*)", response_text, re.DOTALL)
    if match:
        agent_name = match.group(1).strip()
        reason = match.group(2).strip()
        valid_agents = _get_valid_agents()
        if agent_name in valid_agents:
            return agent_name, reason
    return None, response_text


# ── Dynamic routing instruction builder ──────────────────


def _build_routing_instruction() -> str:
    """
    Tạo chỉ dẫn định tuyến cho system prompt.
    Đọc tên và mô tả agent từ YAML, không mã hóa cứng.
    """
    from hagent.bridge.config import get_routing_config

    valid_agents = _get_valid_agents()
    routing = get_routing_config()

    lines = [
        "\n## Routing Protocol",
        "Bạn là Lead Agent (Coordinator). Dựa trên yêu cầu, quyết định route tới sub-agent phù hợp:\n",
    ]

    for agent_name in sorted(valid_agents):
        # Lấy keywords làm mô tả ngắn
        keywords = routing.get(agent_name, {})
        if isinstance(keywords, dict):
            keywords = keywords.get("keywords", [])
        kw_preview = ", ".join(keywords[:5]) if keywords else "N/A"
        lines.append(f"- **{agent_name}**: Khi yêu cầu liên quan tới: {kw_preview}")

    lines.extend(
        [
            "\n### Cách route:",
            "- Nếu cần chuyển cho sub-agent: bắt đầu response bằng `[ROUTE:agent_name]` rồi giải thích ngắn.",
            "- Nếu trả lời trực tiếp (chào hỏi, câu hỏi chung): KHÔNG dùng [ROUTE:], trả lời bình thường.",
            "- Nếu cần dùng tools: gọi tools trực tiếp (không cần route).",
        ]
    )

    return "\n".join(lines)


# ── System prompt loader ─────────────────────────────────


def _load_system_prompt(world_model: dict[str, Any] | None = None) -> str:
    """
    Tải system prompt từ file được cấu hình trong hagent.yaml.
    Chèn bản tóm tắt World Model và chỉ dẫn định tuyến động.
    """
    from hagent.bridge.config import load_prompt_file

    try:
        template = load_prompt_file()
    except FileNotFoundError:
        logger.warning("Không tìm thấy prompt file, dùng fallback minimal.")
        template = (
            "Bạn là HAgent, trợ lý AI cho HAutoML. "
            "Trả lời bằng ngôn ngữ người dùng sử dụng.\n\n"
            "## World Model\n{world_model_summary}"
        )

    base_prompt = template.format(
        world_model_summary=_format_world_model_summary(world_model),
    )

    # Routing instruction build dynamic từ YAML
    routing_instruction = _build_routing_instruction()
    return base_prompt + "\n" + routing_instruction


# ── Coordinator node function ────────────────────────────


async def coordinator_node(state: AutoMLState) -> dict:
    """
    Node LangGraph nơi agent điều phối quyết định định tuyến hoặc trả lời trực tiếp.

    Chiến lược định tuyến gồm hai tầng:
    1. Định tuyến nhanh theo từ khóa trong cấu hình YAML.
    2. Định tuyến bằng LLM khi response chứa [ROUTE:X].
    """
    from hagent.agent.llm import create_chat_model
    from hagent.agent.tools.automl_tools import ALL_TOOLS

    messages = state["messages"]
    world_model = state.get("world_model")

    # ── Bước 1: Quick keyword route ──────────────────────
    last_user_msg = ""
    for msg in reversed(messages):
        if hasattr(msg, "content") and getattr(msg, "type", "") == "human":
            last_user_msg = msg.content or ""
            break

    # Tạo plan latent theo LeWM trước khi định tuyến và gắn goal, plan, verification.
    plan_fields = _attach_latent_plan(state, last_user_msg)

    quick_route = keyword_route(last_user_msg)
    valid_agents = _get_valid_agents()

    # Ưu tiên chuyên gia trong plan nếu thiếu định tuyến theo từ khóa.
    suggested = plan_fields.pop("_suggested_agent", None)

    # Với plan latent hoặc goal huấn luyện, dùng campaign hoặc plan_executor.
    if plan_fields.get("selected_plan") or plan_fields.get("goal"):
        goal = plan_fields.get("goal") or {}
        gtype = str(goal.get("goal_type") or "")
        if gtype and gtype != "respond":
            verification = plan_fields.get("plan_verification") or {}
            plan_fields.setdefault("plan_status", "ready")
            plan_fields.setdefault("plan_step_index", 0)
            plan_fields.setdefault("revision_count", 0)

            # Giai đoạn 6: campaign nhiều ứng viên cho goal huấn luyện.
            use_campaign = False
            try:
                from hagent.bridge.config import get_campaign_config

                ccfg = get_campaign_config()
                prefer = {
                    str(t).lower()
                    for t in (ccfg.get("prefer_for_goal_types") or ["train"])
                }
                use_campaign = (
                    bool(ccfg.get("enabled", True)) and gtype.lower() in prefer
                )
            except Exception:  # noqa: BLE001
                use_campaign = gtype.lower() == "train"

            # Hierarchy thích ứng với bộ điều khiển trực tiếp cho huấn luyện hoặc đánh giá.
            hierarchy_payload: dict = {}
            try:
                from hagent.agent.planning.hierarchy import (
                    apply_smart_skips,
                    decompose_goal,
                )
                from hagent.bridge.config import get_hierarchy_config

                hcfg = get_hierarchy_config()
                if hcfg.get("enabled", True) and gtype in ("train", "evaluate"):
                    hier = decompose_goal(goal)
                    skips = apply_smart_skips(
                        hier,
                        world_model=state.get("world_model"),
                    )
                    hierarchy_payload = {
                        "hierarchy": hier.to_dict(),
                        "hierarchy_status": "running",
                    }
                    if skips:
                        hierarchy_payload.setdefault("execution_events", [])
                        # Graph sẽ gộp sự kiện sau; tạm lưu số lượt bỏ qua trong message.
                        hierarchy_payload["_skip_count"] = len(skips)
            except Exception:  # noqa: BLE001, S110
                pass

            live_h = bool(hierarchy_payload.get("hierarchy")) and bool(
                hierarchy_payload.get("hierarchy_status") == "running"
            )
            try:
                from hagent.bridge.config import get_hierarchy_config as _gh

                live_h = live_h and bool(_gh().get("live_controller", True))
            except Exception:  # noqa: BLE001, S110
                pass

            if (
                live_h
                and goal.get("dataset_id")
                and (gtype != "train" or goal.get("target_column"))
            ):
                n_sub = len(
                    (hierarchy_payload.get("hierarchy") or {}).get("subgoals") or []
                )
                n_skip = int(hierarchy_payload.pop("_skip_count", 0) or 0)
                route_msg = AIMessage(
                    content=(
                        f"Goal `{gtype}` → adaptive hierarchy "
                        f"({n_sub} subgoals, smart-skip={n_skip})."
                    )
                )
                return {
                    "messages": [route_msg],
                    "next_agent": None,
                    **plan_fields,
                    **hierarchy_payload,
                }

            if use_campaign and goal.get("dataset_id") and goal.get("target_column"):
                route_msg = AIMessage(
                    content=(
                        f"Goal `{gtype}` → multi-candidate campaign "
                        f"(warm-start + parallel jobs)."
                    )
                )
                return {
                    "messages": [route_msg],
                    "next_agent": None,
                    **plan_fields,
                    **{
                        k: v for k, v in hierarchy_payload.items() if k != "_skip_count"
                    },
                }

            route_msg = AIMessage(
                content=(
                    f"Đã lập latent plan ({gtype}). "
                    f"Chuyển plan_executor. "
                    f"verify={verification.get('ok', verification.get('pass', True))}."
                )
            )
            return {
                "messages": [route_msg],
                "next_agent": None,
                **plan_fields,
            }

    if quick_route and quick_route in valid_agents:
        logger.info("Quick route → %s (keyword match)", quick_route)
        route_msg = AIMessage(
            content=f"[ROUTE:{quick_route}] Chuyển yêu cầu tới {quick_route}."
        )
        return {
            "messages": [route_msg],
            "next_agent": quick_route,
            **plan_fields,
        }

    if suggested and suggested in valid_agents and plan_fields.get("selected_plan"):
        logger.info("Plan-suggested route → %s", suggested)
        route_msg = AIMessage(
            content=f"[ROUTE:{suggested}] Theo latent plan → {suggested}."
        )
        return {
            "messages": [route_msg],
            "next_agent": suggested,
            **plan_fields,
        }

    # ── Bước 2: LLM-based routing ────────────────────────
    system_prompt = _load_system_prompt(world_model)

    # Chèn bản tóm tắt plan khi có.
    if plan_fields.get("selected_plan"):
        sp = plan_fields["selected_plan"]
        system_prompt += (
            f"\n\n## Latent Plan (CEM-lite)\n"
            f"title={sp.get('title')}, cost={sp.get('cost')}, "
            f"steps={sp.get('meta', {}).get('action_types') or sp.get('steps')}\n"
            f"verification={plan_fields.get('plan_verification')}"
        )

    try:
        llm = create_chat_model()
        llm_with_tools = llm.bind_tools(ALL_TOOLS)

        prompt_parts = [system_prompt]
        memory_ctx = state.get("memory_context")
        if memory_ctx:
            prompt_parts.append(f"\n## Trí nhớ dài hạn\n{memory_ctx}")

        full_system = "\n".join(prompt_parts)
        full_messages = [SystemMessage(content=full_system)] + list(messages)

        response = await llm_with_tools.ainvoke(full_messages)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Coordinator LLM invocation/classification failed: %s. Falling back to direct coordinator response.",
            exc,
        )
        fallback_msg = AIMessage(
            content="Xin chào, tôi là trợ lý HAgent. Tôi có thể hỗ trợ bạn khám phá dữ liệu, chọn mô hình hoặc huấn luyện tự động. Bạn muốn bắt đầu tác vụ gì?"
        )
        return {
            "messages": [fallback_msg],
            "next_agent": None,
            **plan_fields,
        }

    # ── Bước 3: Parse routing decision ───────────────────
    state_update: dict[str, Any] = {"messages": [response], **plan_fields}

    if response.content and not (
        hasattr(response, "tool_calls") and response.tool_calls
    ):
        route_target, _ = parse_coordinator_response(response.content)
        if route_target:
            state_update["next_agent"] = route_target
            logger.info("LLM route → %s", route_target)
        else:
            state_update["next_agent"] = None

    return state_update
