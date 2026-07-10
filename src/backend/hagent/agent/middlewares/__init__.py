"""
DeerFlow-AutoML — Middleware Stack (Phase 3).

Pipeline pre/post processing cho agent graph:
- Pre: World model loading, memory injection, input validation
- Post: Fact extraction, world state update, response audit

SOLID:
  S — Mỗi middleware chỉ 1 trách nhiệm
  O — Thêm middleware mới qua YAML config
  D — Middleware chain inject, không hardcode
"""

from __future__ import annotations

import abc
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


# ── Base Middleware ───────────────────────────────────────


class Middleware(abc.ABC):
    """Abstract base cho middleware."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    async def pre_process(self, state: dict[str, Any]) -> dict[str, Any]:
        """Chạy trước graph invoke. Override để xử lý."""
        return state

    async def post_process(
        self, state: dict[str, Any], result: dict[str, Any],
    ) -> dict[str, Any]:
        """Chạy sau graph invoke. Override để xử lý."""
        return result


# ── Middleware Chain ──────────────────────────────────────


class MiddlewareChain:
    """
    Chạy danh sách middlewares theo thứ tự.
    Pre: đầu → cuối. Post: cuối → đầu (onion model).
    """

    def __init__(self, middlewares: list[Middleware] | None = None):
        self._middlewares = middlewares or []

    def add(self, middleware: Middleware) -> None:
        self._middlewares.append(middleware)
        logger.debug("Middleware added: %s", middleware.name)

    async def run_pre(self, state: dict[str, Any]) -> dict[str, Any]:
        for mw in self._middlewares:
            try:
                state = await mw.pre_process(state)
            except Exception as e:
                logger.error("Middleware '%s' pre_process failed: %s", mw.name, e)
        return state

    async def run_post(
        self, state: dict[str, Any], result: dict[str, Any],
    ) -> dict[str, Any]:
        for mw in reversed(self._middlewares):
            try:
                result = await mw.post_process(state, result)
            except Exception as e:
                logger.error("Middleware '%s' post_process failed: %s", mw.name, e)
        return result


# ── Concrete Middlewares ─────────────────────────────────


class TimingMiddleware(Middleware):
    """Đo thời gian xử lý."""

    @property
    def name(self) -> str:
        return "timing"

    async def pre_process(self, state: dict[str, Any]) -> dict[str, Any]:
        state["_start_time"] = time.time()
        return state

    async def post_process(
        self, state: dict[str, Any], result: dict[str, Any],
    ) -> dict[str, Any]:
        start = state.get("_start_time", 0)
        elapsed = time.time() - start
        result["_elapsed_seconds"] = round(elapsed, 3)
        logger.info("Agent processing time: %.3fs", elapsed)
        return result


class MemoryMiddleware(Middleware):
    """Pre: inject memory. Post: extract và save facts."""

    @property
    def name(self) -> str:
        return "memory"

    async def pre_process(self, state: dict[str, Any]) -> dict[str, Any]:
        from hagent.agent.memory import create_fact_store
        from hagent.agent.memory.injection import inject_memory_into_state

        store = create_fact_store()
        state["_fact_store"] = store
        return await inject_memory_into_state(store, state)

    async def post_process(
        self, state: dict[str, Any], result: dict[str, Any],
    ) -> dict[str, Any]:
        from hagent.agent.memory.extractor import extract_from_tool_message

        store = state.get("_fact_store")
        user_id = state.get("user_id")
        if not store or not user_id:
            return result

        messages = result.get("messages", [])
        for msg in messages:
            facts = extract_from_tool_message(msg, source="graph_output")
            for fact in facts:
                await store.save(user_id, fact)

        return result


class WorldModelMiddleware(Middleware):
    """
    Pre: ensure world_model snapshot + WorldModelService on state.
    Post: apply tool_outputs / plan fields back into snapshot patch (in-memory);
          persist when state_store is attached on state['_world_store'].
    """

    @property
    def name(self) -> str:
        return "world_model"

    async def pre_process(self, state: dict[str, Any]) -> dict[str, Any]:
        from hagent.world.service import WorldModelService

        # Always attach WM service (config-driven; works offline without Mongo)
        if "_wm_service" not in state:
            try:
                state["_wm_service"] = WorldModelService.from_config()
            except Exception as exc:
                logger.warning("WorldModelService init failed: %s", exc)

        user_id = state.get("user_id")
        store = state.get("_world_store")

        # Load snapshot from store when available and missing
        if user_id and store is not None and not state.get("world_model"):
            try:
                await store.ensure(str(user_id))
                snapshot = await store.get_snapshot(str(user_id))
                if snapshot:
                    state["world_model"] = snapshot
            except Exception as exc:
                logger.debug("WM middleware load failed: %s", exc)

        # Normalize empty snapshot
        if not state.get("world_model") and user_id:
            state["world_model"] = {
                "user_id": str(user_id),
                "datasets": {},
                "jobs": {},
                "plans": {},
                "goals": [],
                "phase": "idle",
            }

        return state

    async def post_process(
        self, state: dict[str, Any], result: dict[str, Any],
    ) -> dict[str, Any]:
        user_id = state.get("user_id")
        if not user_id:
            return result

        try:
            from hagent.world.schema import WorldState
            from hagent.world.updater import apply_plan_event, apply_tool_output

            snapshot = dict(state.get("world_model") or {"user_id": str(user_id)})
            ws = WorldState(
                user_id=str(user_id),
                datasets=dict(snapshot.get("datasets") or {}),
                jobs=dict(snapshot.get("jobs") or {}),
                goals=list(snapshot.get("goals") or []),
                plans=dict(snapshot.get("plans") or {}),
                active_plan_id=snapshot.get("active_plan_id"),
                active_dataset_id=snapshot.get("active_dataset_id"),
                active_job_id=snapshot.get("active_job_id"),
                active_goal=snapshot.get("active_goal"),
                phase=str(snapshot.get("phase") or "idle"),
                last_verification=snapshot.get("last_verification"),
                last_surprise=snapshot.get("last_surprise"),
                cost_metrics=dict(snapshot.get("cost_metrics") or {}),
            )

            merged_patch: dict[str, Any] = {}

            for tout in result.get("tool_outputs") or []:
                name = tout.get("tool_name") or tout.get("name")
                payload = tout.get("payload") or tout.get("content") or {}
                if isinstance(payload, str):
                    import json
                    try:
                        payload = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                if not name or not isinstance(payload, dict):
                    continue
                patch = apply_tool_output(ws, name, payload)
                if patch:
                    for k, v in patch.items():
                        setattr(ws, k, v) if hasattr(ws, k) else None
                        merged_patch[k] = v

            selected = result.get("selected_plan") or state.get("selected_plan")
            if isinstance(selected, dict) and selected.get("plan_id"):
                pe = apply_plan_event(
                    ws,
                    "plan_selected",
                    {**selected, "plan_id": selected["plan_id"]},
                )
                merged_patch.update(pe)

            surprise = result.get("surprise") or state.get("surprise")
            if isinstance(surprise, dict):
                pe = apply_plan_event(ws, "surprise_recorded", {"surprise": surprise})
                merged_patch.update(pe)

            updated_snapshot = ws.to_dict()
            result["world_model"] = updated_snapshot
            state["world_model"] = updated_snapshot

            store = state.get("_world_store")
            if store is not None and merged_patch:
                await store.apply_patch(str(user_id), merged_patch)
                result["world_model_updated"] = True
        except Exception as exc:
            logger.debug("WM middleware post failed: %s", exc)

        return result


class InputSanitizer(Middleware):
    """Pre: sanitize user input."""

    @property
    def name(self) -> str:
        return "input_sanitizer"

    async def pre_process(self, state: dict[str, Any]) -> dict[str, Any]:
        messages = state.get("messages", [])
        if messages:
            last = messages[-1]
            if hasattr(last, "content") and last.content:
                content = last.content.strip()
                if len(content) > 10000:
                    content = content[:10000] + "\n\n[... truncated]"
                    logger.warning("Input truncated to 10000 chars")
        return state


# ── Factory ──────────────────────────────────────────────


def create_default_chain() -> MiddlewareChain:
    """
    Tạo middleware chain mặc định từ config.
    Thứ tự: timing → input_sanitizer → world_model → memory
    """
    try:
        from hagent.bridge.config import load_config
        cfg = load_config()
        memory_enabled = cfg.get("memory", {}).get("enabled", True)
    except Exception:
        memory_enabled = True

    chain = MiddlewareChain()
    chain.add(TimingMiddleware())
    chain.add(InputSanitizer())
    chain.add(WorldModelMiddleware())

    if memory_enabled:
        chain.add(MemoryMiddleware())

    logger.info(
        "Middleware chain created: %s",
        [mw.name for mw in chain._middlewares],
    )
    return chain
