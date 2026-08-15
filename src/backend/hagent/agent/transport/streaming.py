"""Mã hóa Server-Sent Events có kiểu cho HAgent runtime."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_EVENT_TYPES = frozenset(
    {
        "meta",
        "route",
        "phase",
        "plan",
        "plan_event",
        "surprise",
        "token",
        "tool_call",
        "tool_result",
        "done",
        "error",
    }
)
_TERMINAL_EVENTS = frozenset({"done", "error"})


def _format_sse(event_name: str, event_id: int, payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event_name}\nid: {event_id}\ndata: {data}\n\n"


async def sse_stream(
    message: str,
    *,
    user_id: str | None = None,
    user_token: str | None = None,
    history: list[dict[str, str]] | None = None,
    world_model: dict[str, Any] | None = None,
    memory_context: str | None = None,
    mongo_client: Any | None = None,
    db_name: str | None = None,
    model_name: str | None = None,
    conversation_id: str | None = None,
) -> AsyncIterator[str]:
    """Mã hóa event AgentRuntime thành SSE frame cũ có kiểu."""
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
        model_name=model_name,
        trace_id=conversation_id,
    )
    agent_events = stream_legacy_events(
        get_agent_runtime(),
        command,
        scope=scope,
    )
    event_id = 0
    terminal_sent = False

    try:
        async for raw_event in agent_events:
            if not isinstance(raw_event, dict):
                raise TypeError("Agent stream event must be an object")
            event_name = raw_event.get("type")
            if event_name not in _EVENT_TYPES:
                raise ValueError("Agent stream emitted an unsupported event type")

            payload = dict(raw_event)
            if event_name == "done":
                response = payload.get("response")
                if not isinstance(response, dict):
                    raise TypeError("Agent done response must be an object")
                response = dict(response)
                if conversation_id:
                    response.setdefault("conversation_id", conversation_id)
                payload["response"] = response
            elif event_name == "error":
                payload = {
                    "type": "error",
                    "error": {
                        "code": "agent_stream_failed",
                        "message": "Agent stream failed",
                    },
                }

            next_event_id = event_id + 1
            frame = _format_sse(str(event_name), next_event_id, payload)
            event_id = next_event_id
            terminal_sent = event_name in _TERMINAL_EVENTS
            yield frame
            if terminal_sent:
                break

        if not terminal_sent:
            raise RuntimeError("Agent stream ended without a terminal event")
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("SSE streaming error: %s", type(exc).__name__)
        if not terminal_sent:
            event_id += 1
            terminal_sent = True
            yield _format_sse(
                "error",
                event_id,
                {
                    "type": "error",
                    "error": {
                        "code": "agent_stream_failed",
                        "message": "Agent stream failed",
                    },
                },
            )
    finally:
        close = getattr(agent_events, "aclose", None)
        if close is not None:
            try:
                await close()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("Agent stream close failed: %s", type(exc).__name__)
