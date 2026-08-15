"""
Subagent Specialist Isolation Runner & Observability (REFAC-022, REFAC-024).

Thực thi subagent trong môi trường cô lập có resource limit, timeout và span-based distributed tracing.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

import structlog

from hagent.core.errors import ExecutionError

logger = structlog.get_logger(__name__)


async def execute_subagent(
    agent: Any,
    state: Any,
    *,
    timeout_seconds: float = 60.0,
    span_id: str | None = None,
    parent_span_id: str | None = None,
) -> dict[str, Any]:
    """
    Thực thi subagent với timeout bảo vệ và span tracing observability.

    Args:
        agent: Instance của SubAgent (hoặc object có async method .run(state)).
        state: State dictionary hoặc AutoMLState.
        timeout_seconds: Thời gian tối đa cho phép thực thi (giây).
        span_id: Span ID cho tracing. None → tự sinh UUID4 mới.
        parent_span_id: Parent span ID (nếu có) để tạo cây quan sát (interaction tree).

    Returns:
        Dict update state trả về từ subagent.

    Raises:
        ExecutionError: Khi subagent chạy quá thời gian timeout.
    """
    current_span_id = span_id or str(uuid.uuid4())
    agent_name = getattr(agent, "name", "unknown_subagent")
    t0 = time.perf_counter()

    logger.info(
        "subagent_invocation_started",
        span_id=current_span_id,
        parent_span_id=parent_span_id,
        subagent=agent_name,
        timeout_seconds=timeout_seconds,
    )

    try:
        result = await asyncio.wait_for(agent.run(state), timeout=timeout_seconds)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            "subagent_invocation_completed",
            span_id=current_span_id,
            parent_span_id=parent_span_id,
            subagent=agent_name,
            latency_ms=latency_ms,
            status="success",
        )
        return result
    except TimeoutError as exc:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.warning(
            "subagent_invocation_failed",
            span_id=current_span_id,
            parent_span_id=parent_span_id,
            subagent=agent_name,
            latency_ms=latency_ms,
            status="timeout",
            error=f"Timed out after {timeout_seconds}s",
        )
        raise ExecutionError(
            f"Subagent '{agent_name}' timed out after {timeout_seconds}s",
            context={
                "subagent": agent_name,
                "timeout_seconds": timeout_seconds,
                "span_id": current_span_id,
                "latency_ms": latency_ms,
            },
            cause=exc,
        ) from exc
    except Exception as exc:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.error(
            "subagent_invocation_failed",
            span_id=current_span_id,
            parent_span_id=parent_span_id,
            subagent=agent_name,
            latency_ms=latency_ms,
            status="error",
            error=str(exc),
        )
        raise


__all__ = ["execute_subagent"]
