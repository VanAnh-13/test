"""
Unit tests for Multi-Agent Observability, Tracing and Metrics (REFAC-024).
"""

from __future__ import annotations

import asyncio

import pytest
from structlog.testing import capture_logs

from hagent.agent.subagents.manager import SubagentManager
from hagent.agent.subagents.protocol import create_request
from hagent.agent.subagents.specialist import execute_subagent
from hagent.core.errors import ExecutionError


class DummySubAgent:
    """Mock subagent for observability verification."""

    def __init__(self, name: str = "observability_agent", delay: float = 0.01) -> None:
        self.name = name
        self.delay = delay

    async def run(self, state: dict) -> dict:
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        return {"result": "success", "agent": self.name}


@pytest.mark.asyncio
async def test_subagent_invocation_tracing_logs() -> None:
    """Kiểm tra log records khi subagent thực thi thành công chứa span_id, latency_ms và status."""
    agent = DummySubAgent(name="test_analyst", delay=0.02)

    with capture_logs() as cap_logs:
        result = await execute_subagent(
            agent,
            state={},
            timeout_seconds=2.0,
            span_id="custom_span_123",
            parent_span_id="parent_span_000",
        )

    assert result["result"] == "success"

    # Tìm các sự kiện bắt đầu và hoàn thành trong captured logs
    start_events = [
        e for e in cap_logs if e.get("event") == "subagent_invocation_started"
    ]
    done_events = [
        e for e in cap_logs if e.get("event") == "subagent_invocation_completed"
    ]

    assert len(start_events) >= 1
    assert start_events[0]["span_id"] == "custom_span_123"
    assert start_events[0]["parent_span_id"] == "parent_span_000"
    assert start_events[0]["subagent"] == "test_analyst"

    assert len(done_events) >= 1
    assert done_events[0]["span_id"] == "custom_span_123"
    assert done_events[0]["status"] == "success"
    assert "latency_ms" in done_events[0]
    assert done_events[0]["latency_ms"] >= 10.0  # ít nhất delay 10ms


@pytest.mark.asyncio
async def test_subagent_invocation_failure_tracing_logs() -> None:
    """Kiểm tra log records khi subagent timeout chứa failure event và error details."""
    slow_agent = DummySubAgent(name="slow_timeout_agent", delay=0.5)

    with capture_logs() as cap_logs, pytest.raises(ExecutionError):
        await execute_subagent(
            slow_agent, state={}, timeout_seconds=0.03, span_id="span_timeout_456"
        )

    failed_events = [
        e for e in cap_logs if e.get("event") == "subagent_invocation_failed"
    ]
    assert len(failed_events) >= 1
    assert failed_events[0]["span_id"] == "span_timeout_456"
    assert failed_events[0]["status"] == "timeout"
    assert failed_events[0]["subagent"] == "slow_timeout_agent"


def test_agent_interaction_graph_logging() -> None:
    """Kiểm tra việc ghi log agent_interaction khi gửi message để phục dựng interaction graph."""
    manager = SubagentManager()
    manager.register_agent("agent_source")
    manager.register_agent("agent_target")

    msg = create_request(
        sender="agent_source",
        recipient="agent_target",
        payload={"action": "query_features"},
        correlation_id="corr_graph_789",
    )

    with capture_logs() as cap_logs:
        manager.send_message(msg)

    interaction_events = [e for e in cap_logs if e.get("event") == "agent_interaction"]
    assert len(interaction_events) == 1
    ev = interaction_events[0]
    assert ev["sender"] == "agent_source"
    assert ev["recipient"] == "agent_target"
    assert ev["correlation_id"] == "corr_graph_789"
    assert ev["message_type"] == "request"
