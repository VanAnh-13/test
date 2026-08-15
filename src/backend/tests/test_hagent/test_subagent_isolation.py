"""
Unit tests for Subagent Isolation, Resource Limits and Timeout (REFAC-022).
"""

from __future__ import annotations

import asyncio

import pytest

from hagent.agent.subagents.manager import SubagentManager
from hagent.agent.subagents.specialist import execute_subagent
from hagent.core.errors import ExecutionError


class MockAgent:
    """Mock subagent để kiểm thử isolation."""

    def __init__(
        self, name: str = "mock_agent", delay: float = 0.01, fail: bool = False
    ) -> None:
        self.name = name
        self.delay = delay
        self.fail = fail

    async def run(self, state: dict) -> dict:
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise ValueError("Subagent internal failure")
        return {"messages": ["done"], "processed_by": self.name}


@pytest.mark.asyncio
async def test_subagent_normal_execution() -> None:
    """Subagent chạy bình thường trả về kết quả chính xác."""
    agent = MockAgent(name="fast_agent", delay=0.01)
    result = await execute_subagent(agent, state={"messages": []}, timeout_seconds=2.0)
    assert result["processed_by"] == "fast_agent"


@pytest.mark.asyncio
async def test_subagent_timeout_raises_execution_error() -> None:
    """Subagent chạy quá thời gian giới hạn sẽ raise ExecutionError chứa context."""
    slow_agent = MockAgent(name="slow_agent", delay=1.0)

    with pytest.raises(ExecutionError) as exc_info:
        await execute_subagent(slow_agent, state={}, timeout_seconds=0.05)

    err = exc_info.value
    assert "slow_agent" in str(err)
    assert err.context["subagent"] == "slow_agent"
    assert err.context["timeout_seconds"] == 0.05


@pytest.mark.asyncio
async def test_subagent_manager_timeout_isolation() -> None:
    """SubagentManager.execute_isolated bắt timeout và raise ExecutionError."""
    manager = SubagentManager(default_timeout=0.05)

    async def slow_work() -> str:
        await asyncio.sleep(0.5)
        return "too late"

    with pytest.raises(ExecutionError) as exc_info:
        await manager.execute_isolated("slow_worker", slow_work)

    assert "slow_worker" in str(exc_info.value)
    assert manager.active_count == 0


@pytest.mark.asyncio
async def test_subagent_manager_max_concurrent_limit() -> None:
    """SubagentManager từ chối thực thi khi số lượng concurrent subagents vượt quá giới hạn."""
    manager = SubagentManager(max_concurrent=2, default_timeout=2.0)

    started_event = asyncio.Event()
    stop_event = asyncio.Event()

    async def long_running_task() -> str:
        started_event.set()
        await stop_event.wait()
        return "completed"

    # Khởi động 2 tác vụ nền để lấp đầy slot concurrency (limit = 2)
    t1 = asyncio.create_task(manager.execute_isolated("worker_1", long_running_task))
    t2 = asyncio.create_task(manager.execute_isolated("worker_2", long_running_task))

    # Chờ 2 task bắt đầu
    await asyncio.sleep(0.05)
    assert manager.active_count == 2

    # Thử gọi task thứ 3 -> Phải bị từ chối với ExecutionError
    async def task_3() -> str:
        return "should not run"

    with pytest.raises(ExecutionError) as exc_info:
        await manager.execute_isolated("worker_3", task_3)

    assert "limit (2) exceeded" in str(exc_info.value)
    assert exc_info.value.context["max_concurrent"] == 2

    # Giải phóng các task nền
    stop_event.set()
    await asyncio.gather(t1, t2)

    assert manager.active_count == 0
