"""
Unit tests for Subagent Load Balancing and Priority Queue Scheduling (REFAC-023).
"""

from __future__ import annotations

import asyncio

import pytest

from hagent.agent.subagents.manager import SubagentManager, TaskPriority
from hagent.core.errors import ExecutionError


@pytest.mark.asyncio
async def test_subagent_schedule_single_task() -> None:
    """Task đơn lẻ được schedule và hoàn thành thành công."""
    manager = SubagentManager(max_concurrent=2, default_timeout=2.0)
    try:

        async def work() -> str:
            await asyncio.sleep(0.01)
            return "done"

        result = await manager.schedule_task(
            "analyst", work, priority=TaskPriority.NORMAL
        )
        assert result == "done"
        metrics = manager.get_metrics()
        assert metrics["completed_tasks"] == 1
        assert metrics["failed_tasks"] == 0
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_subagent_priority_queue_ordering() -> None:
    """Tác vụ có độ ưu tiên cao (HIGH) được thực thi trước tác vụ độ ưu tiên thấp (BACKGROUND) khi slot mở."""
    manager = SubagentManager(max_concurrent=1, default_timeout=2.0)
    execution_order: list[str] = []
    task1_release = asyncio.Event()

    try:
        # Task 1: Chiếm slot duy nhất (max_concurrent=1)
        async def task_1() -> str:
            await task1_release.wait()
            execution_order.append("task_1")
            return "t1"

        # Task 2: BACKGROUND (priority=2)
        async def task_bg() -> str:
            execution_order.append("task_bg")
            return "t_bg"

        # Task 3: HIGH (priority=0)
        async def task_high() -> str:
            execution_order.append("task_high")
            return "t_high"

        # Bắt đầu task 1 trước
        fut1 = asyncio.create_task(
            manager.schedule_task("worker_1", task_1, priority=TaskPriority.NORMAL)
        )
        await asyncio.sleep(0.02)  # Đảm bảo task_1 đã chiếm slot

        # Đẩy 2 task tiếp theo vào hàng đợi
        fut_bg = asyncio.create_task(
            manager.schedule_task(
                "worker_bg", task_bg, priority=TaskPriority.BACKGROUND
            )
        )
        fut_high = asyncio.create_task(
            manager.schedule_task("worker_high", task_high, priority=TaskPriority.HIGH)
        )

        await asyncio.sleep(0.02)
        assert manager.queued_count == 2

        # Cho phép task 1 hoàn thành để giải phóng slot
        task1_release.set()

        await asyncio.gather(fut1, fut_bg, fut_high)

        # Thứ tự hoàn thành phải là: task_1 -> task_high -> task_bg
        assert execution_order == ["task_1", "task_high", "task_bg"]
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_subagent_scheduled_task_timeout() -> None:
    """Tác vụ trong hàng đợi bị timeout sẽ raise ExecutionError và tăng failed_count."""
    manager = SubagentManager(max_concurrent=2, default_timeout=0.05)
    try:

        async def slow_work() -> str:
            await asyncio.sleep(0.5)
            return "finished"

        with pytest.raises(ExecutionError) as exc_info:
            await manager.schedule_task("slow_agent", slow_work)

        assert "slow_agent" in str(exc_info.value)
        metrics = manager.get_metrics()
        assert metrics["failed_tasks"] == 1
    finally:
        await manager.shutdown()
