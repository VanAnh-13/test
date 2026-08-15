"""
Subagent Manager (REFAC-021, REFAC-022, REFAC-023).

Quản lý đăng ký, điều phối tin nhắn typed, cô lập tài nguyên và load balancing theo hàng đợi ưu tiên (Priority Queue Scheduling).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import structlog

from hagent.agent.subagents.protocol import (
    AgentMessage,
    MessageType,
    create_error,
    create_event,
    create_request,
    create_response,
)
from hagent.core.errors import ExecutionError

logger = structlog.get_logger(__name__)


class TaskPriority(IntEnum):
    """Mức độ ưu tiên của tác vụ subagent (số nhỏ hơn = ưu tiên cao hơn)."""

    HIGH = 0
    NORMAL = 1
    BACKGROUND = 2


@dataclass(order=True)
class QueuedTask:
    """Tác vụ chờ trong hàng đợi ưu tiên."""

    priority: int
    entry_time: float
    agent_name: str = field(compare=False)
    coroutine_factory: Callable[[], Coroutine[Any, Any, Any]] = field(compare=False)
    future: asyncio.Future[Any] = field(compare=False)
    timeout: float = field(compare=False)


def _load_subagents_config() -> tuple[int, float]:
    """Tải cấu hình subagents từ hagent.yaml với fallback an toàn."""
    try:
        from hagent.bridge.config import load_config

        cfg = load_config().get("subagents", {})
        max_concurrent = int(cfg.get("max_concurrent_subagents", 4))
        timeout_seconds = float(cfg.get("timeout_seconds", 60.0))
        return max_concurrent, timeout_seconds
    except Exception:  # noqa: BLE001
        return 4, 60.0


class SubagentManager:
    """Quản lý các sub-agents, hàng đợi tin nhắn, resource limits và priority scheduling."""

    def __init__(
        self,
        *,
        max_concurrent: int | None = None,
        default_timeout: float | None = None,
    ) -> None:
        cfg_max_concurrent, cfg_timeout = _load_subagents_config()
        self.max_concurrent_subagents = (
            max_concurrent if max_concurrent is not None else cfg_max_concurrent
        )
        self.default_timeout = (
            default_timeout if default_timeout is not None else cfg_timeout
        )

        self._inbox: dict[str, list[AgentMessage]] = {}
        self._registered_agents: set[str] = set()

        # Concurrency & Metrics
        self._active_count: int = 0
        self._completed_count: int = 0
        self._failed_count: int = 0

        # Priority Queue Scheduling
        self._queue: asyncio.PriorityQueue[QueuedTask] = asyncio.PriorityQueue()
        self._workers: list[asyncio.Task[None]] = []
        self._lock: asyncio.Lock | None = None
        self._running: bool = False

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @property
    def active_count(self) -> int:
        """Số lượng subagent task đang chạy đồng thời."""
        return self._active_count

    @property
    def queued_count(self) -> int:
        """Số lượng subagent task đang chờ trong hàng đợi."""
        return self._queue.qsize()

    @property
    def completed_count(self) -> int:
        """Tổng số tác vụ đã hoàn thành thành công."""
        return self._completed_count

    @property
    def failed_count(self) -> int:
        """Tổng số tác vụ bị lỗi hoặc timeout."""
        return self._failed_count

    def get_metrics(self) -> dict[str, Any]:
        """Trả về thống kê tải và tài nguyên của hệ thống subagents."""
        return {
            "max_concurrent": self.max_concurrent_subagents,
            "active_tasks": self.active_count,
            "queued_tasks": self.queued_count,
            "completed_tasks": self.completed_count,
            "failed_tasks": self.failed_count,
            "registered_agents": list(self._registered_agents),
        }

    def register_agent(self, agent_name: str) -> None:
        """Đăng ký agent vào manager."""
        self._registered_agents.add(agent_name)
        self._inbox.setdefault(agent_name, [])
        logger.debug("Registered subagent: %s", agent_name)

    def is_registered(self, agent_name: str) -> bool:
        """Kiểm tra agent đã đăng ký hay chưa."""
        return agent_name in self._registered_agents

    def send_message(self, message: AgentMessage) -> None:
        """Gửi message tới recipient hoặc broadcast tới toàn bộ registered agents."""
        logger.info(
            "agent_interaction",
            sender=message.sender,
            recipient=message.recipient,
            message_id=message.id,
            message_type=message.type.value
            if hasattr(message.type, "value")
            else str(message.type),
            correlation_id=message.correlation_id,
        )
        if message.recipient == "broadcast":
            for agent in self._registered_agents:
                if agent != message.sender:
                    self._inbox.setdefault(agent, []).append(message)
            logger.debug("Broadcasted message %s from %s", message.id, message.sender)
        else:
            self._inbox.setdefault(message.recipient, []).append(message)
            logger.debug(
                "Routed message %s from %s to %s",
                message.id,
                message.sender,
                message.recipient,
            )

    def receive_messages(self, agent_name: str) -> list[AgentMessage]:
        """Lấy toàn bộ tin nhắn đang chờ trong hộp thư của agent."""
        messages = self._inbox.get(agent_name, [])
        self._inbox[agent_name] = []
        return messages

    async def _worker_loop(self) -> None:
        """Worker background loop lấy task từ priority queue và thực thi."""
        while self._running:
            try:
                task = await self._queue.get()
            except asyncio.CancelledError:
                break

            lock = self._get_lock()
            async with lock:
                self._active_count += 1

            try:
                coro = task.coroutine_factory()
                result = await asyncio.wait_for(coro, timeout=task.timeout)
                async with lock:
                    self._completed_count += 1
                if not task.future.done():
                    task.future.set_result(result)
            except TimeoutError as exc:
                async with lock:
                    self._failed_count += 1
                err = ExecutionError(
                    f"Subagent '{task.agent_name}' timed out after {task.timeout}s",
                    context={
                        "subagent": task.agent_name,
                        "timeout_seconds": task.timeout,
                    },
                    cause=exc,
                )
                if not task.future.done():
                    task.future.set_exception(err)
            except Exception as exc:  # noqa: BLE001
                async with lock:
                    self._failed_count += 1
                if not task.future.done():
                    task.future.set_exception(exc)
            finally:
                async with lock:
                    self._active_count = max(0, self._active_count - 1)
                self._queue.task_done()

    def _ensure_workers(self) -> None:
        """Khởi động worker pool nếu chưa chạy."""
        if not self._running:
            self._running = True
            for _ in range(self.max_concurrent_subagents):
                worker = asyncio.create_task(self._worker_loop())
                self._workers.append(worker)

    async def schedule_task(
        self,
        agent_name: str,
        coroutine_factory: Callable[[], Coroutine[Any, Any, Any]],
        *,
        priority: TaskPriority | int = TaskPriority.NORMAL,
        timeout: float | None = None,
    ) -> Any:
        """
        Đưa tác vụ vào hàng đợi ưu tiên để thực thi theo tài nguyên khả dụng.

        Args:
            agent_name: Tên của subagent.
            coroutine_factory: Hàm trả về coroutine để thực thi.
            priority: Mức ưu tiên (0: HIGH, 1: NORMAL, 2: BACKGROUND).
            timeout: Thời gian timeout (giây).

        Returns:
            Kết quả thực thi từ coroutine.

        Raises:
            ExecutionError: Khi timeout hoặc lỗi thực thi.
        """
        self._ensure_workers()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        effective_timeout = timeout if timeout is not None else self.default_timeout

        queued_task = QueuedTask(
            priority=int(priority),
            entry_time=time.monotonic(),
            agent_name=agent_name,
            coroutine_factory=coroutine_factory,
            future=future,
            timeout=effective_timeout,
        )

        await self._queue.put(queued_task)
        logger.debug(
            "Scheduled task for subagent '%s' with priority %s (queue size: %d)",
            agent_name,
            priority,
            self.queued_count,
        )
        return await future

    async def execute_isolated(
        self,
        agent_name: str,
        coroutine_factory: Callable[[], Coroutine[Any, Any, Any]],
        *,
        timeout: float | None = None,
    ) -> Any:
        """
        Thực thi subagent trực tiếp có kiểm soát concurrency và timeout.
        Nếu vượt quá concurrency limit ngay lập tức, raise ExecutionError (fail-fast).
        Dùng schedule_task nếu muốn xếp hàng (queue-based).
        """
        lock = self._get_lock()
        async with lock:
            if self._active_count >= self.max_concurrent_subagents:
                logger.warning(
                    "Exceeded max concurrent subagents limit (%d, active: %d)",
                    self.max_concurrent_subagents,
                    self._active_count,
                )
                raise ExecutionError(
                    f"Maximum concurrent subagents limit ({self.max_concurrent_subagents}) exceeded",
                    context={
                        "max_concurrent": self.max_concurrent_subagents,
                        "active": self._active_count,
                        "agent": agent_name,
                    },
                )
            self._active_count += 1

        effective_timeout = timeout if timeout is not None else self.default_timeout
        try:
            coro = coroutine_factory()
            result = await asyncio.wait_for(coro, timeout=effective_timeout)
            async with lock:
                self._completed_count += 1
            return result
        except TimeoutError as exc:
            async with lock:
                self._failed_count += 1
            logger.warning(
                "Subagent '%s' timed out after %.1fs", agent_name, effective_timeout
            )
            raise ExecutionError(
                f"Subagent '{agent_name}' timed out after {effective_timeout}s",
                context={"subagent": agent_name, "timeout_seconds": effective_timeout},
                cause=exc,
            ) from exc
        except Exception:
            async with lock:
                self._failed_count += 1
            raise
        finally:
            async with lock:
                self._active_count = max(0, self._active_count - 1)

    async def shutdown(self) -> None:
        """Dừng tất cả worker loops và xóa hàng đợi."""
        self._running = False
        for worker in self._workers:
            worker.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    def clear(self) -> None:
        """Xóa toàn bộ hàng đợi tin nhắn và đặt lại trạng thái."""
        self._inbox.clear()
        self._registered_agents.clear()
        self._active_count = 0
        self._completed_count = 0
        self._failed_count = 0


__all__ = [
    "AgentMessage",
    "MessageType",
    "SubagentManager",
    "TaskPriority",
    "create_error",
    "create_event",
    "create_request",
    "create_response",
]
