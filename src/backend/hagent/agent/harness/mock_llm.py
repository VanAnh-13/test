"""
Scripted chat model for graph-layer harness (no real LLM).

Coordinator may call create_chat_model(); we patch it to a dummy that
returns a fixed AIMessage so graph can route without API keys.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch


class FakeAIMessage:
    def __init__(self, content: str = "", tool_calls: list | None = None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.type = "ai"


class FakeChatModel:
    """Minimal chat model with bind_tools / ainvoke / invoke."""

    def __init__(self, responses: list[str] | None = None):
        self._responses = list(
            responses
            or [
                "Tôi đã hiểu yêu cầu AutoML. Đang chuyển xử lý multi-agent.",
            ]
        )
        self._i = 0

    def bind_tools(self, tools: list) -> FakeChatModel:
        return self

    def _next(self) -> FakeAIMessage:
        if not self._responses:
            return FakeAIMessage("OK")
        msg = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return FakeAIMessage(msg)

    def invoke(self, messages: Any, **kwargs: Any) -> FakeAIMessage:
        return self._next()

    async def ainvoke(self, messages: Any, **kwargs: Any) -> FakeAIMessage:
        return self._next()


def create_fake_chat_model(model_name: str | None = None, **kwargs: Any) -> FakeChatModel:
    return FakeChatModel()


@contextmanager
def patch_chat_model(responses: list[str] | None = None) -> Iterator[FakeChatModel]:
    """Thay factory canonical bằng model giả trong lúc chạy harness."""
    fake = FakeChatModel(responses=responses)

    def _factory(*args: Any, **kwargs: Any) -> FakeChatModel:
        return fake

    # Patch public interface và call site import sẵn nếu có.
    targets = [
        "hagent.agent.llm.create_chat_model",
        "hagent.agent.orchestration.coordinator.create_chat_model",
    ]
    patches = []
    for t in targets:
        try:
            patches.append(patch(t, _factory))
        except Exception:
            continue
    for p in patches:
        p.start()
    try:
        # Giữ patch package hoạt động với import nằm bên trong hàm coordinator.
        with patch(
            "hagent.agent.llm.create_chat_model",
            _factory,
        ):
            yield fake
    finally:
        for p in patches:
            try:
                p.stop()
            except Exception:
                pass
