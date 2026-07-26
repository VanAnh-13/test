"""
Tests cho T1 — bugfix production:
1. get_default_model_config raise khi tên model không tồn tại (chống âm thầm
   rơi về openai-gpt4o-mini tính phí / chạy nhầm model trong thí nghiệm).
2. stream_agent không NameError vì thiếu import get_agent_registry.
"""

from __future__ import annotations

import asyncio

import pytest

import hagent.agent.graph as graph_mod
from hagent.agent.llm_config import get_default_model_config


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestStrictModelResolution:
    def test_unknown_default_model_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_DEFAULT_MODEL", "gpt-4o-minii-typo")
        with pytest.raises(ValueError) as exc:
            get_default_model_config()
        # Thông báo phải liệt kê tên hợp lệ để sửa nhanh
        assert "gpt-4o-minii-typo" in str(exc.value)
        assert "openai-gpt4o-mini" in str(exc.value)

    def test_known_name_still_resolves(self, monkeypatch):
        monkeypatch.setenv("LLM_DEFAULT_MODEL", "ollama-ci")
        cfg = get_default_model_config()
        assert cfg.name == "ollama-ci"
        assert cfg.provider == "ollama"

    def test_empty_default_falls_back_to_flagged_model(self, monkeypatch):
        """Không đặt default_model → hành vi cũ (default: true) giữ nguyên."""
        monkeypatch.setenv("LLM_DEFAULT_MODEL", "")
        cfg = get_default_model_config()
        assert cfg.name == "openai-gpt4o-mini"


class _StubGraph:
    """Graph giả: astream_events trả một event rồi hết — đủ để đi qua đoạn
    registry (nơi từng NameError) mà không cần LLM."""

    async def astream_events(self, initial_state, version="v2"):
        yield {
            "event": "on_chain_end",
            "name": "synthesize",
            "data": {"output": {}},
        }


class TestStreamAgentImport:
    def test_stream_agent_no_nameerror(self, monkeypatch):
        """Regression: graph.py dùng get_agent_registry trong stream_agent mà
        không import → NameError ngay lần streaming đầu tiên."""
        monkeypatch.setattr(graph_mod, "get_automl_graph", lambda: _StubGraph())

        async def consume():
            events = []
            async for ev in graph_mod.stream_agent("xin chào", user_id="t1"):
                events.append(ev)
                if len(events) > 20:
                    break
            return events

        # Không được raise NameError; nội dung event không quan trọng ở đây
        run(consume())
