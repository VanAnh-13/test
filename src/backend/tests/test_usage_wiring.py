"""
Tests cho T2 — usage tracker nối vào vòng đời agent qua contextvar.
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from hagent.agent.llm_config import create_chat_model
from hagent.agent.middlewares.usage_tracker import (
    UsageTracker,
    UsageTrackingCallback,
    get_current_tracker,
    reset_current_tracker,
    set_current_tracker,
)


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _fake_llm_end(tracker_cb: UsageTrackingCallback, n_in=10, n_out=5, model="m"):
    msg = AIMessage(
        content="ok",
        usage_metadata={"input_tokens": n_in, "output_tokens": n_out, "total_tokens": n_in + n_out},
        response_metadata={"model_name": model},
    )
    tracker_cb.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]], llm_output={}))


class TestContextvarLifecycle:
    def test_set_get_reset(self):
        assert get_current_tracker() is None
        t = UsageTracker()
        token = set_current_tracker(t)
        assert get_current_tracker() is t
        reset_current_tracker(token)
        assert get_current_tracker() is None

    def test_parallel_tasks_isolated(self):
        """Hai run song song — mỗi task thấy đúng tracker của mình."""

        async def one_run(n_tokens):
            t = UsageTracker()
            set_current_tracker(t)
            await asyncio.sleep(0.01)  # nhường điều khiển để đan xen
            cb = UsageTrackingCallback(get_current_tracker())
            _fake_llm_end(cb, n_in=n_tokens)
            await asyncio.sleep(0.01)
            return get_current_tracker().summary()["total_input_tokens"]

        async def main():
            # mỗi task copy context lúc tạo → set trong task không lẫn nhau
            return await asyncio.gather(one_run(100), one_run(7))

        assert run(main()) == [100, 7]


class TestCreateChatModelPicksUpTracker:
    def test_callbacks_attached_from_contextvar(self):
        t = UsageTracker()
        token = set_current_tracker(t)
        try:
            model = create_chat_model("ci-mock")
            cbs = model.callbacks or []
            assert any(isinstance(c, UsageTrackingCallback) for c in cbs)
            # đúng tracker của run này, không phải instance nào khác
            cb = next(c for c in cbs if isinstance(c, UsageTrackingCallback))
            assert cb.tracker is t
        finally:
            reset_current_tracker(token)

    def test_no_tracker_no_callbacks(self):
        model = create_chat_model("ci-mock")
        assert not any(
            isinstance(c, UsageTrackingCallback) for c in (model.callbacks or [])
        )

    def test_explicit_callbacks_win(self):
        t = UsageTracker()
        token = set_current_tracker(t)
        try:
            marker = UsageTrackingCallback(UsageTracker())
            model = create_chat_model("ci-mock", callbacks=[marker])
            cbs = model.callbacks or []
            assert marker in cbs
            assert all(
                (not isinstance(c, UsageTrackingCallback)) or c is marker for c in cbs
            )
        finally:
            reset_current_tracker(token)

    def test_bind_tools_still_works(self):
        """Callbacks vào constructor phải giữ nguyên class → bind_tools sống."""
        t = UsageTracker()
        token = set_current_tracker(t)
        try:
            model = create_chat_model("ci-mock")
            bound = model.bind_tools([])
            assert bound is not None
        finally:
            reset_current_tracker(token)


class TestRunAgentMergesUsage:
    def test_cost_metrics_contains_usage(self, monkeypatch):
        """run_agent trả cost_metrics có các trường usage (qua graph stub +
        LLM call giả lập trong node)."""
        import hagent.agent.graph as graph_mod

        class _StubGraph:
            async def ainvoke(self, state):
                # Node giả: thực hiện một "LLM call" ghi vào tracker hiện tại
                cur = get_current_tracker()
                if cur is not None:
                    _fake_llm_end(UsageTrackingCallback(cur), n_in=42, n_out=13)
                state = dict(state)
                state["messages"] = list(state.get("messages") or []) + [
                    AIMessage(content="xong")
                ]
                return state

        monkeypatch.setattr(graph_mod, "get_automl_graph", lambda: _StubGraph())
        result = run(graph_mod.run_agent("chạy thử", user_id="t2"))
        cost = result["cost_metrics"]
        assert cost["total_input_tokens"] == 42
        assert cost["total_output_tokens"] == 13
        assert cost["total_calls"] == 1
        assert "total_cost_usd" in cost
        # tracker đã được reset sau run
        assert get_current_tracker() is None
