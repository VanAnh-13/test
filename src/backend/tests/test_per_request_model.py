"""
Tests cho T12 — per-request model qua contextvar + validate strict.
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage

from hagent.agent.llm import (
    create_chat_model,
    get_current_model_name,
    require_model_config,
    reset_current_model_name,
    set_current_model_name,
)


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestRequireModelConfig:
    def test_bad_name_raises_with_list(self):
        with pytest.raises(ValueError) as exc:
            require_model_config("gpt-100-turbo-max")
        assert "gpt-100-turbo-max" in str(exc.value)
        assert "ci-mock" in str(exc.value)

    def test_good_name_resolves(self):
        cfg = require_model_config("ollama-ci")
        assert cfg.provider == "ollama"


class TestContextvarModelSelection:
    def test_create_chat_model_uses_current(self):
        token = set_current_model_name("ci-mock")
        try:
            model = create_chat_model()  # không truyền tên
            assert model.model_name == "mock-model"
        finally:
            reset_current_model_name(token)

    def test_explicit_name_wins_over_contextvar(self):
        token = set_current_model_name("ollama-ci")
        try:
            model = create_chat_model("ci-mock")
            assert model.model_name == "mock-model"
        finally:
            reset_current_model_name(token)

    def test_parallel_runs_isolated(self):
        async def one(name):
            set_current_model_name(name)
            await asyncio.sleep(0.01)
            return get_current_model_name()

        async def main():
            return await asyncio.gather(one("ci-mock"), one("ollama-ci"))

        assert run(main()) == ["ci-mock", "ollama-ci"]


class TestRunAgentModelName:
    def test_bad_model_raises_before_graph(self, monkeypatch):
        import hagent.agent.orchestration.graph as graph_mod

        invoked = {"n": 0}

        class _StubGraph:
            async def astream_events(self, state, version="v2"):
                invoked["n"] += 1
                yield {
                    "event": "on_chain_end",
                    "name": "LangGraph",
                    "parent_ids": [],
                    "data": {"output": state},
                }

        monkeypatch.setattr(graph_mod, "get_automl_graph", lambda: _StubGraph())
        with pytest.raises(ValueError):
            run(graph_mod.run_agent("hi", user_id="t12", model_name="bogus-model"))
        assert invoked["n"] == 0  # nổ TRƯỚC khi chạy graph

    def test_model_name_visible_inside_run_and_in_result(self, monkeypatch):
        import hagent.agent.orchestration.graph as graph_mod

        seen = {}

        class _StubGraph:
            async def astream_events(self, state, version="v2"):
                seen["model"] = get_current_model_name()
                state = dict(state)
                state["messages"] = list(state.get("messages") or []) + [
                    AIMessage(content="ok")
                ]
                yield {
                    "event": "on_chain_end",
                    "name": "LangGraph",
                    "parent_ids": [],
                    "data": {"output": state},
                }

        monkeypatch.setattr(graph_mod, "get_automl_graph", lambda: _StubGraph())
        result = run(graph_mod.run_agent("hi", user_id="t12", model_name="ci-mock"))
        assert seen["model"] == "ci-mock"  # subagent/coordinator sẽ dùng đúng model
        assert result["model"] == "ci-mock"

    def test_no_model_keeps_default_behavior(self, monkeypatch):
        import hagent.agent.orchestration.graph as graph_mod

        class _StubGraph:
            async def astream_events(self, state, version="v2"):
                state = dict(state)
                state["messages"] = list(state.get("messages") or []) + [
                    AIMessage(content="ok")
                ]
                yield {
                    "event": "on_chain_end",
                    "name": "LangGraph",
                    "parent_ids": [],
                    "data": {"output": state},
                }

        monkeypatch.setattr(graph_mod, "get_automl_graph", lambda: _StubGraph())
        result = run(graph_mod.run_agent("hi", user_id="t12"))
        assert result["model"] == "multi-agent"
