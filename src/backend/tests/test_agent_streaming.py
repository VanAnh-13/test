from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from hagent.agent import graph as graph_module


async def _collect(iterator):
    return [item async for item in iterator]


def _parse_frame(frame: str) -> tuple[str, int, dict]:
    fields = {}
    for line in frame.strip().splitlines():
        key, value = line.split(":", 1)
        fields[key] = value.strip()
    return fields["event"], int(fields["id"]), json.loads(fields["data"])


class _Registry:
    def agent_names(self):
        return []


class _Middleware:
    def __init__(self):
        self.pre_state = None
        self.post_result = None

    async def run_pre(self, state):
        self.pre_state = state
        return state

    async def run_post(self, state, result):
        self.post_result = dict(result)
        return result


class _FinalStateGraph:
    def __init__(self):
        self.initial_state = None

    async def astream_events(self, initial_state, version="v2"):
        self.initial_state = initial_state
        yield {
            "event": "on_chat_model_stream",
            "name": "coordinator",
            "parent_ids": ["root"],
            "data": {"chunk": SimpleNamespace(content="internal draft")},
        }
        yield {
            "event": "on_chain_end",
            "name": "LangGraph",
            "parent_ids": [],
            "data": {
                "output": {
                    "messages": [
                        HumanMessage(content="current"),
                        ToolMessage(
                            content='{"rows": 150}',
                            name="get_dataset_info",
                            tool_call_id="call-1",
                        ),
                        AIMessage(content="final answer"),
                    ],
                    "current_phase": "synthesize",
                    "plan_status": "done",
                    "selected_plan": {"plan_id": "plan-1"},
                    "surprise": {"level": "low"},
                    "cost_metrics": {"planner_calls": 1},
                    "execution_events": [{"type": "step_end"}],
                    "execution_log": [{"step": 1}],
                    "revision_count": 2,
                    "world_model": {"phase": "respond"},
                    "campaign": {"status": "done"},
                    "campaign_status": "done",
                    "evaluation": {"score": 0.9},
                    "hierarchy": {"status": "done"},
                    "hierarchy_status": "done",
                }
            },
        }


@pytest.mark.asyncio
async def test_stream_agent_done_uses_root_final_state_and_resets_context(monkeypatch):
    from hagent.agent import llm_config, middlewares
    from hagent.agent import registry as registry_module
    from hagent.agent.middlewares import usage_tracker

    stub_graph = _FinalStateGraph()
    middleware = _Middleware()
    monkeypatch.setattr(graph_module, "get_automl_graph", lambda: stub_graph)
    monkeypatch.setattr(registry_module, "get_agent_registry", lambda: _Registry())
    monkeypatch.setattr(middlewares, "create_default_chain", lambda: middleware)

    outer_tracker = usage_tracker.UsageTracker()
    outer_usage_token = usage_tracker.set_current_tracker(outer_tracker)
    outer_model_token = llm_config.set_current_model_name("outer-model")
    try:
        events = await _collect(
            graph_module.stream_agent(
                "current",
                user_id="owner",
                history=[
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "answer"},
                ],
                world_store=object(),
                wm_service=object(),
                model_name="ci-mock",
            )
        )
        assert usage_tracker.get_current_tracker() is outer_tracker
        assert llm_config.get_current_model_name() == "outer-model"
    finally:
        usage_tracker.reset_current_tracker(outer_usage_token)
        llm_config.reset_current_model_name(outer_model_token)

    token = next(event for event in events if event["type"] == "token")
    done = next(event for event in events if event["type"] == "done")
    assert token["content"] == "internal draft"
    response = done["response"]
    assert response["message"] == "final answer"
    assert response["message"] != token["content"]
    assert response["provider"] == "hagent"
    assert response["model"] == "ci-mock"
    assert response["route"] == "synthesize"
    assert response["tool_outputs"] == [
        {"tool_name": "get_dataset_info", "payload": {"rows": 150}}
    ]
    assert response["selected_plan"] == {"plan_id": "plan-1"}
    assert response["world_model"] == {"phase": "respond"}
    assert response["campaign_status"] == "done"
    assert response["hierarchy_status"] == "done"
    assert response["evaluation"] == {"score": 0.9}
    assert response["execution_log"] == [{"step": 1}]
    assert middleware.post_result["response"] == "final answer"

    messages = stub_graph.initial_state["messages"]
    assert [type(message) for message in messages] == [
        HumanMessage,
        AIMessage,
        HumanMessage,
    ]
    assert [message.content for message in messages] == ["first", "answer", "current"]

@pytest.mark.asyncio
async def test_stream_agent_failure_resets_request_context(monkeypatch):
    from hagent.agent import llm_config, middlewares
    from hagent.agent import registry as registry_module
    from hagent.agent.middlewares import usage_tracker

    class _FailingGraph:
        async def astream_events(self, initial_state, version="v2"):
            raise RuntimeError("graph failed")
            if False:
                yield {}

    monkeypatch.setattr(graph_module, "get_automl_graph", lambda: _FailingGraph())
    monkeypatch.setattr(registry_module, "get_agent_registry", lambda: _Registry())
    monkeypatch.setattr(middlewares, "create_default_chain", lambda: _Middleware())

    outer_tracker = usage_tracker.UsageTracker()
    outer_usage_token = usage_tracker.set_current_tracker(outer_tracker)
    outer_model_token = llm_config.set_current_model_name("outer-model")
    try:
        with pytest.raises(RuntimeError, match="graph failed"):
            await _collect(
                graph_module.stream_agent(
                    "fail",
                    user_id="owner",
                    world_store=object(),
                    wm_service=object(),
                    model_name="ci-mock",
                )
            )
        assert usage_tracker.get_current_tracker() is outer_tracker
        assert llm_config.get_current_model_name() == "outer-model"
    finally:
        usage_tracker.reset_current_tracker(outer_usage_token)
        llm_config.reset_current_model_name(outer_model_token)

@pytest.mark.asyncio
async def test_stream_agent_cancellation_propagates_after_context_cleanup(monkeypatch):
    from hagent.agent import llm_config, middlewares
    from hagent.agent import registry as registry_module
    from hagent.agent.middlewares import usage_tracker

    started = asyncio.Event()
    cleanup_calls = []

    class _BlockingGraph:
        async def astream_events(self, initial_state, version="v2"):
            started.set()
            await asyncio.Future()
            if False:
                yield {}

    original_model_reset = llm_config.reset_current_model_name
    original_usage_reset = usage_tracker.reset_current_tracker

    def reset_model(token):
        cleanup_calls.append("model")
        original_model_reset(token)

    def reset_usage(token):
        cleanup_calls.append("usage")
        original_usage_reset(token)

    monkeypatch.setattr(graph_module, "get_automl_graph", lambda: _BlockingGraph())
    monkeypatch.setattr(registry_module, "get_agent_registry", lambda: _Registry())
    monkeypatch.setattr(middlewares, "create_default_chain", lambda: _Middleware())
    monkeypatch.setattr(llm_config, "reset_current_model_name", reset_model)
    monkeypatch.setattr(usage_tracker, "reset_current_tracker", reset_usage)

    task = asyncio.create_task(
        _collect(
            graph_module.stream_agent(
                "cancel me",
                user_id="owner",
                world_store=object(),
                wm_service=object(),
                model_name="ci-mock",
            )
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert cleanup_calls.count("usage") == 1
    assert cleanup_calls.count("model") == 1


@pytest.mark.asyncio
async def test_sse_stream_emits_typed_monotonic_frames_and_one_terminal(monkeypatch):
    async def fake_stream_agent(*args, **kwargs):
        yield {"type": "route", "agent": "coordinator"}
        yield {"type": "token", "content": "hello"}
        yield {"type": "done", "response": {"message": "final"}}
        yield {"type": "done", "response": {"message": "duplicate"}}

    monkeypatch.setattr(graph_module, "stream_agent", fake_stream_agent)
    from hagent.agent.streaming import sse_stream

    frames = await _collect(sse_stream("hello", conversation_id="conversation-1"))
    parsed = [_parse_frame(frame) for frame in frames]

    assert [event for event, _, _ in parsed] == ["route", "token", "done"]
    assert [event_id for _, event_id, _ in parsed] == [1, 2, 3]
    assert [data["type"] for _, _, data in parsed] == ["route", "token", "done"]
    assert parsed[-1][2]["response"]["conversation_id"] == "conversation-1"
    assert sum(event in {"done", "error"} for event, _, _ in parsed) == 1
    assert all("[DONE]" not in frame for frame in frames)


@pytest.mark.asyncio
async def test_sse_stream_converts_failure_to_safe_error_terminal(monkeypatch):
    async def failing_stream_agent(*args, **kwargs):
        raise RuntimeError("credential-value-must-not-leak")
        if False:
            yield {}

    monkeypatch.setattr(graph_module, "stream_agent", failing_stream_agent)
    from hagent.agent.streaming import sse_stream

    frames = await _collect(sse_stream("hello"))
    assert len(frames) == 1
    event, event_id, data = _parse_frame(frames[0])
    assert (event, event_id, data["type"]) == ("error", 1, "error")
    assert "credential-value-must-not-leak" not in frames[0]

@pytest.mark.asyncio
async def test_sse_stream_serialization_failure_still_emits_error_terminal(monkeypatch):
    async def unserializable_stream_agent(*args, **kwargs):
        yield {"type": "done", "response": {"message": object()}}

    monkeypatch.setattr(graph_module, "stream_agent", unserializable_stream_agent)
    from hagent.agent.streaming import sse_stream

    frames = await _collect(sse_stream("hello"))
    assert len(frames) == 1
    event, event_id, data = _parse_frame(frames[0])
    assert (event, event_id, data["type"]) == ("error", 1, "error")

@pytest.mark.asyncio
async def test_sse_stream_propagates_cancellation_and_closes_agent(monkeypatch):
    started = asyncio.Event()
    closed = asyncio.Event()

    async def blocking_stream_agent(*args, **kwargs):
        try:
            started.set()
            await asyncio.Future()
            if False:
                yield {}
        finally:
            closed.set()

    monkeypatch.setattr(graph_module, "stream_agent", blocking_stream_agent)
    from hagent.agent.streaming import sse_stream

    task = asyncio.create_task(_collect(sse_stream("hello")))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()


@pytest.mark.asyncio
async def test_agent_run_stream_is_stateless_and_forwards_chat_contract(monkeypatch):
    from hagent import chat_router
    from hagent.agent import streaming

    captured = {}

    async def load_world_model(*args, **kwargs):
        return {
            "user_id": "owner",
            "phase": "server",
            "datasets": {},
            "jobs": {},
        }

    async def fake_sse_stream(message, **kwargs):
        captured["message"] = message
        captured.update(kwargs)
        yield "event: done\nid: 1\ndata: {\"type\": \"done\"}\n\n"

    add_message = AsyncMock()
    monkeypatch.setattr(chat_router, "_load_world_model", load_world_model)
    monkeypatch.setattr(streaming, "sse_stream", fake_sse_stream)
    monkeypatch.setattr(chat_router.chat_store, "add_message", add_message)

    response = await chat_router.agent_run_stream(
        req=chat_router.ChatRequest(
            message="current",
            conversation_id="conversation-1",
            context={
                "dataset_id": "dataset-1",
                "history": [
                    {"role": "user", "content": "first"},
                    {"role": "system", "content": "ignore"},
                    {"role": "assistant", "content": "answer"},
                ],
                "world_state": {"phase": "spoofed"},
            },
            model="ci-mock",
        ),
        request=SimpleNamespace(headers={"Authorization": "Bearer jwt"}),
        db=SimpleNamespace(client="mongo-client", name="test-db"),
        current_user={"_id": "owner"},
    )
    chunks = await _collect(response.body_iterator)

    assert chunks
    assert response.headers["x-conversation-id"] == "conversation-1"
    assert captured["message"] == "current"
    assert captured["user_id"] == "owner"
    assert captured["user_token"] == "jwt"
    assert captured["history"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
    ]
    assert captured["world_model"]["phase"] == "server"
    assert captured["world_model"]["request_context"]["dataset_id"] == "dataset-1"
    assert captured["mongo_client"] == "mongo-client"
    assert captured["db_name"] == "test-db"
    assert captured["model_name"] == "ci-mock"
    add_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_run_stream_rejects_unknown_model_before_stream(monkeypatch):
    from hagent import chat_router

    monkeypatch.setattr(chat_router, "_load_world_model", AsyncMock(return_value=None))
    with pytest.raises(HTTPException) as exc:
        await chat_router.agent_run_stream(
            req=chat_router.ChatRequest(message="hello", model="unknown-model"),
            request=SimpleNamespace(headers={}),
            db=SimpleNamespace(client=None, name="test-db"),
            current_user={"_id": "owner"},
        )
    assert exc.value.status_code == 400
