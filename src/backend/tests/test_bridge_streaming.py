from __future__ import annotations

import asyncio
import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

try:
    import motor.motor_asyncio  # noqa: F401
except ModuleNotFoundError:
    motor_module = types.ModuleType("motor")
    motor_asyncio_module = types.ModuleType("motor.motor_asyncio")
    motor_asyncio_module.AsyncIOMotorClient = type("AsyncIOMotorClient", (), {})
    motor_asyncio_module.AsyncIOMotorDatabase = type("AsyncIOMotorDatabase", (), {})
    motor_module.motor_asyncio = motor_asyncio_module
    sys.modules["motor"] = motor_module
    sys.modules["motor.motor_asyncio"] = motor_asyncio_module


from hagent.bridge import app as bridge_app
from hagent.bridge import conversation as conv_store
from hagent.bridge.auth import TokenPayload
from hagent.bridge.models import ChatRequest


class _WorldState:
    def to_dict(self):
        return {"user_id": "owner", "phase": "idle", "datasets": {}, "jobs": {}}


class _WorldStore:
    def __init__(self):
        self.state = _WorldState()

    async def ensure(self, user_id):
        return self.state

    async def get(self, user_id):
        return self.state


async def _collect(iterator):
    return [item async for item in iterator]


def _request():
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(world_state_store=_WorldStore()))
    )


def _user():
    return TokenPayload({"sub": "owner"}, raw_token="jwt")


def _frame_lines(event: str, event_id: int, data: dict) -> list[str]:
    return [
        f"event: {event}",
        f"id: {event_id}",
        f"data: {json.dumps(data)}",
        "",
    ]


def _parse_frame(frame: str) -> tuple[str, int, dict]:
    fields = {}
    for line in frame.strip().splitlines():
        key, value = line.split(":", 1)
        fields[key] = value.strip()
    return fields["event"], int(fields["id"]), json.loads(fields["data"])


def _install_conversation_stubs(
    monkeypatch, *, assistant_error: Exception | None = None
):
    order = []
    assistant_calls = []

    async def get_history(conversation_id, user_id, limit=50):
        order.append("history")
        assert (conversation_id, user_id, limit) == ("conversation-1", "owner", 20)
        return [
            SimpleNamespace(role="user", content="first"),
            SimpleNamespace(role="assistant", content="answer"),
        ]

    async def add_message(**kwargs):
        order.append("user")
        assert kwargs["role"] == "user"

    async def add_assistant_once(**kwargs):
        order.append("assistant")
        assistant_calls.append(kwargs)
        if assistant_error is not None:
            raise assistant_error
        return True

    monkeypatch.setattr(conv_store, "get_message_history", get_history)
    monkeypatch.setattr(conv_store, "add_message", add_message)
    monkeypatch.setattr(
        conv_store,
        "add_assistant_message_once",
        add_assistant_once,
        raising=False,
    )
    monkeypatch.setattr(
        bridge_app,
        "_apply_tool_outputs_to_world_state",
        AsyncMock(),
    )
    return order, assistant_calls


@pytest.mark.asyncio
async def test_bridge_stream_forwards_typed_events_and_persists_before_done(
    monkeypatch,
):
    order, assistant_calls = _install_conversation_stubs(monkeypatch)
    captured = {}
    closed = asyncio.Event()

    final_response = {
        "message": "final answer",
        "provider": "hagent",
        "model": "ci-mock",
        "route": "synthesize",
        "tool_outputs": [{"tool_name": "list_datasets", "payload": {}}],
        "planning": {"status": "done"},
        "campaign": {"status": "done"},
        "campaign_status": "done",
        "hierarchy": {"status": "done"},
        "hierarchy_status": "done",
        "world_model": {"phase": "respond"},
        "evaluation": {"score": 0.9},
        "execution_events": [{"type": "step_end"}],
        "execution_log": [{"step": 1}],
        "revision_count": 1,
        "cost_metrics": {"total_tokens": 3},
    }

    async def upstream_lines(**kwargs):
        captured.update(kwargs)
        try:
            events = [
                ("route", 1, {"type": "route", "agent": "coordinator"}),
                ("token", 2, {"type": "token", "content": "draft"}),
                ("done", 3, {"type": "done", "response": final_response}),
                (
                    "done",
                    4,
                    {"type": "done", "response": {**final_response, "message": "dup"}},
                ),
            ]
            for event, event_id, data in events:
                for line in _frame_lines(event, event_id, data):
                    yield line
        finally:
            closed.set()

    monkeypatch.setattr(
        bridge_app,
        "_stream_agent_runtime_lines",
        upstream_lines,
        raising=False,
    )

    response = await bridge_app.chat_stream(
        request=_request(),
        req=ChatRequest(
            message="current",
            conversation_id="conversation-1",
            context={"dataset_id": "dataset-1"},
            model="ci-mock",
        ),
        user=_user(),
    )
    assert order == ["history", "user"]
    assert response.headers["x-conversation-id"] == "conversation-1"

    iterator = response.body_iterator
    route_frame = await anext(iterator)
    token_frame = await anext(iterator)
    assert not assistant_calls
    done_frame = await anext(iterator)
    with pytest.raises(StopAsyncIteration):
        await anext(iterator)

    parsed = [_parse_frame(frame) for frame in (route_frame, token_frame, done_frame)]
    assert [event for event, _, _ in parsed] == ["route", "token", "done"]
    assert [event_id for _, event_id, _ in parsed] == [1, 2, 3]
    assert parsed[-1][2]["response"]["message"] == "final answer"
    assert parsed[-1][2]["response"]["conversation_id"] == "conversation-1"
    assert order == ["history", "user", "assistant"]
    assert len(assistant_calls) == 1
    assert assistant_calls[0]["content"] == "final answer"
    assert assistant_calls[0]["message_id"]
    assert captured["model_name"] == "ci-mock"
    assert captured["context_extra"]["history"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
    ]
    assert closed.is_set()


@pytest.mark.asyncio
async def test_bridge_stream_malformed_upstream_emits_error_without_assistant(
    monkeypatch,
):
    _, assistant_calls = _install_conversation_stubs(monkeypatch)

    async def upstream_lines(**kwargs):
        for line in ("event: done", "id: 1", "data: not-json", ""):
            yield line

    monkeypatch.setattr(
        bridge_app,
        "_stream_agent_runtime_lines",
        upstream_lines,
        raising=False,
    )
    response = await bridge_app.chat_stream(
        request=_request(),
        req=ChatRequest(message="current", conversation_id="conversation-1"),
        user=_user(),
    )
    frames = await _collect(response.body_iterator)

    assert len(frames) == 1
    event, event_id, data = _parse_frame(frames[0])
    assert (event, event_id, data["type"]) == ("error", 1, "error")
    assert not assistant_calls
    assert all("done" not in frame.splitlines()[0] for frame in frames)


@pytest.mark.asyncio
async def test_bridge_stream_persist_failure_replaces_done_with_error(monkeypatch):
    _, assistant_calls = _install_conversation_stubs(
        monkeypatch,
        assistant_error=RuntimeError("mongo unavailable"),
    )

    async def upstream_lines(**kwargs):
        data = {"type": "done", "response": {"message": "final"}}
        for line in _frame_lines("done", 1, data):
            yield line

    monkeypatch.setattr(
        bridge_app,
        "_stream_agent_runtime_lines",
        upstream_lines,
        raising=False,
    )
    response = await bridge_app.chat_stream(
        request=_request(),
        req=ChatRequest(message="current", conversation_id="conversation-1"),
        user=_user(),
    )
    frames = await _collect(response.body_iterator)

    assert len(assistant_calls) == 1
    assert len(frames) == 1
    event, event_id, data = _parse_frame(frames[0])
    assert (event, event_id, data["type"]) == ("error", 2, "error")
    assert "done" not in frames[0].splitlines()[0]


@pytest.mark.asyncio
async def test_bridge_stream_network_failure_emits_error_terminal(monkeypatch):
    _, assistant_calls = _install_conversation_stubs(monkeypatch)

    async def upstream_lines(**kwargs):
        raise httpx.ConnectError("cannot connect")
        if False:
            yield ""

    monkeypatch.setattr(
        bridge_app,
        "_stream_agent_runtime_lines",
        upstream_lines,
        raising=False,
    )
    response = await bridge_app.chat_stream(
        request=_request(),
        req=ChatRequest(message="current", conversation_id="conversation-1"),
        user=_user(),
    )
    frames = await _collect(response.body_iterator)

    event, event_id, data = _parse_frame(frames[0])
    assert (event, event_id, data["type"]) == ("error", 1, "error")
    assert len(frames) == 1
    assert not assistant_calls
    assert "cannot connect" not in frames[0]


@pytest.mark.asyncio
async def test_bridge_stream_abort_closes_upstream_without_partial_assistant(
    monkeypatch,
):
    _, assistant_calls = _install_conversation_stubs(monkeypatch)
    waiting = asyncio.Event()
    closed = asyncio.Event()

    async def upstream_lines(**kwargs):
        try:
            for line in _frame_lines(
                "token", 1, {"type": "token", "content": "partial"}
            ):
                yield line
            waiting.set()
            await asyncio.Future()
        finally:
            closed.set()

    monkeypatch.setattr(
        bridge_app,
        "_stream_agent_runtime_lines",
        upstream_lines,
        raising=False,
    )
    response = await bridge_app.chat_stream(
        request=_request(),
        req=ChatRequest(message="current", conversation_id="conversation-1"),
        user=_user(),
    )
    iterator = response.body_iterator
    token_frame = await anext(iterator)
    assert _parse_frame(token_frame)[0] == "token"

    pending = asyncio.create_task(anext(iterator))
    await asyncio.wait_for(waiting.wait(), timeout=1)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    assert closed.is_set()
    assert not assistant_calls


@pytest.mark.asyncio
async def test_add_assistant_message_once_uses_owner_scoped_atomic_filter(monkeypatch):
    collection = Mock()
    collection.update_one = AsyncMock(
        side_effect=[
            SimpleNamespace(matched_count=1, modified_count=1),
            SimpleNamespace(matched_count=0, modified_count=0),
        ]
    )
    collection.find_one = AsyncMock(return_value={"_id": "conversation"})
    monkeypatch.setattr(conv_store, "_get_collection", lambda: collection)

    first = await conv_store.add_assistant_message_once(
        conversation_id="conversation-1",
        user_id="owner",
        content="final",
        message_id="stream-turn-1",
        provider="hagent",
        model="ci-mock",
    )
    second = await conv_store.add_assistant_message_once(
        conversation_id="conversation-1",
        user_id="owner",
        content="final",
        message_id="stream-turn-1",
        provider="hagent",
        model="ci-mock",
    )

    assert first is True
    assert second is False
    query, update = collection.update_one.await_args_list[0].args
    assert query == {
        "conversation_id": "conversation-1",
        "user_id": "owner",
        "messages.message_id": {"$ne": "stream-turn-1"},
    }
    assert update["$push"]["messages"]["message_id"] == "stream-turn-1"
    assert update["$push"]["messages"]["role"] == "assistant"
    collection.find_one.assert_awaited_once_with(
        {"conversation_id": "conversation-1", "user_id": "owner"},
        {"_id": 1},
    )
