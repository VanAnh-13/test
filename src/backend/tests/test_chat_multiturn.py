from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace

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

from hagent import chat_router
from hagent.agent import graph
from hagent.bridge import app as bridge_app
from hagent.bridge.auth import TokenPayload
from hagent.bridge.models import ChatRequest


@dataclass
class _StoredMessage:
    role: str
    content: object


class _WorldState:
    def __init__(self, owner: str):
        self.owner = owner

    def to_dict(self):
        return {"user_id": self.owner, "phase": "server"}


class _WorldStateStore:
    async def ensure(self, user_id):
        return None

    async def get(self, user_id):
        return _WorldState(user_id)


def _bridge_request():
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(world_state_store=_WorldStateStore()))
    )


def _install_bridge_store(monkeypatch):
    stored: dict[tuple[str, str], list[_StoredMessage]] = {}
    history_calls: list[tuple[str, str, int]] = []
    gateway_calls: list[dict] = []

    async def add_message(
        conversation_id,
        user_id,
        role,
        content,
        provider="",
        model="",
    ):
        del provider, model
        stored.setdefault((conversation_id, user_id), []).append(
            _StoredMessage(role=role, content=content)
        )

    async def get_message_history(conversation_id, user_id, limit=50):
        history_calls.append((conversation_id, user_id, limit))
        return list(stored.get((conversation_id, user_id), []))[-limit:]

    async def gateway(**kwargs):
        gateway_calls.append(kwargs)
        return {
            "message": f"reply-{len(gateway_calls)}",
            "provider": "hagent",
            "model": "ci-mock",
            "tool_outputs": [],
        }

    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(bridge_app.conv_store, "add_message", add_message)
    monkeypatch.setattr(
        bridge_app.conv_store,
        "get_message_history",
        get_message_history,
    )
    monkeypatch.setattr(bridge_app, "_call_hagent_gateway", gateway)
    monkeypatch.setattr(bridge_app, "_apply_tool_outputs_to_world_state", noop)
    return stored, history_calls, gateway_calls


@pytest.mark.asyncio
async def test_bridge_second_turn_forwards_prior_owner_history_once(monkeypatch):
    stored, history_calls, gateway_calls = _install_bridge_store(monkeypatch)
    request = _bridge_request()
    owner_a = TokenPayload({"sub": "owner-a"}, raw_token="jwt-a")
    owner_b = TokenPayload({"sub": "owner-b"}, raw_token="jwt-b")

    await bridge_app.chat(
        request,
        ChatRequest(message="first", conversation_id="shared"),
        owner_a,
    )
    await bridge_app.chat(
        request,
        ChatRequest(message="second", conversation_id="shared"),
        owner_a,
    )
    await bridge_app.chat(
        request,
        ChatRequest(message="private", conversation_id="shared"),
        owner_b,
    )

    assert gateway_calls[0]["context_extra"]["history"] == []
    assert gateway_calls[1]["context_extra"]["history"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply-1"},
    ]
    assert gateway_calls[2]["context_extra"]["history"] == []
    assert all(call[2] == 20 for call in history_calls)
    assert history_calls == [
        ("shared", "owner-a", 20),
        ("shared", "owner-a", 20),
        ("shared", "owner-b", 20),
    ]
    assert [message.content for message in stored[("shared", "owner-a")]] == [
        "first",
        "reply-1",
        "second",
        "reply-2",
    ]
    assert "second" not in {
        item["content"] for item in gateway_calls[1]["context_extra"]["history"]
    }


@pytest.mark.asyncio
async def test_bridge_filters_roles_and_bounds_history(monkeypatch):
    stored, _, gateway_calls = _install_bridge_store(monkeypatch)
    key = ("conversation-1", "owner")
    stored[key] = [
        _StoredMessage(
            role="tool" if index == 5 else ("user" if index % 2 == 0 else "assistant"),
            content=f"message-{index}",
        )
        for index in range(25)
    ]
    stored[key][-1] = _StoredMessage(role="assistant", content=123)

    await bridge_app.chat(
        _bridge_request(),
        ChatRequest(message="current", conversation_id="conversation-1"),
        TokenPayload({"sub": "owner"}, raw_token="jwt"),
    )

    history = gateway_calls[0]["context_extra"]["history"]
    assert len(history) == 18
    assert history[0] == {"role": "user", "content": "message-6"}
    assert history[-1] == {"role": "assistant", "content": "message-23"}
    assert {item["role"] for item in history} <= {"user", "assistant"}
    assert all(isinstance(item["content"], str) for item in history)


@pytest.mark.asyncio
async def test_toolkit_agent_run_passes_only_sanitized_history_to_graph(monkeypatch):
    captured = {}

    async def load_world_model(*args, **kwargs):
        return {"user_id": "owner", "datasets": {}, "jobs": {}}

    async def call_agent(*args, **kwargs):
        captured.update(kwargs)
        return {
            "message": "ok",
            "provider": "hagent",
            "model": "ci-mock",
            "tool_outputs": [],
        }

    monkeypatch.setattr(chat_router, "_load_world_model", load_world_model)
    monkeypatch.setattr(chat_router, "_call_agent", call_agent)

    await chat_router.agent_run(
        req=chat_router.ChatRequest(
            message="current",
            conversation_id="conversation-1",
            context={
                "history": [
                    {"role": "user", "content": "first"},
                    {"role": "system", "content": "ignore"},
                    {"role": "assistant", "content": "answer"},
                    {"role": "user", "content": 123},
                ]
            },
        ),
        request=SimpleNamespace(headers={"Authorization": "Bearer jwt"}),
        db=SimpleNamespace(client=None, name="test"),
        current_user={"_id": "owner"},
    )

    assert captured["history"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
    ]
    assert captured["message"] == "current"


def test_graph_builds_ordered_messages_and_appends_current_once():
    from langchain_core.messages import AIMessage, HumanMessage

    messages = graph._build_initial_messages(
        "current",
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
        ],
    )

    assert [type(message) for message in messages] == [
        HumanMessage,
        AIMessage,
        HumanMessage,
    ]
    assert [message.content for message in messages] == ["first", "answer", "current"]
    assert (
        sum(
            isinstance(message, HumanMessage) and message.content == "current"
            for message in messages
        )
        == 1
    )
