"""Regression cho Mongo URI và log khởi động của HAgent Bridge."""

from __future__ import annotations

import sys
import types
from typing import Any, ClassVar

import pytest
from fastapi import FastAPI

try:
    import motor.motor_asyncio  # noqa: F401
except ModuleNotFoundError:
    motor_module = types.ModuleType("motor")
    motor_asyncio_module = types.ModuleType("motor.motor_asyncio")
    motor_asyncio_module.AsyncIOMotorClient = type("AsyncIOMotorClient", (), {})
    motor_asyncio_module.AsyncIOMotorDatabase = type(
        "AsyncIOMotorDatabase",
        (),
        {},
    )
    motor_module.motor_asyncio = motor_asyncio_module
    sys.modules["motor"] = motor_module
    sys.modules["motor.motor_asyncio"] = motor_asyncio_module

from hagent.bridge import app as bridge_app
from hagent.bridge import conversation


class _FakeCollection:
    async def create_index(self, *args: Any, **kwargs: Any) -> None:
        return None


class _FakeDatabase:
    def __init__(self) -> None:
        self.conversations = _FakeCollection()


class _FakeMotorClient:
    created_uris: ClassVar[list[str]] = []

    def __init__(self, uri: str) -> None:
        self.created_uris.append(uri)
        self.database = _FakeDatabase()

    def __getitem__(self, name: str) -> _FakeDatabase:
        assert name == "hagent_test"
        return self.database

    def close(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _restore_conversation_state() -> None:
    previous_client = conversation._client
    previous_database = conversation._db
    _FakeMotorClient.created_uris.clear()
    yield
    conversation._client = previous_client
    conversation._db = previous_database


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (
            "mongodb://app-user:unit-test-password@mongo:27017/?authSource=admin",
            "mongodb://app-user:unit-test-password@mongo:27017/?authSource=admin",
        ),
        ("mongo:27017", "mongodb://mongo:27017"),
    ],
)
async def test_init_db_normalizes_mongodb_uri_once(
    monkeypatch: pytest.MonkeyPatch,
    configured: str,
    expected: str,
) -> None:
    monkeypatch.setattr(conversation, "AsyncIOMotorClient", _FakeMotorClient)
    monkeypatch.setattr(
        conversation,
        "get_mongodb_config",
        lambda: {
            "connect": configured,
            "db_name": "hagent_test",
            "conversation_ttl_hours": 24,
        },
    )

    await conversation.init_db()

    assert _FakeMotorClient.created_uris == [expected]


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", ["", "   ", "https://mongo:27017"])
async def test_init_db_rejects_invalid_connect_without_echo(
    monkeypatch: pytest.MonkeyPatch,
    configured: str,
) -> None:
    monkeypatch.setattr(conversation, "AsyncIOMotorClient", _FakeMotorClient)
    monkeypatch.setattr(
        conversation,
        "get_mongodb_config",
        lambda: {
            "connect": configured,
            "db_name": "hagent_test",
            "conversation_ttl_hours": 24,
        },
    )

    with pytest.raises(ValueError) as error:
        await conversation.init_db()

    assert str(error.value) == "Cấu hình kết nối MongoDB không hợp lệ"
    assert _FakeMotorClient.created_uris == []


@pytest.mark.asyncio
async def test_bridge_startup_log_does_not_include_mongodb_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_uri = "mongodb://secret-user:secret-password@mongo:27017/?authSource=admin"
    info_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    monkeypatch.setattr(
        bridge_app,
        "get_mongodb_config",
        lambda: {
            "connect": secret_uri,
            "db_name": "hagent_test",
            "conversation_ttl_hours": 24,
        },
    )

    async def fail_after_log() -> None:
        raise RuntimeError("dừng sau khi quan sát log")

    monkeypatch.setattr(bridge_app.conv_store, "init_db", fail_after_log)
    monkeypatch.setattr(
        bridge_app.logger,
        "info",
        lambda *args, **kwargs: info_calls.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="dừng sau khi quan sát log"):
        async with bridge_app.lifespan(FastAPI()):
            pass

    serialized_calls = repr(info_calls)
    assert secret_uri not in serialized_calls
    assert "secret-password" not in serialized_calls
