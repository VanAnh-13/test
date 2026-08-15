"""Regression cho liveness và readiness của HAgent Bridge."""

from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace

from fastapi.testclient import TestClient

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


class _Response:
    def __init__(self, status_code: int, payload: object | None = None) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> object:
        return self._payload


class _HttpClient:
    def __init__(
        self, responses: dict[str, _Response | Exception], calls: list[str]
    ) -> None:
        self._responses = responses
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None

    async def get(self, url: str) -> _Response:
        self._calls.append(url)
        result = self._responses[url]
        if isinstance(result, Exception):
            raise result
        return result


class _SlowHttpClient(_HttpClient):
    def __init__(self, calls: list[str]) -> None:
        super().__init__({}, calls)
        self.cancelled = False

    async def get(self, url: str) -> _Response:
        self._calls.append(url)
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise AssertionError("Probe chậm phải bị hủy trước khi hoàn tất")


class _MongoAdmin:
    def __init__(self, *, error: Exception | None = None, delay: float = 0) -> None:
        self.error = error
        self.delay = delay
        self.commands: list[str] = []
        self.cancelled = False

    async def command(self, name: str) -> dict[str, int]:
        self.commands.append(name)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if self.error is not None:
            raise self.error
        return {"ok": 1}


def _patch_http(monkeypatch, responses: dict[str, _Response | Exception]) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(
        bridge_app.httpx,
        "AsyncClient",
        lambda *args, **kwargs: _HttpClient(responses, calls),
    )
    return calls


def _patch_urls(monkeypatch) -> None:
    monkeypatch.setenv(
        "HAGENT_RUN_API_URL",
        "http://runtime.internal:8585/api/v1/runs",
    )
    monkeypatch.setattr(
        bridge_app,
        "get_hautoml_config",
        lambda: {"base_url": "http://automl.internal:8585"},
    )


def _client() -> TestClient:
    return TestClient(bridge_app.hagent_bridge)


def test_legacy_health_uses_configured_runtime_and_keeps_response_contract(monkeypatch):
    _patch_urls(monkeypatch)
    calls = _patch_http(
        monkeypatch,
        {
            "http://runtime.internal:8585/api/v1/chat/health": _Response(200, {}),
            "http://automl.internal:8585/home": _Response(200, {}),
        },
    )

    response = _client().get("/api/v1/chat/health")

    assert response.status_code == 200
    assert response.json() == {
        "hagent_url": "/api/hagent",
        "connected": True,
        "hautoml_connected": True,
        "mode": "hagent",
        "active_provider": "hagent",
        "active_model": "hagent-agent",
        "available_providers": ["hagent"],
    }
    assert calls == [
        "http://runtime.internal:8585/api/v1/chat/health",
        "http://automl.internal:8585/home",
    ]
    assert "runtime.internal" not in response.text
    assert "automl.internal" not in response.text


def test_readiness_returns_200_only_when_mongo_and_toolkit_are_ready(monkeypatch):
    _patch_urls(monkeypatch)
    mongo_admin = _MongoAdmin()
    monkeypatch.setattr(
        bridge_app.conv_store,
        "get_db_client",
        lambda: SimpleNamespace(admin=mongo_admin),
    )
    calls = _patch_http(
        monkeypatch,
        {
            "http://runtime.internal:8585/api/v1/chat/health": _Response(
                200,
                {"hautoml_connected": True, "available_models": ["model-a"]},
            )
        },
    )

    response = _client().get("/api/v1/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "dependencies": {"mongodb": "ready", "toolkit": "ready"},
    }
    assert mongo_admin.commands == ["ping"]
    assert calls == ["http://runtime.internal:8585/api/v1/chat/health"]


def test_readiness_returns_sanitized_503_when_mongo_fails(monkeypatch):
    _patch_urls(monkeypatch)
    mongo_admin = _MongoAdmin(error=RuntimeError("mongodb://root:secret@mongo"))
    monkeypatch.setattr(
        bridge_app.conv_store,
        "get_db_client",
        lambda: SimpleNamespace(admin=mongo_admin),
    )
    _patch_http(
        monkeypatch,
        {
            "http://runtime.internal:8585/api/v1/chat/health": _Response(
                200,
                {"hautoml_connected": True, "available_models": ["model-a"]},
            )
        },
    )

    response = _client().get("/api/v1/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "dependencies": {"mongodb": "unavailable", "toolkit": "ready"},
    }
    assert "secret" not in response.text
    assert "mongo" not in response.text.replace("mongodb", "")


def test_readiness_requires_toolkit_dependency_status_not_only_http_200(monkeypatch):
    _patch_urls(monkeypatch)
    monkeypatch.setattr(
        bridge_app.conv_store,
        "get_db_client",
        lambda: SimpleNamespace(admin=_MongoAdmin()),
    )
    _patch_http(
        monkeypatch,
        {
            "http://runtime.internal:8585/api/v1/chat/health": _Response(
                200,
                {"hautoml_connected": False, "available_models": ["model-a"]},
            )
        },
    )

    response = _client().get("/api/v1/ready")

    assert response.status_code == 503
    assert response.json()["dependencies"] == {
        "mongodb": "ready",
        "toolkit": "unavailable",
    }
    assert "runtime.internal" not in response.text


def test_readiness_bounds_and_cancels_slow_dependency_probe(monkeypatch):
    _patch_urls(monkeypatch)
    monkeypatch.setenv("HAGENT_READINESS_TIMEOUT_SECONDS", "0.01")
    mongo_admin = _MongoAdmin(delay=1)
    monkeypatch.setattr(
        bridge_app.conv_store,
        "get_db_client",
        lambda: SimpleNamespace(admin=mongo_admin),
    )
    _patch_http(
        monkeypatch,
        {
            "http://runtime.internal:8585/api/v1/chat/health": _Response(
                200,
                {"hautoml_connected": True, "available_models": ["model-a"]},
            )
        },
    )

    response = _client().get("/api/v1/ready")

    assert response.status_code == 503
    assert response.json()["dependencies"]["mongodb"] == "unavailable"
    assert mongo_admin.cancelled is True


def test_readiness_fails_closed_when_timeout_config_is_invalid(monkeypatch):
    _patch_urls(monkeypatch)
    monkeypatch.setenv("HAGENT_READINESS_TIMEOUT_SECONDS", "not-a-number")
    mongo_admin = _MongoAdmin()
    monkeypatch.setattr(
        bridge_app.conv_store,
        "get_db_client",
        lambda: SimpleNamespace(admin=mongo_admin),
    )
    calls = _patch_http(monkeypatch, {})

    response = _client().get("/api/v1/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "dependencies": {"mongodb": "unavailable", "toolkit": "unavailable"},
    }
    assert mongo_admin.commands == []
    assert calls == []


def test_readiness_sanitizes_malformed_toolkit_url_before_any_probe(monkeypatch):
    monkeypatch.setenv(
        "HAGENT_RUN_API_URL",
        "http://runtime.internal:not-a-port/api/v1/runs/sentinel-secret",
    )
    mongo_admin = _MongoAdmin()
    monkeypatch.setattr(
        bridge_app.conv_store,
        "get_db_client",
        lambda: SimpleNamespace(admin=mongo_admin),
    )
    calls = _patch_http(monkeypatch, {})

    response = _client().get("/api/v1/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "dependencies": {"mongodb": "unavailable", "toolkit": "unavailable"},
    }
    assert "runtime.internal" not in response.text
    assert "sentinel-secret" not in response.text
    assert mongo_admin.commands == []
    assert calls == []


def test_readiness_bounds_and_cancels_slow_toolkit_probe(monkeypatch):
    _patch_urls(monkeypatch)
    monkeypatch.setenv("HAGENT_READINESS_TIMEOUT_SECONDS", "0.01")
    monkeypatch.setattr(
        bridge_app.conv_store,
        "get_db_client",
        lambda: SimpleNamespace(admin=_MongoAdmin()),
    )
    calls: list[str] = []
    slow_client = _SlowHttpClient(calls)
    monkeypatch.setattr(
        bridge_app.httpx,
        "AsyncClient",
        lambda *args, **kwargs: slow_client,
    )

    response = _client().get("/api/v1/ready")

    assert response.status_code == 503
    assert response.json()["dependencies"] == {
        "mongodb": "ready",
        "toolkit": "unavailable",
    }
    assert calls == ["http://runtime.internal:8585/api/v1/chat/health"]
    assert slow_client.cancelled is True


def test_readiness_sanitizes_toolkit_transport_exception(monkeypatch):
    _patch_urls(monkeypatch)
    monkeypatch.setattr(
        bridge_app.conv_store,
        "get_db_client",
        lambda: SimpleNamespace(admin=_MongoAdmin()),
    )
    _patch_http(
        monkeypatch,
        {
            "http://runtime.internal:8585/api/v1/chat/health": RuntimeError(
                "sentinel-secret at runtime.internal"
            )
        },
    )

    response = _client().get("/api/v1/ready")

    assert response.status_code == 503
    assert response.json()["dependencies"] == {
        "mongodb": "ready",
        "toolkit": "unavailable",
    }
    assert "sentinel-secret" not in response.text
    assert "runtime.internal" not in response.text
