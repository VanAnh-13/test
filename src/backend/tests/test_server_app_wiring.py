from __future__ import annotations

import asyncio
import concurrent.futures
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

_APP_IMPORT_ENV = {
    "DEPLOY_MODE": "test",
    "MINIO_ENDPOINT": "localhost:9000",
    "MINIO_ACCESS_KEY": "test-access",
    "MINIO_SECRET_KEY": "test-secret-0123456789",
}
_PREVIOUS_ENV = {key: os.environ.get(key) for key in _APP_IMPORT_ENV}
os.environ.update(_APP_IMPORT_ENV)
try:
    from server import application as app_module
finally:
    for _key, _value in _PREVIOUS_ENV.items():
        if _value is None:
            os.environ.pop(_key, None)
        else:
            os.environ[_key] = _value
from automl.v2.minio import MinIOStorage, _healthcheck_timeout_seconds
from config.server_runtime import load_server_runtime_config


def test_application_composition_root_is_owned_by_server_package() -> None:
    backend_root = Path(__file__).resolve().parents[1]

    assert app_module.__name__ == "server.application"
    assert not (backend_root / "app.py").exists()


def _private_config(**overrides: str):
    environment = {
        "DEPLOY_MODE": "private",
        "APP_ORIGIN": "http://localhost:8080",
        "SUPER_SECRET_KEY": "private-runtime-8f92c4ab31d7e650-AZURE",
        "SESSION_HTTPS_ONLY": "false",
        "BACKEND_RELOAD": "false",
        "HAGENT_RUNTIME_MODE": "journey",
        "HAGENT_CHECKPOINT_BACKEND": "mongodb",
        "MONGODB_CONNECT": ("mongodb://runtime-user:runtime-password@mongo:27017/"),
        "HAGENT_RUNTIME_DB_NAME": "hagent_journey",
        "HAGENT_CHECKPOINT_TTL_SECONDS": "2592000",
        "HAGENT_EVENT_RETENTION_DAYS": "30",
        "HAGENT_RUNTIME_SERVER_SELECTION_TIMEOUT_MS": "2000",
        "SERVER_READINESS_TIMEOUT_SECONDS": "0.05",
    }
    environment.update(overrides)
    return load_server_runtime_config(environment)


def _middleware_options(application: FastAPI, middleware_type: type) -> dict[str, Any]:
    middleware = next(
        item for item in application.user_middleware if item.cls is middleware_type
    )
    return dict(middleware.kwargs)


def test_application_wires_exact_http_boundary_and_run_routes() -> None:
    cors = _middleware_options(app_module.app, CORSMiddleware)
    session = _middleware_options(app_module.app, SessionMiddleware)
    paths = {route.path for route in app_module.app.routes}

    assert cors["allow_origins"] == list(app_module.SERVER_CONFIG.cors_origins)
    assert "*" not in cors["allow_origins"]
    assert session["secret_key"] == app_module.SERVER_CONFIG.session_secret
    assert session["https_only"] is app_module.SERVER_CONFIG.session_https_only
    assert "/api/v1/runs" in paths
    assert "/ready" in paths


class _FakeAdmin:
    def __init__(self, *, error: Exception | None = None, delay: float = 0) -> None:
        self.error = error
        self.delay = delay

    async def command(self, name: str) -> dict[str, int]:
        assert name == "ping"
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return {"ok": 1}


class _FakeClient:
    def __init__(self, admin: _FakeAdmin | None = None) -> None:
        self.admin = admin or _FakeAdmin()
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _PendingTask:
    def done(self) -> bool:
        return False


def _ready_application(config=None) -> FastAPI:
    application = FastAPI()
    application.state.server_config = config or _private_config()
    application.state.client = _FakeClient()
    application.state.kafka_available = True
    application.state.kafka_task = _PendingTask()
    application.state.monitor_task = _PendingTask()
    application.state.agent_runtime_handle = SimpleNamespace(
        mode="journey",
        healthcheck=lambda: True,
    )
    return application


def _configured_default_model():
    return SimpleNamespace(
        name="model-a",
        provider="ollama",
        model="qwen",
        base_url="http://ollama:11434",
        resolve_api_key=lambda: None,
    )


@pytest.mark.asyncio
async def test_readiness_returns_200_only_when_every_dependency_is_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = _ready_application()
    monkeypatch.setattr(app_module.minIOStorage, "healthcheck", lambda: None)
    monkeypatch.setattr(
        app_module,
        "list_available_models",
        lambda: [{"name": "model-a", "provider": "ollama", "model": "qwen"}],
    )
    monkeypatch.setattr(
        app_module, "get_default_model_config", _configured_default_model
    )

    response = await app_module._readiness_response(application)

    assert response.status_code == 200
    assert response.body == b'{"ready":true}'
    assert b"runtime-password" not in response.body
    assert b"mongo:27017" not in response.body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dependency", ["mongodb", "kafka", "minio", "providers", "runtime"]
)
async def test_readiness_is_sanitized_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    dependency: str,
) -> None:
    sentinel = "secret-user:secret-password@internal-host:27017"
    application = _ready_application()
    monkeypatch.setattr(app_module.minIOStorage, "healthcheck", lambda: None)
    monkeypatch.setattr(
        app_module,
        "list_available_models",
        lambda: [{"name": "model-a", "provider": "ollama", "model": "qwen"}],
    )
    monkeypatch.setattr(
        app_module, "get_default_model_config", _configured_default_model
    )
    if dependency == "mongodb":
        application.state.client = _FakeClient(_FakeAdmin(error=RuntimeError(sentinel)))
    elif dependency == "kafka":
        application.state.kafka_available = False
    elif dependency == "minio":
        monkeypatch.setattr(
            app_module.minIOStorage,
            "healthcheck",
            lambda: (_ for _ in ()).throw(RuntimeError(sentinel)),
        )
    elif dependency == "providers":
        monkeypatch.setattr(app_module, "list_available_models", list)
    else:
        application.state.agent_runtime_handle = None

    response = await app_module._readiness_response(application)

    assert response.status_code == 503
    assert sentinel.encode() not in response.body
    assert response.body == b'{"ready":false}'
    assert dependency.encode() not in response.body


@pytest.mark.asyncio
async def test_readiness_bounds_slow_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = _ready_application()
    application.state.client = _FakeClient(_FakeAdmin(delay=0.2))
    monkeypatch.setattr(app_module.minIOStorage, "healthcheck", lambda: None)
    monkeypatch.setattr(
        app_module,
        "list_available_models",
        lambda: [{"name": "model-a", "provider": "ollama", "model": "qwen"}],
    )
    monkeypatch.setattr(
        app_module, "get_default_model_config", _configured_default_model
    )

    response = await asyncio.wait_for(
        app_module._readiness_response(application), timeout=0.15
    )

    assert response.status_code == 503
    assert response.body == b'{"ready":false}'


@pytest.mark.asyncio
async def test_provider_probe_rejects_placeholder_cloud_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = _ready_application()
    default_model = SimpleNamespace(
        name="cloud-model",
        provider="openai",
        model="gpt-test",
        base_url=None,
        resolve_api_key=lambda: "CHANGE_ME_OPENAI_API_KEY",
    )
    monkeypatch.setattr(app_module.minIOStorage, "healthcheck", lambda: None)
    monkeypatch.setattr(
        app_module,
        "list_available_models",
        lambda: [{"name": "cloud-model", "provider": "openai", "model": "gpt-test"}],
    )
    monkeypatch.setattr(app_module, "get_default_model_config", lambda: default_model)

    response = await app_module._readiness_response(application)

    assert response.status_code == 503
    assert response.body == b'{"ready":false}'
    assert b"OPENAI_API_KEY" not in response.body


@pytest.mark.asyncio
async def test_provider_probe_ignores_unconfigured_optional_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = _ready_application()
    default_model = SimpleNamespace(
        name="cloud-model",
        provider="openai",
        model="gpt-test",
        base_url=None,
        resolve_api_key=lambda: "unit-test-provider-key",
    )
    monkeypatch.setattr(app_module.minIOStorage, "healthcheck", lambda: None)
    monkeypatch.setattr(
        app_module,
        "list_available_models",
        lambda: [
            {"name": "cloud-model", "provider": "openai", "model": "gpt-test"},
            {
                "name": "local-compatible",
                "provider": "openai_compatible",
                "model": "",
            },
        ],
    )
    monkeypatch.setattr(app_module, "get_default_model_config", lambda: default_model)

    response = await app_module._readiness_response(application)

    assert response.status_code == 200
    assert response.body == b'{"ready":true}'


@pytest.mark.asyncio
async def test_provider_probe_rejects_default_missing_from_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = _ready_application()
    default_model = SimpleNamespace(
        name="cloud-model",
        provider="openai",
        model="gpt-test",
        base_url=None,
        resolve_api_key=lambda: "unit-test-provider-key",
    )
    monkeypatch.setattr(app_module.minIOStorage, "healthcheck", lambda: None)
    monkeypatch.setattr(
        app_module,
        "list_available_models",
        lambda: [{"name": "other-model", "provider": "openai", "model": "gpt-other"}],
    )
    monkeypatch.setattr(app_module, "get_default_model_config", lambda: default_model)

    response = await app_module._readiness_response(application)

    assert response.status_code == 503
    assert response.body == b'{"ready":false}'


class _FakeMinioClient:
    def __init__(self) -> None:
        self.list_calls = 0

    def list_buckets(self) -> list[Any]:
        self.list_calls += 1
        return []


def test_minio_healthcheck_is_read_only() -> None:
    client = _FakeMinioClient()
    storage = object.__new__(MinIOStorage)
    storage._MinIOStorage__healthcheck_client = client
    storage._MinIOStorage__healthcheck_lock = threading.Lock()

    storage.healthcheck()

    assert client.list_calls == 1


class _HungMinioClient:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.list_calls = 0

    def list_buckets(self) -> list[Any]:
        self.list_calls += 1
        self.started.set()
        assert self.release.wait(timeout=1)
        return []


def test_minio_healthcheck_allows_only_one_inflight_probe() -> None:
    client = _HungMinioClient()
    storage = object.__new__(MinIOStorage)
    storage._MinIOStorage__healthcheck_client = client
    storage._MinIOStorage__healthcheck_lock = threading.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(storage.healthcheck)
        assert client.started.wait(timeout=1)
        with pytest.raises(RuntimeError, match="healthcheck"):
            storage.healthcheck()
        client.release.set()
        first.result(timeout=1)

    assert client.list_calls == 1


@pytest.mark.parametrize("value", ["nan", "inf", "0", "invalid"])
def test_minio_healthcheck_timeout_rejects_invalid_value(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("MINIO_HEALTHCHECK_TIMEOUT_SECONDS", value)

    with pytest.raises(ValueError, match="MINIO_HEALTHCHECK_TIMEOUT_SECONDS"):
        _healthcheck_timeout_seconds()


@dataclass
class _FakeRuntimeHandle:
    runtime: object
    mode: str = "journey"
    closed: bool = False

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("retention_override", "expected_retention_days"),
    [(None, 180), ("731", 731)],
)
async def test_lifespan_owns_runtime_and_restores_previous_global(
    monkeypatch: pytest.MonkeyPatch,
    retention_override: str | None,
    expected_retention_days: int,
) -> None:
    application = FastAPI()
    overrides = (
        {"HAGENT_ARTIFACT_RETENTION_DAYS": retention_override}
        if retention_override is not None
        else {}
    )
    application.state.server_config = _private_config(**overrides)
    client = _FakeClient()
    handle = _FakeRuntimeHandle(runtime=object())
    previous_runtime = object()
    runtime_changes: list[object | None] = []
    runtime_options: list[dict[str, Any]] = []
    stop_calls: list[bool] = []

    async def idle(*_args: Any) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(
        app_module, "connection", lambda: _async_value((object(), client))
    )

    def create_runtime(**options: Any) -> _FakeRuntimeHandle:
        runtime_options.append(options)
        return handle

    monkeypatch.setattr(app_module, "create_agent_runtime", create_runtime)
    monkeypatch.setattr(
        app_module,
        "set_agent_runtime",
        lambda value: runtime_changes.append(value) or previous_runtime,
    )
    monkeypatch.setattr(app_module, "start_producer", _async_noop)
    monkeypatch.setattr(app_module, "_probe_kafka_consumer_startup", _async_noop)
    monkeypatch.setattr(app_module, "stop_producer", lambda: _record_async(stop_calls))
    monkeypatch.setattr(app_module, "kafka_consumer_process", idle)
    monkeypatch.setattr(app_module, "monitor_tasks", idle)
    monkeypatch.setattr(app_module.chat_store, "ensure_indexes", _async_noop)

    async with app_module.lifespan(application):
        assert application.state.agent_runtime_handle is handle
        assert runtime_changes == [handle.runtime]
        assert application.state.kafka_available is True
        assert runtime_options == [
            {
                "mode": "journey",
                "persistence_mode": "mongodb",
                "mongodb_uri": ("mongodb://runtime-user:runtime-password@mongo:27017/"),
                "db_name": "hagent_journey",
                "checkpoint_ttl_seconds": 2592000,
                "event_retention_days": 30,
                "artifact_retention_days": expected_retention_days,
                "server_selection_timeout_ms": 2000,
                "allow_memory": False,
            }
        ]

    assert runtime_changes == [handle.runtime, previous_runtime]
    assert handle.closed is True
    assert client.closed is True
    assert stop_calls == [True]


async def _async_value(value: Any) -> Any:
    return value


async def _async_noop(*_args: Any, **_kwargs: Any) -> None:
    return None


async def _record_async(target: list[bool]) -> None:
    target.append(True)


@pytest.mark.asyncio
async def test_server_startup_failure_cleans_runtime_without_leaking_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = FastAPI()
    application.state.server_config = _private_config()
    client = _FakeClient()
    handle = _FakeRuntimeHandle(runtime=object())
    previous_runtime = object()
    runtime_changes: list[object | None] = []
    sentinel = "secret-user:secret-password@internal-host:9092"

    async def fail_producer() -> None:
        raise RuntimeError(sentinel)

    monkeypatch.setattr(
        app_module, "connection", lambda: _async_value((object(), client))
    )
    monkeypatch.setattr(app_module, "create_agent_runtime", lambda **_kwargs: handle)
    monkeypatch.setattr(
        app_module,
        "set_agent_runtime",
        lambda value: runtime_changes.append(value) or previous_runtime,
    )
    monkeypatch.setattr(app_module, "start_producer", fail_producer)
    monkeypatch.setattr(app_module, "_probe_kafka_consumer_startup", _async_noop)
    monkeypatch.setattr(app_module.chat_store, "ensure_indexes", _async_noop)
    monkeypatch.setattr(app_module, "stop_producer", _async_noop)

    with pytest.raises(RuntimeError) as error:
        async with app_module.lifespan(application):
            pytest.fail("Lifespan không được yield khi Kafka startup lỗi")

    assert sentinel not in str(error.value)
    assert runtime_changes == [handle.runtime, previous_runtime]
    assert handle.closed is True
    assert client.closed is True


@pytest.mark.asyncio
async def test_durable_runtime_startup_failure_is_sanitized_and_closes_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = FastAPI()
    application.state.server_config = _private_config()
    client = _FakeClient()
    sentinel = "mongodb://runtime-user:secret-password@internal-host:27017"

    def fail_runtime(**_kwargs: Any):
        raise RuntimeError(sentinel)

    monkeypatch.setattr(
        app_module, "connection", lambda: _async_value((object(), client))
    )
    monkeypatch.setattr(app_module, "create_agent_runtime", fail_runtime)
    monkeypatch.setattr(app_module, "stop_producer", _async_noop)

    with pytest.raises(RuntimeError, match="Agent Runtime startup") as error:
        async with app_module.lifespan(application):
            pytest.fail("Lifespan không được yield khi durable runtime lỗi")

    assert sentinel not in str(error.value)
    assert client.closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize("startup_stage", ["database", "runtime"])
async def test_non_server_startup_errors_are_also_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    startup_stage: str,
) -> None:
    application = FastAPI()
    application.state.server_config = load_server_runtime_config(
        {"DEPLOY_MODE": "test"}
    )
    client = _FakeClient()
    sentinel = "mongodb://runtime-user:secret-password@internal-host:27017"

    async def fail_database():
        raise RuntimeError(sentinel)

    def fail_runtime(**_kwargs: Any):
        raise RuntimeError(sentinel)

    if startup_stage == "database":
        monkeypatch.setattr(app_module, "connection", fail_database)
    else:
        monkeypatch.setattr(
            app_module, "connection", lambda: _async_value((object(), client))
        )
        monkeypatch.setattr(app_module, "create_agent_runtime", fail_runtime)
        monkeypatch.setattr(app_module, "stop_producer", _async_noop)

    with pytest.raises(RuntimeError) as error:
        async with app_module.lifespan(application):
            pytest.fail("Lifespan không được yield khi startup lỗi")

    assert sentinel not in str(error.value)


@pytest.mark.asyncio
async def test_development_can_start_degraded_when_kafka_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = FastAPI()
    application.state.server_config = load_server_runtime_config(
        {"DEPLOY_MODE": "test"}
    )
    client = _FakeClient()
    handle = _FakeRuntimeHandle(runtime=object(), mode="legacy")
    previous_runtime = object()

    async def fail_producer() -> None:
        raise RuntimeError("local-kafka-offline")

    monkeypatch.setattr(
        app_module, "connection", lambda: _async_value((object(), client))
    )
    monkeypatch.setattr(app_module, "create_agent_runtime", lambda **_kwargs: handle)
    monkeypatch.setattr(
        app_module, "set_agent_runtime", lambda _value: previous_runtime
    )
    monkeypatch.setattr(app_module, "start_producer", fail_producer)
    monkeypatch.setattr(app_module, "stop_producer", _async_noop)
    monkeypatch.setattr(app_module.chat_store, "ensure_indexes", _async_noop)

    async with app_module.lifespan(application):
        assert application.state.kafka_available is False
        assert application.state.kafka_task is None

    assert handle.closed is True
    assert client.closed is True


@pytest.mark.asyncio
async def test_server_rejects_background_task_that_fails_before_yield(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = FastAPI()
    application.state.server_config = _private_config()
    client = _FakeClient()
    handle = _FakeRuntimeHandle(runtime=object())
    previous_runtime = object()
    sentinel = "secret-user:secret-password@internal-kafka:9092"

    async def fail_immediately(*_args: Any) -> None:
        raise RuntimeError(sentinel)

    async def idle(*_args: Any) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(
        app_module, "connection", lambda: _async_value((object(), client))
    )
    monkeypatch.setattr(app_module, "create_agent_runtime", lambda **_kwargs: handle)
    monkeypatch.setattr(
        app_module, "set_agent_runtime", lambda _value: previous_runtime
    )
    monkeypatch.setattr(app_module, "start_producer", _async_noop)
    monkeypatch.setattr(app_module, "_probe_kafka_consumer_startup", _async_noop)
    monkeypatch.setattr(app_module, "stop_producer", _async_noop)
    monkeypatch.setattr(app_module, "kafka_consumer_process", fail_immediately)
    monkeypatch.setattr(app_module, "monitor_tasks", idle)
    monkeypatch.setattr(app_module.chat_store, "ensure_indexes", _async_noop)

    with pytest.raises(RuntimeError, match="background task") as error:
        async with app_module.lifespan(application):
            pytest.fail("Lifespan không được yield khi background task lỗi ngay")

    assert sentinel not in str(error.value)
    assert handle.closed is True
    assert client.closed is True


@pytest.mark.asyncio
async def test_server_rejects_consumer_probe_failure_after_await(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = FastAPI()
    application.state.server_config = _private_config()
    client = _FakeClient()
    handle = _FakeRuntimeHandle(runtime=object())
    previous_runtime = object()
    sentinel = "secret-user:secret-password@internal-kafka:9092"

    async def fail_consumer_probe(*_args: Any) -> None:
        await asyncio.sleep(0)
        raise RuntimeError(sentinel)

    monkeypatch.setattr(
        app_module, "connection", lambda: _async_value((object(), client))
    )
    monkeypatch.setattr(app_module, "create_agent_runtime", lambda **_kwargs: handle)
    monkeypatch.setattr(
        app_module, "set_agent_runtime", lambda _value: previous_runtime
    )
    monkeypatch.setattr(app_module, "start_producer", _async_noop)
    monkeypatch.setattr(
        app_module, "_probe_kafka_consumer_startup", fail_consumer_probe
    )
    monkeypatch.setattr(app_module, "stop_producer", _async_noop)

    with pytest.raises(RuntimeError, match="Kafka startup") as error:
        async with app_module.lifespan(application):
            pytest.fail("Lifespan không được yield khi consumer probe lỗi")

    assert sentinel not in str(error.value)
    assert handle.closed is True
    assert client.closed is True


class _ReplayRuntime:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    async def replay(self, run_id: str, *, after_sequence: int, scope):
        self.calls += 1
        assert run_id == app_module._READINESS_RUN_ID
        assert after_sequence == 0
        assert scope.principal_id == app_module._READINESS_PRINCIPAL_ID
        if self.error is not None:
            raise self.error
        if False:
            yield None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["journey", "shadow"])
async def test_durable_readiness_uses_public_runtime_replay_contract(
    mode: str,
) -> None:
    application = _ready_application()
    application.state.server_config = SimpleNamespace(
        agent_runtime=SimpleNamespace(mode=mode),
    )
    runtime = _ReplayRuntime(app_module.RuntimeRunNotFound())
    application.state.agent_runtime_handle = SimpleNamespace(
        mode=mode,
        runtime=runtime,
        _closed=False,
    )

    assert await app_module._probe_agent_runtime(application) is True
    assert runtime.calls == 1

    application.state.agent_runtime_handle._closed = True
    assert await app_module._probe_agent_runtime(application) is False
    assert runtime.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["journey", "shadow"])
async def test_durable_readiness_fails_when_public_replay_fails(mode: str) -> None:
    application = _ready_application()
    application.state.server_config = SimpleNamespace(
        agent_runtime=SimpleNamespace(mode=mode),
    )
    runtime = _ReplayRuntime(RuntimeError("private-runtime-storage"))
    application.state.agent_runtime_handle = SimpleNamespace(
        mode=mode,
        runtime=runtime,
        _closed=False,
    )

    name, ready = await app_module._bounded_readiness_probe(
        "runtime",
        app_module._probe_agent_runtime,
        application,
        0.05,
    )

    assert name == "runtime"
    assert ready is False
    assert runtime.calls == 1


@pytest.mark.asyncio
async def test_sync_runtime_healthcheck_is_bounded_and_single_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = _ready_application()
    started = threading.Event()
    release = threading.Event()

    def hung_healthcheck() -> bool:
        started.set()
        assert release.wait(timeout=1)
        return True

    application.state.agent_runtime_handle = SimpleNamespace(
        mode="journey",
        healthcheck=hung_healthcheck,
    )
    monkeypatch.setattr(app_module.minIOStorage, "healthcheck", lambda: None)
    monkeypatch.setattr(
        app_module,
        "list_available_models",
        lambda: [{"name": "model-a", "provider": "ollama", "model": "qwen"}],
    )
    monkeypatch.setattr(
        app_module, "get_default_model_config", _configured_default_model
    )

    response = await asyncio.wait_for(
        app_module._readiness_response(application), timeout=0.15
    )

    assert started.is_set()
    assert response.status_code == 503
    assert response.body == b'{"ready":false}'
    release.set()
    await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_hung_cleanup_does_not_block_later_resource_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = FastAPI()
    application.state.server_config = _private_config()
    client = _FakeClient()
    handle = _FakeRuntimeHandle(runtime=object())
    previous_runtime = object()
    release = asyncio.Event()

    async def hung_stop() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()

    async def idle(*_args: Any) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(
        app_module, "connection", lambda: _async_value((object(), client))
    )
    monkeypatch.setattr(app_module, "create_agent_runtime", lambda **_kwargs: handle)
    monkeypatch.setattr(
        app_module, "set_agent_runtime", lambda _value: previous_runtime
    )
    monkeypatch.setattr(app_module, "start_producer", _async_noop)
    monkeypatch.setattr(app_module, "_probe_kafka_consumer_startup", _async_noop)
    monkeypatch.setattr(app_module, "stop_producer", hung_stop)
    monkeypatch.setattr(app_module, "kafka_consumer_process", idle)
    monkeypatch.setattr(app_module, "monitor_tasks", idle)
    monkeypatch.setattr(app_module.chat_store, "ensure_indexes", _async_noop)

    async with app_module.lifespan(application):
        pass

    assert handle.closed is True
    assert client.closed is True
    release.set()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_cancelled_cleanup_does_not_block_later_resource_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = FastAPI()
    application.state.server_config = _private_config()
    client = _FakeClient()
    handle = _FakeRuntimeHandle(runtime=object())
    previous_runtime = object()

    async def cancelled_stop() -> None:
        raise asyncio.CancelledError

    async def idle(*_args: Any) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(
        app_module, "connection", lambda: _async_value((object(), client))
    )
    monkeypatch.setattr(app_module, "create_agent_runtime", lambda **_kwargs: handle)
    monkeypatch.setattr(
        app_module, "set_agent_runtime", lambda _value: previous_runtime
    )
    monkeypatch.setattr(app_module, "start_producer", _async_noop)
    monkeypatch.setattr(app_module, "_probe_kafka_consumer_startup", _async_noop)
    monkeypatch.setattr(app_module, "stop_producer", cancelled_stop)
    monkeypatch.setattr(app_module, "kafka_consumer_process", idle)
    monkeypatch.setattr(app_module, "monitor_tasks", idle)
    monkeypatch.setattr(app_module.chat_store, "ensure_indexes", _async_noop)

    async with app_module.lifespan(application):
        pass

    assert handle.closed is True
    assert client.closed is True
