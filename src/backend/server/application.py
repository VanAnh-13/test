"""Composition root FastAPI của HAutoML."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
import threading
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from aiokafka import AIOKafkaConsumer
from dotenv import load_dotenv
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.sessions import SessionMiddleware

# Nạp .env TRƯỚC các import nội bộ: nhiều module (master, minio, kafka, users)
# đọc biến môi trường ngay lúc import.
load_dotenv()

from api import api_v1_router
from api.experiment import exp
from automl.v2.master import master, monitor_tasks
from automl.v2.minio import minIOStorage
from config.server_runtime import ServerRuntimeConfig, load_server_runtime_config
from database.database import connection
from hagent.agent.llm import get_default_model_config, list_available_models
from hagent.agent.runtime import (
    RequestScope,
    RuntimeRunNotFound,
    create_agent_runtime,
    set_agent_runtime,
)
from hagent.chat import store as chat_store
from hagent.chat.router import router as chat_router
from hagent.run.router import router as run_router
from infrastructure.messaging.kafka import (
    kafka_consumer_process,
    start_producer,
    stop_producer,
)

logger = logging.getLogger(__name__)
_RUNTIME_PROBE_LOCK = threading.Lock()
_READINESS_RUN_ID = "__hagent_runtime_readiness__"
_READINESS_PRINCIPAL_ID = "__hagent_runtime_readiness__"
_DEVELOPMENT_KAFKA_SERVER = "localhost:9092"
_DEVELOPMENT_KAFKA_TOPIC = "example-topic"

SERVER_CONFIG = load_server_runtime_config()


def _consume_detached_task(task: asyncio.Task[Any]) -> None:
    try:
        task.exception()
    except (asyncio.CancelledError, Exception):  # noqa: BLE001, S110
        pass


async def _cancel_background_tasks(
    application: FastAPI, *, timeout_seconds: float
) -> None:
    tasks = [
        task
        for task in (
            getattr(application.state, "kafka_task", None),
            getattr(application.state, "monitor_task", None),
        )
        if task is not None
    ]
    for task in tasks:
        task.cancel()
    if tasks:
        done, pending = await asyncio.wait(tasks, timeout=timeout_seconds)
        for task in done:
            _consume_detached_task(task)
        for task in pending:
            task.add_done_callback(_consume_detached_task)
        if pending:
            logger.error(
                "Background task không dừng trong thời hạn",
                extra={"pending_count": len(pending)},
            )


async def _run_bounded_cleanup(
    awaitable: Any, *, resource: str, timeout_seconds: float
) -> bool:
    task = asyncio.create_task(awaitable)
    _done, pending = await asyncio.wait({task}, timeout=timeout_seconds)
    if pending:
        task.cancel()
        task.add_done_callback(_consume_detached_task)
        logger.error("Resource cleanup vượt quá thời hạn", extra={"resource": resource})
        return False
    try:
        task.result()
    except asyncio.CancelledError:
        logger.error("Resource cleanup bị hủy", extra={"resource": resource})
        return False
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Resource cleanup thất bại",
            extra={"resource": resource, "error_type": type(exc).__name__},
        )
        return False
    return True


async def _verify_background_tasks_started(
    application: FastAPI, *, server_mode: bool, timeout_seconds: float
) -> None:
    await asyncio.sleep(0)
    tasks = (application.state.kafka_task, application.state.monitor_task)
    if all(task is not None and not task.done() for task in tasks):
        return
    await _cancel_background_tasks(application, timeout_seconds=timeout_seconds)
    application.state.kafka_task = None
    application.state.monitor_task = None
    application.state.kafka_available = False
    logger.warning("Kafka background task không khởi động được")
    if server_mode:
        raise RuntimeError("Kafka background task startup is unavailable")


async def _close_runtime_handle(
    application: FastAPI, previous_runtime: Any, *, restore_runtime: bool
) -> None:
    handle = getattr(application.state, "agent_runtime_handle", None)
    if handle is None:
        return
    if restore_runtime:
        try:
            set_agent_runtime(previous_runtime)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Không khôi phục được Agent Runtime trước đó",
                extra={"error_type": type(exc).__name__},
            )
    application.state.agent_runtime_handle = None
    await handle.aclose()


async def _probe_kafka_consumer_startup(config: ServerRuntimeConfig) -> None:
    bootstrap = os.getenv("KAFKA_SERVER")
    topic = os.getenv("KAFKA_TOPIC")
    if config.server_mode and (not bootstrap or not topic):
        raise RuntimeError("Kafka consumer configuration is unavailable")
    consumer = AIOKafkaConsumer(
        topic or _DEVELOPMENT_KAFKA_TOPIC,
        bootstrap_servers=bootstrap or _DEVELOPMENT_KAFKA_SERVER,
        enable_auto_commit=False,
        group_id=None,
    )
    cleanup_succeeded = False
    try:
        await asyncio.wait_for(
            consumer.start(), timeout=config.readiness_timeout_seconds
        )
    finally:
        cleanup_succeeded = await _run_bounded_cleanup(
            consumer.stop(),
            resource="kafka_startup_probe",
            timeout_seconds=config.readiness_timeout_seconds,
        )
    if not cleanup_succeeded:
        raise RuntimeError("Kafka consumer startup probe cleanup failed")


def _runtime_factory_options(config: ServerRuntimeConfig) -> dict[str, Any]:
    runtime = config.agent_runtime
    return {
        "mode": runtime.mode,
        "persistence_mode": runtime.persistence_mode,
        "mongodb_uri": runtime.mongodb_uri,
        "db_name": runtime.db_name,
        "checkpoint_ttl_seconds": runtime.checkpoint_ttl_seconds,
        "event_retention_days": runtime.event_retention_days,
        "artifact_retention_days": runtime.artifact_retention_days,
        "server_selection_timeout_ms": runtime.server_selection_timeout_ms,
        "allow_memory": runtime.allow_memory,
    }


def _log_sanitized_startup_error(stage: str, exc: Exception) -> None:
    """Ghi log lỗi khởi động CHỈ với error_type, không bao giờ ghi message gốc.

    AUDIT-003: exception gốc từ pymongo/aiokafka có thể chứa nguyên văn connection
    string (username:password@host) — không được để lọt vào log hay vào exception
    lan truyền ra ngoài lifespan().
    """
    logger.error(
        "Startup stage thất bại, đã sanitize error",
        extra={"stage": stage, "error_type": type(exc).__name__},
    )


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Quản lý vòng đời nhất quán cho database, Agent Runtime và background workers."""
    config = getattr(application.state, "server_config", SERVER_CONFIG)
    application.state.kafka_task = None
    application.state.monitor_task = None
    application.state.kafka_available = False
    application.state.agent_runtime_handle = None
    application.state.client = None
    previous_runtime = None
    runtime_registered = False

    try:
        try:
            application.state.db, application.state.client = await connection()
        except Exception as exc:  # noqa: BLE001
            _log_sanitized_startup_error("database", exc)
            raise RuntimeError("Database startup failed") from None

        try:
            application.state.agent_runtime_handle = create_agent_runtime(
                **_runtime_factory_options(config)
            )
            # AUDIT-003: set_agent_runtime() trả về runtime CŨ nó vừa thay thế —
            # đây là nguồn previous_runtime chính xác duy nhất để khôi phục khi
            # startup thất bại (không phải attribute không tồn tại trên factory).
            previous_runtime = set_agent_runtime(
                application.state.agent_runtime_handle.runtime
            )
            runtime_registered = True
        except Exception as exc:  # noqa: BLE001
            _log_sanitized_startup_error("agent_runtime", exc)
            raise RuntimeError("Agent Runtime startup failed") from None

        if config.server_mode:
            try:
                await _probe_kafka_consumer_startup(config)
            except Exception as exc:  # noqa: BLE001
                _log_sanitized_startup_error("kafka_probe", exc)
                raise RuntimeError("Kafka startup failed") from None

        try:
            await chat_store.ensure_indexes(application.state.db)
        except Exception as exc:  # noqa: BLE001
            _log_sanitized_startup_error("chat_store", exc)
            raise RuntimeError("Chat store startup failed") from None

        kafka_started = True
        try:
            await start_producer()
        except Exception as exc:  # noqa: BLE001
            if config.server_mode:
                _log_sanitized_startup_error("kafka_producer", exc)
                raise RuntimeError("Kafka producer startup failed") from None
            kafka_started = False
            logger.warning(
                "Kafka không khả dụng ngoài server mode, chạy chế độ degraded",
                extra={"error_type": type(exc).__name__},
            )

        if kafka_started:
            application.state.kafka_task = asyncio.create_task(kafka_consumer_process())
            application.state.monitor_task = asyncio.create_task(monitor_tasks())
            await _verify_background_tasks_started(
                application,
                server_mode=config.server_mode,
                timeout_seconds=config.readiness_timeout_seconds,
            )
            application.state.kafka_available = True
    except Exception:
        await _cancel_background_tasks(
            application, timeout_seconds=config.readiness_timeout_seconds
        )
        await _close_runtime_handle(
            application, previous_runtime, restore_runtime=runtime_registered
        )
        await _run_bounded_cleanup(
            stop_producer(),
            resource="kafka_producer",
            timeout_seconds=config.readiness_timeout_seconds,
        )
        client = getattr(application.state, "client", None)
        if client is not None:
            application.state.client = None
            await _run_bounded_cleanup(
                client.close(),
                resource="mongodb_client",
                timeout_seconds=config.readiness_timeout_seconds,
            )
        raise

    yield

    await _cancel_background_tasks(
        application, timeout_seconds=config.readiness_timeout_seconds
    )
    await _close_runtime_handle(
        application, previous_runtime, restore_runtime=runtime_registered
    )
    await _run_bounded_cleanup(
        stop_producer(),
        resource="kafka_producer",
        timeout_seconds=config.readiness_timeout_seconds,
    )
    client = getattr(application.state, "client", None)
    if client is not None:
        application.state.client = None
        await _run_bounded_cleanup(
            client.close(),
            resource="mongodb_client",
            timeout_seconds=config.readiness_timeout_seconds,
        )


app = FastAPI(lifespan=lifespan)
app.state.server_config = SERVER_CONFIG

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(SERVER_CONFIG.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    SessionMiddleware,
    secret_key=SERVER_CONFIG.session_secret,
    https_only=SERVER_CONFIG.session_https_only,
    same_site="lax",
)

# Gắn các router theo từng miền API.
app.include_router(api_v1_router)
app.include_router(exp)
app.include_router(master)
app.include_router(chat_router)
app.include_router(run_router)


@app.get("/")
async def read_root() -> dict[str, str]:
    return {
        "HAutoML": "Open-Source for Automated Machine Learning",
        "Authors": "Đỗ Mạnh Quang, Chử Thị Ánh, Ngọ Công Bình, Bùi Huy Nam, Nguyễn Thị Mỹ Khánh, Nguyễn Thị Minh",
        "Lab": "OptiVisionLab",
        "University": "School of Information and Communications Technology, Hanoi University of Industry",
    }


@app.get("/home")
async def ping() -> dict[str, str]:
    return {"AutoML": "version 1.0", "message": "Hi there :P"}


async def _probe_mongodb(application: FastAPI) -> bool:
    client = getattr(application.state, "client", None)
    if client is None:
        return False
    result = await client.admin.command("ping")
    return bool(result.get("ok"))


async def _probe_kafka(application: FastAPI) -> bool:
    if not getattr(application.state, "kafka_available", False):
        return False
    tasks = (
        getattr(application.state, "kafka_task", None),
        getattr(application.state, "monitor_task", None),
    )
    return all(task is not None and not task.done() for task in tasks)


async def _probe_minio(_application: FastAPI) -> bool:
    await asyncio.to_thread(minIOStorage.healthcheck)
    return True


async def _probe_providers(_application: FastAPI) -> bool:
    models = await asyncio.to_thread(list_available_models)
    config = await asyncio.to_thread(get_default_model_config)
    config_values = tuple(
        getattr(config, field, None) for field in ("name", "provider", "model")
    )
    if not all(isinstance(value, str) and value.strip() for value in config_values):
        return False
    default_name, provider, default_model = (value.strip() for value in config_values)
    provider = provider.lower()
    if not isinstance(models, list) or not any(
        isinstance(model, dict)
        and all(
            isinstance(model.get(f), str) and bool(model[f].strip())
            for f in ("name", "provider", "model")
        )
        and model["name"].strip() == default_name
        and model["provider"].strip().lower() == provider
        and model["model"].strip() == default_model
        for model in models
    ):
        return False
    if provider in {"openai", "anthropic"}:
        api_key = config.resolve_api_key()
        if not isinstance(api_key, str) or not api_key.strip():
            return False
        compact_key = re.sub(r"[^a-z0-9]", "", api_key.strip().lower())
        return not any(
            marker in compact_key
            for marker in ("changeme", "replaceme", "placeholder", "exampleonly")
        )
    if provider == "ollama":
        base_url = config.base_url or os.getenv("OLLAMA_BASE_URL")
        return isinstance(base_url, str) and bool(base_url.strip())
    if provider == "openai_compatible":
        return isinstance(config.base_url, str) and bool(config.base_url.strip())
    return False


async def _probe_agent_runtime(application: FastAPI) -> bool:
    handle = getattr(application.state, "agent_runtime_handle", None)
    config = getattr(application.state, "server_config", SERVER_CONFIG)
    if (
        handle is None
        or handle.mode != config.agent_runtime.mode
        or getattr(handle, "_closed", False)
    ):
        return False
    explicit_healthcheck = getattr(handle, "healthcheck", None)
    if callable(explicit_healthcheck):
        if inspect.iscoroutinefunction(explicit_healthcheck):
            result = await explicit_healthcheck()
        else:
            result = await asyncio.to_thread(
                _run_sync_runtime_healthcheck, explicit_healthcheck
            )
            if inspect.isawaitable(result):
                result = await result
        return result is not False
    if handle.mode == "legacy":
        return True
    scope = RequestScope(principal_id=_READINESS_PRINCIPAL_ID)
    try:
        async for _event in handle.runtime.replay(
            _READINESS_RUN_ID, after_sequence=0, scope=scope
        ):
            pass
    except RuntimeRunNotFound:
        return True
    return True


def _run_sync_runtime_healthcheck(healthcheck: Any) -> bool:
    if not _RUNTIME_PROBE_LOCK.acquire(blocking=False):
        return False
    try:
        return healthcheck()
    finally:
        _RUNTIME_PROBE_LOCK.release()


async def _bounded_readiness_probe(
    name: str, probe: Any, application: FastAPI, timeout_seconds: float
) -> tuple[str, bool]:
    try:
        ready = await asyncio.wait_for(probe(application), timeout=timeout_seconds)
        return name, bool(ready)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Readiness dependency không khả dụng",
            extra={"dependency": name, "error_type": type(exc).__name__},
        )
        return name, False


async def _readiness_response(application: FastAPI) -> JSONResponse:
    config = getattr(application.state, "server_config", SERVER_CONFIG)
    probes = {
        "mongodb": _probe_mongodb,
        "kafka": _probe_kafka,
        "minio": _probe_minio,
        "providers": _probe_providers,
        "runtime": _probe_agent_runtime,
    }
    results = await asyncio.gather(
        *(
            _bounded_readiness_probe(
                name, probe, application, config.readiness_timeout_seconds
            )
            for name, probe in probes.items()
        )
    )
    ready = all(is_ready for _, is_ready in results)
    return JSONResponse(
        status_code=status.HTTP_200_OK
        if ready
        else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"ready": ready},
    )


@app.get("/ready", include_in_schema=False)
async def readiness() -> JSONResponse:
    return await _readiness_response(app)


if __name__ == "__main__":
    uvicorn.run(
        "server.application:app",
        host=os.getenv("HOST_BACK_END", "0.0.0.0"),
        port=int(os.getenv("PORT_BACK_END", "8080")),
        reload=SERVER_CONFIG.reload_enabled,
    )
