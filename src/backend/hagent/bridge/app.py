"""
HAgent Bridge — Ứng dụng FastAPI chính

Lớp trung gian giữa frontend ChatWidget và HAgent runtime (LangGraph).
Tất cả cấu hình được tải từ hagent.yaml — KHÔNG có hard-code.

Xử lý:
  - Xác thực JWT
  - Lưu trữ cuộc hội thoại (MongoDB)
  - Điều phối provider/model (tải từ YAML)
  - Chuyển tiếp upload file
"""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from ..world import updater as world_updater
from ..world.schema import WorldState
from ..world.state_store import WorldStateStore
from . import conversation as conv_store
from .auth import TokenPayload, get_current_user, get_optional_user
from .config import (
    get_bridge_config,
    get_hautoml_config,
    get_llm_config,
    get_llm_models,
    get_mongodb_config,
    get_world_state_config,
)
from .models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ProviderInfo,
    ProvidersResponse,
    SuggestionsResponse,
)

logger = logging.getLogger("hagent.bridge")

UUID_PATTERN = (
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
JOB_ID_LINE_RE = re.compile(rf"Job\s*ID\s*[:\-]\s*`?({UUID_PATTERN})`?", re.IGNORECASE)
GENERIC_UUID_RE = re.compile(rf"\b({UUID_PATTERN})\b")
TRAINING_STARTED_RE = re.compile(
    r"training\s+started|training\s+job\s+initiated|bắt\s+đầu\s+huấn\s+luyện|huấn\s+luyện\s+đã\s+bắt\s+đầu",
    re.IGNORECASE,
)
TRAINING_POLL_INTERVAL_SECONDS = float(
    os.getenv("HAGENT_TRAINING_POLL_INTERVAL_SECONDS", "10")
)
TRAINING_POLL_TIMEOUT_SECONDS = int(
    os.getenv("HAGENT_TRAINING_POLL_TIMEOUT_SECONDS", "7200")
)
_training_watch_tasks: dict[str, asyncio.Task] = {}


# ─── Vòng đời ứng dụng ──────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý startup/shutdown."""
    mongo_cfg = get_mongodb_config()
    bridge_cfg = get_bridge_config()
    world_state_cfg = get_world_state_config()

    # Kết nối MongoDB
    logger.info("Đang kết nối MongoDB tại %s ...", mongo_cfg["connect"])
    await conv_store.init_db()
    logger.info("Kết nối MongoDB thành công ✓")

    # Khởi tạo WorldStateStore
    app.state.world_state_store = WorldStateStore(
        client=conv_store.get_db_client(),
        db_name=mongo_cfg["db_name"],
        collection_name=world_state_cfg["collection_name"],
        ttl_seconds=world_state_cfg["ttl_seconds"],
    )
    await app.state.world_state_store.ensure_indexes()
    logger.info("WorldStateStore đã khởi tạo ✓")

    # Trajectory store for LeWM offline learning
    try:
        from hagent.world.trajectory_store import create_trajectory_store

        app.state.trajectory_store = create_trajectory_store(
            conv_store.get_db_client(),
            db_name=mongo_cfg["db_name"],
        )
        logger.info("TrajectoryStore đã khởi tạo ✓")
    except Exception as exc:
        app.state.trajectory_store = None
        logger.warning("TrajectoryStore init skipped: %s", exc)

    logger.info(
        "HAgent Bridge khởi chạy trên port %d — runtime=langgraph",
        bridge_cfg["port"],
    )

    yield

    if _training_watch_tasks:
        for task in list(_training_watch_tasks.values()):
            if not task.done():
                task.cancel()
        await asyncio.gather(
            *list(_training_watch_tasks.values()), return_exceptions=True
        )
        _training_watch_tasks.clear()

    await conv_store.close_db()
    logger.info("HAgent Bridge đã dừng.")


# ─── Khởi tạo ứng dụng ──────────────────────────────────


hagent_bridge = FastAPI(
    title="HAgent Bridge",
    description="Lớp trung gian giữa frontend ChatWidget và HAgent runtime (LangGraph) — Đa provider LLM, cấu hình từ YAML",
    version="2.0.0",
    lifespan=lifespan,
)

# Cấu hình CORS từ YAML
_bridge_cfg = get_bridge_config()
hagent_bridge.add_middleware(
    CORSMiddleware,
    allow_origins=_bridge_cfg.get("cors_origins", ["http://localhost:3000"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Hàm gọi LLM ───────────────────────────────────────
_RESERVED_RUNTIME_CONTEXT_KEYS = {"user_id", "user_token"}


def _validate_model_name(model_name: str | None) -> None:
    if model_name is None:
        return
    available = [
        str(model.get("name")) for model in get_llm_models() if model.get("name")
    ]
    if model_name not in available:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model {model_name!r} does not exist in the configured registry. "
                f"Valid names: {available}"
            ),
        )


def _upstream_error_detail(response: httpx.Response) -> object:
    try:
        payload = response.json()
    except ValueError:
        return f"HAgent runtime returned HTTP {response.status_code}"
    if isinstance(payload, dict) and "detail" in payload:
        return payload["detail"]
    return f"HAgent runtime returned HTTP {response.status_code}"


async def _call_agent_runtime(
    message: str,
    *,
    user_token: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    context_extra: dict | None = None,
    model_name: str | None = None,
) -> dict:
    """
    HAgent path: Bridge → toolkit /api/v1/chat/agent-run (LangGraph).

    Conversation history is owned by Bridge; toolkit only runs the agent graph.
    """
    hautoml_cfg = get_hautoml_config()
    base = hautoml_cfg["base_url"].rstrip("/")
    # Allow override for split deployments
    agent_url = os.getenv(
        "HAGENT_AGENT_RUN_URL",
        f"{base}/api/v1/chat/agent-run",
    )
    _validate_model_name(model_name)
    context = dict(context_extra or {})
    for key in _RESERVED_RUNTIME_CONTEXT_KEYS:
        context.pop(key, None)
    context["hautoml_url"] = base
    payload = {
        "message": message,
        "conversation_id": session_id or "hagent_session",
        "context": context,
    }
    if model_name is not None:
        payload["model"] = model_name
    headers = {"Content-Type": "application/json"}
    if user_token:
        headers["Authorization"] = f"Bearer {user_token}"

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(agent_url, json=payload, headers=headers)
        if 200 <= resp.status_code < 300:
            data = resp.json()
            if not isinstance(data, dict):
                raise HTTPException(status_code=502, detail="Invalid runtime response")
            data = dict(data)
            message = data.get("message", data.get("response"))
            if not isinstance(message, str):
                raise HTTPException(status_code=502, detail="Invalid runtime response")
            data["message"] = message
            return data
        status_code = resp.status_code if 400 <= resp.status_code < 500 else 502
        logger.warning("HAgent runtime returned HTTP %d", resp.status_code)
        raise HTTPException(
            status_code=status_code,
            detail=_upstream_error_detail(resp),
        )
    except HTTPException:
        raise
    except httpx.TimeoutException as exc:
        logger.warning("HAgent runtime timed out at %s", agent_url)
        raise HTTPException(status_code=504, detail="HAgent runtime timed out") from exc
    except httpx.RequestError as exc:
        logger.warning("Cannot reach HAgent runtime at %s", agent_url)
        raise HTTPException(
            status_code=502, detail="HAgent runtime unavailable"
        ) from exc
    except Exception as exc:
        logger.exception("Invalid HAgent runtime response")
        raise HTTPException(status_code=502, detail="Invalid runtime response") from exc


async def _call_hagent_gateway(
    message: str,
    user_token: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    context_extra: dict | None = None,
    model_name: str | None = None,
) -> dict:
    """Gọi HAgent runtime (LangGraph trên toolkit) — runtime duy nhất."""
    return await _call_agent_runtime(
        message,
        user_token=user_token,
        user_id=user_id,
        session_id=session_id,
        context_extra=context_extra,
        model_name=model_name,
    )


_BRIDGE_SSE_EVENTS = frozenset(
    {
        "meta",
        "route",
        "phase",
        "plan",
        "plan_event",
        "surprise",
        "token",
        "tool_call",
        "tool_result",
        "done",
        "error",
    }
)
_BRIDGE_TERMINALS = frozenset({"done", "error"})


class _UpstreamStatusError(RuntimeError):
    def __init__(self, status_code: int):
        super().__init__("Agent runtime returned an HTTP error")
        self.status_code = status_code


class _BridgePersistenceError(RuntimeError):
    pass


def _stream_runtime_url() -> str:
    base = get_hautoml_config()["base_url"].rstrip("/")
    sync_url = os.getenv(
        "HAGENT_AGENT_RUN_URL",
        f"{base}/api/v1/chat/agent-run",
    ).rstrip("/")
    return os.getenv(
        "HAGENT_AGENT_RUN_STREAM_URL",
        sync_url if sync_url.endswith("/stream") else f"{sync_url}/stream",
    )


async def _stream_agent_runtime_lines(
    *,
    message: str,
    user_token: str | None,
    user_id: str,
    session_id: str,
    context_extra: dict | None,
    model_name: str | None,
) -> AsyncIterator[str]:
    """Open the private toolkit stream and yield decoded SSE lines."""
    del user_id
    _validate_model_name(model_name)
    base = get_hautoml_config()["base_url"].rstrip("/")
    context = dict(context_extra or {})
    for key in _RESERVED_RUNTIME_CONTEXT_KEYS:
        context.pop(key, None)
    context["hautoml_url"] = base
    payload = {
        "message": message,
        "conversation_id": session_id,
        "context": context,
    }
    if model_name is not None:
        payload["model"] = model_name
    headers = {"Content-Type": "application/json"}
    if user_token:
        headers["Authorization"] = f"Bearer {user_token}"

    timeout = httpx.Timeout(300.0, read=None)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST",
            _stream_runtime_url(),
            json=payload,
            headers=headers,
        ) as response:
            if not 200 <= response.status_code < 300:
                raise _UpstreamStatusError(response.status_code)
            async for line in response.aiter_lines():
                yield line


def _decode_sse_frame(
    event_name: str | None,
    event_id: str | None,
    data_lines: list[str],
) -> tuple[str, int, dict]:
    if not event_name or event_id is None or not data_lines:
        raise ValueError("Incomplete upstream SSE frame")
    parsed_id = int(event_id)
    payload = json.loads("\n".join(data_lines))
    if not isinstance(payload, dict):
        raise ValueError("Upstream SSE data must be an object")
    return event_name, parsed_id, payload


async def _iter_upstream_sse(
    lines: AsyncIterator[str],
) -> AsyncIterator[tuple[str, int, dict]]:
    event_name = None
    event_id = None
    data_lines: list[str] = []

    async for raw_line in lines:
        if not isinstance(raw_line, str):
            raise TypeError("Upstream SSE line must be text")
        line = raw_line.rstrip("\r")
        if not line:
            if event_name is not None or event_id is not None or data_lines:
                yield _decode_sse_frame(event_name, event_id, data_lines)
            event_name = None
            event_id = None
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if not separator:
            raise ValueError("Malformed upstream SSE line")
        value = value.lstrip(" ")
        if field == "event":
            event_name = value
        elif field == "id":
            event_id = value
        elif field == "data":
            data_lines.append(value)

    if event_name is not None or event_id is not None or data_lines:
        yield _decode_sse_frame(event_name, event_id, data_lines)


def _format_bridge_sse(event_name: str, event_id: int, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event_name}\nid: {event_id}\ndata: {payload}\n\n"


def _bridge_error_code(exc: Exception) -> str:
    if isinstance(exc, _BridgePersistenceError):
        return "persistence_failed"
    if isinstance(exc, _UpstreamStatusError):
        return "upstream_http_error"
    if isinstance(exc, httpx.TimeoutException):
        return "upstream_timeout"
    if isinstance(exc, httpx.RequestError):
        return "upstream_unavailable"
    if isinstance(exc, (TypeError, ValueError, json.JSONDecodeError)):
        return "invalid_upstream_stream"
    return "bridge_stream_failed"


def _bridge_error_frame(event_id: int, code: str) -> str:
    return _format_bridge_sse(
        "error",
        event_id,
        {
            "type": "error",
            "error": {
                "code": code,
                "message": "Chat stream failed",
            },
        },
    )


async def _bridge_event_stream(
    *,
    message: str,
    user: TokenPayload,
    conversation_id: str,
    context_extra: dict,
    model_name: str | None,
    world_state_store: WorldStateStore,
    message_id: str,
) -> AsyncIterator[str]:
    upstream = _stream_agent_runtime_lines(
        message=message,
        user_token=user.raw_token,
        user_id=user.user_id,
        session_id=conversation_id,
        context_extra=context_extra,
        model_name=model_name,
    )
    last_event_id = 0
    terminal_sent = False

    try:
        async for event_name, event_id, data in _iter_upstream_sse(upstream):
            if event_name not in _BRIDGE_SSE_EVENTS:
                raise ValueError("Unsupported upstream SSE event")
            if data.get("type") != event_name:
                raise ValueError("Upstream SSE event/data mismatch")
            if event_id <= last_event_id:
                raise ValueError("Upstream SSE IDs must be strictly increasing")
            last_event_id = event_id

            if event_name == "done":
                upstream_response = data.get("response")
                if not isinstance(upstream_response, dict):
                    raise ValueError("Upstream done response must be an object")
                result = dict(upstream_response)
                if not isinstance(result.get("message"), str):
                    raise ValueError("Upstream done response has no message")
                normalized = _to_chat_response(result, conversation_id).model_dump()

                try:
                    await _apply_tool_outputs_to_world_state(
                        world_state_store,
                        user.user_id,
                        result,
                    )
                    await conv_store.add_assistant_message_once(
                        conversation_id=conversation_id,
                        user_id=user.user_id,
                        content=normalized["message"],
                        message_id=message_id,
                        provider=normalized.get("provider", ""),
                        model=normalized.get("model", ""),
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    raise _BridgePersistenceError from exc

                tracked_job_id = _extract_training_job_id(normalized["message"])
                if tracked_job_id:
                    try:
                        _schedule_training_result_notification(
                            conversation_id=conversation_id,
                            user_id=user.user_id,
                            user_token=user.raw_token,
                            job_id=tracked_job_id,
                        )
                    except Exception as exc:
                        logger.error(
                            "Stream notification scheduling failed: %s",
                            type(exc).__name__,
                        )

                data = {"type": "done", "response": normalized}
                frame = _format_bridge_sse("done", event_id, data)
                terminal_sent = True
                yield frame
                break

            if event_name == "error":
                frame = _format_bridge_sse("error", event_id, data)
                terminal_sent = True
                yield frame
                break

            yield _format_bridge_sse(event_name, event_id, data)

        if not terminal_sent:
            raise ValueError("Upstream stream ended without a terminal event")
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("Bridge SSE failed: %s", type(exc).__name__)
        if not terminal_sent:
            terminal_sent = True
            yield _bridge_error_frame(last_event_id + 1, _bridge_error_code(exc))
    finally:
        close = getattr(upstream, "aclose", None)
        if close is not None:
            try:
                await close()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("Upstream SSE close failed: %s", type(exc).__name__)

def _extract_training_job_id(message: str) -> str | None:
    """Chỉ lấy job_id khi nội dung cho thấy vừa submit training."""
    if not message:
        return None

    tagged_match = JOB_ID_LINE_RE.search(message)
    if tagged_match:
        return tagged_match.group(1)

    if not TRAINING_STARTED_RE.search(message):
        return None

    generic_match = GENERIC_UUID_RE.search(message)
    return generic_match.group(1) if generic_match else None


def _format_score(score) -> str:
    if score is None:
        return "N/A"

    if isinstance(score, (int, float)):
        if 0 <= score <= 1:
            return f"{score * 100:.2f}%"
        return f"{score:.6g}"

    return str(score)


def _format_training_success_message(job_id: str, job: dict) -> str:
    data = job.get("data") if isinstance(job.get("data"), dict) else {}
    data_name = data.get("name")
    best_model = job.get("best_model")
    best_score = _format_score(job.get("best_score"))
    time_limit_reached = job.get("time_limit_reached")

    lines = [
        "✅ Job training đã hoàn tất.",
        f"- Job ID: {job_id}",
    ]

    if data_name:
        lines.append(f"- Dataset: {data_name}")
    if best_model:
        lines.append(f"- Best model: {best_model}")

    lines.append(f"- Best score: {best_score}")

    if time_limit_reached is not None:
        lines.append(
            f"- Chạm giới hạn thời gian: {'Có' if bool(time_limit_reached) else 'Không'}"
        )

    lines.append("Bạn có thể mở Training History để xem đầy đủ chi tiết.")
    return "\n".join(lines)


def _format_training_failed_message(job_id: str, job: dict) -> str:
    error_detail = job.get("infor") or job.get("error") or "Không rõ nguyên nhân"
    return "\n".join(
        [
            "❌ Job training kết thúc với trạng thái lỗi.",
            f"- Job ID: {job_id}",
            f"- Chi tiết: {error_detail}",
            "Bạn có thể gửi lại lệnh train hoặc kiểm tra cấu hình dataset/features.",
        ]
    )


async def _poll_training_result_and_notify(
    conversation_id: str,
    user_id: str,
    user_token: str,
    job_id: str,
):
    """Theo dõi trạng thái job và ghi kết quả vào hội thoại khi kết thúc."""
    task_key = f"{conversation_id}:{job_id}"
    hautoml_base = get_hautoml_config()["base_url"].rstrip("/")
    deadline = time.monotonic() + TRAINING_POLL_TIMEOUT_SECONDS

    try:
        while time.monotonic() < deadline:
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.post(
                        f"{hautoml_base}/get-job-info",
                        params={"id": job_id},
                        headers={"Authorization": f"Bearer {user_token}"},
                    )

                if resp.status_code == 200:
                    job = resp.json()
                    raw_status = job.get("status")

                    try:
                        job_status = int(raw_status)
                    except Exception:
                        job_status = raw_status

                    if job_status == 1:
                        await conv_store.add_message(
                            conversation_id=conversation_id,
                            user_id=user_id,
                            role="assistant",
                            content=_format_training_success_message(job_id, job),
                            provider="hagent",
                            model="hagent-agent",
                        )
                        return

                    if job_status == -1:
                        await conv_store.add_message(
                            conversation_id=conversation_id,
                            user_id=user_id,
                            role="assistant",
                            content=_format_training_failed_message(job_id, job),
                            provider="hagent",
                            model="hagent-agent",
                        )
                        return

                elif resp.status_code in {401, 403}:
                    logger.warning(
                        "Không thể tiếp tục theo dõi job %s do lỗi xác thực (%d)",
                        job_id,
                        resp.status_code,
                    )
                    await conv_store.add_message(
                        conversation_id=conversation_id,
                        user_id=user_id,
                        role="assistant",
                        content=(
                            f"⚠️ Đã submit job {job_id}, nhưng không thể theo dõi kết quả tự động "
                            "do token hết hạn hoặc không hợp lệ. Bạn có thể kiểm tra ở Training History."
                        ),
                        provider="hagent",
                        model="hagent-agent",
                    )
                    return

            except Exception as e:
                logger.warning("Lỗi khi theo dõi job %s: %s", job_id, e)

            await asyncio.sleep(TRAINING_POLL_INTERVAL_SECONDS)

        await conv_store.add_message(
            conversation_id=conversation_id,
            user_id=user_id,
            role="assistant",
            content=(
                f"ℹ️ Job {job_id} vẫn đang xử lý sau {TRAINING_POLL_TIMEOUT_SECONDS} giây. "
                "Mình sẽ dừng theo dõi tự động, bạn có thể mở Training History để xem tiến trình mới nhất."
            ),
            provider="hagent",
            model="hagent-agent",
        )
    finally:
        _training_watch_tasks.pop(task_key, None)


def _schedule_training_result_notification(
    conversation_id: str,
    user_id: str,
    user_token: str,
    job_id: str,
):
    task_key = f"{conversation_id}:{job_id}"
    existing = _training_watch_tasks.get(task_key)

    if existing and not existing.done():
        return

    _training_watch_tasks[task_key] = asyncio.create_task(
        _poll_training_result_and_notify(
            conversation_id=conversation_id,
            user_id=user_id,
            user_token=user_token,
            job_id=job_id,
        )
    )
    logger.info("Đã schedule theo dõi kết quả training cho job %s", job_id)


def _world_state_context(world_state: WorldState | None) -> dict:
    return world_state.to_dict() if world_state else {}


_PUBLIC_CHAT_CONTEXT_FIELDS = (
    "dataset_id",
    "dataset_name",
    "target_column",
    "problem_type",
    "metric",
    "models",
)


def _normalize_history(messages: list | None) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in list(messages or [])[-20:]:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
        else:
            role = getattr(item, "role", None)
            content = getattr(item, "content", None)
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def _runtime_context(
    client_context: dict | None,
    world_state: WorldState | None,
    history: list | None = None,
) -> dict:
    context = {
        key: client_context[key]
        for key in _PUBLIC_CHAT_CONTEXT_FIELDS
        if isinstance(client_context, dict)
        and key in client_context
        and client_context[key] is not None
    }
    context["world_state"] = _world_state_context(world_state)
    context["history"] = _normalize_history(history)
    return context


async def _apply_tool_outputs_to_world_state(
    world_state_store: WorldStateStore,
    user_id: str,
    result: dict,
) -> None:
    tool_outputs = result.get("tool_outputs")
    if not isinstance(tool_outputs, list):
        tool_outputs = []

    current_state = await world_state_store.get(user_id)
    if not current_state:
        await world_state_store.ensure(user_id)
        current_state = await world_state_store.get(user_id)
    if not current_state:
        return

    for tool_call in tool_outputs:
        if not isinstance(tool_call, dict):
            continue

        tool_name = tool_call.get("tool_name")
        payload = tool_call.get("payload")
        if not tool_name or not isinstance(payload, dict):
            continue

        patch = world_updater.apply_tool_output(current_state, tool_name, payload)
        if patch:
            updated_state = await world_state_store.upsert(user_id, patch)
            if updated_state:
                current_state = updated_state

    # Persist plan / surprise / agent world_model fields from agent result
    meta_patch: dict = {}
    if isinstance(result.get("surprise"), dict):
        meta_patch["last_surprise"] = result["surprise"]
        pe = world_updater.apply_plan_event(
            current_state, "surprise_recorded", {"surprise": result["surprise"]}
        )
        meta_patch.update(pe)
    selected = result.get("selected_plan")
    if isinstance(selected, dict) and selected.get("plan_id"):
        pe = world_updater.apply_plan_event(
            current_state,
            "plan_selected",
            {**selected, "plan_id": selected["plan_id"]},
        )
        meta_patch.update(pe)
    wm = result.get("world_model")
    if isinstance(wm, dict):
        for k in (
            "datasets",
            "jobs",
            "phase",
            "active_dataset_id",
            "active_job_id",
            "active_plan_id",
            "active_goal",
            "cost_metrics",
        ):
            if k in wm and wm[k] is not None:
                meta_patch[k] = wm[k]
    if meta_patch:
        await world_state_store.upsert(user_id, meta_patch)


def _to_chat_response(result: dict, conversation_id: str) -> ChatResponse:
    return ChatResponse(
        message=result["message"],
        conversation_id=conversation_id,
        sources=result.get("sources", []),
        suggestions=result.get("suggestions", []),
        provider=result.get("provider", "hagent"),
        model=result.get("model", ""),
        route=result.get("route", "direct"),
        tool_outputs=result.get("tool_outputs", []),
        plan_status=result.get("plan_status"),
        selected_plan=result.get("selected_plan"),
        planning=result.get("planning"),
        surprise=result.get("surprise"),
        cost_metrics=result.get("cost_metrics"),
        execution_events=result.get("execution_events") or [],
        execution_log=result.get("execution_log") or [],
        revision_count=result.get("revision_count") or 0,
        world_model=result.get("world_model"),
        campaign=result.get("campaign"),
        campaign_status=result.get("campaign_status"),
        hierarchy=result.get("hierarchy"),
        hierarchy_status=result.get("hierarchy_status"),
        evaluation=result.get("evaluation"),
    )


# ─── Endpoints ───────────────────────────────────────────


@hagent_bridge.post("/api/v1/chat/", response_model=ChatResponse)
async def chat(
    request: Request,
    req: ChatRequest,
    user: TokenPayload = Depends(get_current_user),
):
    _validate_model_name(req.model)
    conversation_id = req.conversation_id or uuid.uuid4().hex
    world_state_store: WorldStateStore = request.app.state.world_state_store

    # Đảm bảo và lấy world state
    await world_state_store.ensure(user.user_id)
    world_state_snapshot = await world_state_store.get(user.user_id)

    # Lấy lịch sử đã persist trước turn hiện tại, đã scope theo owner.
    history = await conv_store.get_message_history(
        conversation_id, user.user_id, limit=20
    )

    # Lưu tin nhắn người dùng sau khi snapshot lịch sử được chụp.
    await conv_store.add_message(
        conversation_id=conversation_id,
        user_id=user.user_id,
        role="user",
        content=req.message,
    )

    # Gọi HAgent runtime với world_state trong context
    result = await _call_hagent_gateway(
        message=req.message,
        user_token=user.raw_token,
        user_id=user.user_id,
        session_id=conversation_id,
        context_extra=_runtime_context(req.context, world_state_snapshot, history),
        model_name=req.model,
    )

    await _apply_tool_outputs_to_world_state(world_state_store, user.user_id, result)

    # Lưu phản hồi trợ lý
    await conv_store.add_message(
        conversation_id=conversation_id,
        user_id=user.user_id,
        role="assistant",
        content=result["message"],
        provider=result.get("provider", ""),
        model=result.get("model", ""),
    )

    tracked_job_id = _extract_training_job_id(result["message"])
    if tracked_job_id:
        _schedule_training_result_notification(
            conversation_id=conversation_id,
            user_id=user.user_id,
            user_token=user.raw_token,
            job_id=tracked_job_id,
        )

    return _to_chat_response(result, conversation_id)


@hagent_bridge.post("/api/v1/chat/stream")
async def chat_stream(
    request: Request,
    req: ChatRequest,
    user: TokenPayload = Depends(get_current_user),
):
    """Public owner-scoped SSE chat; Bridge owns history and persistence."""
    _validate_model_name(req.model)
    conversation_id = req.conversation_id or uuid.uuid4().hex
    world_state_store: WorldStateStore = request.app.state.world_state_store

    await world_state_store.ensure(user.user_id)
    world_state_snapshot = await world_state_store.get(user.user_id)
    history = await conv_store.get_message_history(
        conversation_id,
        user.user_id,
        limit=20,
    )
    await conv_store.add_message(
        conversation_id=conversation_id,
        user_id=user.user_id,
        role="user",
        content=req.message,
    )

    context_extra = _runtime_context(req.context, world_state_snapshot, history)
    message_id = f"stream:{uuid.uuid4().hex}"
    return StreamingResponse(
        _bridge_event_stream(
            message=req.message,
            user=user,
            conversation_id=conversation_id,
            context_extra=context_extra,
            model_name=req.model,
            world_state_store=world_state_store,
            message_id=message_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Conversation-Id": conversation_id,
        },
    )

@hagent_bridge.post("/api/v1/chat/upload", response_model=ChatResponse)
async def chat_with_file(
    request: Request,
    message: str = Form(...),
    file: UploadFile = File(...),
    conversation_id: str | None = Form(None),
    model: str | None = Form(None),
    user: TokenPayload = Depends(get_current_user),
):
    """Chat kèm upload file — chuyển tiếp file tới HAutoML."""
    _validate_model_name(model)
    hautoml_cfg = get_hautoml_config()
    conv_id = conversation_id or uuid.uuid4().hex
    world_state_store: WorldStateStore = request.app.state.world_state_store

    # Đảm bảo và lấy world state
    await world_state_store.ensure(user.user_id)
    world_state_snapshot = await world_state_store.get(user.user_id)

    # Chuyển tiếp file tới HAutoML để upload data
    file_info = ""
    try:
        file_content = await file.read()

        # Xác định data_type từ đuôi file
        filename = file.filename or "uploaded_data.csv"
        data_type = filename.split(".")[-1].lower() if "." in filename else "csv"

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{hautoml_cfg['base_url']}/upload-dataset?user_id={user.user_id}",
                data={
                    "data_name": filename,
                    "data_type": data_type,
                },
                files={"file_data": (filename, file_content, file.content_type)},
                headers={"Authorization": f"Bearer {user.raw_token}"},
            )
            if resp.status_code == 200:
                file_info = f"\n[File đã upload vào hệ thống dataset: {filename} — {len(file_content)} bytes]"
            else:
                file_info = (
                    f"\n[Upload file thất bại: {resp.status_code} - {resp.text}]"
                )
    except Exception as e:
        file_info = f"\n[Lỗi upload file: {e}]"

    history = await conv_store.get_message_history(conv_id, user.user_id, limit=20)

    full_message = f"{message}{file_info}"
    await conv_store.add_message(conv_id, user.user_id, "user", full_message)

    result = await _call_hagent_gateway(
        message=full_message,
        user_token=user.raw_token,
        user_id=user.user_id,
        session_id=conv_id,
        context_extra=_runtime_context(None, world_state_snapshot, history),
        model_name=model,
    )

    await _apply_tool_outputs_to_world_state(world_state_store, user.user_id, result)

    await conv_store.add_message(
        conv_id,
        user.user_id,
        "assistant",
        result["message"],
        result.get("provider", ""),
        result.get("model", ""),
    )

    tracked_job_id = _extract_training_job_id(result["message"])
    if tracked_job_id:
        _schedule_training_result_notification(
            conversation_id=conv_id,
            user_id=user.user_id,
            user_token=user.raw_token,
            job_id=tracked_job_id,
        )

    return _to_chat_response(result, conv_id)


@hagent_bridge.get("/api/v1/chat/health", response_model=HealthResponse)
async def health_check():
    """Kiểm tra HAgent runtime (toolkit) + HAutoML."""
    hautoml_cfg = get_hautoml_config()
    base = hautoml_cfg["base_url"].rstrip("/")
    # Ưu tiên URL agent/toolkit tường minh, fallback HAutoML base
    runtime_url = os.getenv("HAGENT_URL") or base

    # Kiểm tra agent runtime
    hagent_ok = False
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            # Toolkit home hoặc chat health
            resp = await client.get(f"{base}/home")
            if resp.status_code != 200:
                resp = await client.get(f"{base}/api/v1/chat/health")
            hagent_ok = resp.status_code == 200
    except Exception:
        pass

    # Kiểm tra HAutoML Backend
    hautoml_ok = False
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{hautoml_cfg['base_url']}/home")
            hautoml_ok = resp.status_code == 200
    except Exception:
        pass

    return HealthResponse(
        hagent_url=runtime_url,
        connected=hagent_ok,
        hautoml_connected=hautoml_ok,
        mode="hagent",
        active_provider="hagent",
        active_model="hagent-agent",
        available_providers=["hagent"],
    )


@hagent_bridge.get("/api/v1/chat/suggestions", response_model=SuggestionsResponse)
async def get_suggestions():
    """Trả về gợi ý chat ban đầu cho widget."""
    return SuggestionsResponse(
        suggestions=[
            "📊 Hiển thị danh sách dataset của tôi",
            "🚀 Huấn luyện model phân loại mới",
            "📈 Có những thuật toán ML nào khả dụng?",
            "🔍 Kiểm tra trạng thái các job training",
            "💡 Giúp tôi chọn model phù hợp",
            "⚙️ Chuyển sang provider AI khác",
        ]
    )


@hagent_bridge.delete("/api/v1/chat/conversation/{conversation_id}")
async def clear_conversation(
    conversation_id: str,
    user: TokenPayload = Depends(get_current_user),
):
    """Xóa cuộc hội thoại và toàn bộ lịch sử."""
    deleted = await conv_store.delete_conversation(conversation_id, user.user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Không tìm thấy cuộc hội thoại")
    return {"status": "deleted", "conversation_id": conversation_id}


@hagent_bridge.get("/api/v1/chat/providers", response_model=ProvidersResponse)
async def list_providers(
    user: TokenPayload | None = Depends(get_optional_user),
):
    """List configured model aliases grouped by their toolkit provider."""
    del user
    registry = [model for model in get_llm_models() if model.get("name")]
    grouped: dict[str, list[str]] = {}
    for model in registry:
        provider_id = str(model.get("provider") or "unknown")
        grouped.setdefault(provider_id, []).append(str(model["name"]))

    default_model = str(get_llm_config().get("default_model") or "")
    if not default_model and registry:
        default_model = str(registry[0]["name"])
    default_entry = next(
        (model for model in registry if str(model["name"]) == default_model),
        None,
    )
    default_provider = (
        str(default_entry.get("provider") or "unknown") if default_entry else ""
    )
    providers = [
        ProviderInfo(
            name=provider_id.replace("_", " ").title(),
            provider_id=provider_id,
            models=model_names,
            available=True,
            description="Configured in the toolkit model registry",
        )
        for provider_id, model_names in grouped.items()
    ]
    return ProvidersResponse(
        default_provider=default_provider,
        default_model=default_model,
        providers=providers,
    )


@hagent_bridge.get("/api/v1/chat/conversations")
async def list_user_conversations(
    user: TokenPayload = Depends(get_current_user),
):
    """Liệt kê các cuộc hội thoại gần nhất của người dùng."""
    conversations = await conv_store.list_conversations(user.user_id)
    return {"conversations": conversations}


@hagent_bridge.get("/api/v1/world-state/{user_id}")
async def get_world_state(
    user_id: str,
    request: Request,
    user: TokenPayload = Depends(get_current_user),
):
    """
    Lấy world state (trạng thái thế giới) của một người dùng cụ thể.
    Chỉ người dùng đó mới có quyền truy cập world state của chính mình.
    """
    # So sánh user_id từ token JWT với user_id từ URL
    if user.user_id != user_id:
        raise HTTPException(
            status_code=403,
            detail="Không có quyền truy cập world state của người dùng khác",
        )

    # Lấy world_state_store từ request.app.state
    world_state_store: WorldStateStore = request.app.state.world_state_store

    # Gọi store.get(user_id) để lấy bản ghi
    world_state = await world_state_store.get(user_id)

    # Nếu không tìm thấy bản ghi, raise 404
    if not world_state:
        raise HTTPException(
            status_code=404,
            detail="Không tìm thấy world state cho người dùng này",
        )

    # Trả về đối tượng world state dạng dict để serialization đáng tin cậy
    return world_state.to_dict()


@hagent_bridge.get("/api/v1/chat/conversation/{conversation_id}")
async def get_conversation_history(
    conversation_id: str,
    user: TokenPayload = Depends(get_current_user),
):
    """Truy xuất toàn bộ lịch sử tin nhắn của một cuộc hội thoại cụ thể."""
    conversation = await conv_store.get_conversation(conversation_id, user.user_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Không tìm thấy cuộc hội thoại")

    return {
        "conversation_id": conversation.conversation_id,
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
        "provider": conversation.provider,
        "model": conversation.model,
        "messages": [
            {
                "id": str(uuid.uuid4()),
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                "provider": msg.provider,
                "model": msg.model,
            }
            for msg in conversation.messages
        ],
    }


# ─── Điểm chạy chính ────────────────────────────────────


def main():
    """Chạy bridge server với uvicorn."""
    import uvicorn

    cfg = get_bridge_config()
    uvicorn.run(
        "hagent.bridge.app:hagent_bridge",
        host=cfg["host"],
        port=cfg["port"],
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
