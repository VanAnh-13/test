"""
Các hàm hỗ trợ Bridge chứa logic nghiệp vụ dùng chung cho những module route.

Module này chứa các pure-function helpers và async utilities. Không chứa
route handlers, không import FastAPI APIRouter trực tiếp.

Lý do thiết kế:
  Tách ra từ bridge/app.py để:
  1. Các routes/*.py có thể import mà không phụ thuộc vòng lẫn nhau.
  2. Dễ kiểm thử độc lập với httpx.AsyncClient giả lập.
  3. Giảm độ phức tạp nhận thức của app.py chính.
"""

# Các khối bắt ngoại lệ rộng tại đây bảo vệ probe, stream và tác vụ nền ở boundary.
# ruff: noqa: BLE001, TRY004

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sys
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
import structlog
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from hagent.bridge import conversation as conv_store
from hagent.bridge.auth import TokenPayload
from hagent.bridge.config import get_hautoml_config, get_llm_models
from hagent.bridge.models import ChatResponse, ProviderInfo, ProvidersResponse
from hagent.world import updater as world_updater
from hagent.world.schema import WorldState
from hagent.world.state_store import WorldStateStore

logger = structlog.get_logger("hagent.bridge")

# ── Biểu thức chính quy và hằng số ──────────────────────────────────────────

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

_RESERVED_RUNTIME_CONTEXT_KEYS = {"user_id", "user_token"}
_PUBLIC_CHAT_CONTEXT_FIELDS = (
    "dataset_id",
    "dataset_name",
    "target_column",
    "problem_type",
    "metric",
    "models",
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

_READINESS_TIMEOUT_ENV = "HAGENT_READINESS_TIMEOUT_SECONDS"
_DEFAULT_READINESS_TIMEOUT_SECONDS = 5.0
_MAX_READINESS_TIMEOUT_SECONDS = 30.0


# ── Các kiểu lỗi nội bộ ─────────────────────────────────────────────────────


class _UpstreamStatusError(RuntimeError):
    def __init__(self, status_code: int):
        super().__init__("Agent runtime returned an HTTP error")
        self.status_code = status_code


class _BridgePersistenceError(RuntimeError):
    pass


# ── Kiểm tra model và LLM ───────────────────────────────────────────────────


def validate_model_name(model_name: str | None) -> None:
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


def upstream_error_detail(response: httpx.Response) -> object:
    try:
        payload = response.json()
    except ValueError:
        return f"HAgent runtime returned HTTP {response.status_code}"
    if isinstance(payload, dict) and "detail" in payload:
        return payload["detail"]
    return f"HAgent runtime returned HTTP {response.status_code}"


# ── Lời gọi agent runtime ────────────────────────────────────────────────────


async def call_agent_runtime(
    message: str,
    *,
    user_token: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    context_extra: dict | None = None,
    model_name: str | None = None,
) -> dict:
    """Gọi từ Bridge đến toolkit /api/v1/chat/agent-run của LangGraph.

    Bridge sở hữu lịch sử hội thoại; toolkit chỉ chạy agent graph.
    """
    hautoml_cfg = get_hautoml_config()
    base = hautoml_cfg["base_url"].rstrip("/")
    agent_url = os.getenv(
        "HAGENT_AGENT_RUN_URL",
        f"{base}/api/v1/chat/agent-run",
    )
    validate_model_name(model_name)
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
            msg = data.get("message", data.get("response"))
            if not isinstance(msg, str):
                raise HTTPException(status_code=502, detail="Invalid runtime response")
            data["message"] = msg
            to_chat_response(data, session_id or "hagent_session")
            return data
        status_code = resp.status_code if 400 <= resp.status_code < 500 else 502
        logger.warning("HAgent runtime returned HTTP %d", resp.status_code)
        raise HTTPException(
            status_code=status_code,
            detail=upstream_error_detail(resp),
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


async def call_hagent_gateway(
    message: str,
    user_token: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    context_extra: dict | None = None,
    model_name: str | None = None,
) -> dict:
    """Gọi HAgent runtime (LangGraph trên toolkit) — runtime duy nhất."""
    return await call_agent_runtime(
        message,
        user_token=user_token,
        user_id=user_id,
        session_id=session_id,
        context_extra=context_extra,
        model_name=model_name,
    )


# ── Hàm hỗ trợ context ───────────────────────────────────────────────────────


def world_state_context(world_state: WorldState | None) -> dict:
    return world_state.to_dict() if world_state else {}


def normalize_history(messages: list | None) -> list[dict[str, str]]:
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


def runtime_context(
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
    context["world_state"] = world_state_context(world_state)
    context["history"] = normalize_history(history)
    return context


# ── Cập nhật world state ─────────────────────────────────────────────────────


async def apply_tool_outputs_to_world_state(
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


# ── Hàm tạo ChatResponse ─────────────────────────────────────────────────────


def to_chat_response(result: dict, conversation_id: str) -> ChatResponse:
    return ChatResponse(
        message=result["message"],
        conversation_id=conversation_id,
        sources=result.get("sources", []),
        suggestions=result.get("suggestions", []),
        provider=result.get("provider", "hagent"),
        model=result.get("model", ""),
        route="direct" if result.get("route") is None else result.get("route"),
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


# ── Hàm hỗ trợ thông báo huấn luyện ──────────────────────────────────────────


def extract_training_job_id(message: str) -> str | None:
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


def format_score(score) -> str:
    if score is None:
        return "N/A"
    if isinstance(score, (int, float)):
        if 0 <= score <= 1:
            return f"{score * 100:.2f}%"
        return f"{score:.6g}"
    return str(score)


def format_training_success_message(job_id: str, job: dict) -> str:
    data = job.get("data") if isinstance(job.get("data"), dict) else {}
    data_name = data.get("name")
    best_model = job.get("best_model")
    best_score = format_score(job.get("best_score"))
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


def format_training_failed_message(job_id: str, job: dict) -> str:
    error_detail = job.get("infor") or job.get("error") or "Không rõ nguyên nhân"
    return "\n".join(
        [
            "❌ Job training kết thúc với trạng thái lỗi.",
            f"- Job ID: {job_id}",
            f"- Chi tiết: {error_detail}",
            "Bạn có thể gửi lại lệnh train hoặc kiểm tra cấu hình dataset/features.",
        ]
    )


async def poll_training_result_and_notify(
    conversation_id: str,
    user_id: str,
    user_token: str,
    job_id: str,
) -> None:
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
                            content=format_training_success_message(job_id, job),
                            provider="hagent",
                            model="hagent-agent",
                        )
                        return
                    if job_status == -1:
                        await conv_store.add_message(
                            conversation_id=conversation_id,
                            user_id=user_id,
                            role="assistant",
                            content=format_training_failed_message(job_id, job),
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


def schedule_training_result_notification(
    conversation_id: str,
    user_id: str,
    user_token: str,
    job_id: str,
) -> None:
    task_key = f"{conversation_id}:{job_id}"
    existing = _training_watch_tasks.get(task_key)
    if existing and not existing.done():
        return
    _training_watch_tasks[task_key] = asyncio.create_task(
        poll_training_result_and_notify(
            conversation_id=conversation_id,
            user_id=user_id,
            user_token=user_token,
            job_id=job_id,
        )
    )
    logger.info("Đã schedule theo dõi kết quả training cho job %s", job_id)


# ── Hàm hỗ trợ SSE streaming ─────────────────────────────────────────────────


def stream_runtime_url() -> str:
    base = get_hautoml_config()["base_url"].rstrip("/")
    sync_url = os.getenv(
        "HAGENT_AGENT_RUN_URL",
        f"{base}/api/v1/chat/agent-run",
    ).rstrip("/")
    return os.getenv(
        "HAGENT_AGENT_RUN_STREAM_URL",
        sync_url if sync_url.endswith("/stream") else f"{sync_url}/stream",
    )


async def stream_agent_runtime_lines(
    *,
    message: str,
    user_token: str | None,
    user_id: str,
    session_id: str,
    context_extra: dict | None,
    model_name: str | None,
) -> AsyncIterator[str]:
    """Mở stream nội bộ của toolkit và trả dần các dòng SSE đã giải mã."""
    del user_id
    validate_model_name(model_name)
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
    async with (
        httpx.AsyncClient(timeout=timeout) as client,
        client.stream(
            "POST",
            stream_runtime_url(),
            json=payload,
            headers=headers,
        ) as response,
    ):
        if not 200 <= response.status_code < 300:
            raise _UpstreamStatusError(response.status_code)
        async for line in response.aiter_lines():
            yield line


def decode_sse_frame(
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


async def iter_upstream_sse(
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
                yield decode_sse_frame(event_name, event_id, data_lines)
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
        yield decode_sse_frame(event_name, event_id, data_lines)


def format_bridge_sse(event_name: str, event_id: int, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event_name}\nid: {event_id}\ndata: {payload}\n\n"


def bridge_error_code(exc: Exception) -> str:
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


def bridge_error_frame(event_id: int, code: str) -> str:
    return format_bridge_sse(
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


async def bridge_event_stream(
    *,
    message: str,
    user: TokenPayload,
    conversation_id: str,
    context_extra: dict,
    model_name: str | None,
    world_state_store: WorldStateStore,
    message_id: str,
    stream_lines_fn=None,
    apply_tool_outputs_fn=None,
) -> AsyncIterator[str]:
    _stream_fn = (
        stream_lines_fn if stream_lines_fn is not None else stream_agent_runtime_lines
    )
    _apply_fn = (
        apply_tool_outputs_fn
        if apply_tool_outputs_fn is not None
        else apply_tool_outputs_to_world_state
    )
    upstream = _stream_fn(
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
        async for event_name, event_id, data in iter_upstream_sse(upstream):
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
                normalized = to_chat_response(result, conversation_id).model_dump()
                try:
                    await _apply_fn(world_state_store, user.user_id, result)
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
                tracked_job_id = extract_training_job_id(normalized["message"])
                if tracked_job_id:
                    try:
                        schedule_training_result_notification(
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
                frame = format_bridge_sse("done", event_id, data)
                terminal_sent = True
                yield frame
                break
            if event_name == "error":
                frame = format_bridge_sse("error", event_id, data)
                terminal_sent = True
                yield frame
                break
            yield format_bridge_sse(event_name, event_id, data)
        if not terminal_sent:
            raise ValueError("Upstream stream ended without a terminal event")
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("Bridge SSE failed: %s", type(exc).__name__)
        if not terminal_sent:
            terminal_sent = True
            yield bridge_error_frame(last_event_id + 1, bridge_error_code(exc))
    finally:
        close = getattr(upstream, "aclose", None)
        if close is not None:
            try:
                await close()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("Upstream SSE close failed: %s", type(exc).__name__)


# ── Hàm hỗ trợ kiểm tra readiness ────────────────────────────────────────────


def readiness_timeout_seconds() -> float:
    """Đọc timeout probe và từ chối cấu hình không hữu hạn hoặc quá rộng."""
    raw_value = os.getenv(
        _READINESS_TIMEOUT_ENV,
        str(_DEFAULT_READINESS_TIMEOUT_SECONDS),
    )
    try:
        timeout = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Cấu hình readiness timeout không hợp lệ") from exc
    if (
        not math.isfinite(timeout)
        or timeout <= 0
        or timeout > _MAX_READINESS_TIMEOUT_SECONDS
    ):
        raise ValueError("Cấu hình readiness timeout không hợp lệ")
    return timeout


def toolkit_url(path: str, run_api_url_fn) -> str:
    """Tạo URL probe từ origin của durable run API đã cấu hình."""
    try:
        configured = httpx.URL(run_api_url_fn())
    except httpx.InvalidURL:
        raise ValueError("Cấu hình toolkit URL không hợp lệ") from None
    if (
        configured.scheme not in {"http", "https"}
        or not configured.host
        or configured.userinfo
    ):
        raise ValueError("Cấu hình toolkit URL không hợp lệ")
    return str(configured.copy_with(path=path, query=None, fragment=None))


async def probe_http_status(url: str, timeout: float) -> bool:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
        return response.status_code == 200
    except Exception:
        return False


async def probe_mongo_readiness() -> bool:
    client = conv_store.get_db_client()
    if client is None:
        return False
    result = await client.admin.command("ping")
    return isinstance(result, dict) and result.get("ok") == 1


async def probe_toolkit_readiness(timeout: float, toolkit_url_fn) -> bool:
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(toolkit_url_fn("/api/v1/chat/health"))
    if response.status_code != 200:
        return False
    try:
        payload = response.json()
    except (TypeError, ValueError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("hautoml_connected") is True
        and isinstance(payload.get("available_models"), list)
        and bool(payload["available_models"])
    )


async def bounded_readiness_probe(probe, timeout: float) -> bool:
    """Giới hạn toàn bộ coroutine để fake hoặc transport lỗi cũng không treo request."""
    try:
        return bool(await asyncio.wait_for(probe, timeout=timeout))
    except Exception:
        return False


# ── Lớp tương thích ngược cho bridge.app ─────────────────────────────────────


def _bridge_app_module():
    """Lấy module facade để các monkeypatch lịch sử tiếp tục có hiệu lực."""
    return sys.modules["hagent.bridge.app"]


@asynccontextmanager
async def bridge_lifespan(app):
    """Quản lý tài nguyên của Bridge mà không làm phình module khởi tạo app."""
    bridge_app = _bridge_app_module()
    mongo_cfg = bridge_app.get_mongodb_config()
    bridge_cfg = bridge_app.get_bridge_config()
    world_state_cfg = bridge_app.get_world_state_config()

    bridge_app.logger.info("Đang kết nối MongoDB ...")
    await bridge_app.conv_store.init_db()
    bridge_app.logger.info("Kết nối MongoDB thành công ✓")
    app.state.world_state_store = WorldStateStore(
        client=bridge_app.conv_store.get_db_client(),
        db_name=mongo_cfg["db_name"],
        collection_name=world_state_cfg["collection_name"],
        ttl_seconds=world_state_cfg["ttl_seconds"],
    )
    await app.state.world_state_store.ensure_indexes()

    try:
        from hagent.world.trajectory_store import create_trajectory_store

        app.state.trajectory_store = create_trajectory_store(
            bridge_app.conv_store.get_db_client(),
            db_name=mongo_cfg["db_name"],
        )
    except Exception as exc:
        app.state.trajectory_store = None
        bridge_app.logger.warning("Bỏ qua khởi tạo TrajectoryStore: %s", exc)

    bridge_app.logger.info(
        "HAgent Bridge khởi chạy trên port %d — runtime=langgraph",
        bridge_cfg["port"],
    )
    yield

    for task in list(_training_watch_tasks.values()):
        if not task.done():
            task.cancel()
    if _training_watch_tasks:
        await asyncio.gather(
            *list(_training_watch_tasks.values()), return_exceptions=True
        )
        _training_watch_tasks.clear()
    await bridge_app.conv_store.close_db()
    bridge_app.logger.info("HAgent Bridge đã dừng.")


async def compat_call_agent_runtime(
    message: str,
    *,
    user_token: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    context_extra: dict | None = None,
    model_name: str | None = None,
) -> dict:
    """Giữ điểm monkeypatch cũ của ``bridge.app._call_agent_runtime``."""
    del user_id
    bridge_app = _bridge_app_module()
    base = bridge_app.get_hautoml_config()["base_url"].rstrip("/")
    agent_url = os.getenv("HAGENT_AGENT_RUN_URL", f"{base}/api/v1/chat/agent-run")
    models = bridge_app.get_llm_models()
    if model_name is not None:
        available = [str(model.get("name")) for model in models if model.get("name")]
        if model_name not in available:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model {model_name!r} does not exist in the configured registry. "
                    f"Valid names: {available}"
                ),
            )

    context = dict(context_extra or {})
    for key in _RESERVED_RUNTIME_CONTEXT_KEYS:
        context.pop(key, None)
    context["hautoml_url"] = base
    payload: dict[str, Any] = {
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
            response = await client.post(agent_url, json=payload, headers=headers)
        if 200 <= response.status_code < 300:
            data = response.json()
            if not isinstance(data, dict):
                raise HTTPException(status_code=502, detail="Invalid runtime response")
            result = dict(data)
            message_value = result.get("message", result.get("response"))
            if not isinstance(message_value, str):
                raise HTTPException(status_code=502, detail="Invalid runtime response")
            result["message"] = message_value
            to_chat_response(result, session_id or "hagent_session")
            return result
        status_code = response.status_code if 400 <= response.status_code < 500 else 502
        raise HTTPException(
            status_code=status_code,
            detail=upstream_error_detail(response),
        )
    except HTTPException:
        raise
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="HAgent runtime timed out") from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502, detail="HAgent runtime unavailable"
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Invalid runtime response") from exc


async def compat_chat(request, req, user) -> ChatResponse:
    """Điểm gọi chat cũ, vẫn tra cứu dependency qua namespace bridge.app."""
    bridge_app = _bridge_app_module()
    bridge_app._validate_model_name(req.model)
    conversation_id = req.conversation_id or uuid.uuid4().hex
    world_state_store = request.app.state.world_state_store
    await world_state_store.ensure(user.user_id)
    snapshot = await world_state_store.get(user.user_id)
    history = await bridge_app.conv_store.get_message_history(
        conversation_id, user.user_id, limit=20
    )
    await bridge_app.conv_store.add_message(
        conversation_id=conversation_id,
        user_id=user.user_id,
        role="user",
        content=req.message,
    )
    result = await bridge_app._call_hagent_gateway(
        message=req.message,
        user_token=user.raw_token,
        user_id=user.user_id,
        session_id=conversation_id,
        context_extra=bridge_app._runtime_context(req.context, snapshot, history),
        model_name=req.model,
    )
    await bridge_app._apply_tool_outputs_to_world_state(
        world_state_store, user.user_id, result
    )
    await bridge_app.conv_store.add_message(
        conversation_id=conversation_id,
        user_id=user.user_id,
        role="assistant",
        content=result["message"],
        provider=result.get("provider", ""),
        model=result.get("model", ""),
    )
    tracked_job_id = bridge_app._extract_training_job_id(result["message"])
    if tracked_job_id:
        bridge_app._schedule_training_result_notification(
            conversation_id=conversation_id,
            user_id=user.user_id,
            user_token=user.raw_token,
            job_id=tracked_job_id,
        )
    return bridge_app._to_chat_response(result, conversation_id)


async def upload_hautoml_dataset(
    file: Any,
    *,
    base_url: str,
    user_id: str,
    raw_token: str,
    on_http_failure: Callable[[int], None] | None = None,
) -> tuple[bytes, str]:
    """Tải lên một file dữ liệu và chuẩn hóa lỗi truyền tải của HAutoML."""
    file_content = await file.read()
    filename = file.filename or "uploaded_data.csv"
    data_type = filename.rsplit(".", 1)[-1].lower() if "." in filename else "csv"
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{base_url}/upload-dataset?user_id={user_id}",
                data={"data_name": filename, "data_type": data_type},
                files={"file_data": (filename, file_content, file.content_type)},
                headers={"Authorization": f"Bearer {raw_token}"},
            )
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="HAutoML upload timed out") from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail="HAutoML upload unavailable",
        ) from exc
    if not 200 <= response.status_code < 300:
        if on_http_failure is not None:
            on_http_failure(response.status_code)
        status_code = response.status_code if 400 <= response.status_code < 500 else 502
        raise HTTPException(
            status_code=status_code,
            detail=f"HAutoML upload returned HTTP {response.status_code}",
        )
    return file_content, filename


async def compat_chat_with_file(
    request,
    message: str,
    file,
    conversation_id: str | None = None,
    model: str | None = None,
    user=None,
) -> ChatResponse:
    """Điểm upload cũ, bảo toàn các seam monkeypatch của test và client nội bộ."""
    bridge_app = _bridge_app_module()
    bridge_app._validate_model_name(model)
    hautoml_cfg = bridge_app.get_hautoml_config()
    conv_id = conversation_id or uuid.uuid4().hex
    world_state_store = request.app.state.world_state_store
    await world_state_store.ensure(user.user_id)
    snapshot = await world_state_store.get(user.user_id)
    file_content, filename = await upload_hautoml_dataset(
        file,
        base_url=hautoml_cfg["base_url"],
        user_id=user.user_id,
        raw_token=user.raw_token,
    )

    full_message = f"{message}\n[File uploaded: {filename} — {len(file_content)} bytes]"
    history = await bridge_app.conv_store.get_message_history(
        conv_id, user.user_id, limit=20
    )
    await bridge_app.conv_store.add_message(conv_id, user.user_id, "user", full_message)
    result = await bridge_app._call_hagent_gateway(
        message=full_message,
        user_token=user.raw_token,
        user_id=user.user_id,
        session_id=conv_id,
        context_extra=bridge_app._runtime_context(None, snapshot, history),
        model_name=model,
    )
    await bridge_app._apply_tool_outputs_to_world_state(
        world_state_store, user.user_id, result
    )
    await bridge_app.conv_store.add_message(
        conv_id,
        user.user_id,
        "assistant",
        result["message"],
        result.get("provider", ""),
        result.get("model", ""),
    )
    tracked_job_id = bridge_app._extract_training_job_id(result["message"])
    if tracked_job_id:
        bridge_app._schedule_training_result_notification(
            conversation_id=conv_id,
            user_id=user.user_id,
            user_token=user.raw_token,
            job_id=tracked_job_id,
        )
    return bridge_app._to_chat_response(result, conv_id)


async def compat_chat_stream(request, req, user):
    """Điểm streaming cũ với dependency được tra cứu động từ bridge.app."""
    bridge_app = _bridge_app_module()
    bridge_app._validate_model_name(req.model)
    conversation_id = req.conversation_id or uuid.uuid4().hex
    world_state_store = request.app.state.world_state_store
    await world_state_store.ensure(user.user_id)
    snapshot = await world_state_store.get(user.user_id)
    history = await bridge_app.conv_store.get_message_history(
        conversation_id, user.user_id, limit=20
    )
    await bridge_app.conv_store.add_message(
        conversation_id=conversation_id,
        user_id=user.user_id,
        role="user",
        content=req.message,
    )
    return StreamingResponse(
        bridge_app._bridge_event_stream(
            message=req.message,
            user=user,
            conversation_id=conversation_id,
            context_extra=bridge_app._runtime_context(req.context, snapshot, history),
            model_name=req.model,
            world_state_store=world_state_store,
            message_id=f"stream:{uuid.uuid4().hex}",
            stream_lines_fn=bridge_app._stream_agent_runtime_lines,
            apply_tool_outputs_fn=bridge_app._apply_tool_outputs_to_world_state,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Conversation-Id": conversation_id,
        },
    )


async def compat_list_providers(user=None) -> ProvidersResponse:
    """Giữ API gọi hàm cũ cho danh sách provider."""
    del user
    bridge_app = _bridge_app_module()
    registry = [model for model in bridge_app.get_llm_models() if model.get("name")]
    grouped: dict[str, list[str]] = {}
    for model in registry:
        provider_id = str(model.get("provider") or "unknown")
        grouped.setdefault(provider_id, []).append(str(model["name"]))
    default_model = str(bridge_app.get_llm_config().get("default_model") or "")
    if not default_model and registry:
        default_model = str(registry[0]["name"])
    default_entry = next(
        (model for model in registry if str(model["name"]) == default_model), None
    )
    default_provider = (
        str(default_entry.get("provider") or "unknown") if default_entry else ""
    )
    return ProvidersResponse(
        default_provider=default_provider,
        default_model=default_model,
        providers=[
            ProviderInfo(
                name=provider_id.replace("_", " ").title(),
                provider_id=provider_id,
                models=model_names,
                available=True,
                description="Configured in the toolkit model registry",
            )
            for provider_id, model_names in grouped.items()
        ],
    )
