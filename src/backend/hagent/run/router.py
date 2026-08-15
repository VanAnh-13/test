"""Transport HTTP/SSE theo chủ sở hữu cho AgentRuntime bền vững."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Path, Query, Request
from fastapi.responses import StreamingResponse

from hagent.agent.runtime import (
    AgentRuntimeError,
    CancelRun,
    RequestScope,
    ResolveApproval,
    RuntimeAccessDenied,
    RuntimeCapacityExceeded,
    RuntimeCommandConflict,
    RuntimeCommandExpired,
    RuntimeEvent,
    RuntimeEventLimitExceeded,
    RuntimeLedgerUnavailable,
    RuntimeRunNotFound,
    StartTurn,
    UnsupportedRuntimeCommand,
    get_agent_runtime,
    runtime_event_to_dict,
)
from hagent.run.models import (
    SAFE_RUNTIME_ID_PATTERN,
    CancelRunRequest,
    ResolveRunApprovalRequest,
    StartRunRequest,
)
from users.routers import get_current_user

router = APIRouter(prefix="/api/v1/runs", tags=["HAgent Runs"])
logger = structlog.get_logger(__name__)

_RUNTIME_SCOPES = (
    "automl.dataset.read",
    "automl.training.read",
    "automl.training.write",
)
_MAX_BEARER_LENGTH = 8192


def _credential(request: Request) -> str:
    authorization = request.headers.get("Authorization", "")
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={"code": "AUTH_REQUIRED"},
        )
    credential = authorization.removeprefix("Bearer ").strip()
    if not credential or len(credential) > _MAX_BEARER_LENGTH:
        raise HTTPException(
            status_code=401,
            detail={"code": "AUTH_REQUIRED"},
        )
    return credential


def _request_scope(request: Request, current_user: dict) -> RequestScope:
    principal_id = str(current_user.get("_id", "")).strip()
    if not principal_id:
        raise HTTPException(
            status_code=401,
            detail={"code": "AUTH_REQUIRED"},
        )
    try:
        return RequestScope(
            principal_id=principal_id,
            credential=_credential(request),
            trace_id=uuid.uuid4().hex,
            services={"scopes": _RUNTIME_SCOPES},
        )
    except ValueError:
        raise HTTPException(
            status_code=401,
            detail={"code": "AUTH_REQUIRED"},
        ) from None


def _runtime_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, RuntimeAccessDenied | RuntimeRunNotFound):
        return HTTPException(status_code=404, detail={"code": "RUN_NOT_FOUND"})
    if isinstance(exc, RuntimeCommandConflict):
        return HTTPException(
            status_code=409,
            detail={"code": "COMMAND_ID_CONFLICT"},
        )
    if isinstance(exc, RuntimeCommandExpired):
        return HTTPException(
            status_code=410,
            detail={"code": "COMMAND_REPLAY_EXPIRED"},
        )
    if isinstance(exc, UnsupportedRuntimeCommand):
        return HTTPException(
            status_code=409,
            detail={"code": "COMMAND_UNSUPPORTED"},
        )
    if isinstance(
        exc,
        RuntimeCapacityExceeded | RuntimeEventLimitExceeded | RuntimeLedgerUnavailable,
    ):
        return HTTPException(
            status_code=503,
            detail={"code": "RUNTIME_UNAVAILABLE"},
        )
    if isinstance(exc, AgentRuntimeError):
        return HTTPException(
            status_code=500,
            detail={"code": "RUNTIME_FAILED"},
        )
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail={"code": "INVALID_REQUEST"})
    logger.error(
        "Toolkit run transport gặp lỗi không dự kiến",
        extra={"error_type": type(exc).__name__},
    )
    return HTTPException(status_code=500, detail={"code": "RUNTIME_FAILED"})


def _sse_frame(event: RuntimeEvent) -> str:
    data = json.dumps(
        runtime_event_to_dict(event),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"id: {event.sequence}\nevent: {event.type}\ndata: {data}\n\n"


async def _stream_response(
    events: AsyncIterator[RuntimeEvent],
    *,
    run_id: str,
) -> StreamingResponse:
    try:
        first = await anext(events)
    except StopAsyncIteration:
        first = None
    except Exception as exc:
        close = getattr(events, "aclose", None)
        if callable(close):
            await close()
        raise _runtime_http_error(exc) from exc

    async def frames():
        try:
            if first is not None:
                yield _sse_frame(first)
            async for event in events:
                yield _sse_frame(event)
        finally:
            close = getattr(events, "aclose", None)
            if callable(close):
                await close()

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Run-Id": run_id,
        },
    )


def _replay_sequence(after_sequence: int, last_event_id: str | None) -> int:
    if last_event_id is None:
        return after_sequence
    try:
        header_sequence = int(last_event_id)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_LAST_EVENT_ID"},
        ) from None
    if header_sequence < 0:
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_LAST_EVENT_ID"},
        )
    return max(after_sequence, header_sequence)


@router.post("")
async def start_run(
    payload: StartRunRequest,
    request: Request,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    scope = _request_scope(request, current_user)
    command = StartTurn(
        message=payload.message,
        run_id=payload.run_id or uuid.uuid4().hex,
        command_id=payload.command_id or uuid.uuid4().hex,
        history=tuple(item.model_dump() for item in payload.history),
        model_name=payload.model,
    )
    return await _stream_response(
        get_agent_runtime().dispatch(command, scope=scope),
        run_id=command.run_id,
    )


@router.get("/{run_id}/events")
async def replay_run_events(
    run_id: Annotated[str, Path(pattern=SAFE_RUNTIME_ID_PATTERN)],
    request: Request,
    current_user: Annotated[dict, Depends(get_current_user)],
    after_sequence: Annotated[int, Query(ge=0)] = 0,
    last_event_id: Annotated[
        str | None,
        Header(alias="Last-Event-ID"),
    ] = None,
):
    scope = _request_scope(request, current_user)
    sequence = _replay_sequence(after_sequence, last_event_id)
    return await _stream_response(
        get_agent_runtime().replay(
            run_id,
            after_sequence=sequence,
            scope=scope,
        ),
        run_id=run_id,
    )


@router.post("/{run_id}/approvals/{approval_id}")
async def resolve_run_approval(
    run_id: Annotated[str, Path(pattern=SAFE_RUNTIME_ID_PATTERN)],
    approval_id: Annotated[str, Path(pattern=SAFE_RUNTIME_ID_PATTERN)],
    payload: ResolveRunApprovalRequest,
    request: Request,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    scope = _request_scope(request, current_user)
    command = ResolveApproval(
        run_id=run_id,
        approval_id=approval_id,
        approved=payload.approved,
        command_id=payload.command_id or uuid.uuid4().hex,
        response=payload.response,
    )
    return await _stream_response(
        get_agent_runtime().dispatch(command, scope=scope),
        run_id=run_id,
    )


@router.post("/{run_id}/cancel")
async def cancel_run(
    run_id: Annotated[str, Path(pattern=SAFE_RUNTIME_ID_PATTERN)],
    payload: CancelRunRequest,
    request: Request,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    scope = _request_scope(request, current_user)
    command = CancelRun(
        run_id=run_id,
        command_id=payload.command_id or uuid.uuid4().hex,
    )
    return await _stream_response(
        get_agent_runtime().dispatch(command, scope=scope),
        run_id=run_id,
    )
