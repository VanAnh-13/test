"""Các endpoint điều khiển durable run, proxy sang HAgent toolkit."""

# ruff: noqa: B008, BLE001

from __future__ import annotations

import os
import re
from typing import Annotated

import httpx
import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Path, Query
from fastapi.responses import StreamingResponse

from hagent.bridge import config as _bridge_config
from hagent.bridge.auth import TokenPayload, get_current_user
from hagent.run.models import (
    SAFE_RUNTIME_ID_PATTERN,
    CancelRunRequest,
    ResolveRunApprovalRequest,
    StartRunRequest,
)

logger = structlog.get_logger("hagent.bridge.routes.agent_control")

router = APIRouter(tags=["agent-runs"])

# ── Helpers ──────────────────────────────────────────────────────────────────

_SAFE_RUN_ERROR_CODE = re.compile(r"[A-Z][A-Z0-9_]{0,63}")
_PRESERVED_RUN_ERROR_STATUSES = frozenset({400, 401, 403, 404, 409, 410, 422, 429, 503})


def _run_api_url(path: str = "") -> str:
    # Lookup via sys.modules to support monkeypatching bridge_app.get_hautoml_config in tests
    import sys as _sys

    _bridge_app = _sys.modules.get("hagent.bridge.app")
    _cfg_fn = (
        _bridge_app.get_hautoml_config
        if _bridge_app is not None and hasattr(_bridge_app, "get_hautoml_config")
        else _bridge_config.get_hautoml_config
    )
    base = _cfg_fn()["base_url"].rstrip("/")
    configured = os.getenv("HAGENT_RUN_API_URL", f"{base}/api/v1/runs")
    return f"{configured.rstrip('/')}{path}"


def _safe_run_error(response: httpx.Response) -> tuple[int, dict]:
    status_code = (
        response.status_code
        if response.status_code in _PRESERVED_RUN_ERROR_STATUSES
        else 502
    )
    code = "UPSTREAM_RUNTIME_ERROR"
    try:
        payload = response.json()
        detail = payload.get("detail") if isinstance(payload, dict) else None
        candidate = detail.get("code") if isinstance(detail, dict) else None
        if isinstance(candidate, str) and _SAFE_RUN_ERROR_CODE.fullmatch(candidate):
            code = candidate
    except (TypeError, ValueError):
        pass
    return status_code, {"code": code}


def _validated_last_event_id(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        sequence = int(value)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_LAST_EVENT_ID"},
        ) from None
    if sequence < 0:
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_LAST_EVENT_ID"},
        )
    return str(sequence)


async def _proxy_run_stream(
    *,
    method: str,
    path: str,
    user: TokenPayload,
    payload: dict | None = None,
    params: dict | None = None,
    last_event_id: str | None = None,
) -> StreamingResponse:
    if not user.raw_token:
        raise HTTPException(status_code=401, detail={"code": "AUTH_REQUIRED"})
    headers = {"Authorization": f"Bearer {user.raw_token}"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    validated_event_id = _validated_last_event_id(last_event_id)
    if validated_event_id is not None:
        headers["Last-Event-ID"] = validated_event_id

    client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, read=None))
    response = None
    try:
        upstream_request = client.build_request(
            method,
            _run_api_url(path),
            json=payload,
            params=params,
            headers=headers,
        )
        response = await client.send(upstream_request, stream=True)
    except httpx.TimeoutException:
        await client.aclose()
        raise HTTPException(
            status_code=504,
            detail={"code": "UPSTREAM_RUNTIME_TIMEOUT"},
        ) from None
    except httpx.RequestError:
        await client.aclose()
        raise HTTPException(
            status_code=502,
            detail={"code": "UPSTREAM_RUNTIME_UNAVAILABLE"},
        ) from None
    except Exception:
        await client.aclose()
        raise HTTPException(
            status_code=502,
            detail={"code": "UPSTREAM_RUNTIME_ERROR"},
        ) from None

    if not 200 <= response.status_code < 300:
        try:
            await response.aread()
            status_code, detail = _safe_run_error(response)
        except (httpx.TimeoutException, httpx.RequestError):
            status_code, detail = 502, {"code": "UPSTREAM_RUNTIME_ERROR"}
        finally:
            await response.aclose()
            await client.aclose()
        raise HTTPException(status_code=status_code, detail=detail)

    content_type = response.headers.get("content-type", "")
    if not content_type.lower().startswith("text/event-stream"):
        await response.aclose()
        await client.aclose()
        raise HTTPException(
            status_code=502,
            detail={"code": "INVALID_UPSTREAM_STREAM"},
        )

    async def relay():
        try:
            async for chunk in response.aiter_bytes():
                if chunk:
                    yield chunk
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            logger.warning(
                "Bridge run SSE bi ngat; client can replay",
                extra={"error_type": type(exc).__name__},
            )
        finally:
            await response.aclose()
            await client.aclose()

    response_headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    upstream_run_id = response.headers.get("x-run-id")
    if isinstance(upstream_run_id, str) and re.fullmatch(
        SAFE_RUNTIME_ID_PATTERN, upstream_run_id
    ):
        response_headers["X-Run-Id"] = upstream_run_id
    return StreamingResponse(
        relay(),
        media_type="text/event-stream",
        headers=response_headers,
    )


# ── Route handlers ────────────────────────────────────────────────────────────


@router.post("/api/v1/runs")
async def bridge_start_run(
    payload: StartRunRequest,
    user: TokenPayload = Depends(get_current_user),
):
    return await _proxy_run_stream(
        method="POST",
        path="",
        user=user,
        payload=payload.model_dump(mode="json"),
    )


@router.get("/api/v1/runs/{run_id}/events")
async def bridge_replay_run(
    run_id: Annotated[str, Path(pattern=SAFE_RUNTIME_ID_PATTERN)],
    after_sequence: Annotated[int, Query(ge=0)] = 0,
    last_event_id: Annotated[
        str | None,
        Header(alias="Last-Event-ID"),
    ] = None,
    user: TokenPayload = Depends(get_current_user),
):
    return await _proxy_run_stream(
        method="GET",
        path=f"/{run_id}/events",
        user=user,
        params={"after_sequence": after_sequence},
        last_event_id=last_event_id,
    )


@router.post("/api/v1/runs/{run_id}/approvals/{approval_id}")
async def bridge_resolve_run_approval(
    run_id: Annotated[str, Path(pattern=SAFE_RUNTIME_ID_PATTERN)],
    approval_id: Annotated[str, Path(pattern=SAFE_RUNTIME_ID_PATTERN)],
    payload: ResolveRunApprovalRequest,
    user: TokenPayload = Depends(get_current_user),
):
    return await _proxy_run_stream(
        method="POST",
        path=f"/{run_id}/approvals/{approval_id}",
        user=user,
        payload=payload.model_dump(mode="json"),
    )


@router.post("/api/v1/runs/{run_id}/cancel")
async def bridge_cancel_run(
    run_id: Annotated[str, Path(pattern=SAFE_RUNTIME_ID_PATTERN)],
    payload: CancelRunRequest,
    user: TokenPayload = Depends(get_current_user),
):
    return await _proxy_run_stream(
        method="POST",
        path=f"/{run_id}/cancel",
        user=user,
        payload=payload.model_dump(mode="json"),
    )
