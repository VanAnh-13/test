"""
Route module: trạng thái thế giới và sức khỏe dịch vụ.

Endpoints:
  GET  /api/v1/world-state/{user_id}
  GET  /api/v1/chat/health
  GET  /api/v1/ready
"""

# FastAPI yêu cầu Depends trong chữ ký endpoint.
# ruff: noqa: B008

from __future__ import annotations

import asyncio

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from hagent.bridge.auth import TokenPayload, get_current_user
from hagent.bridge.config import get_hautoml_config
from hagent.bridge.models import HealthResponse
from hagent.bridge.routes.route_support import (
    bounded_readiness_probe,
    probe_http_status,
    probe_mongo_readiness,
    probe_toolkit_readiness,
    readiness_timeout_seconds,
    toolkit_url,
)
from hagent.world.state_store import WorldStateStore

logger = structlog.get_logger("hagent.bridge.routes.world_model")

router = APIRouter(tags=["world-model"])

_DEFAULT_READINESS_TIMEOUT_SECONDS = 5.0


def _get_run_api_url() -> str:
    """Lazy import to avoid circular dependency."""
    from hagent.bridge.routes.agent_control import _run_api_url

    return _run_api_url()


def _toolkit_url_local(path: str) -> str:
    return toolkit_url(path, _get_run_api_url)


def _readiness_response(mongodb_ready: bool, toolkit_ready: bool) -> JSONResponse:
    ready = mongodb_ready and toolkit_ready
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ready" if ready else "not_ready",
            "dependencies": {
                "mongodb": "ready" if mongodb_ready else "unavailable",
                "toolkit": "ready" if toolkit_ready else "unavailable",
            },
        },
    )


# ── Route handlers ─────────────────────────────────────────────────────────────


@router.get("/api/v1/world-state/{user_id}")
async def get_world_state(
    user_id: str,
    request: Request,
    user: TokenPayload = Depends(get_current_user),
):
    """Lay world state cua mot nguoi dung cu the.
    Chi nguoi dung do moi co quyen truy cap world state cua chinh minh.
    """
    if user.user_id != user_id:
        raise HTTPException(
            status_code=403,
            detail="Khong co quyen truy cap world state cua nguoi dung khac",
        )
    world_state_store: WorldStateStore = request.app.state.world_state_store
    world_state = await world_state_store.get(user_id)
    if not world_state:
        raise HTTPException(
            status_code=404,
            detail="Khong tim thay world state cho nguoi dung nay",
        )
    return world_state.to_dict()


@router.get("/api/v1/chat/health", response_model=HealthResponse)
async def health_check():
    """Giữ hợp đồng health cũ mà không biến lỗi dependency thành lỗi liveness."""
    import sys

    bridge_app = sys.modules.get("hagent.bridge.app")
    config_loader = (
        bridge_app.get_hautoml_config
        if bridge_app is not None and hasattr(bridge_app, "get_hautoml_config")
        else get_hautoml_config
    )
    hautoml_cfg = config_loader()
    try:
        timeout = readiness_timeout_seconds()
        toolkit_health_url = _toolkit_url_local("/api/v1/chat/health")
    except (TypeError, ValueError):
        timeout = _DEFAULT_READINESS_TIMEOUT_SECONDS
        toolkit_health_url = ""

    hagent_ready = bool(toolkit_health_url) and await probe_http_status(
        toolkit_health_url, timeout
    )
    hautoml_ready = await probe_http_status(
        f"{hautoml_cfg['base_url'].rstrip('/')}/home", timeout
    )
    return HealthResponse(
        hagent_url="/api/hagent",
        connected=hagent_ready,
        hautoml_connected=hautoml_ready,
        mode="hagent",
        active_provider="hagent",
        active_model="hagent-agent",
        available_providers=["hagent"],
    )


@router.get("/api/v1/ready")
async def readiness_check():
    """Chi bao ready khi Mongo va toolkit deu phuc vu duoc request thuc."""
    try:
        timeout = readiness_timeout_seconds()
        _toolkit_url_local("/api/v1/chat/health")
    except (TypeError, ValueError):
        return _readiness_response(False, False)

    mongodb_ready, toolkit_ready = await asyncio.gather(
        bounded_readiness_probe(probe_mongo_readiness(), timeout),
        bounded_readiness_probe(
            probe_toolkit_readiness(timeout, _toolkit_url_local), timeout
        ),
    )
    return _readiness_response(mongodb_ready, toolkit_ready)
