"""Regression cho Bridge proxy của durable run API."""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import httpx
import pytest
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
from hagent.bridge.auth import TokenPayload, get_current_user


class _StreamResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        chunks: tuple[bytes, ...] = (),
        payload: object | None = None,
        headers: dict[str, str] | None = None,
        read_error: Exception | None = None,
        stream_error: Exception | None = None,
    ) -> None:
        self.status_code = status_code
        self._chunks = chunks
        self._payload = payload
        self._read_error = read_error
        self._stream_error = stream_error
        self.headers = headers or {
            "content-type": "text/event-stream",
            "x-run-id": "bridge-run-1",
        }
        self.closed = False

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk
        if self._stream_error is not None:
            raise self._stream_error

    async def aread(self):
        if self._read_error is not None:
            raise self._read_error
        if self._payload is None:
            return b""
        return json.dumps(self._payload).encode()

    def json(self):
        return self._payload

    async def aclose(self):
        self.closed = True


class _StreamClient:
    def __init__(self, *, response=None, error=None, capture=None) -> None:
        self.response = response
        self.error = error
        self.capture = capture if capture is not None else {}
        self.closed = False

    def build_request(self, method, url, **kwargs):
        request = SimpleNamespace(method=method, url=url, **kwargs)
        self.capture.update(
            method=method,
            url=url,
            json=kwargs.get("json"),
            headers=kwargs.get("headers"),
            params=kwargs.get("params"),
        )
        return request

    async def send(self, request, *, stream):
        self.capture["stream"] = stream
        if self.error is not None:
            raise self.error
        return self.response

    async def aclose(self):
        self.closed = True


@pytest.fixture
def public_client():
    bridge_app.hagent_bridge.dependency_overrides[get_current_user] = lambda: (
        TokenPayload(
            {"sub": "owner-1", "type": "access"},
            raw_token="trusted-bridge-token",
        )
    )
    try:
        yield TestClient(bridge_app.hagent_bridge)
    finally:
        bridge_app.hagent_bridge.dependency_overrides.pop(get_current_user, None)


def _patch_upstream(monkeypatch, *, response=None, error=None):
    capture = {}
    client = _StreamClient(response=response, error=error, capture=capture)
    monkeypatch.setattr(
        bridge_app.httpx,
        "AsyncClient",
        lambda *args, **kwargs: client,
    )
    monkeypatch.setattr(
        bridge_app,
        "get_hautoml_config",
        lambda: {"base_url": "http://toolkit:8585"},
    )
    monkeypatch.delenv("HAGENT_RUN_API_URL", raising=False)
    return capture, client


def test_start_relays_raw_sse_and_forwards_only_authenticated_token(
    monkeypatch,
    public_client,
):
    raw_stream = (
        b'id: 1\nevent: run_started\ndata: {"sequence":1}\n\n',
        b'id: 2\nevent: approval_required\ndata: {"sequence":2}\n\n',
    )
    upstream = _StreamResponse(chunks=raw_stream)
    capture, stream_client = _patch_upstream(monkeypatch, response=upstream)

    response = public_client.post(
        "/api/v1/runs",
        headers={"Authorization": "Bearer caller-header-is-not-forwarded"},
        json={
            "message": "Train dataset dataset-1 target target",
            "run_id": "bridge-run-1",
            "command_id": "bridge-command-1",
        },
    )

    assert response.status_code == 200
    assert response.content == b"".join(raw_stream)
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-run-id"] == "bridge-run-1"
    assert capture == {
        "method": "POST",
        "url": "http://toolkit:8585/api/v1/runs",
        "json": {
            "message": "Train dataset dataset-1 target target",
            "run_id": "bridge-run-1",
            "command_id": "bridge-command-1",
            "history": [],
            "model": None,
        },
        "headers": {
            "Authorization": "Bearer trusted-bridge-token",
            "Content-Type": "application/json",
        },
        "params": None,
        "stream": True,
    }
    assert upstream.closed
    assert stream_client.closed
    assert b"trusted-bridge-token" not in response.content


@pytest.mark.parametrize(
    ("public_path", "method", "body", "expected_path"),
    [
        (
            "/api/v1/runs/bridge-run-1/approvals/approval-1",
            "POST",
            {"approved": True, "command_id": "approve-command"},
            "/api/v1/runs/bridge-run-1/approvals/approval-1",
        ),
        (
            "/api/v1/runs/bridge-run-1/cancel",
            "POST",
            {"command_id": "cancel-command"},
            "/api/v1/runs/bridge-run-1/cancel",
        ),
    ],
)
def test_approval_and_cancel_forward_exact_path_and_body(
    monkeypatch,
    public_client,
    public_path,
    method,
    body,
    expected_path,
):
    upstream = _StreamResponse(
        chunks=(b'id: 3\nevent: run_completed\ndata: {"sequence":3}\n\n',)
    )
    capture, _ = _patch_upstream(monkeypatch, response=upstream)

    response = public_client.request(method, public_path, json=body)

    assert response.status_code == 200
    assert capture["url"] == f"http://toolkit:8585{expected_path}"
    assert capture["json"] == {
        **body,
        **({"response": {}} if "approved" in body else {}),
    }


def test_replay_forwards_query_and_last_event_id(monkeypatch, public_client):
    upstream = _StreamResponse(
        chunks=(b'id: 5\nevent: run_completed\ndata: {"sequence":5}\n\n',)
    )
    capture, _ = _patch_upstream(monkeypatch, response=upstream)

    response = public_client.get(
        "/api/v1/runs/bridge-run-1/events?after_sequence=2",
        headers={"Last-Event-ID": "4"},
    )

    assert response.status_code == 200
    assert capture["method"] == "GET"
    assert capture["params"] == {"after_sequence": 2}
    assert capture["headers"]["Last-Event-ID"] == "4"
    assert capture["headers"]["Authorization"] == "Bearer trusted-bridge-token"


@pytest.mark.parametrize("status_code", [400, 404, 409, 410, 422, 503])
def test_safe_upstream_error_code_is_preserved(
    monkeypatch,
    public_client,
    status_code,
):
    upstream = _StreamResponse(
        status_code=status_code,
        payload={"detail": {"code": "COMMAND_ID_CONFLICT"}},
    )
    _, stream_client = _patch_upstream(monkeypatch, response=upstream)

    response = public_client.post(
        "/api/v1/runs",
        json={"message": "Audit dataset dataset-1"},
    )

    assert response.status_code == status_code
    assert response.json() == {"detail": {"code": "COMMAND_ID_CONFLICT"}}
    assert upstream.closed
    assert stream_client.closed


def test_timeout_invalid_upstream_and_extra_body_fail_closed(
    monkeypatch,
    public_client,
):
    _, timeout_client = _patch_upstream(
        monkeypatch,
        error=httpx.ReadTimeout("contains-upstream-secret"),
    )
    timeout = public_client.post(
        "/api/v1/runs",
        json={"message": "Audit dataset dataset-1"},
    )
    assert timeout.status_code == 504
    assert "contains-upstream-secret" not in timeout.text
    assert timeout_client.closed

    invalid_upstream = _StreamResponse(
        status_code=500,
        payload={"detail": "internal-database-secret"},
    )
    _patch_upstream(monkeypatch, response=invalid_upstream)
    invalid = public_client.post(
        "/api/v1/runs",
        json={"message": "Audit dataset dataset-1"},
    )
    assert invalid.status_code == 502
    assert "internal-database-secret" not in invalid.text

    extra = public_client.post(
        "/api/v1/runs",
        json={"message": "Audit dataset dataset-1", "token": "forged"},
    )
    assert extra.status_code == 422


def test_read_and_midstream_network_errors_close_upstream_without_synthetic_event(
    monkeypatch,
    public_client,
):
    unreadable = _StreamResponse(
        status_code=503,
        read_error=httpx.ReadError("contains-upstream-secret"),
    )
    _, unreadable_client = _patch_upstream(monkeypatch, response=unreadable)

    error_response = public_client.post(
        "/api/v1/runs",
        json={"message": "Audit dataset dataset-1"},
    )

    assert error_response.status_code == 502
    assert error_response.json() == {"detail": {"code": "UPSTREAM_RUNTIME_ERROR"}}
    assert "contains-upstream-secret" not in error_response.text
    assert unreadable.closed
    assert unreadable_client.closed

    first_event = b'id: 1\nevent: run_started\ndata: {"sequence":1}\n\n'
    interrupted = _StreamResponse(
        chunks=(first_event,),
        stream_error=httpx.ReadError("contains-stream-secret"),
    )
    _, interrupted_client = _patch_upstream(monkeypatch, response=interrupted)

    stream_response = public_client.post(
        "/api/v1/runs",
        json={"message": "Audit dataset dataset-1"},
    )

    assert stream_response.status_code == 200
    assert stream_response.content == first_event
    assert b"contains-stream-secret" not in stream_response.content
    assert interrupted.closed
    assert interrupted_client.closed
