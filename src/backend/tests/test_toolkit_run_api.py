"""Regression HTTP/SSE cho owner-scoped Toolkit run API."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi import FastAPI, Request
from langgraph.checkpoint.memory import InMemorySaver

from hagent.agent.capabilities.catalog import CapabilityCatalog
from hagent.agent.capabilities.models import CapabilityDescriptor
from hagent.agent.journey.runtime_adapter import JourneyRuntime
from hagent.agent.runtime import set_agent_runtime
from users.routers import get_current_user


class _DatasetAdapter:
    async def invoke(self, _capability_id, arguments, *, scope):
        return {
            "_id": arguments["dataset_id"],
            "columns": ["feature", "target"],
            "target": "target",
            "missingness": {},
            "class_balance": {"0": 0.5, "1": 0.5},
            "leakage_risks": [],
        }


def _runtime() -> JourneyRuntime:
    descriptor = CapabilityDescriptor(
        id="automl.dataset.inspect@1",
        input_schema={
            "type": "object",
            "required": ["dataset_id"],
            "properties": {"dataset_id": {"type": "string"}},
            "additionalProperties": False,
        },
        output_schema={"type": "object"},
        effect="read",
        required_scopes=frozenset({"automl.dataset.read"}),
        provider_id="toolkit-run-test",
    )
    catalog = CapabilityCatalog()
    catalog.register_provider("toolkit-run-test", [descriptor], _DatasetAdapter())
    return JourneyRuntime(
        capability_snapshot=catalog.snapshot(),
        checkpointer=InMemorySaver(),
    )


@pytest.fixture
async def client():
    from hagent.run.router import router

    app = FastAPI()
    app.include_router(router)

    async def current_user(request: Request):
        return {"_id": request.headers.get("X-Test-Owner", "owner-1")}

    app.dependency_overrides[get_current_user] = current_user
    previous = set_agent_runtime(_runtime())
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as http_client:
            yield http_client
    finally:
        set_agent_runtime(previous)


def _frames(response: httpx.Response) -> list[dict]:
    frames = []
    for block in response.text.strip().split("\n\n"):
        if not block.strip():
            continue
        fields = {}
        for line in block.splitlines():
            name, value = line.split(":", 1)
            fields[name] = value.strip()
        frames.append(
            {
                "id": int(fields["id"]),
                "event": fields["event"],
                "data": json.loads(fields["data"]),
            }
        )
    return frames


async def _start(client: httpx.AsyncClient, *, suffix: str):
    response = await client.post(
        "/api/v1/runs",
        headers={"Authorization": "Bearer toolkit-request-secret"},
        json={
            "message": "Train dataset dataset-1 target target",
            "run_id": f"toolkit-run-{suffix}",
            "command_id": f"toolkit-start-{suffix}",
            "history": [{"role": "user", "content": "Need a safe experiment"}],
        },
    )
    assert response.status_code == 200
    return response, _frames(response)


@pytest.mark.asyncio
async def test_start_replay_approval_and_duplicate_use_runtime_sequence(client):
    response, started = await _start(client, suffix="approval")
    duplicate_start, duplicate_started = await _start(client, suffix="approval")

    assert response.headers["content-type"].startswith("text/event-stream")
    assert [frame["id"] for frame in started] == list(range(1, len(started) + 1))
    assert all(frame["data"]["sequence"] == frame["id"] for frame in started)
    assert started[-1]["event"] == "approval_required"
    assert duplicate_start.text == response.text
    assert duplicate_started == started
    assert "toolkit-request-secret" not in response.text

    last_seen = started[-2]["id"]
    replay = await client.get(
        "/api/v1/runs/toolkit-run-approval/events",
        headers={
            "Authorization": "Bearer toolkit-request-secret",
            "Last-Event-ID": str(last_seen),
        },
    )
    assert [frame["id"] for frame in _frames(replay)] == [started[-1]["id"]]

    approval_id = started[-1]["data"]["approval_id"]
    approval_body = {
        "approved": True,
        "command_id": "toolkit-approval-command",
    }
    approved = await client.post(
        f"/api/v1/runs/toolkit-run-approval/approvals/{approval_id}",
        headers={"Authorization": "Bearer toolkit-request-secret"},
        json=approval_body,
    )
    duplicate = await client.post(
        f"/api/v1/runs/toolkit-run-approval/approvals/{approval_id}",
        headers={"Authorization": "Bearer toolkit-request-secret"},
        json=approval_body,
    )
    approved_frames = _frames(approved)

    assert approved.status_code == 200
    assert duplicate.text == approved.text
    assert approved_frames[-1]["event"] == "run_completed"
    assert approved_frames[0]["id"] == started[-1]["id"] + 1


@pytest.mark.asyncio
async def test_cancel_is_terminal_and_wrong_owner_cannot_replay(client):
    _, started = await _start(client, suffix="cancel")
    cancelled = await client.post(
        "/api/v1/runs/toolkit-run-cancel/cancel",
        headers={"Authorization": "Bearer toolkit-request-secret"},
        json={"command_id": "toolkit-cancel-command"},
    )

    cancelled_frames = _frames(cancelled)
    assert cancelled.status_code == 200
    assert cancelled_frames == [
        {
            "id": started[-1]["id"] + 1,
            "event": "run_cancelled",
            "data": cancelled_frames[0]["data"],
        }
    ]
    assert cancelled_frames[0]["data"]["reason"] == "user_requested"

    hidden = await client.get(
        "/api/v1/runs/toolkit-run-cancel/events",
        headers={
            "Authorization": "Bearer other-owner-secret",
            "X-Test-Owner": "owner-2",
        },
    )
    assert hidden.status_code == 404
    assert hidden.json() == {"detail": {"code": "RUN_NOT_FOUND"}}


@pytest.mark.asyncio
async def test_boundary_rejects_auth_replay_id_and_forbidden_authority_fields(client):
    missing_auth = await client.post(
        "/api/v1/runs",
        json={"message": "Audit dataset dataset-1"},
    )
    malformed_auth = await client.post(
        "/api/v1/runs",
        headers={"Authorization": "Basic unsafe"},
        json={"message": "Audit dataset dataset-1"},
    )
    forbidden_authority = await client.post(
        "/api/v1/runs",
        headers={"Authorization": "Bearer safe"},
        json={
            "message": "Audit dataset dataset-1",
            "principal_id": "forged-owner",
            "token": "forged-token",
        },
    )
    malformed_replay = await client.get(
        "/api/v1/runs/any-run/events",
        headers={
            "Authorization": "Bearer safe",
            "Last-Event-ID": "not-a-sequence",
        },
    )

    assert missing_auth.status_code == 401
    assert malformed_auth.status_code == 401
    assert forbidden_authority.status_code == 422
    assert malformed_replay.status_code == 400
    assert "unsafe" not in malformed_auth.text
