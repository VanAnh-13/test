from __future__ import annotations

import inspect
import sys
import types
from io import BytesIO
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from hagent import chat_router

try:
    import motor.motor_asyncio  # noqa: F401
except ModuleNotFoundError:
    motor_module = types.ModuleType("motor")
    motor_asyncio_module = types.ModuleType("motor.motor_asyncio")
    motor_asyncio_module.AsyncIOMotorClient = type("AsyncIOMotorClient", (), {})
    motor_asyncio_module.AsyncIOMotorDatabase = type("AsyncIOMotorDatabase", (), {})
    motor_module.motor_asyncio = motor_asyncio_module
    sys.modules["motor"] = motor_module
    sys.modules["motor.motor_asyncio"] = motor_asyncio_module

from hagent.bridge import app as bridge_app
from hagent.bridge.auth import TokenPayload
from hagent.bridge.models import ChatRequest


class _StubResponse:
    def __init__(self, status_code: int, payload: object):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class _StubAsyncClient:
    def __init__(self, *, response=None, error=None, capture=None):
        self.response = response
        self.error = error
        self.capture = capture if capture is not None else {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, *, json=None, headers=None, **kwargs):
        self.capture.update(url=url, json=json, headers=headers, kwargs=kwargs)
        if self.error is not None:
            raise self.error
        return self.response


def _patch_runtime_client(monkeypatch, *, response=None, error=None, capture=None):
    monkeypatch.setattr(
        bridge_app.httpx,
        "AsyncClient",
        lambda *args, **kwargs: _StubAsyncClient(
            response=response,
            error=error,
            capture=capture,
        ),
    )
    monkeypatch.setattr(
        bridge_app,
        "get_hautoml_config",
        lambda: {"base_url": "http://toolkit:8000"},
    )
    monkeypatch.setattr(
        bridge_app,
        "get_llm_models",
        lambda: [
            {
                "name": "ci-mock",
                "provider": "openai_compatible",
                "model": "mock-model",
            }
        ],
        raising=False,
    )


@pytest.mark.asyncio
async def test_runtime_forwards_contract_without_credentials_in_json(monkeypatch):
    upstream = {
        "message": "ok",
        "provider": "hagent",
        "model": "ci-mock",
        "route": "campaign",
        "tool_outputs": [{"tool_name": "list_datasets", "payload": {}}],
        "planning": {"status": "done"},
        "campaign": {"status": "done"},
        "hierarchy": {"status": "done"},
        "world_model": {"phase": "trained"},
        "evaluation": {"score": 0.9},
        "execution_events": [{"type": "done"}],
        "execution_log": [{"step": 1}],
        "revision_count": 2,
        "cost_metrics": {"total_calls": 1},
    }
    capture = {}
    _patch_runtime_client(
        monkeypatch,
        response=_StubResponse(200, upstream),
        capture=capture,
    )

    result = await bridge_app._call_agent_runtime(
        "train",
        user_token="jwt-secret",
        user_id="server-user",
        session_id="conversation-1",
        context_extra={
            "dataset_id": "ds-1",
            "dataset_name": "Dataset One",
            "target_column": "label",
            "problem_type": "classification",
            "metric": "f1",
            "models": ["RandomForestClassifier"],
            "user_id": "spoofed",
            "user_token": "spoofed-secret",
            "hautoml_url": "http://evil.invalid",
            "world_state": {"phase": "ready"},
        },
        model_name="ci-mock",
    )

    assert capture["json"]["message"] == "train"
    assert capture["json"]["conversation_id"] == "conversation-1"
    assert capture["json"]["model"] == "ci-mock"
    assert capture["json"]["context"]["dataset_id"] == "ds-1"
    assert capture["json"]["context"]["hautoml_url"] == "http://toolkit:8000"
    assert capture["json"]["context"]["dataset_name"] == "Dataset One"
    assert capture["json"]["context"]["target_column"] == "label"
    assert capture["json"]["context"]["problem_type"] == "classification"
    assert capture["json"]["context"]["metric"] == "f1"
    assert capture["json"]["context"]["models"] == ["RandomForestClassifier"]
    assert capture["json"]["context"]["world_state"] == {"phase": "ready"}
    assert "user_id" not in capture["json"]["context"]
    assert "user_token" not in capture["json"]["context"]
    assert capture["headers"]["Authorization"] == "Bearer jwt-secret"
    assert result == upstream


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 409, 422, 429])
async def test_runtime_preserves_upstream_4xx(monkeypatch, status_code):
    _patch_runtime_client(
        monkeypatch,
        response=_StubResponse(status_code, {"detail": "upstream rejected"}),
    )

    with pytest.raises(HTTPException) as exc:
        await bridge_app._call_agent_runtime("hello", model_name="ci-mock")

    assert exc.value.status_code == status_code
    assert exc.value.detail == "upstream rejected"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        _StubResponse(500, {"detail": "runtime crashed"}),
        _StubResponse(200, ["not", "an", "object"]),
        _StubResponse(200, {}),
        _StubResponse(200, {"message": None}),
    ],
)
async def test_runtime_maps_bad_upstream_response_to_502(monkeypatch, response):
    _patch_runtime_client(monkeypatch, response=response)

    with pytest.raises(HTTPException) as exc:
        await bridge_app._call_agent_runtime("hello", model_name="ci-mock")

    assert exc.value.status_code == 502


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (
            httpx.ConnectError(
                "connection failed",
                request=httpx.Request("POST", "http://toolkit:8000"),
            ),
            502,
        ),
        (
            httpx.ReadTimeout(
                "timed out",
                request=httpx.Request("POST", "http://toolkit:8000"),
            ),
            504,
        ),
    ],
)
async def test_runtime_maps_transport_errors(monkeypatch, error, expected_status):
    _patch_runtime_client(monkeypatch, error=error)

    with pytest.raises(HTTPException) as exc:
        await bridge_app._call_agent_runtime("hello", model_name="ci-mock")

    assert exc.value.status_code == expected_status


@pytest.mark.asyncio
async def test_runtime_rejects_unknown_model_before_request(monkeypatch):
    capture = {}
    _patch_runtime_client(
        monkeypatch,
        response=_StubResponse(200, {"message": "not reached"}),
        capture=capture,
    )

    with pytest.raises(HTTPException) as exc:
        await bridge_app._call_agent_runtime("hello", model_name="missing-model")

    assert exc.value.status_code == 400
    assert capture == {}


@pytest.mark.asyncio
async def test_bridge_chat_merges_client_context_and_forwards_model(monkeypatch):
    class _WorldState:
        def to_dict(self):
            return {"user_id": "owner", "phase": "server"}

    class _WorldStateStore:
        async def ensure(self, user_id):
            return None

        async def get(self, user_id):
            return _WorldState()

    async def _noop(*args, **kwargs):
        return None

    async def _history(*args, **kwargs):
        return []

    captured = {}

    async def _gateway(**kwargs):
        captured.update(kwargs)
        return {
            "message": "ok",
            "provider": "hagent",
            "model": "ci-mock",
            "route": "campaign",
            "tool_outputs": [{"tool_name": "list_datasets", "payload": {}}],
            "planning": {"status": "done"},
            "campaign": {"status": "done"},
            "hierarchy": {"status": "done"},
            "world_model": {"phase": "trained"},
            "evaluation": {"score": 0.9},
            "execution_events": [{"type": "done"}],
            "execution_log": [{"step": 1}],
            "revision_count": 1,
            "cost_metrics": {"total_calls": 1},
        }

    monkeypatch.setattr(bridge_app.conv_store, "add_message", _noop)
    monkeypatch.setattr(bridge_app.conv_store, "get_message_history", _history)
    monkeypatch.setattr(bridge_app, "_apply_tool_outputs_to_world_state", _noop)
    monkeypatch.setattr(bridge_app, "_call_hagent_gateway", _gateway)

    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(world_state_store=_WorldStateStore()))
    )
    user = TokenPayload({"sub": "owner"}, raw_token="jwt")
    response = await bridge_app.chat(
        request,
        ChatRequest(
            message="hello",
            conversation_id="conversation-1",
            context={
                "dataset_id": "ds-1",
                "dataset_name": "Dataset One",
                "target_column": "label",
                "problem_type": "classification",
                "metric": "f1",
                "models": ["RandomForestClassifier"],
                "world_state": {"phase": "spoofed"},
                "user_id": "spoofed",
            },
            model="ci-mock",
        ),
        user,
    )

    assert captured["model_name"] == "ci-mock"
    context = captured["context_extra"]
    assert context["dataset_id"] == "ds-1"
    assert context["dataset_name"] == "Dataset One"
    assert context["target_column"] == "label"
    assert context["problem_type"] == "classification"
    assert context["metric"] == "f1"
    assert context["models"] == ["RandomForestClassifier"]
    assert "user_id" not in context
    assert context["world_state"] == {
        "user_id": "owner",
        "phase": "server",
    }
    assert response.route == "campaign"
    assert response.planning == {"status": "done"}
    assert response.tool_outputs[0]["tool_name"] == "list_datasets"
    assert response.campaign == {"status": "done"}
    assert response.hierarchy == {"status": "done"}
    assert response.world_model == {"phase": "trained"}
    assert response.evaluation == {"score": 0.9}
    assert response.execution_events == [{"type": "done"}]
    assert response.execution_log == [{"step": 1}]
    assert response.revision_count == 1
    assert response.cost_metrics == {"total_calls": 1}


@pytest.mark.asyncio
async def test_provider_discovery_uses_configured_model_registry(monkeypatch):
    monkeypatch.setattr(
        bridge_app,
        "get_llm_models",
        lambda: [
            {"name": "gpt-small", "provider": "openai", "model": "gpt-x"},
            {"name": "local", "provider": "ollama", "model": "qwen"},
        ],
        raising=False,
    )
    monkeypatch.setattr(
        bridge_app,
        "get_llm_config",
        lambda: {"default_model": "local"},
        raising=False,
    )

    response = await bridge_app.list_providers(user=None)

    assert response.default_provider == "ollama"
    assert response.default_model == "local"
    assert {
        provider.provider_id: provider.models for provider in response.providers
    } == {
        "openai": ["gpt-small"],
        "ollama": ["local"],
    }


@pytest.mark.asyncio
async def test_provider_discovery_matches_real_toolkit_registry():
    from hagent.agent.llm_config import (
        get_default_model_config,
        list_available_models,
    )

    response = await bridge_app.list_providers(user=None)
    selectable = {
        model_name for provider in response.providers for model_name in provider.models
    }
    expected = {model["name"] for model in list_available_models()}
    default = get_default_model_config()

    assert selectable == expected
    assert response.default_model == default.name
    assert response.default_provider == default.provider


def test_upload_endpoints_accept_model_form_field():
    assert "model" in inspect.signature(bridge_app.chat_with_file).parameters
    assert "model" in inspect.signature(chat_router.chat_with_file).parameters


@pytest.mark.asyncio
async def test_bridge_upload_forwards_model(monkeypatch):
    class _WorldStateStore:
        async def ensure(self, user_id):
            return None

        async def get(self, user_id):
            return None

    async def _noop(*args, **kwargs):
        return None

    async def _history(*args, **kwargs):
        return []

    captured = {}

    async def _gateway(**kwargs):
        captured.update(kwargs)
        return {
            "message": "ok",
            "provider": "hagent",
            "model": "ci-mock",
            "tool_outputs": [],
        }

    monkeypatch.setattr(
        bridge_app.httpx,
        "AsyncClient",
        lambda *args, **kwargs: _StubAsyncClient(
            response=_StubResponse(200, {"ok": True})
        ),
    )
    monkeypatch.setattr(
        bridge_app,
        "get_hautoml_config",
        lambda: {"base_url": "http://toolkit:8000"},
    )
    monkeypatch.setattr(bridge_app.conv_store, "add_message", _noop)
    monkeypatch.setattr(bridge_app.conv_store, "get_message_history", _history)
    monkeypatch.setattr(bridge_app, "_apply_tool_outputs_to_world_state", _noop)
    monkeypatch.setattr(bridge_app, "_call_hagent_gateway", _gateway)

    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(world_state_store=_WorldStateStore()))
    )
    upload = UploadFile(filename="data.csv", file=BytesIO(b"x\n1\n"))
    await bridge_app.chat_with_file(
        request=request,
        message="train",
        file=upload,
        conversation_id="conversation-1",
        model="ci-mock",
        user=TokenPayload({"sub": "owner"}, raw_token="jwt"),
    )

    assert captured["model_name"] == "ci-mock"


@pytest.mark.asyncio
async def test_toolkit_upload_forwards_model(monkeypatch):
    async def _noop(*args, **kwargs):
        return None

    captured = {}

    async def _call_agent(*args, **kwargs):
        captured.update(kwargs)
        return {
            "message": "ok",
            "provider": "hagent",
            "model": "ci-mock",
            "tool_outputs": [],
        }

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *args, **kwargs: _StubAsyncClient(
            response=_StubResponse(200, {"ok": True})
        ),
    )
    monkeypatch.setattr(
        chat_router,
        "get_hautoml_config",
        lambda: {"base_url": "http://toolkit:8000"},
    )
    monkeypatch.setattr(chat_router.chat_store, "add_message", _noop)
    monkeypatch.setattr(chat_router, "_load_world_model", _noop)
    monkeypatch.setattr(chat_router, "_call_agent", _call_agent)

    upload = UploadFile(filename="data.csv", file=BytesIO(b"x\n1\n"))
    await chat_router.chat_with_file(
        request=SimpleNamespace(headers={"Authorization": "Bearer jwt"}),
        message="train",
        file=upload,
        conversation_id="conversation-1",
        model="ci-mock",
        db=SimpleNamespace(),
        current_user={"_id": "owner"},
    )

    assert captured["model_name"] == "ci-mock"


@pytest.mark.asyncio
async def test_toolkit_server_world_model_overrides_forwarded_snapshot(monkeypatch):
    captured = {}

    async def _load_world_model(*args, **kwargs):
        return {
            "user_id": "owner",
            "phase": "server",
            "datasets": {},
            "jobs": {"server-job": {"status": "done"}},
        }

    async def _call_agent(*args, **kwargs):
        captured.update(kwargs)
        return {
            "message": "ok",
            "provider": "hagent",
            "model": "ci-mock",
            "tool_outputs": [],
        }

    monkeypatch.setattr(chat_router, "_load_world_model", _load_world_model)
    monkeypatch.setattr(chat_router, "_call_agent", _call_agent)

    await chat_router.agent_run(
        req=chat_router.ChatRequest(
            message="hello",
            conversation_id="conversation-1",
            context={
                "dataset_id": "ds-1",
                "world_state": {
                    "phase": "spoofed",
                    "jobs": {"client-job": {"status": "running"}},
                },
            },
            model="ci-mock",
        ),
        request=SimpleNamespace(headers={"Authorization": "Bearer jwt"}),
        db=SimpleNamespace(client=None, name="test"),
        current_user={"_id": "owner"},
    )

    assert captured["world_model"]["phase"] == "server"
    assert captured["world_model"]["jobs"] == {"server-job": {"status": "done"}}
    assert captured["world_model"]["request_context"]["dataset_id"] == "ds-1"


@pytest.mark.asyncio
async def test_toolkit_agent_mapping_keeps_complete_metadata(monkeypatch):
    from hagent.agent import graph

    async def _run_agent(*args, **kwargs):
        return {
            "response": "ok",
            "provider": "hagent",
            "model": "ci-mock",
            "route": "plan_executor",
            "tool_outputs": [{"tool_name": "list_datasets", "payload": {}}],
            "plan_status": "done",
            "selected_plan": {"plan_id": "p1"},
            "campaign": {"status": "done"},
            "hierarchy": {"status": "done"},
            "world_model": {"phase": "trained"},
            "evaluation": {"score": 0.8},
            "execution_events": [{"type": "done"}],
            "execution_log": [{"step": 1}],
            "revision_count": 3,
            "cost_metrics": {"total_calls": 1},
        }

    monkeypatch.setattr(graph, "run_agent", _run_agent)
    result = await chat_router._call_agent("hello", model_name="ci-mock")

    assert result["route"] == "plan_executor"
    assert result["planning"] == {
        "status": "done",
        "selected_plan": {"plan_id": "p1"},
    }
    assert result["campaign"] == {"status": "done"}
    assert result["tool_outputs"][0]["tool_name"] == "list_datasets"
    assert result["world_model"] == {"phase": "trained"}
    assert result["evaluation"] == {"score": 0.8}
    assert result["execution_events"] == [{"type": "done"}]
    assert result["cost_metrics"] == {"total_calls": 1}
    assert result["hierarchy"] == {"status": "done"}
    assert result["execution_log"] == [{"step": 1}]
    assert result["revision_count"] == 3


@pytest.mark.asyncio
async def test_toolkit_rejects_unknown_model_with_400():
    with pytest.raises(HTTPException) as exc:
        await chat_router._call_agent(
            "hello",
            model_name="missing-model",
        )

    assert exc.value.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (RuntimeError("graph failed"), 500),
        (TimeoutError("graph timed out"), 504),
    ],
)
async def test_toolkit_agent_failure_is_not_fake_success(
    monkeypatch,
    error,
    expected_status,
):
    from hagent.agent import graph

    async def _run_agent(*args, **kwargs):
        raise error

    monkeypatch.setattr(graph, "run_agent", _run_agent)

    with pytest.raises(HTTPException) as exc:
        await chat_router._call_agent("hello")

    assert exc.value.status_code == expected_status


def test_response_schemas_expose_complete_contract():
    required = {
        "route",
        "planning",
        "campaign",
        "hierarchy",
        "execution_events",
        "execution_log",
        "revision_count",
        "cost_metrics",
    }
    assert required <= set(bridge_app.ChatResponse.model_fields)
    assert required <= set(chat_router.ChatResponse.model_fields)
