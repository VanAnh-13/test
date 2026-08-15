from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.tools import tool


def test_graph_request_context_is_immutable_and_hides_sensitive_values():
    from hagent.agent.runtime.context import GraphRequestContext

    service_handles = {"world_model": object()}
    context = GraphRequestContext(
        principal_id="owner-1",
        credential="sentinel-secret",
        services=service_handles,
    )

    service_handles["late"] = object()

    assert context.principal_id == "owner-1"
    assert GraphRequestContext.__module__ == "hagent.agent.runtime.context"
    assert "sentinel-secret" not in repr(context)
    assert "world_model" not in repr(context)
    assert "late" not in context.services
    with pytest.raises(AttributeError):
        context.principal_id = "spoofed"
    with pytest.raises(TypeError):
        context.services["late"] = object()


@pytest.mark.parametrize(
    ("principal_id", "error_type"),
    [
        ("", ValueError),
        ("   ", ValueError),
        ("owner\x00suffix", ValueError),
        (None, TypeError),
        (42, TypeError),
    ],
)
def test_graph_request_context_rejects_invalid_principal(principal_id, error_type):
    from hagent.agent.runtime.context import GraphRequestContext

    with pytest.raises(error_type):
        GraphRequestContext(principal_id=principal_id)


def test_automl_graph_declares_request_context_schema(monkeypatch):
    from hagent.agent.orchestration import registry as registry_module
    from hagent.agent.orchestration.graph import build_automl_graph
    from hagent.agent.runtime.context import GraphRequestContext

    @tool
    def harmless_probe(value: str) -> str:
        """Trả lại giá trị để dựng ToolNode trong test contract."""
        return value

    class _Registry:
        def agent_names(self):
            return []

        def get_all_tools(self):
            return [harmless_probe]

        def get_node_functions(self):
            return {}

    monkeypatch.setattr(registry_module, "get_agent_registry", lambda: _Registry())

    graph = build_automl_graph()

    assert graph.context_schema is GraphRequestContext


@pytest.mark.asyncio
async def test_tool_boundary_rejects_missing_runtime_context():
    from hagent.agent.orchestration.graph import _inject_request_scope_into_tool_call

    class _Tool:
        def __init__(self):
            self.args = {"token": {}, "user_id": {}, "dataset_id": {}}

    class _Request:
        def __init__(self):
            self.tool = _Tool()
            self.state = {
                "user_id": "state-owner",
                "user_token": "state-secret",
            }
            self.runtime = SimpleNamespace(context=None)
            self.tool_call = {
                "name": "get_dataset_info",
                "id": "call-1",
                "type": "tool_call",
                "args": {
                    "dataset_id": "dataset-1",
                    "user_id": "model-owner",
                    "token": "model-secret",
                },
            }

        def override(self, *, tool_call):
            raise AssertionError("Tool không được chạy khi thiếu request context")

    async def execute(_request):
        raise AssertionError("Tool không được chạy khi thiếu request context")

    result = await _inject_request_scope_into_tool_call(_Request(), execute)

    assert result.status == "error"
    assert "AUTH_SCOPE_REQUIRED" in result.content
    assert "state-secret" not in result.content
    assert "model-secret" not in result.content


@pytest.mark.asyncio
async def test_context_node_uses_ephemeral_authority_and_scrubs_output():
    from hagent.agent.runtime.context import GraphRequestContext, bind_request_context

    captured = {}
    trusted_wm_service = object()
    trusted_world_store = object()

    async def legacy_node(state):
        captured.update(state)
        result = dict(state)
        result["execution_log"] = [
            {
                "summary": "Authorization: Bearer runtime-sentinel",
                "token": "runtime-sentinel",
            }
        ]
        return result

    node = bind_request_context(legacy_node)
    runtime = SimpleNamespace(
        context=GraphRequestContext(
            principal_id="trusted-owner",
            credential="runtime-sentinel",
            services={
                "wm_service": trusted_wm_service,
                "world_store": trusted_world_store,
            },
        )
    )

    result = await node(
        {
            "user_id": "state-owner",
            "user_token": "state-sentinel",
            "_wm_service": "state-service",
            "_world_store": "state-store",
            "safe_value": "kept",
        },
        runtime,
    )

    assert captured["user_id"] == "trusted-owner"
    assert captured["user_token"] == "runtime-sentinel"
    assert captured["_wm_service"] is trusted_wm_service
    assert captured["_world_store"] is trusted_world_store
    assert result["user_id"] == "trusted-owner"
    assert result["safe_value"] == "kept"
    assert "user_token" not in result
    assert "_wm_service" not in result
    assert "_world_store" not in result
    assert "runtime-sentinel" not in repr(result)
    assert "state-sentinel" not in repr(result)


@pytest.mark.asyncio
async def test_context_node_fails_closed_before_legacy_node_without_context():
    from hagent.agent.runtime.context import bind_request_context

    called = False

    async def legacy_node(_state):
        nonlocal called
        called = True
        return {}

    node = bind_request_context(legacy_node)

    with pytest.raises(TypeError, match="LANGGRAPH_REQUEST_CONTEXT_REQUIRED"):
        await node({}, SimpleNamespace(context=None))

    assert not called


@pytest.mark.asyncio
async def test_context_node_can_withhold_credential_from_read_only_node():
    from hagent.agent.runtime.context import GraphRequestContext, bind_request_context

    captured = {}

    async def read_only_node(state):
        captured.update(state)
        return {"status": "done"}

    node = bind_request_context(read_only_node, include_credential=False)
    result = await node(
        {"user_token": "state-sentinel"},
        SimpleNamespace(
            context=GraphRequestContext(
                principal_id="trusted-owner",
                credential="runtime-sentinel",
            )
        ),
    )

    assert captured["user_id"] == "trusted-owner"
    assert "user_token" not in captured
    assert result == {"status": "done"}


@pytest.mark.asyncio
async def test_compiled_langgraph_injects_context_without_persisting_credential():
    from typing import TypedDict

    from langgraph.graph import END, StateGraph

    from hagent.agent.runtime.context import GraphRequestContext, bind_request_context

    class _State(TypedDict, total=False):
        status: str
        user_id: str

    captured = {}

    async def legacy_node(state):
        captured.update(state)
        return {
            "status": "done",
            "user_id": state["user_id"],
            "user_token": state["user_token"],
        }

    builder = StateGraph(_State, context_schema=GraphRequestContext)
    builder.add_node("legacy", bind_request_context(legacy_node))
    builder.set_entry_point("legacy")
    builder.add_edge("legacy", END)

    result = await builder.compile().ainvoke(
        {"status": "new", "user_id": "state-owner"},
        context=GraphRequestContext(
            principal_id="trusted-owner",
            credential="runtime-sentinel",
        ),
    )

    assert captured["user_id"] == "trusted-owner"
    assert captured["user_token"] == "runtime-sentinel"
    assert result == {"status": "done", "user_id": "trusted-owner"}
    assert "runtime-sentinel" not in repr(result)


def test_automl_state_has_no_persistent_credential_field():
    from hagent.agent.orchestration import AutoMLState

    assert "user_token" not in AutoMLState.__annotations__
