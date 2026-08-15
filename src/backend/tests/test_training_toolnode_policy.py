"""Regression policy cho raw ToolNode training mutation."""

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from hagent.agent.orchestration.graph import _inject_request_scope_into_tool_call
from hagent.agent.runtime.context import GraphRequestContext
from hagent.agent.tools import automl_tools


def _build_tool_graph():
    builder = StateGraph(MessagesState, context_schema=GraphRequestContext)
    builder.add_node(
        "tools",
        ToolNode(
            [automl_tools.start_training],
            awrap_tool_call=_inject_request_scope_into_tool_call,
        ),
    )
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    return builder.compile()


def _tool_args(*, idempotency_key: str | None) -> dict:
    args = {
        "user_id": "model-owner",
        "dataset_id": "dataset-1",
        "problem_type": "classification",
        "target_column": "label",
        "list_feature": ["feature"],
        "token": "model-token",
    }
    if idempotency_key is not None:
        args["idempotency_key"] = idempotency_key
    return args


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_key",
    [None, "forged-key-a", "forged-key-b"],
)
async def test_real_toolnode_denies_direct_training_before_api(
    monkeypatch,
    model_key,
):
    api_post = AsyncMock(return_value={"status": "success", "job_id": "unsafe-job"})
    monkeypatch.setattr(automl_tools, "_api_post", api_post)
    graph = _build_tool_graph()
    context = GraphRequestContext(
        principal_id="trusted-owner",
        credential="trusted-token",
    )
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "id": f"model-call-{model_key or 'missing'}",
                "name": "start_training",
                "args": _tool_args(idempotency_key=model_key),
            }
        ],
    )

    result = await graph.ainvoke({"messages": [message]}, context=context)
    tool_message = result["messages"][-1]

    assert tool_message.status == "error"
    assert "TRAINING_TRUSTED_ACTION_REQUIRED" in tool_message.content
    assert "model-owner" not in tool_message.content
    assert "model-token" not in tool_message.content
    assert "trusted-token" not in tool_message.content
    assert not model_key or model_key not in tool_message.content
    api_post.assert_not_awaited()
