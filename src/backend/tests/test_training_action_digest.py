"""Regression cho action digest của các caller training HAgent."""

from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from hagent.agent.campaign.runner import campaign_step
from hagent.agent.campaign.schema import Campaign, CampaignVariant
from hagent.agent.execution.plan_executor import plan_executor_node
from hagent.agent.execution.tool_runner import (
    enrich_params,
    invoke_tool,
    set_tool_invoker,
)
from hagent.agent.tools import automl_tools

OWNER_ID = "64b64b64b64b64b64b64b641"
DATASET_ID = "64b64b64b64b64b64b64b643"


@pytest.fixture(autouse=True)
def _reset_tool_invoker():
    set_tool_invoker(None)
    yield
    set_tool_invoker(None)


def _training_params() -> dict:
    return {
        "dataset_id": DATASET_ID,
        "problem_type": "classification",
        "target_column": "label",
        "list_feature": ["feature"],
        "idempotency_key": "model-supplied-key",
    }


def _enrich(*, action_id: str | None) -> dict:
    return enrich_params(
        "start_training",
        _training_params(),
        user_id=OWNER_ID,
        user_token="credential-sentinel",
        goal={},
        world_model={},
        action_id=action_id,
    )


def test_enrich_params_uses_only_trusted_action_identity():
    first = _enrich(action_id="plan:plan-a:step:0")
    retry = _enrich(action_id="plan:plan-a:step:0")
    next_action = _enrich(action_id="plan:plan-a:step:1")
    missing_identity = _enrich(action_id=None)

    assert first["idempotency_key"] == retry["idempotency_key"]
    assert first["idempotency_key"] != next_action["idempotency_key"]
    assert first["idempotency_key"].startswith("hagent-")
    assert "plan-a" not in first["idempotency_key"]
    assert "credential-sentinel" not in first["idempotency_key"]
    assert "model-supplied-key" not in first["idempotency_key"]
    assert "idempotency_key" not in missing_identity


@pytest.mark.asyncio
async def test_runtime_digest_invokes_the_registered_langchain_tool(monkeypatch):
    api_post = AsyncMock(return_value={"status": "success", "job_id": "job-1"})
    monkeypatch.setattr(automl_tools, "_api_post", api_post)
    params = _enrich(action_id="plan:plan-a:step:0")

    result = await invoke_tool("start_training", params)

    assert result == {"status": "success", "job_id": "job-1"}
    api_post.assert_awaited_once()
    assert api_post.await_args.kwargs["idempotency_key"] == params["idempotency_key"]


def _plan_state(plan_id: str) -> dict:
    return {
        "messages": [],
        "user_id": OWNER_ID,
        "user_token": "credential-sentinel",
        "world_model": {
            "user_id": OWNER_ID,
            "datasets": {
                DATASET_ID: {
                    "id": DATASET_ID,
                    "features": ["feature", "label"],
                }
            },
            "jobs": {},
            "active_dataset_id": DATASET_ID,
        },
        "goal": {
            "goal_type": "train",
            "dataset_id": DATASET_ID,
            "problem_type": "classification",
            "target_column": "label",
        },
        "selected_plan": {
            "plan_id": plan_id,
            "steps": [
                {"action": {"type": "start_training", "params": _training_params()}}
            ],
        },
        "plan_status": "ready",
        "plan_step_index": 0,
        "revision_count": 0,
        "execution_log": [],
        "execution_events": [],
        "cost_metrics": {},
    }


@pytest.mark.asyncio
async def test_plan_executor_keeps_digest_per_plan_step(monkeypatch):
    invoke = AsyncMock(return_value={"status": "success", "job_id": "job-1"})
    monkeypatch.setattr("hagent.agent.execution.plan_executor.invoke_tool", invoke)

    first_output = await plan_executor_node(_plan_state("plan-a"))
    await plan_executor_node(_plan_state("plan-a"))
    await plan_executor_node(_plan_state("plan-b"))

    keys = [
        call.args[1]["idempotency_key"]
        for call in invoke.await_args_list
        if call.args[0] == "start_training"
    ]
    assert keys[0] == keys[1]
    assert keys[0] != keys[2]
    assert keys[0] not in str(first_output)


@pytest.mark.asyncio
async def test_plan_executor_without_plan_identity_fails_closed(monkeypatch):
    invoke = AsyncMock()
    monkeypatch.setattr("hagent.agent.execution.plan_executor.invoke_tool", invoke)
    state = _plan_state("plan-a")
    state["selected_plan"].pop("plan_id")

    output = await plan_executor_node(state)

    assert output["plan_status"] == "need_revise"
    assert output["last_step_error"] == "training action identity required"
    invoke.assert_not_awaited()


def _campaign(campaign_id: str, variant_id: str) -> Campaign:
    return Campaign(
        campaign_id=campaign_id,
        goal={
            "dataset_id": DATASET_ID,
            "problem_type": "classification",
            "target_column": "label",
        },
        variants=[CampaignVariant(variant_id, "candidate", _training_params())],
        status="submitting",
        max_concurrent=1,
    )


@pytest.mark.asyncio
async def test_campaign_keeps_digest_per_variant(monkeypatch):
    invoke = AsyncMock(return_value={"status": "success", "job_id": "job-1"})
    monkeypatch.setattr("hagent.agent.campaign.runner.invoke_tool", invoke)

    for campaign in (
        _campaign("campaign-a", "variant-a"),
        deepcopy(_campaign("campaign-a", "variant-a")),
        _campaign("campaign-a", "variant-b"),
    ):
        await campaign_step(
            campaign,
            user_id=OWNER_ID,
            user_token="credential-sentinel",
            world_model={},
            outcome_model=None,
        )

    keys = [
        call.args[1]["idempotency_key"]
        for call in invoke.await_args_list
        if call.args[0] == "start_training"
    ]
    assert keys[0] == keys[1]
    assert keys[0] != keys[2]
    for campaign in (
        _campaign("campaign-a", "variant-a"),
        _campaign("campaign-a", "variant-b"),
    ):
        await campaign_step(
            campaign,
            user_id=OWNER_ID,
            user_token="credential-sentinel",
            world_model={},
            outcome_model=None,
        )
        assert "idempotency_key" not in campaign.variants[0].params


@pytest.mark.asyncio
async def test_campaign_without_variant_identity_fails_closed(monkeypatch):
    invoke = AsyncMock()
    monkeypatch.setattr("hagent.agent.campaign.runner.invoke_tool", invoke)
    campaign = _campaign("campaign-a", "")

    await campaign_step(
        campaign,
        user_id=OWNER_ID,
        user_token="credential-sentinel",
        world_model={},
        outcome_model=None,
    )

    assert campaign.variants[0].status == "failed"
    assert "trusted action identity" in str(campaign.variants[0].error)
    invoke.assert_not_awaited()
