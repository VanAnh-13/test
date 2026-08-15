"""Contract test cho ExperimentSpec draft và LangGraph approval interrupt."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from hagent.agent.capabilities.catalog import CapabilityCatalog
from hagent.agent.capabilities.models import CapabilityDescriptor
from hagent.agent.journey.graph import (
    build_audit_graph,
    compile_experiment_graph,
    initial_audit_state,
)
from hagent.agent.journey.persistence import journey_graph_config
from hagent.agent.runtime.context import GraphRequestContext


class _DatasetAdapter:
    def __init__(self, *, target="target", leakage=(), classification=True):
        self.target = target
        self.leakage = leakage
        self.classification = classification
        self.calls = 0

    async def invoke(self, capability_id, arguments, *, scope):
        self.calls += 1
        return {
            "_id": arguments["dataset_id"],
            "columns": ["feature", "target"],
            "target": self.target,
            "missingness": {},
            "class_balance": {"0": 0.5, "1": 0.5} if self.classification else {},
            "leakage_risks": list(self.leakage),
        }


def _snapshot(adapter):
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
        provider_id="experiment-fake",
    )
    catalog = CapabilityCatalog()
    catalog.register_provider("experiment-fake", [descriptor], adapter)
    return catalog.snapshot()


def _context(adapter, *, max_jobs=5):
    return GraphRequestContext(
        principal_id="owner-1",
        credential="experiment-sentinel",
        services={
            "scopes": ("automl.dataset.read",),
            "max_training_jobs": max_jobs,
        },
        capability_snapshot=_snapshot(adapter),
    )


async def _start(graph, message, context, run_id="experiment-run"):
    return await graph.ainvoke(
        initial_audit_state(message=message, run_id=run_id),
        config=journey_graph_config(principal_id="owner-1", run_id=run_id),
        context=context,
        durability="sync",
    )


def _proposal(result):
    interrupts = result.get("__interrupt__", ())
    assert len(interrupts) == 1
    return interrupts[0].value


@pytest.mark.asyncio
async def test_draft_defaults_are_typed_checked_and_approval_can_resume_after_rebuild():
    saver = InMemorySaver()
    adapter = _DatasetAdapter(classification=True)
    context = _context(adapter)
    graph = compile_experiment_graph(checkpointer=saver)
    config = journey_graph_config(principal_id="owner-1", run_id="experiment-run")

    interrupted = await _start(
        graph,
        "Huấn luyện model cho dataset dataset-1 target target",
        context,
    )
    proposal = _proposal(interrupted)
    spec = interrupted["experiment_spec"]

    assert spec.metric == "accuracy"
    assert spec.metric_direction == "maximize"
    assert spec.split_strategy == "stratified_holdout"
    assert spec.max_training_jobs == 3
    assert set(spec.default_reasons) == {
        "metric",
        "split_strategy",
        "max_training_jobs",
    }
    assert spec.dataset_audit_id in spec.lineage
    assert len(interrupted["experiment_verdicts"]) == 3
    assert not any(item.blocked for item in interrupted["experiment_verdicts"])
    assert adapter.calls == 1

    recreated = compile_experiment_graph(checkpointer=saver)
    approved = await recreated.ainvoke(
        Command(
            resume={
                "approval_id": proposal["approval_id"],
                "decision": "approve",
            }
        ),
        config=config,
        context=context,
        durability="sync",
    )

    assert approved["result"]["status"] == "approved"
    assert approved["result"]["artifact_id"] == spec.artifact_id
    assert adapter.calls == 1
    assert "experiment-sentinel" not in repr(list(saver.storage.values()))


@pytest.mark.asyncio
async def test_explicit_regression_metric_minimizes_and_reject_is_terminal():
    adapter = _DatasetAdapter(classification=False)
    context = _context(adapter)
    graph = compile_experiment_graph(checkpointer=InMemorySaver())
    run_id = "regression-run"
    interrupted = await _start(
        graph,
        "Train experiment dataset dataset-1 target target metric rmse budget 2 kfold",
        context,
        run_id=run_id,
    )
    proposal = _proposal(interrupted)

    assert interrupted["experiment_spec"].metric_direction == "minimize"
    assert interrupted["experiment_spec"].max_training_jobs == 2
    assert interrupted["experiment_spec"].split_strategy == "kfold"
    rejected = await graph.ainvoke(
        Command(resume={"approval_id": proposal["approval_id"], "decision": "reject"}),
        config=journey_graph_config(principal_id="owner-1", run_id=run_id),
        context=context,
        durability="sync",
    )
    assert rejected["result"]["status"] == "rejected"


@pytest.mark.asyncio
async def test_stale_or_invalid_edit_response_finishes_without_revision():
    context = _context(_DatasetAdapter())
    graph = compile_experiment_graph(checkpointer=InMemorySaver())
    run_id = "stale-run"
    interrupted = await _start(
        graph,
        "Train dataset dataset-1 target target",
        context,
        run_id=run_id,
    )
    stale = await graph.ainvoke(
        Command(resume={"approval_id": "approval-stale", "decision": "approve"}),
        config=journey_graph_config(principal_id="owner-1", run_id=run_id),
        context=context,
        durability="sync",
    )
    assert stale["result"]["status"] == "stale_approval"
    assert (
        stale["experiment_spec"].artifact_id
        == interrupted["experiment_spec"].artifact_id
    )

    invalid_graph = compile_experiment_graph(checkpointer=InMemorySaver())
    invalid_run = "invalid-edit-run"
    invalid_first = await _start(
        invalid_graph,
        "Train dataset dataset-1 target target",
        context,
        run_id=invalid_run,
    )
    invalid_proposal = _proposal(invalid_first)
    invalid = await invalid_graph.ainvoke(
        Command(
            resume={
                "approval_id": invalid_proposal["approval_id"],
                "decision": "edit",
                "changes": {"max_training_jobs": "not-an-integer"},
            }
        ),
        config=journey_graph_config(principal_id="owner-1", run_id=invalid_run),
        context=context,
        durability="sync",
    )
    assert invalid["result"]["status"] == "invalid_approval"
    assert invalid["experiment_spec"].version == 1


@pytest.mark.asyncio
async def test_expired_approval_cannot_be_resumed(monkeypatch):
    import hagent.agent.journey.graph as journey_graph

    class FutureDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2100, 1, 1, tzinfo=tz or UTC)

    context = _context(_DatasetAdapter())
    graph = compile_experiment_graph(checkpointer=InMemorySaver())
    run_id = "expired-run"
    interrupted = await _start(
        graph,
        "Train dataset dataset-1 target target",
        context,
        run_id=run_id,
    )
    proposal = _proposal(interrupted)
    monkeypatch.setattr(journey_graph, "datetime", FutureDateTime)
    expired = await graph.ainvoke(
        Command(
            resume={
                "approval_id": proposal["approval_id"],
                "decision": "approve",
            }
        ),
        config=journey_graph_config(principal_id="owner-1", run_id=run_id),
        context=context,
        durability="sync",
    )
    assert expired["result"]["status"] == "approval_expired"


@pytest.mark.asyncio
async def test_resume_payload_cannot_persist_unknown_sensitive_fields():
    saver = InMemorySaver()
    context = _context(_DatasetAdapter())
    graph = compile_experiment_graph(checkpointer=saver)
    run_id = "resume-injection-run"
    interrupted = await _start(
        graph,
        "Train dataset dataset-1 target target",
        context,
        run_id=run_id,
    )
    proposal = _proposal(interrupted)
    result = await graph.ainvoke(
        Command(
            resume={
                "approval_id": proposal["approval_id"],
                "decision": "approve",
                "token": "resume-payload-sentinel",
            }
        ),
        config=journey_graph_config(principal_id="owner-1", run_id=run_id),
        context=context,
        durability="sync",
    )

    assert result["result"]["status"] == "invalid_approval"
    assert "resume-payload-sentinel" not in repr(list(saver.storage.values()))


@pytest.mark.asyncio
async def test_edit_creates_superseding_version_and_requires_new_approval():
    context = _context(_DatasetAdapter())
    graph = compile_experiment_graph(checkpointer=InMemorySaver())
    run_id = "edit-run"
    first = await _start(
        graph,
        "Train dataset dataset-1 target target",
        context,
        run_id=run_id,
    )
    first_proposal = _proposal(first)
    first_spec = first["experiment_spec"]
    edited = await graph.ainvoke(
        Command(
            resume={
                "approval_id": first_proposal["approval_id"],
                "decision": "edit",
                "changes": {"metric": "f1", "max_training_jobs": 2},
            }
        ),
        config=journey_graph_config(principal_id="owner-1", run_id=run_id),
        context=context,
        durability="sync",
    )
    second_proposal = _proposal(edited)
    second_spec = edited["experiment_spec"]

    assert second_spec.version == 2
    assert second_spec.supersedes == first_spec.artifact_id
    assert second_spec.artifact_id != first_spec.artifact_id
    assert second_spec.metric == "f1"
    assert set(second_spec.default_reasons) == {"split_strategy"}
    assert second_proposal["approval_id"] != first_proposal["approval_id"]
    assert len(edited["experiment_verdicts"]) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("adapter", "message"),
    [
        (_DatasetAdapter(target=None), "Train dataset dataset-1"),
        (
            _DatasetAdapter(leakage=("target_copy",)),
            "Huấn luyện dataset dataset-1 target target",
        ),
    ],
)
async def test_audit_blocker_never_reaches_approval(adapter, message):
    graph = compile_experiment_graph(checkpointer=InMemorySaver())
    result = await _start(graph, message, _context(adapter), run_id=uuid_for(adapter))

    assert result["result"]["status"] == "blocked"
    assert "__interrupt__" not in result
    assert "experiment_spec" not in result
    assert adapter.calls == 1


def uuid_for(adapter):
    return "blocker-" + str(id(adapter))


@pytest.mark.asyncio
async def test_audit_only_keeps_legacy_topology_and_never_interrupts():
    assert set(build_audit_graph().nodes) == {
        "contract_checker",
        "dataset_profiler",
        "finalize",
        "interpret",
        "policy_checker",
        "statistical_checker",
    }
    result = await _start(
        compile_experiment_graph(checkpointer=InMemorySaver()),
        "Audit dataset dataset-1 target target",
        _context(_DatasetAdapter()),
        run_id="audit-only-run",
    )
    assert result["result"]["status"] == "completed"
    assert "__interrupt__" not in result
