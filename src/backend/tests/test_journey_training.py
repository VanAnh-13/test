"""Regression cho training dispatch sau approval trong AutoML Journey."""

from __future__ import annotations

import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from hagent.agent.capabilities.catalog import CapabilityCatalog
from hagent.agent.capabilities.models import CapabilityDescriptor
from hagent.agent.journey.runtime_adapter import JourneyRuntime
from hagent.agent.journey.training_operator import (
    TRAINING_LOOKUP_CAPABILITY_ID,
    TRAINING_START_CAPABILITY_ID,
)
from hagent.agent.runtime import (
    ActionCompleted,
    ApprovalRequired,
    ArtifactProduced,
    InMemoryRuntimeEventStore,
    RequestScope,
    ResolveApproval,
    RunCompleted,
    RunFailed,
    StartTurn,
)


class _JourneyTrainingAdapter:
    def __init__(self, *, start_mode: str = "success") -> None:
        self.start_mode = start_mode
        self.inspect_calls = 0
        self.start_calls: list[dict] = []
        self.lookup_calls: list[dict] = []

    async def invoke(self, capability_id, arguments, *, scope):
        if capability_id == "automl.dataset.inspect@1":
            self.inspect_calls += 1
            return {
                "_id": arguments["dataset_id"],
                "columns": ["feature", "target"],
                "target": "target",
                "missingness": {},
                "class_balance": {"0": 0.5, "1": 0.5},
                "leakage_risks": [],
            }
        if capability_id == TRAINING_START_CAPABILITY_ID:
            self.start_calls.append(dict(arguments))
            if self.start_mode.startswith("timeout"):
                raise TimeoutError("response lost")
            if self.start_mode == "invalid_output_found":
                return {
                    "status": "success",
                    "job_id": "job-1",
                    "unsafe_echo": scope.credential,
                }
            return {
                "status": "success",
                "job_id": "job-1",
                "replayed": False,
                "dispatch_status": "sent",
                "cost": 1.25,
            }
        if capability_id == TRAINING_LOOKUP_CAPABILITY_ID:
            self.lookup_calls.append(dict(arguments))
            if self.start_mode in {"invalid_output_found", "timeout_found"}:
                return {
                    "found": True,
                    "job_id": "job-1",
                    "dispatch_status": "sent",
                    "cost": 1.25,
                }
            return {"found": False}
        raise AssertionError(f"Capability không mong đợi: {capability_id}")


def _descriptor(capability_id: str, *, effect: str, required: list[str]):
    properties = {
        field: {"type": "array" if field in {"models", "list_feature"} else "string"}
        for field in required
    }
    for field in ("models", "list_feature"):
        if field in properties:
            properties[field]["items"] = {"type": "string"}
    if "time_limit" in properties:
        properties["time_limit"] = {"type": "integer"}
    return CapabilityDescriptor(
        id=capability_id,
        input_schema={
            "type": "object",
            "required": required,
            "properties": properties,
            "additionalProperties": False,
        },
        output_schema={"type": "object"},
        effect=effect,
        required_scopes=frozenset(
            {
                "automl.training.write"
                if capability_id == TRAINING_START_CAPABILITY_ID
                else "automl.training.read"
            }
        ),
        provider_id="journey-training-fake",
    )


def _snapshot(adapter: _JourneyTrainingAdapter, *, include_training: bool = True):
    inspect = CapabilityDescriptor(
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
        provider_id="journey-training-fake",
    )
    start = _descriptor(
        TRAINING_START_CAPABILITY_ID,
        effect="write",
        required=[
            "dataset_id",
            "problem_type",
            "target_column",
            "metric",
            "models",
            "time_limit",
            "list_feature",
            "idempotency_key",
        ],
    )
    lookup = _descriptor(
        TRAINING_LOOKUP_CAPABILITY_ID,
        effect="read",
        required=["idempotency_key"],
    )
    catalog = CapabilityCatalog()
    descriptors = [inspect, start, lookup] if include_training else [inspect]
    catalog.register_provider(
        "journey-training-fake",
        descriptors,
        adapter,
    )
    return catalog.snapshot()


def _scope(*, owner: str = "owner-1", credential: str = "training-secret"):
    return RequestScope(
        principal_id=owner,
        credential=credential,
        services={
            "scopes": (
                "automl.dataset.read",
                "automl.training.read",
                "automl.training.write",
            ),
            "max_training_jobs": 3,
        },
    )


async def _collect(stream):
    return [event async for event in stream]


async def _start(runtime: JourneyRuntime, *, suffix: str):
    command = StartTurn(
        message="Train dataset dataset-1 target target",
        run_id=f"training-{suffix}-run",
        command_id=f"training-{suffix}-start",
    )
    events = await _collect(runtime.dispatch(command, scope=_scope()))
    assert isinstance(events[-1], ApprovalRequired)
    return command, events


@pytest.mark.asyncio
async def test_approved_spec_dispatches_once_and_replays_training_artifact():
    adapter = _JourneyTrainingAdapter()
    runtime = JourneyRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=InMemorySaver(),
    )
    start, initial = await _start(runtime, suffix="success")
    approval = ResolveApproval(
        run_id=start.run_id,
        approval_id=initial[-1].approval_id,
        approved=True,
        command_id="training-success-approve",
    )

    resolved = await _collect(runtime.dispatch(approval, scope=_scope()))
    duplicate = await _collect(runtime.dispatch(approval, scope=_scope()))
    replayed = await _collect(
        runtime.replay(start.run_id, after_sequence=0, scope=_scope())
    )

    training_artifacts = [
        event
        for event in resolved
        if isinstance(event, ArtifactProduced)
        and event.artifact_type == "TrainingRunSet"
    ]
    assert training_artifacts, resolved
    artifact_event = training_artifacts[0]
    action_event = next(
        event for event in resolved if isinstance(event, ActionCompleted)
    )
    assert adapter.inspect_calls == 1
    assert len(adapter.start_calls) == 1
    assert adapter.lookup_calls == []
    assert resolved == duplicate
    assert replayed == initial + resolved
    assert [event.sequence for event in replayed] == list(range(1, len(replayed) + 1))
    assert artifact_event.artifact["job_ids"] == ["job-1"]
    assert artifact_event.artifact["reconciliation_status"] == "not_required"
    assert artifact_event.artifact["lineage"]
    assert action_event.action == "automl.training.start"
    assert action_event.outcome == "submitted"
    assert isinstance(resolved[-1], RunCompleted)
    assert resolved[-1].result["status"] == "training_dispatched"

    call = adapter.start_calls[0]
    assert call["dataset_id"] == "dataset-1"
    assert call["target_column"] == "target"
    assert call["list_feature"] == ["feature"]
    assert len(call["idempotency_key"]) == 64
    assert not ({"user_id", "token", "credential"} & set(call))


@pytest.mark.asyncio
async def test_reject_finishes_without_training_mutation():
    adapter = _JourneyTrainingAdapter()
    runtime = JourneyRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=InMemorySaver(),
    )
    start, initial = await _start(runtime, suffix="reject")

    rejected = await _collect(
        runtime.dispatch(
            ResolveApproval(
                run_id=start.run_id,
                approval_id=initial[-1].approval_id,
                approved=False,
                command_id="training-reject-approval",
            ),
            scope=_scope(),
        )
    )

    assert rejected[-1].result["status"] == "rejected"
    assert adapter.start_calls == []
    assert adapter.lookup_calls == []


@pytest.mark.asyncio
async def test_lost_submit_response_reconciles_by_same_digest_without_resubmit():
    adapter = _JourneyTrainingAdapter(start_mode="timeout_found")
    runtime = JourneyRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=InMemorySaver(),
    )
    start, initial = await _start(runtime, suffix="reconciled")

    resolved = await _collect(
        runtime.dispatch(
            ResolveApproval(
                run_id=start.run_id,
                approval_id=initial[-1].approval_id,
                approved=True,
                command_id="training-reconciled-approval",
            ),
            scope=_scope(),
        )
    )

    artifact = next(
        event.artifact
        for event in resolved
        if isinstance(event, ArtifactProduced)
        and event.artifact_type == "TrainingRunSet"
    )
    assert len(adapter.start_calls) == 1
    assert len(adapter.lookup_calls) == 1
    assert (
        adapter.start_calls[0]["idempotency_key"]
        == adapter.lookup_calls[0]["idempotency_key"]
    )
    assert artifact["reconciliation_status"] == "reconciled"
    assert artifact["job_ids"] == ["job-1"]
    assert resolved[-1].result["status"] == "training_dispatched"


@pytest.mark.asyncio
async def test_unknown_submit_outcome_is_terminal_reconciliation_without_retry():
    adapter = _JourneyTrainingAdapter(start_mode="timeout_unknown")
    runtime = JourneyRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=InMemorySaver(),
    )
    start, initial = await _start(runtime, suffix="unknown")
    approval = ResolveApproval(
        run_id=start.run_id,
        approval_id=initial[-1].approval_id,
        approved=True,
        command_id="training-unknown-approval",
    )

    resolved = await _collect(runtime.dispatch(approval, scope=_scope()))
    duplicate = await _collect(runtime.dispatch(approval, scope=_scope()))

    action = next(event for event in resolved if isinstance(event, ActionCompleted))
    assert len(adapter.start_calls) == 1
    assert len(adapter.lookup_calls) == 1
    assert resolved == duplicate
    assert action.outcome == "needs_reconciliation"
    assert resolved[-1].result["status"] == "needs_reconciliation"


@pytest.mark.asyncio
async def test_invalid_output_after_submit_uses_lookup_without_resubmit():
    adapter = _JourneyTrainingAdapter(start_mode="invalid_output_found")
    runtime = JourneyRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=InMemorySaver(),
    )
    start, initial = await _start(runtime, suffix="invalid-output")

    resolved = await _collect(
        runtime.dispatch(
            ResolveApproval(
                run_id=start.run_id,
                approval_id=initial[-1].approval_id,
                approved=True,
                command_id="training-invalid-output-approval",
            ),
            scope=_scope(),
        )
    )

    assert len(adapter.start_calls) == 1
    assert len(adapter.lookup_calls) == 1
    assert (
        adapter.start_calls[0]["idempotency_key"]
        == adapter.lookup_calls[0]["idempotency_key"]
    )
    assert resolved[-1].result["status"] == "training_dispatched"


@pytest.mark.asyncio
async def test_restart_with_different_capability_snapshot_fails_before_mutation():
    adapter = _JourneyTrainingAdapter()
    checkpointer = InMemorySaver()
    event_store = InMemoryRuntimeEventStore()
    experiment_runtime = JourneyRuntime(
        capability_snapshot=_snapshot(adapter, include_training=False),
        checkpointer=checkpointer,
        event_store=event_store,
    )
    start, initial = await _start(experiment_runtime, suffix="snapshot-change")
    training_runtime = JourneyRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=checkpointer,
        event_store=event_store,
    )

    resolved = await _collect(
        training_runtime.dispatch(
            ResolveApproval(
                run_id=start.run_id,
                approval_id=initial[-1].approval_id,
                approved=True,
                command_id="training-snapshot-change-approval",
            ),
            scope=_scope(),
        )
    )

    assert isinstance(resolved[-1], RunFailed)
    assert resolved[-1].error_code == "CAPABILITY_SNAPSHOT_MISMATCH"
    assert adapter.start_calls == []
    assert adapter.lookup_calls == []


@pytest.mark.asyncio
async def test_restart_with_training_provider_disabled_fails_before_resume():
    adapter = _JourneyTrainingAdapter()
    checkpointer = InMemorySaver()
    event_store = InMemoryRuntimeEventStore()
    training_runtime = JourneyRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=checkpointer,
        event_store=event_store,
    )
    start, initial = await _start(training_runtime, suffix="provider-disabled")
    experiment_runtime = JourneyRuntime(
        capability_snapshot=_snapshot(adapter, include_training=False),
        checkpointer=checkpointer,
        event_store=event_store,
    )

    resolved = await _collect(
        experiment_runtime.dispatch(
            ResolveApproval(
                run_id=start.run_id,
                approval_id=initial[-1].approval_id,
                approved=True,
                command_id="training-provider-disabled-approval",
            ),
            scope=_scope(),
        )
    )

    assert isinstance(resolved[-1], RunFailed)
    assert resolved[-1].error_code == "CAPABILITY_SNAPSHOT_MISMATCH"
    assert adapter.start_calls == []
    assert adapter.lookup_calls == []


@pytest.mark.asyncio
async def test_training_credential_never_enters_checkpoint_or_runtime_events():
    credential = "journey-training-credential-sentinel"
    adapter = _JourneyTrainingAdapter()
    runtime = JourneyRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=InMemorySaver(),
    )
    scope = _scope(credential=credential)
    command = StartTurn(
        message="Train dataset dataset-1 target target",
        run_id="training-secret-run",
        command_id="training-secret-start",
    )
    initial = await _collect(runtime.dispatch(command, scope=scope))
    resolved = await _collect(
        runtime.dispatch(
            ResolveApproval(
                run_id=command.run_id,
                approval_id=initial[-1].approval_id,
                approved=True,
                command_id="training-secret-approval",
            ),
            scope=scope,
        )
    )
    checkpoint = await runtime.get_checkpoint_state(run_id=command.run_id, scope=scope)

    serialized = json.dumps(
        {
            "events": [
                event.__dict__ if hasattr(event, "__dict__") else repr(event)
                for event in initial + resolved
            ],
            "checkpoint": checkpoint,
        },
        default=str,
    )
    assert credential not in serialized
    assert credential not in repr(adapter.start_calls)
    assert credential not in repr(adapter.lookup_calls)
