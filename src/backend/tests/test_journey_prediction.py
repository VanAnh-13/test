"""Regression qua AgentRuntime cho prediction có schema gate và provenance."""

from __future__ import annotations

import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from hagent.agent.capabilities.catalog import CapabilityCatalog
from hagent.agent.capabilities.models import CapabilityDescriptor
from hagent.agent.journey.prediction_operator import (
    PREDICTION_INPUT_INSPECT_CAPABILITY_ID,
    PREDICTION_WRITE_CAPABILITY_ID,
)
from hagent.agent.journey.result_critic import TRAINING_RESULTS_CAPABILITY_ID
from hagent.agent.journey.runtime_adapter import JourneyRuntime
from hagent.agent.journey.training_operator import (
    TRAINING_LOOKUP_CAPABILITY_ID,
    TRAINING_START_CAPABILITY_ID,
)
from hagent.agent.runtime import (
    ActionCompleted,
    ApprovalRequired,
    ArtifactProduced,
    CheckCompleted,
    InMemoryRuntimeEventStore,
    RequestScope,
    ResolveApproval,
    RunCompleted,
    RunFailed,
    StartTurn,
)

_INPUT_HASH = "a" * 64


class _PredictionAdapter:
    def __init__(
        self,
        *,
        input_schema: dict | None = None,
        prediction_output: dict | None = None,
    ) -> None:
        self.input_schema = input_schema or {"feature": "number"}
        self.prediction_output = prediction_output or {
            "status": "success",
            "prediction_id": "prediction-1",
            "result_uri": "artifact://prediction-1",
            "row_errors": {"4": "invalid numeric value"},
        }
        self.training_calls: list[dict] = []
        self.result_calls: list[dict] = []
        self.inspect_calls: list[dict] = []
        self.prediction_calls: list[dict] = []

    async def invoke(self, capability_id, arguments, *, scope):
        if capability_id == "automl.dataset.inspect@1":
            return {
                "_id": arguments["dataset_id"],
                "columns": ["feature", "target"],
                "target": "target",
                "missingness": {},
                "class_balance": {"0": 0.5, "1": 0.5},
                "leakage_risks": [],
            }
        if capability_id == TRAINING_START_CAPABILITY_ID:
            self.training_calls.append(dict(arguments))
            return {
                "status": "success",
                "job_id": "job-1",
                "dispatch_status": "sent",
                "replayed": False,
            }
        if capability_id == TRAINING_LOOKUP_CAPABILITY_ID:
            raise AssertionError("Không lookup khi training đã xác nhận thành công")
        if capability_id == TRAINING_RESULTS_CAPABILITY_ID:
            self.result_calls.append(dict(arguments))
            return {
                "status": "completed",
                "job_id": "job-1",
                "metric": "accuracy",
                "metric_value": 0.82,
                "baseline_value": 0.70,
                "cv_scores": [0.80, 0.82, 0.84],
                "train_metric": 0.85,
                "calibration_error": 0.03,
                "model_version": "model-v1",
                "input_schema": {"feature": "number"},
                "decision_threshold": 0.5,
            }
        if capability_id == PREDICTION_INPUT_INSPECT_CAPABILITY_ID:
            self.inspect_calls.append(dict(arguments))
            return {
                "input_artifact_id": arguments["input_artifact_id"],
                "content_hash": _INPUT_HASH,
                "schema": self.input_schema,
                "row_count": 3,
            }
        if capability_id == PREDICTION_WRITE_CAPABILITY_ID:
            self.prediction_calls.append(dict(arguments))
            output = dict(self.prediction_output)
            if output.pop("reflect_credential", False):
                output["unsafe_echo"] = scope.credential
            return output
        raise AssertionError(f"Capability không mong đợi: {capability_id}")


def _descriptor(capability_id: str, *, effect: str, scope: str):
    return CapabilityDescriptor(
        id=capability_id,
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        effect=effect,
        required_scopes=frozenset({scope}),
        provider_id="journey-prediction-fake",
    )


def _snapshot(adapter: _PredictionAdapter, *, include_prediction: bool = True):
    descriptors = [
        _descriptor(
            "automl.dataset.inspect@1",
            effect="read",
            scope="automl.dataset.read",
        ),
        _descriptor(
            TRAINING_START_CAPABILITY_ID,
            effect="write",
            scope="automl.training.write",
        ),
        _descriptor(
            TRAINING_LOOKUP_CAPABILITY_ID,
            effect="read",
            scope="automl.training.read",
        ),
        _descriptor(
            TRAINING_RESULTS_CAPABILITY_ID,
            effect="read",
            scope="automl.training.read",
        ),
    ]
    if include_prediction:
        descriptors.extend(
            [
                _descriptor(
                    PREDICTION_INPUT_INSPECT_CAPABILITY_ID,
                    effect="read",
                    scope="automl.prediction.read",
                ),
                _descriptor(
                    PREDICTION_WRITE_CAPABILITY_ID,
                    effect="write",
                    scope="automl.prediction.write",
                ),
            ]
        )
    catalog = CapabilityCatalog()
    catalog.register_provider("journey-prediction-fake", descriptors, adapter)
    return catalog.snapshot()


def _scope(
    *,
    credential: str = "prediction-credential",
    prediction_write: bool = True,
):
    scopes = [
        "automl.dataset.read",
        "automl.training.read",
        "automl.training.write",
        "automl.prediction.read",
    ]
    if prediction_write:
        scopes.append("automl.prediction.write")
    return RequestScope(
        principal_id="owner-1",
        credential=credential,
        services={"scopes": tuple(scopes), "max_training_jobs": 3},
    )


async def _collect(stream):
    return [event async for event in stream]


async def _run_approved(
    runtime: JourneyRuntime,
    *,
    suffix: str,
    message: str,
    scope: RequestScope | None = None,
):
    request_scope = scope or _scope()
    start = StartTurn(
        message=message,
        run_id=f"prediction-{suffix}-run",
        command_id=f"prediction-{suffix}-start",
    )
    initial = await _collect(runtime.dispatch(start, scope=request_scope))
    assert isinstance(initial[-1], ApprovalRequired)
    resolved = await _collect(
        runtime.dispatch(
            ResolveApproval(
                run_id=start.run_id,
                approval_id=initial[-1].approval_id,
                approved=True,
                command_id=f"prediction-{suffix}-approve",
            ),
            scope=request_scope,
        )
    )
    return start, initial, resolved


def _artifact(events, artifact_type: str):
    return next(
        event.artifact
        for event in events
        if isinstance(event, ArtifactProduced) and event.artifact_type == artifact_type
    )


@pytest.mark.asyncio
async def test_explicit_prediction_creates_provenance_artifact_after_schema_gate():
    adapter = _PredictionAdapter()
    runtime = JourneyRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=InMemorySaver(),
    )

    _, initial, resolved = await _run_approved(
        runtime,
        suffix="happy",
        message=(
            "Train dataset dataset-1 target target, then predict input artifact input-1"
        ),
    )

    release = _artifact(resolved, "ReleaseCandidate")
    prediction = _artifact(resolved, "PredictionArtifact")
    replayed = await _collect(
        runtime.replay("prediction-happy-run", after_sequence=0, scope=_scope())
    )
    actions = [event.action for event in resolved if isinstance(event, ActionCompleted)]
    assert adapter.inspect_calls == [{"input_artifact_id": "input-1"}]
    assert len(adapter.prediction_calls) == 1
    assert adapter.prediction_calls[0]["release_candidate_id"] == release["artifact_id"]
    assert adapter.prediction_calls[0]["input_artifact_id"] == "input-1"
    assert len(adapter.prediction_calls[0]["action_digest"]) == 64
    assert prediction["release_candidate_id"] == release["artifact_id"]
    assert prediction["status"] == "accepted"
    assert prediction["lineage"] == [release["artifact_id"]]
    assert prediction["result_uri"] == "artifact://prediction-1"
    assert prediction["row_errors"] == {"4": "invalid numeric value"}
    assert prediction["provenance"]["input_artifact_id"] == "input-1"
    assert prediction["model_input_hash"] != _INPUT_HASH
    assert actions == ["automl.training.start", "automl.prediction.batch"]
    assert [event.checker for event in resolved if isinstance(event, CheckCompleted)][
        -2:
    ] == ["contract", "policy"]
    assert isinstance(resolved[-1], RunCompleted)
    assert resolved[-1].result["status"] == "prediction_completed"
    assert replayed == initial + resolved
    assert [event.sequence for event in replayed] == list(range(1, len(replayed) + 1))


@pytest.mark.asyncio
async def test_prediction_capabilities_are_not_called_without_explicit_request():
    adapter = _PredictionAdapter()
    runtime = JourneyRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=InMemorySaver(),
    )

    _, _, resolved = await _run_approved(
        runtime,
        suffix="not-requested",
        message="Train dataset dataset-1 target target",
    )

    assert adapter.inspect_calls == []
    assert adapter.prediction_calls == []
    assert resolved[-1].result["status"] == "release_ready"


@pytest.mark.asyncio
async def test_schema_mismatch_stops_before_prediction_mutation():
    adapter = _PredictionAdapter(input_schema={"other": "number"})
    runtime = JourneyRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=InMemorySaver(),
    )

    _, _, resolved = await _run_approved(
        runtime,
        suffix="schema-mismatch",
        message="Train dataset dataset-1 target target; predict input artifact input-1",
    )

    assert adapter.inspect_calls == [{"input_artifact_id": "input-1"}]
    assert adapter.prediction_calls == []
    assert not any(
        isinstance(event, ArtifactProduced)
        and event.artifact_type == "PredictionArtifact"
        for event in resolved
    )
    assert resolved[-1].result == {
        "status": "prediction_failed",
        "error_code": "PREDICTION_SCHEMA_MISMATCH",
    }


@pytest.mark.asyncio
async def test_raw_file_path_is_not_accepted_as_prediction_artifact_reference():
    adapter = _PredictionAdapter()
    runtime = JourneyRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=InMemorySaver(),
    )

    events = await _collect(
        runtime.dispatch(
            StartTurn(
                run_id="prediction-raw-path-run",
                command_id="prediction-raw-path-start",
                message=(
                    "Train dataset dataset-1 target target; predict input artifact "
                    "C:\\temp\\input.csv"
                ),
            ),
            scope=_scope(),
        )
    )

    assert adapter.training_calls == []
    assert adapter.inspect_calls == []
    assert adapter.prediction_calls == []
    assert events[-1].result == {
        "status": "prediction_failed",
        "error_code": "PREDICTION_INPUT_REQUIRED",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("suffix", "message"),
    [
        (
            "missing-prediction",
            "Train dataset dataset-1 target target; predict input artifact input-1",
        ),
        ("deploy", "Train dataset dataset-1 target target then deploy model"),
    ],
)
async def test_missing_prediction_or_deploy_capability_is_explicitly_unavailable(
    suffix,
    message,
):
    adapter = _PredictionAdapter()
    runtime = JourneyRuntime(
        capability_snapshot=_snapshot(adapter, include_prediction=False),
        checkpointer=InMemorySaver(),
    )

    events = await _collect(
        runtime.dispatch(
            StartTurn(
                run_id=f"prediction-{suffix}-run",
                command_id=f"prediction-{suffix}-start",
                message=message,
            ),
            scope=_scope(),
        )
    )

    assert adapter.training_calls == []
    assert adapter.result_calls == []
    assert adapter.inspect_calls == []
    assert adapter.prediction_calls == []
    assert events[-1].result["status"] == "capability_unavailable"
    assert events[-1].result["error_code"] == "CAPABILITY_UNAVAILABLE"


@pytest.mark.asyncio
async def test_prediction_write_scope_is_checked_before_adapter_mutation():
    adapter = _PredictionAdapter()
    runtime = JourneyRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=InMemorySaver(),
    )

    _, _, resolved = await _run_approved(
        runtime,
        suffix="scope-denied",
        message=(
            "Train dataset dataset-1 target target; predict input artifact input-1"
        ),
        scope=_scope(prediction_write=False),
    )

    assert len(adapter.inspect_calls) == 1
    assert adapter.prediction_calls == []
    assert resolved[-1].result == {
        "status": "prediction_failed",
        "error_code": "SCOPE_DENIED",
    }


@pytest.mark.asyncio
async def test_invalid_prediction_output_never_leaks_credential_or_creates_artifact():
    credential = "prediction-secret-sentinel"
    adapter = _PredictionAdapter(
        prediction_output={
            "status": "success",
            "prediction_id": "prediction-1",
            "result_uri": "artifact://prediction-1",
            "row_errors": {},
            "reflect_credential": True,
        }
    )
    runtime = JourneyRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=InMemorySaver(),
    )

    start, initial, resolved = await _run_approved(
        runtime,
        suffix="credential",
        message="Train dataset dataset-1 target target; predict input artifact input-1",
        scope=_scope(credential=credential),
    )
    checkpoint = await runtime.get_checkpoint_state(
        run_id=start.run_id,
        scope=_scope(credential=credential),
    )

    serialized = json.dumps(
        {"initial": repr(initial), "resolved": repr(resolved), "state": checkpoint},
        default=str,
    )
    assert credential not in serialized
    assert not any(
        isinstance(event, ArtifactProduced)
        and event.artifact_type == "PredictionArtifact"
        for event in resolved
    )
    assert resolved[-1].result["status"] == "prediction_needs_reconciliation"
    assert resolved[-1].result["error_code"] == "INVALID_OUTPUT"


@pytest.mark.asyncio
async def test_prediction_provider_drift_fails_before_any_mutation_on_resume():
    adapter = _PredictionAdapter()
    checkpointer = InMemorySaver()
    event_store = InMemoryRuntimeEventStore()
    prediction_runtime = JourneyRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=checkpointer,
        event_store=event_store,
    )
    start = StartTurn(
        message="Train dataset dataset-1 target target; predict input artifact input-1",
        run_id="prediction-provider-drift-run",
        command_id="prediction-provider-drift-start",
    )
    initial = await _collect(prediction_runtime.dispatch(start, scope=_scope()))
    evaluation_runtime = JourneyRuntime(
        capability_snapshot=_snapshot(adapter, include_prediction=False),
        checkpointer=checkpointer,
        event_store=event_store,
    )

    resolved = await _collect(
        evaluation_runtime.dispatch(
            ResolveApproval(
                run_id=start.run_id,
                approval_id=initial[-1].approval_id,
                approved=True,
                command_id="prediction-provider-drift-approve",
            ),
            scope=_scope(),
        )
    )

    assert isinstance(resolved[-1], RunFailed)
    assert resolved[-1].error_code == "CAPABILITY_SNAPSHOT_MISMATCH"
    assert adapter.training_calls == []
    assert adapter.result_calls == []
    assert adapter.inspect_calls == []
    assert adapter.prediction_calls == []
