"""Regression qua AgentRuntime cho EvaluationReport và ReleaseCandidate."""

from __future__ import annotations

import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from hagent.agent.capabilities.catalog import CapabilityCatalog
from hagent.agent.capabilities.models import CapabilityDescriptor
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


class _EvaluationAdapter:
    def __init__(
        self,
        *,
        result: dict,
        regression: bool = False,
    ) -> None:
        self.result = result
        self.regression = regression
        self.start_calls: list[dict] = []
        self.result_calls: list[dict] = []

    async def invoke(self, capability_id, arguments, *, scope):
        if capability_id == "automl.dataset.inspect@1":
            return {
                "_id": arguments["dataset_id"],
                "columns": ["feature", "target"],
                "target": "target",
                "missingness": {},
                "class_balance": {} if self.regression else {"0": 0.5, "1": 0.5},
                "leakage_risks": [],
            }
        if capability_id == TRAINING_START_CAPABILITY_ID:
            self.start_calls.append(dict(arguments))
            return {
                "status": "success",
                "job_id": "job-1",
                "dispatch_status": "sent",
                "replayed": False,
            }
        if capability_id == TRAINING_LOOKUP_CAPABILITY_ID:
            raise AssertionError("Không được lookup khi submit đã xác nhận thành công")
        if capability_id == TRAINING_RESULTS_CAPABILITY_ID:
            self.result_calls.append(dict(arguments))
            output = dict(self.result)
            if output.pop("reflect_credential", False):
                output["unsafe_echo"] = scope.credential
            return output
        raise AssertionError(f"Capability không mong đợi: {capability_id}")


class _RiskCritic:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def __call__(self, payload):
        self.calls.append(dict(payload))
        return {"verdict": "ready", "reasons": ["LLM muốn override blocker"]}


def _descriptor(capability_id: str, *, effect: str, scope: str):
    return CapabilityDescriptor(
        id=capability_id,
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        effect=effect,
        required_scopes=frozenset({scope}),
        provider_id="journey-evaluation-fake",
    )


def _snapshot(adapter: _EvaluationAdapter, *, include_results: bool = True):
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
    ]
    if include_results:
        descriptors.append(
            _descriptor(
                TRAINING_RESULTS_CAPABILITY_ID,
                effect="read",
                scope="automl.training.read",
            )
        )
    catalog = CapabilityCatalog()
    catalog.register_provider("journey-evaluation-fake", descriptors, adapter)
    return catalog.snapshot()


def _scope(*, credential: str = "evaluation-credential", critic=None):
    services = {
        "scopes": (
            "automl.dataset.read",
            "automl.training.read",
            "automl.training.write",
        ),
        "max_training_jobs": 3,
    }
    if critic is not None:
        services["evaluation_critic"] = critic
    return RequestScope(
        principal_id="owner-1",
        credential=credential,
        services=services,
    )


async def _collect(stream):
    return [event async for event in stream]


async def _run_approved(
    runtime: JourneyRuntime,
    *,
    suffix: str,
    message: str = "Train dataset dataset-1 target target",
    scope: RequestScope | None = None,
):
    request_scope = scope or _scope()
    start = StartTurn(
        message=message,
        run_id=f"evaluation-{suffix}-run",
        command_id=f"evaluation-{suffix}-start",
    )
    initial = await _collect(runtime.dispatch(start, scope=request_scope))
    assert isinstance(initial[-1], ApprovalRequired)
    resolved = await _collect(
        runtime.dispatch(
            ResolveApproval(
                run_id=start.run_id,
                approval_id=initial[-1].approval_id,
                approved=True,
                command_id=f"evaluation-{suffix}-approve",
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


def _classification_result(**overrides):
    result = {
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
    result.update(overrides)
    return result


@pytest.mark.asyncio
async def test_maximize_metric_produces_ready_release_candidate():
    adapter = _EvaluationAdapter(result=_classification_result())
    runtime = JourneyRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=InMemorySaver(),
    )

    _, initial, resolved = await _run_approved(runtime, suffix="maximize")

    report = _artifact(resolved, "EvaluationReport")
    release = _artifact(resolved, "ReleaseCandidate")
    replayed = await _collect(
        runtime.replay("evaluation-maximize-run", after_sequence=0, scope=_scope())
    )
    assert report["metric_direction"] == "maximize"
    assert report["baseline_delta"] == pytest.approx(0.12)
    assert report["cv_mean"] == pytest.approx(0.82)
    assert report["variance"] == pytest.approx(0.00026666666666666666)
    assert release["readiness_verdict"] == "ready"
    assert release["lineage"] == [report["artifact_id"]]
    assert isinstance(resolved[-1], RunCompleted)
    assert resolved[-1].result["status"] == "release_ready"
    assert replayed == initial + resolved
    assert [event.sequence for event in replayed] == list(range(1, len(replayed) + 1))
    training_action_index = next(
        index
        for index, event in enumerate(resolved)
        if isinstance(event, ActionCompleted)
    )
    evaluation_index = next(
        index
        for index, event in enumerate(resolved)
        if isinstance(event, ArtifactProduced)
        and event.artifact_type == "EvaluationReport"
    )
    release_index = next(
        index
        for index, event in enumerate(resolved)
        if isinstance(event, ArtifactProduced)
        and event.artifact_type == "ReleaseCandidate"
    )
    checker_indices = [
        index
        for index, event in enumerate(resolved)
        if isinstance(event, CheckCompleted)
    ]
    assert training_action_index < evaluation_index
    assert len(checker_indices) == 3
    assert max(checker_indices) < release_index
    assert len(adapter.start_calls) == 1
    assert adapter.result_calls == [{"job_ids": ["job-1"]}]


@pytest.mark.asyncio
async def test_minimize_metric_uses_direction_aware_delta_and_overfit_gap():
    adapter = _EvaluationAdapter(
        regression=True,
        result={
            "status": "completed",
            "job_id": "job-1",
            "metric": "rmse",
            "metric_value": 0.8,
            "baseline_value": 1.0,
            "cv_scores": [0.79, 0.80, 0.81],
            "train_metric": 0.74,
            "calibration_error": None,
            "model_version": "model-v2",
            "input_schema": {"feature": "number"},
            "decision_threshold": None,
        },
    )
    runtime = JourneyRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=InMemorySaver(),
    )

    _, _, resolved = await _run_approved(
        runtime,
        suffix="minimize",
        message="Train dataset dataset-1 target target metric rmse",
    )

    report = _artifact(resolved, "EvaluationReport")
    release = _artifact(resolved, "ReleaseCandidate")
    assert report["metric_direction"] == "minimize"
    assert report["baseline_delta"] == pytest.approx(0.2)
    assert report["overfit_gap"] == pytest.approx(0.06)
    assert release["readiness_verdict"] == "ready"


@pytest.mark.asyncio
async def test_high_risk_checker_blockers_reject_even_when_critic_says_ready():
    credential = "risk-critic-secret-sentinel"
    critic = _RiskCritic()
    adapter = _EvaluationAdapter(
        result=_classification_result(
            cv_scores=[0.40, 0.80, 1.20],
            train_metric=0.98,
        )
    )
    runtime = JourneyRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=InMemorySaver(),
    )

    _, _, resolved = await _run_approved(
        runtime,
        suffix="risk",
        scope=_scope(credential=credential, critic=critic),
    )

    report = _artifact(resolved, "EvaluationReport")
    release = _artifact(resolved, "ReleaseCandidate")
    findings = [
        finding["code"]
        for event in resolved
        if isinstance(event, CheckCompleted)
        for finding in event.details["findings"]
    ]
    assert report["variance"] > 0.05
    assert report["overfit_gap"] > 0.1
    assert {"HIGH_VARIANCE", "OVERFIT_RISK"}.issubset(findings)
    assert release["readiness_verdict"] == "rejected"
    assert resolved[-1].result["status"] == "evaluation_rejected"
    assert len(critic.calls) == 1
    assert critic.calls[0]["deterministic_blocked"] is True
    assert credential not in json.dumps(critic.calls)


@pytest.mark.asyncio
async def test_pending_training_result_stops_without_evaluation_artifact():
    adapter = _EvaluationAdapter(result={"status": "running"})
    runtime = JourneyRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=InMemorySaver(),
    )

    _, _, resolved = await _run_approved(runtime, suffix="pending")

    artifact_types = [
        event.artifact_type for event in resolved if isinstance(event, ArtifactProduced)
    ]
    assert artifact_types == ["TrainingRunSet"]
    assert resolved[-1].result["status"] == "evaluation_pending"
    assert len(adapter.start_calls) == 1
    assert len(adapter.result_calls) == 1


@pytest.mark.asyncio
async def test_metric_mismatch_and_credential_reflection_fail_closed():
    credential = "evaluation-secret-sentinel"
    adapter = _EvaluationAdapter(
        result=_classification_result(
            metric="rmse",
            reflect_credential=True,
        )
    )
    runtime = JourneyRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=InMemorySaver(),
    )

    start, initial, resolved = await _run_approved(
        runtime,
        suffix="invalid-evidence",
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
        and event.artifact_type in {"EvaluationReport", "ReleaseCandidate"}
        for event in resolved
    )
    assert resolved[-1].result["status"] == "evaluation_failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("suffix", "overrides"),
    [
        ("short-cv", {"cv_scores": [0.8]}),
        ("non-finite", {"metric_value": float("nan")}),
        ("unsafe-model", {"model_version": "../model"}),
        ("non-json-schema", {"input_schema": {"feature": object()}}),
    ],
)
async def test_invalid_provider_evidence_fails_before_report(suffix, overrides):
    adapter = _EvaluationAdapter(result=_classification_result(**overrides))
    runtime = JourneyRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=InMemorySaver(),
    )

    _, _, resolved = await _run_approved(runtime, suffix=suffix)

    assert not any(
        isinstance(event, ArtifactProduced)
        and event.artifact_type in {"EvaluationReport", "ReleaseCandidate"}
        for event in resolved
    )
    assert resolved[-1].result["status"] == "evaluation_failed"
    assert resolved[-1].result["error_code"] in {
        "INVALID_EVALUATION_EVIDENCE",
        "INVALID_OUTPUT",
    }


@pytest.mark.asyncio
async def test_restart_with_results_provider_disabled_fails_before_training():
    adapter = _EvaluationAdapter(result=_classification_result())
    checkpointer = InMemorySaver()
    event_store = InMemoryRuntimeEventStore()
    evaluation_runtime = JourneyRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=checkpointer,
        event_store=event_store,
    )
    start = StartTurn(
        message="Train dataset dataset-1 target target",
        run_id="evaluation-provider-change-run",
        command_id="evaluation-provider-change-start",
    )
    initial = await _collect(evaluation_runtime.dispatch(start, scope=_scope()))
    training_runtime = JourneyRuntime(
        capability_snapshot=_snapshot(adapter, include_results=False),
        checkpointer=checkpointer,
        event_store=event_store,
    )

    resolved = await _collect(
        training_runtime.dispatch(
            ResolveApproval(
                run_id=start.run_id,
                approval_id=initial[-1].approval_id,
                approved=True,
                command_id="evaluation-provider-change-approve",
            ),
            scope=_scope(),
        )
    )

    assert isinstance(resolved[-1], RunFailed)
    assert resolved[-1].error_code == "CAPABILITY_SNAPSHOT_MISMATCH"
    assert adapter.start_calls == []
    assert adapter.result_calls == []
