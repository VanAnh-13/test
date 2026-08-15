"""Checkpoint-shaped state tối thiểu của read-only DatasetAudit journey."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from hagent.agent.journey.artifacts import (
    DatasetAudit,
    EvaluationReport,
    ExperimentSpec,
    PredictionArtifact,
    ReleaseCandidate,
    TrainingRunSet,
)
from hagent.agent.journey.checkers import CheckerVerdict


class JourneyAuditState(TypedDict):
    message: str
    run_id: str
    capability_snapshot_digest: NotRequired[str]
    training_enabled: NotRequired[bool]
    evaluation_enabled: NotRequired[bool]
    prediction_enabled: NotRequired[bool]
    goal: NotRequired[dict[str, Any]]
    artifact: NotRequired[DatasetAudit]
    verdicts: NotRequired[tuple[CheckerVerdict, ...]]
    experiment_spec: NotRequired[ExperimentSpec]
    experiment_verdicts: NotRequired[tuple[CheckerVerdict, ...]]
    approval: NotRequired[dict[str, Any]]
    approval_decision: NotRequired[str]
    approval_response: NotRequired[dict[str, Any]]
    training_run_set: NotRequired[TrainingRunSet]
    training_outcome: NotRequired[str]
    training_error_code: NotRequired[str]
    evaluation_report: NotRequired[EvaluationReport]
    evaluation_status: NotRequired[str]
    evaluation_error_code: NotRequired[str]
    evaluation_verdicts: NotRequired[tuple[CheckerVerdict, ...]]
    release_metadata: NotRequired[dict[str, Any]]
    release_candidate: NotRequired[ReleaseCandidate]
    critic_assessment: NotRequired[dict[str, Any]]
    prediction_artifact: NotRequired[PredictionArtifact]
    prediction_verdicts: NotRequired[tuple[CheckerVerdict, ...]]
    prediction_action: NotRequired[dict[str, Any]]
    prediction_outcome: NotRequired[str]
    prediction_error_code: NotRequired[str]
    error_code: NotRequired[str]
    error_message: NotRequired[str]
    result: NotRequired[dict[str, Any]]
