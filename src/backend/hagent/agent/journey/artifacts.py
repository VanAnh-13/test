"""Immutable artifacts là contract bền vững giữa các stage AutoML journey."""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from hagent.agent.capabilities.models import freeze_json

ArtifactStatus = Literal["draft", "accepted", "rejected"]
MetricDirection = Literal["maximize", "minimize"]

_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_HASH_PATTERN = re.compile(r"[a-fA-F0-9]{64}")


def _new_artifact_id() -> str:
    return uuid.uuid4().hex


def _validate_id(name: str, value: str) -> None:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a safe identifier")


def _freeze_mapping(value: Mapping[str, Any], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return freeze_json(value)


def _freeze_strings(value: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a string sequence")
    frozen = tuple(value)
    if any(not isinstance(item, str) or not item for item in frozen):
        raise ValueError(f"{name} must contain non-empty strings")
    return frozen


@dataclass(frozen=True, slots=True, kw_only=True)
class EvidenceRef:
    """Tham chiếu evidence theo hash; không nhúng raw payload lớn vào artifact."""

    evidence_id: str
    source: str
    content_hash: str
    summary: str

    def __post_init__(self) -> None:
        _validate_id("EvidenceRef.evidence_id", self.evidence_id)
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("EvidenceRef.source must not be empty")
        if not isinstance(self.content_hash, str) or not _HASH_PATTERN.fullmatch(
            self.content_hash
        ):
            raise ValueError("EvidenceRef.content_hash must be a SHA-256 hex digest")
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ValueError("EvidenceRef.summary must not be empty")


@dataclass(frozen=True, slots=True, kw_only=True)
class Artifact:
    """Metadata chung cho mọi artifact, ổn định sau lần append đầu tiên."""

    owner_id: str
    run_id: str
    artifact_id: str = field(default_factory=_new_artifact_id)
    version: int = 1
    status: ArtifactStatus = "draft"
    evidence: tuple[EvidenceRef, ...] = ()
    lineage: tuple[str, ...] = ()
    supersedes: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now().astimezone())

    def __post_init__(self) -> None:
        _validate_id("Artifact.artifact_id", self.artifact_id)
        _validate_id("Artifact.owner_id", self.owner_id)
        _validate_id("Artifact.run_id", self.run_id)
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 1:
            raise ValueError("Artifact.version must be a positive integer")
        if self.status not in ("draft", "accepted", "rejected"):
            raise ValueError("Artifact.status is invalid")
        frozen_evidence = tuple(self.evidence)
        if any(not isinstance(item, EvidenceRef) for item in frozen_evidence):
            raise TypeError("Artifact.evidence must be a tuple of EvidenceRef")
        frozen_lineage = _freeze_strings(self.lineage, name="Artifact.lineage")
        object.__setattr__(self, "evidence", frozen_evidence)
        object.__setattr__(self, "lineage", frozen_lineage)
        for parent_id in frozen_lineage:
            _validate_id("Artifact.lineage item", parent_id)
        if self.supersedes is not None:
            _validate_id("Artifact.supersedes", self.supersedes)
        if self.created_at.tzinfo is None:
            raise ValueError("Artifact.created_at must be timezone-aware")


@dataclass(frozen=True, slots=True, kw_only=True)
class DatasetAudit(Artifact):
    dataset_id: str
    dataset_fingerprint: str
    target_hypothesis: str | None
    columns: tuple[str, ...]
    missingness: Mapping[str, float]
    class_balance: Mapping[str, float]
    leakage_risks: tuple[str, ...]
    quality_blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        super(DatasetAudit, self).__post_init__()
        _validate_id("DatasetAudit.dataset_id", self.dataset_id)
        if not _HASH_PATTERN.fullmatch(self.dataset_fingerprint):
            raise ValueError("DatasetAudit.dataset_fingerprint must be a SHA-256 digest")
        object.__setattr__(
            self,
            "columns",
            _freeze_strings(self.columns, name="DatasetAudit.columns"),
        )
        object.__setattr__(
            self,
            "leakage_risks",
            _freeze_strings(self.leakage_risks, name="DatasetAudit.leakage_risks"),
        )
        object.__setattr__(
            self,
            "quality_blockers",
            _freeze_strings(self.quality_blockers, name="DatasetAudit.quality_blockers"),
        )
        object.__setattr__(
            self,
            "missingness",
            _freeze_mapping(self.missingness, name="DatasetAudit.missingness"),
        )
        object.__setattr__(
            self,
            "class_balance",
            _freeze_mapping(self.class_balance, name="DatasetAudit.class_balance"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ExperimentSpec(Artifact):
    dataset_audit_id: str
    problem_type: str
    target_column: str
    metric: str
    metric_direction: MetricDirection
    split_strategy: str
    model_families: tuple[str, ...]
    max_training_jobs: int
    baseline_value: float
    acceptance_threshold: float
    default_reasons: Mapping[str, str]

    def __post_init__(self) -> None:
        super(ExperimentSpec, self).__post_init__()
        _validate_id("ExperimentSpec.dataset_audit_id", self.dataset_audit_id)
        object.__setattr__(
            self,
            "model_families",
            _freeze_strings(self.model_families, name="ExperimentSpec.model_families"),
        )
        object.__setattr__(
            self,
            "default_reasons",
            _freeze_mapping(self.default_reasons, name="ExperimentSpec.default_reasons"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingRunSet(Artifact):
    experiment_spec_id: str
    config_hash: str
    idempotency_key: str
    job_ids: tuple[str, ...]
    job_statuses: Mapping[str, str]
    cost: float
    reconciliation_status: str

    def __post_init__(self) -> None:
        super(TrainingRunSet, self).__post_init__()
        _validate_id("TrainingRunSet.experiment_spec_id", self.experiment_spec_id)
        if not _HASH_PATTERN.fullmatch(self.config_hash):
            raise ValueError("TrainingRunSet.config_hash must be a SHA-256 digest")
        _validate_id("TrainingRunSet.idempotency_key", self.idempotency_key)
        object.__setattr__(
            self,
            "job_ids",
            _freeze_strings(self.job_ids, name="TrainingRunSet.job_ids"),
        )
        object.__setattr__(
            self,
            "job_statuses",
            _freeze_mapping(self.job_statuses, name="TrainingRunSet.job_statuses"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationReport(Artifact):
    training_run_set_id: str
    metric: str
    metric_direction: MetricDirection
    metric_value: float
    baseline_value: float
    baseline_delta: float
    cv_mean: float
    variance: float
    overfit_gap: float
    calibration_error: float | None
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        super(EvaluationReport, self).__post_init__()
        _validate_id("EvaluationReport.training_run_set_id", self.training_run_set_id)
        object.__setattr__(
            self,
            "rejection_reasons",
            _freeze_strings(
                self.rejection_reasons,
                name="EvaluationReport.rejection_reasons",
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ReleaseCandidate(Artifact):
    evaluation_report_id: str
    model_version: str
    input_schema: Mapping[str, Any]
    decision_threshold: float | None
    readiness_verdict: str

    def __post_init__(self) -> None:
        super(ReleaseCandidate, self).__post_init__()
        _validate_id("ReleaseCandidate.evaluation_report_id", self.evaluation_report_id)
        _validate_id("ReleaseCandidate.model_version", self.model_version)
        object.__setattr__(
            self,
            "input_schema",
            _freeze_mapping(self.input_schema, name="ReleaseCandidate.input_schema"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class PredictionArtifact(Artifact):
    release_candidate_id: str
    model_input_hash: str
    result_uri: str
    row_errors: Mapping[str, str]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        super(PredictionArtifact, self).__post_init__()
        _validate_id("PredictionArtifact.release_candidate_id", self.release_candidate_id)
        if not _HASH_PATTERN.fullmatch(self.model_input_hash):
            raise ValueError("PredictionArtifact.model_input_hash must be a SHA-256 digest")
        if not isinstance(self.result_uri, str) or not self.result_uri.strip():
            raise ValueError("PredictionArtifact.result_uri must not be empty")
        object.__setattr__(
            self,
            "row_errors",
            _freeze_mapping(self.row_errors, name="PredictionArtifact.row_errors"),
        )
        object.__setattr__(
            self,
            "provenance",
            _freeze_mapping(self.provenance, name="PredictionArtifact.provenance"),
        )
