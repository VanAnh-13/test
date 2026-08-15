"""Deterministic checkers cho contract, thống kê và policy của journey."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from hagent.agent.journey.artifacts import (
    Artifact,
    DatasetAudit,
    EvaluationReport,
    ExperimentSpec,
    PredictionArtifact,
    ReleaseCandidate,
    TrainingRunSet,
)
from hagent.agent.journey.ledger import ArtifactLedger

FindingSeverity = Literal["blocker", "warning", "info"]

_METRIC_DIRECTIONS = {
    "accuracy": "maximize",
    "auc": "maximize",
    "f1": "maximize",
    "r2": "maximize",
    "mae": "minimize",
    "mse": "minimize",
    "rmse": "minimize",
    "log_loss": "minimize",
    "logloss": "minimize",
}
_ALLOWED_SPLIT_STRATEGIES = frozenset(
    {
        "holdout",
        "kfold",
        "stratified_holdout",
        "stratified_kfold",
        "time_series",
    }
)


def _normalize_metric(metric: str) -> str:
    return metric.strip().lower().replace("-", "_").replace(" ", "_")


def metric_direction(metric: str) -> str:
    """Trả direction chuẩn hoặc fail fast khi metric chưa được đăng ký."""
    normalized = _normalize_metric(metric)
    try:
        return _METRIC_DIRECTIONS[normalized]
    except KeyError as exc:
        raise ValueError(f"Unknown metric: {metric}") from exc


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckFinding:
    code: str
    message: str
    severity: FindingSeverity = "blocker"
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.severity not in ("blocker", "warning", "info"):
            raise ValueError("CheckFinding.severity is invalid")


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckerVerdict:
    checker: str
    findings: tuple[CheckFinding, ...]
    computed: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "computed", MappingProxyType(dict(self.computed)))

    @property
    def blocked(self) -> bool:
        return any(finding.severity == "blocker" for finding in self.findings)

    @property
    def passed(self) -> bool:
        return not self.blocked


def merge_verdicts(*verdicts: CheckerVerdict) -> CheckerVerdict:
    """Hợp nhất thuần append; verdict khác không thể xóa deterministic blocker."""
    findings = tuple(finding for verdict in verdicts for finding in verdict.findings)
    computed = {
        f"{verdict.checker}.{key}": value
        for verdict in verdicts
        for key, value in verdict.computed.items()
    }
    return CheckerVerdict(checker="aggregate", findings=findings, computed=computed)


class ContractChecker:
    def check(
        self,
        artifact: Artifact,
        *,
        ledger: ArtifactLedger | None = None,
    ) -> CheckerVerdict:
        findings: list[CheckFinding] = []
        if not artifact.evidence:
            findings.append(
                CheckFinding(
                    code="EVIDENCE_REQUIRED",
                    message="Artifact phải có ít nhất một evidence reference.",
                )
            )
        if ledger is not None:
            for parent_id in artifact.lineage:
                try:
                    ledger.get(parent_id)
                except KeyError:
                    findings.append(
                        CheckFinding(
                            code="LINEAGE_MISSING",
                            message="Artifact lineage không tồn tại trong ledger.",
                        )
                    )

        if isinstance(artifact, DatasetAudit) and (
            not artifact.target_hypothesis
            or artifact.target_hypothesis not in artifact.columns
        ):
            findings.append(
                CheckFinding(
                    code="TARGET_REQUIRED",
                    message="Dataset audit chưa xác nhận target tồn tại trong schema.",
                )
            )
        relation = _expected_parent_relation(artifact)
        if relation is not None:
            parent_id, relation_name, expected_type = relation
            if parent_id not in artifact.lineage:
                findings.append(
                    CheckFinding(
                        code="LINEAGE_CONTRACT_MISMATCH",
                        message=f"{relation_name} phải xuất hiện trong lineage.",
                    )
                )
            elif ledger is not None:
                try:
                    parent = ledger.get(parent_id)
                except KeyError:
                    pass
                else:
                    if not isinstance(parent, expected_type):
                        findings.append(
                            CheckFinding(
                                code="LINEAGE_TYPE_MISMATCH",
                                message=f"{relation_name} tham chiếu sai loại artifact.",
                            )
                        )
        return CheckerVerdict(checker="contract", findings=tuple(findings))


def _expected_parent_relation(
    artifact: Artifact,
) -> tuple[str, str, type[Artifact]] | None:
    if isinstance(artifact, ExperimentSpec):
        return artifact.dataset_audit_id, "dataset_audit_id", DatasetAudit
    if isinstance(artifact, TrainingRunSet):
        return artifact.experiment_spec_id, "experiment_spec_id", ExperimentSpec
    if isinstance(artifact, EvaluationReport):
        return artifact.training_run_set_id, "training_run_set_id", TrainingRunSet
    if isinstance(artifact, ReleaseCandidate):
        return artifact.evaluation_report_id, "evaluation_report_id", EvaluationReport
    if isinstance(artifact, PredictionArtifact):
        return artifact.release_candidate_id, "release_candidate_id", ReleaseCandidate
    return None


class StatisticalChecker:
    def __init__(self, *, max_variance: float = 0.05, max_overfit_gap: float = 0.1):
        if max_variance < 0 or max_overfit_gap < 0:
            raise ValueError("Statistical thresholds must not be negative")
        self._max_variance = max_variance
        self._max_overfit_gap = max_overfit_gap

    def check(self, artifact: Artifact) -> CheckerVerdict:
        findings: list[CheckFinding] = []
        computed: dict[str, Any] = {}
        if isinstance(artifact, DatasetAudit) and artifact.leakage_risks:
            findings.append(
                CheckFinding(
                    code="LEAKAGE_RISK",
                    message="Dataset audit phát hiện nguy cơ target leakage.",
                    evidence_ids=tuple(item.evidence_id for item in artifact.evidence),
                )
            )
        if isinstance(artifact, ExperimentSpec):
            self._check_experiment(artifact, findings)
        if isinstance(artifact, EvaluationReport):
            self._check_evaluation(artifact, findings, computed)
        return CheckerVerdict(
            checker="statistical",
            findings=tuple(findings),
            computed=computed,
        )

    def _check_experiment(
        self,
        artifact: ExperimentSpec,
        findings: list[CheckFinding],
    ) -> None:
        try:
            expected_direction = metric_direction(artifact.metric)
        except ValueError:
            findings.append(
                CheckFinding(code="UNKNOWN_METRIC", message="Metric chưa được đăng ký.")
            )
        else:
            if artifact.metric_direction != expected_direction:
                findings.append(
                    CheckFinding(
                        code="METRIC_DIRECTION_MISMATCH",
                        message="Metric direction không khớp metric registry.",
                    )
                )
        if artifact.split_strategy not in _ALLOWED_SPLIT_STRATEGIES:
            findings.append(
                CheckFinding(
                    code="INVALID_SPLIT_STRATEGY",
                    message="Split strategy không thuộc allowlist xác định.",
                )
            )
        if artifact.max_training_jobs < 1:
            findings.append(
                CheckFinding(
                    code="INVALID_TRAINING_BUDGET",
                    message="Training budget phải có ít nhất một job.",
                )
            )

    def _check_evaluation(
        self,
        artifact: EvaluationReport,
        findings: list[CheckFinding],
        computed: dict[str, Any],
    ) -> None:
        try:
            expected_direction = metric_direction(artifact.metric)
        except ValueError:
            findings.append(
                CheckFinding(code="UNKNOWN_METRIC", message="Metric chưa được đăng ký.")
            )
            return
        if artifact.metric_direction != expected_direction:
            findings.append(
                CheckFinding(
                    code="METRIC_DIRECTION_MISMATCH",
                    message="Evaluation direction không khớp metric registry.",
                )
            )
        expected_delta = (
            artifact.metric_value - artifact.baseline_value
            if expected_direction == "maximize"
            else artifact.baseline_value - artifact.metric_value
        )
        computed["baseline_delta"] = expected_delta
        if not math.isclose(artifact.baseline_delta, expected_delta, abs_tol=1e-9):
            findings.append(
                CheckFinding(
                    code="BASELINE_DELTA_MISMATCH",
                    message="Baseline delta không khớp metric direction.",
                )
            )
        if expected_delta <= 0:
            findings.append(
                CheckFinding(
                    code="NO_BASELINE_IMPROVEMENT",
                    message="Candidate không cải thiện so với baseline.",
                )
            )
        if artifact.variance < 0:
            findings.append(
                CheckFinding(
                    code="INVALID_VARIANCE",
                    message="Variance không được âm.",
                )
            )
        elif artifact.variance > self._max_variance:
            findings.append(
                CheckFinding(
                    code="HIGH_VARIANCE",
                    message="Cross-validation variance vượt ngưỡng.",
                )
            )
        if artifact.overfit_gap > self._max_overfit_gap:
            findings.append(
                CheckFinding(
                    code="OVERFIT_RISK",
                    message="Train/holdout gap cho thấy nguy cơ overfit.",
                )
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class PolicyContext:
    owner_id: str
    granted_scopes: frozenset[str]
    max_training_jobs: int
    approved_artifact_ids: frozenset[str]

    def __post_init__(self) -> None:
        if not isinstance(self.owner_id, str) or not self.owner_id:
            raise ValueError("PolicyContext.owner_id must not be empty")
        if (
            not isinstance(self.max_training_jobs, int)
            or isinstance(self.max_training_jobs, bool)
            or self.max_training_jobs < 0
        ):
            raise ValueError("PolicyContext.max_training_jobs must not be negative")
        object.__setattr__(self, "granted_scopes", frozenset(self.granted_scopes))
        object.__setattr__(
            self,
            "approved_artifact_ids",
            frozenset(self.approved_artifact_ids),
        )


class PolicyChecker:
    def __init__(self, context: PolicyContext):
        self._context = context

    def check(self, artifact: Artifact) -> CheckerVerdict:
        findings: list[CheckFinding] = []
        if artifact.owner_id != self._context.owner_id:
            findings.append(
                CheckFinding(
                    code="OWNER_MISMATCH",
                    message="Artifact không thuộc request principal.",
                )
            )
        if isinstance(artifact, DatasetAudit):
            self._require_scope("automl.dataset.read", findings)
        if isinstance(artifact, ExperimentSpec) and (
            artifact.max_training_jobs > self._context.max_training_jobs
        ):
            findings.append(
                CheckFinding(
                    code="BUDGET_EXCEEDED",
                    message="ExperimentSpec vượt training budget đã cấp.",
                )
            )
        if isinstance(artifact, TrainingRunSet):
            self._require_scope("automl.training.write", findings)
            if artifact.experiment_spec_id not in self._context.approved_artifact_ids:
                findings.append(
                    CheckFinding(
                        code="APPROVAL_REQUIRED",
                        message="Training mutation chưa có ExperimentSpec được duyệt.",
                    )
                )
        if isinstance(artifact, PredictionArtifact):
            self._require_scope("automl.prediction.write", findings)
            if artifact.release_candidate_id not in self._context.approved_artifact_ids:
                findings.append(
                    CheckFinding(
                        code="APPROVAL_REQUIRED",
                        message="Prediction chưa có ReleaseCandidate được duyệt.",
                    )
                )
        return CheckerVerdict(checker="policy", findings=tuple(findings))

    def _require_scope(
        self,
        required_scope: str,
        findings: list[CheckFinding],
    ) -> None:
        if required_scope not in self._context.granted_scopes:
            findings.append(
                CheckFinding(
                    code="SCOPE_DENIED",
                    message="Request scope không cho phép stage này.",
                )
            )
