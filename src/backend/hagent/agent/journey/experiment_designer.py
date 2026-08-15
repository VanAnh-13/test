"""Thiết kế ExperimentSpec xác định từ DatasetAudit và constraint người dùng."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import Any

from hagent.agent.journey.artifacts import DatasetAudit, ExperimentSpec
from hagent.agent.journey.checkers import metric_direction

_EXPERIMENT_PATTERN = re.compile(
    r"\b(train|training|experiment|model|huấn\s*luyện|thí\s*nghiệm|mô\s*hình)\b",
    re.IGNORECASE,
)
_METRIC_PATTERN = re.compile(r"\bmetric\s*[:=]?\s*([A-Za-z0-9_-]+)", re.IGNORECASE)
_BUDGET_PATTERN = re.compile(
    r"(?:budget|max[_\s-]*jobs?|ngân\s*sách)\s*[:=]?\s*(\d+)",
    re.IGNORECASE,
)
_SPLIT_PATTERN = re.compile(
    r"\b(holdout|kfold|stratified_holdout|stratified_kfold|time_series)\b",
    re.IGNORECASE,
)
_EDITABLE_FIELDS = frozenset(
    {"metric", "split_strategy", "max_training_jobs", "model_families"}
)


def requests_experiment(message: str) -> bool:
    return bool(_EXPERIMENT_PATTERN.search(message))


def valid_edit_changes(value: Any) -> bool:
    if not isinstance(value, Mapping) or not set(value).issubset(_EDITABLE_FIELDS):
        return False
    if "metric" in value and (
        not isinstance(value["metric"], str) or not value["metric"].strip()
    ):
        return False
    if "split_strategy" in value and (
        not isinstance(value["split_strategy"], str)
        or not value["split_strategy"].strip()
    ):
        return False
    if "max_training_jobs" in value and (
        not isinstance(value["max_training_jobs"], int)
        or isinstance(value["max_training_jobs"], bool)
    ):
        return False
    if "model_families" in value:
        families = value["model_families"]
        if (
            isinstance(families, str)
            or not isinstance(families, Sequence)
            or not families
            or any(not isinstance(item, str) or not item for item in families)
        ):
            return False
    return True


def _model_families(value: Any) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError("model_families must be a non-empty string sequence")
    result = tuple(item for item in value if isinstance(item, str) and item)
    if not result:
        raise ValueError("model_families must not be empty")
    return result


def design_experiment(
    audit: DatasetAudit,
    message: str,
    *,
    previous: ExperimentSpec | None = None,
    changes: Mapping[str, Any] | None = None,
) -> ExperimentSpec:
    """Tạo draft mới; revision không bao giờ sửa artifact cũ tại chỗ."""
    edits = dict(changes or {})
    unknown_fields = set(edits) - _EDITABLE_FIELDS
    if unknown_fields:
        raise ValueError("ExperimentSpec edit contains unsupported fields")
    classification = bool(audit.class_balance)
    default_metric = "accuracy" if classification else "rmse"
    default_split = "stratified_holdout" if classification else "holdout"
    metric_match = _METRIC_PATTERN.search(message)
    budget_match = _BUDGET_PATTERN.search(message)
    split_match = _SPLIT_PATTERN.search(message)
    metric = str(
        edits.get(
            "metric",
            previous.metric if previous is not None else (
                metric_match.group(1).lower() if metric_match else default_metric
            ),
        )
    ).lower()
    try:
        direction = metric_direction(metric)
    except ValueError:
        direction = "maximize"
    split_strategy = str(
        edits.get(
            "split_strategy",
            previous.split_strategy if previous is not None else (
                split_match.group(1).lower() if split_match else default_split
            ),
        )
    )
    max_jobs = edits.get(
        "max_training_jobs",
        previous.max_training_jobs if previous is not None else (
            int(budget_match.group(1)) if budget_match else 3
        ),
    )
    if not isinstance(max_jobs, int) or isinstance(max_jobs, bool):
        raise TypeError("max_training_jobs must be an integer")
    families = _model_families(
        edits.get(
            "model_families",
            previous.model_families
            if previous is not None
            else ("random_forest", "xgboost"),
        )
    )
    default_reasons = dict(previous.default_reasons) if previous is not None else {}
    for edited_field in edits:
        default_reasons.pop(edited_field, None)
    if previous is None and metric_match is None:
        default_reasons["metric"] = "Chọn theo problem type suy ra từ dataset audit."
    if previous is None and split_match is None:
        default_reasons["split_strategy"] = "Chọn split an toàn theo problem type."
    if previous is None and budget_match is None:
        default_reasons["max_training_jobs"] = "Giới hạn beta mặc định là 3 jobs."
    version = previous.version + 1 if previous is not None else 1
    supersedes = previous.artifact_id if previous is not None else None
    digest_payload = {
        "audit": audit.artifact_id,
        "metric": metric,
        "split": split_strategy,
        "families": families,
        "jobs": max_jobs,
        "version": version,
        "supersedes": supersedes,
    }
    digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    target = audit.target_hypothesis or "target_required"
    return ExperimentSpec(
        artifact_id=f"experiment-{digest[:24]}",
        owner_id=audit.owner_id,
        run_id=audit.run_id,
        version=version,
        status="draft",
        evidence=audit.evidence,
        lineage=(audit.artifact_id,),
        supersedes=supersedes,
        dataset_audit_id=audit.artifact_id,
        problem_type="classification" if classification else "regression",
        target_column=target,
        metric=metric,
        metric_direction=direction,
        split_strategy=split_strategy,
        model_families=families,
        max_training_jobs=max_jobs,
        baseline_value=0.0,
        acceptance_threshold=0.0,
        default_reasons=default_reasons,
    )


def approval_proposal(spec: ExperimentSpec) -> dict[str, Any]:
    approval_id = "approval-" + hashlib.sha256(
        f"{spec.owner_id}\0{spec.run_id}\0{spec.artifact_id}\0{spec.version}".encode()
    ).hexdigest()[:24]
    return {
        "approval_id": approval_id,
        "artifact_id": spec.artifact_id,
        "version": spec.version,
        "requested_at": spec.created_at.isoformat(),
        "expires_at": (spec.created_at + timedelta(hours=24)).isoformat(),
    }
