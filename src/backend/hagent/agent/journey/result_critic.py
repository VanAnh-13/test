"""Đánh giá training evidence và tạo release verdict xác định."""

from __future__ import annotations

import inspect
import json
import math
import re
from collections.abc import Mapping, Sequence
from statistics import fmean, pvariance
from typing import Any

from hagent.agent.capabilities.broker import InvocationBroker
from hagent.agent.capabilities.models import (
    CapabilityInvocationError,
    freeze_json,
    thaw_json,
)
from hagent.agent.journey.artifacts import (
    EvaluationReport,
    EvidenceRef,
    ReleaseCandidate,
)
from hagent.agent.journey.canonical import canonical_mapping_hash
from hagent.agent.journey.checkers import metric_direction
from hagent.agent.journey.request_scope import request_scope_from_context
from hagent.agent.runtime.context import GraphRequestContext

TRAINING_RESULTS_CAPABILITY_ID = "automl.training.results@1"

_TERMINAL_RESULT_STATUSES = frozenset({"completed", "success"})
_PENDING_RESULT_STATUSES = frozenset({"pending", "queued", "running", "training"})
_FAILED_RESULT_STATUSES = frozenset({"cancelled", "error", "failed"})
_MAX_CRITIC_REASONS = 8
_MAX_CRITIC_REASON_LENGTH = 512
_SAFE_MODEL_VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


class _InvalidEvaluationEvidence(ValueError):
    pass


def _number(value: Any, *, field_name: str, minimum: float | None = None) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise _InvalidEvaluationEvidence(f"{field_name} phải là số")
    result = float(value)
    if not math.isfinite(result):
        raise _InvalidEvaluationEvidence(f"{field_name} phải hữu hạn")
    if minimum is not None and result < minimum:
        raise _InvalidEvaluationEvidence(f"{field_name} nhỏ hơn giới hạn")
    return result


def _optional_number(
    value: Any,
    *,
    field_name: str,
    minimum: float | None = None,
) -> float | None:
    if value is None:
        return None
    return _number(value, field_name=field_name, minimum=minimum)


def _normalized_metric(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _InvalidEvaluationEvidence("metric không hợp lệ")
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _cv_scores(value: Any) -> tuple[float, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence) or len(value) < 2:
        raise _InvalidEvaluationEvidence("cv_scores cần ít nhất hai fold")
    return tuple(_number(item, field_name="cv_scores") for item in value)


def _cv_summary(
    output: Mapping[str, Any],
) -> tuple[float, float, tuple[float, ...] | None]:
    has_scores = "cv_scores" in output
    has_mean = "cv_mean" in output
    has_variance = "cv_variance" in output
    if has_mean != has_variance:
        raise _InvalidEvaluationEvidence("CV aggregate phải có đủ mean và variance")

    scores = _cv_scores(output.get("cv_scores")) if has_scores else None
    aggregate = None
    if has_mean:
        aggregate = (
            _number(output.get("cv_mean"), field_name="cv_mean"),
            _number(
                output.get("cv_variance"),
                field_name="cv_variance",
                minimum=0.0,
            ),
        )
    if scores is None and aggregate is None:
        raise _InvalidEvaluationEvidence("Thiếu CV evidence")

    if scores is not None:
        score_summary = (fmean(scores), pvariance(scores))
        if aggregate is not None and not all(
            math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12)
            for actual, expected in zip(score_summary, aggregate, strict=True)
        ):
            raise _InvalidEvaluationEvidence("CV fold và aggregate không nhất quán")
        if aggregate is None:
            aggregate = score_summary
    assert aggregate is not None
    return aggregate[0], aggregate[1], scores


def _safe_model_version(value: Any) -> str:
    if not isinstance(value, str) or not _SAFE_MODEL_VERSION_PATTERN.fullmatch(value):
        raise _InvalidEvaluationEvidence("model_version không hợp lệ")
    return value


def _safe_input_schema(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise _InvalidEvaluationEvidence("input_schema không hợp lệ")
    try:
        safe_schema = thaw_json(freeze_json(value))
        json.dumps(safe_schema, allow_nan=False, ensure_ascii=False)
    except (TypeError, ValueError):
        raise _InvalidEvaluationEvidence("input_schema không hợp lệ") from None
    return safe_schema


def _evaluation_evidence(output: Mapping[str, Any]) -> EvidenceRef:
    evidence_payload = {
        "baseline_value": output["baseline_value"],
        "cv_mean": output["cv_mean"],
        "cv_variance": output["variance"],
        "job_id": output["job_id"],
        "metric": output["metric"],
        "metric_value": output["metric_value"],
        "model_version": output["model_version"],
        "train_metric": output["train_metric"],
    }
    if output["cv_scores"] is not None:
        evidence_payload["cv_scores"] = list(output["cv_scores"])
    content_hash = canonical_mapping_hash(evidence_payload)
    return EvidenceRef(
        evidence_id=f"evaluation-{content_hash[:24]}",
        source=f"capability:{TRAINING_RESULTS_CAPABILITY_ID}",
        content_hash=content_hash,
        summary=(
            f"Evaluation metric={output['metric']} value={output['metric_value']} "
            f"baseline={output['baseline_value']}."
        ),
    )


def _parse_completed_output(
    output: Mapping[str, Any],
    *,
    expected_metric: str,
) -> dict[str, Any]:
    metric = _normalized_metric(output.get("metric"))
    if metric != _normalized_metric(expected_metric):
        raise _InvalidEvaluationEvidence("metric không khớp ExperimentSpec")
    direction = metric_direction(metric)
    metric_value = _number(output.get("metric_value"), field_name="metric_value")
    baseline_value = _number(
        output.get("baseline_value"),
        field_name="baseline_value",
    )
    cv_mean, cv_variance, scores = _cv_summary(output)
    train_metric = _number(output.get("train_metric"), field_name="train_metric")
    calibration_error = _optional_number(
        output.get("calibration_error"),
        field_name="calibration_error",
        minimum=0.0,
    )
    baseline_delta = (
        metric_value - baseline_value
        if direction == "maximize"
        else baseline_value - metric_value
    )
    overfit_gap = max(
        0.0,
        train_metric - metric_value
        if direction == "maximize"
        else metric_value - train_metric,
    )
    job_id = output.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise _InvalidEvaluationEvidence("job_id không hợp lệ")
    return {
        "baseline_delta": baseline_delta,
        "baseline_value": baseline_value,
        "calibration_error": calibration_error,
        "cv_mean": cv_mean,
        "cv_scores": scores,
        "decision_threshold": _optional_number(
            output.get("decision_threshold"),
            field_name="decision_threshold",
        ),
        "input_schema": _safe_input_schema(output.get("input_schema")),
        "job_id": job_id,
        "metric": metric,
        "metric_direction": direction,
        "metric_value": metric_value,
        "model_version": _safe_model_version(output.get("model_version")),
        "overfit_gap": overfit_gap,
        "train_metric": train_metric,
        "variance": cv_variance,
    }


async def evaluate_training(
    state: Mapping[str, Any],
    *,
    context: GraphRequestContext,
) -> dict[str, Any]:
    """Đọc kết quả typed; không coi pending hoặc evidence lỗi là evaluation."""
    training_run_set = state["training_run_set"]
    experiment_spec = state["experiment_spec"]
    if not training_run_set.job_ids:
        return {
            "evaluation_error_code": "TRAINING_JOB_REQUIRED",
            "result": {"status": "evaluation_failed"},
        }
    broker = InvocationBroker(context.capability_snapshot, max_cache_entries=0)
    try:
        result = await broker.invoke(
            TRAINING_RESULTS_CAPABILITY_ID,
            {"job_ids": list(training_run_set.job_ids)},
            scope=request_scope_from_context(context),
        )
    except CapabilityInvocationError as exc:
        return {
            "evaluation_error_code": exc.code,
            "result": {
                "status": "evaluation_failed",
                "error_code": exc.code,
            },
        }
    if not isinstance(result.output, Mapping):
        return {
            "evaluation_error_code": "INVALID_OUTPUT",
            "result": {"status": "evaluation_failed"},
        }
    raw_status = result.output.get("status")
    status = str(raw_status).strip().lower() if isinstance(raw_status, str) else ""
    if status in _PENDING_RESULT_STATUSES:
        return {
            "evaluation_status": "pending",
            "result": {"status": "evaluation_pending"},
        }
    if status in _FAILED_RESULT_STATUSES:
        return {
            "evaluation_error_code": "TRAINING_FAILED",
            "result": {
                "status": "evaluation_failed",
                "error_code": "TRAINING_FAILED",
            },
        }
    if status not in _TERMINAL_RESULT_STATUSES:
        return {
            "evaluation_error_code": "INVALID_EVALUATION_STATUS",
            "result": {"status": "evaluation_failed"},
        }
    try:
        parsed = _parse_completed_output(
            result.output,
            expected_metric=experiment_spec.metric,
        )
        evidence = _evaluation_evidence(parsed)
        report = EvaluationReport(
            owner_id=context.principal_id,
            run_id=state["run_id"],
            status="draft",
            evidence=(evidence,),
            lineage=(training_run_set.artifact_id,),
            training_run_set_id=training_run_set.artifact_id,
            metric=parsed["metric"],
            metric_direction=parsed["metric_direction"],
            metric_value=parsed["metric_value"],
            baseline_value=parsed["baseline_value"],
            baseline_delta=parsed["baseline_delta"],
            cv_mean=parsed["cv_mean"],
            variance=parsed["variance"],
            overfit_gap=parsed["overfit_gap"],
            calibration_error=parsed["calibration_error"],
            rejection_reasons=(),
        )
    except (TypeError, ValueError, _InvalidEvaluationEvidence):
        return {
            "evaluation_error_code": "INVALID_EVALUATION_EVIDENCE",
            "result": {
                "status": "evaluation_failed",
                "error_code": "INVALID_EVALUATION_EVIDENCE",
            },
        }
    return {
        "evaluation_report": report,
        "evaluation_status": "completed",
        "evaluation_verdicts": (),
        "release_metadata": {
            "decision_threshold": parsed["decision_threshold"],
            "input_schema": parsed["input_schema"],
            "model_version": parsed["model_version"],
        },
    }


def _critic_payload(state: Mapping[str, Any], *, blocked: bool) -> dict[str, Any]:
    report = state["evaluation_report"]
    finding_codes = [
        finding.code
        for verdict in state.get("evaluation_verdicts", ())
        for finding in verdict.findings
    ]
    return {
        "baseline_delta": report.baseline_delta,
        "deterministic_blocked": blocked,
        "finding_codes": finding_codes,
        "metric": report.metric,
        "metric_value": report.metric_value,
        "overfit_gap": report.overfit_gap,
        "variance": report.variance,
    }


async def _risk_critic(
    state: Mapping[str, Any],
    *,
    context: GraphRequestContext,
    blocked: bool,
) -> Mapping[str, Any]:
    if not blocked:
        return {}
    critic = context.services.get("evaluation_critic")
    if not callable(critic):
        return {"status": "not_configured"}
    payload = _critic_payload(state, blocked=blocked)
    try:
        response = critic(payload)
        if inspect.isawaitable(response):
            response = await response
    except Exception:  # noqa: BLE001 - critic ngoài là tùy chọn nhưng phải từ chối an toàn
        return {"status": "unavailable"}
    if not isinstance(response, Mapping):
        return {"status": "invalid"}
    verdict = response.get("verdict")
    raw_reasons = response.get("reasons", ())
    reasons = (
        [
            reason[:_MAX_CRITIC_REASON_LENGTH]
            for reason in raw_reasons[:_MAX_CRITIC_REASONS]
            if isinstance(reason, str) and reason
        ]
        if isinstance(raw_reasons, list | tuple)
        else []
    )
    return {
        "status": "completed",
        "verdict": verdict if verdict in {"ready", "reject", "revise"} else "invalid",
        "reasons": reasons,
    }


async def finalize_release_candidate(
    state: Mapping[str, Any],
    *,
    context: GraphRequestContext,
) -> dict[str, Any]:
    """Tạo release verdict; critic chỉ bổ sung nhận xét, không sửa blocker."""
    report = state["evaluation_report"]
    blocked = any(verdict.blocked for verdict in state.get("evaluation_verdicts", ()))
    critic_assessment = await _risk_critic(
        state,
        context=context,
        blocked=blocked,
    )
    metadata = state["release_metadata"]
    release = ReleaseCandidate(
        owner_id=context.principal_id,
        run_id=state["run_id"],
        status="rejected" if blocked else "accepted",
        evidence=report.evidence,
        lineage=(report.artifact_id,),
        evaluation_report_id=report.artifact_id,
        model_version=metadata["model_version"],
        input_schema=metadata["input_schema"],
        decision_threshold=metadata["decision_threshold"],
        readiness_verdict="rejected" if blocked else "ready",
    )
    return {
        "critic_assessment": dict(critic_assessment),
        "release_candidate": release,
        "result": {
            "status": "evaluation_rejected" if blocked else "release_ready",
            "artifact_id": release.artifact_id,
            "model_version": release.model_version,
        },
    }
