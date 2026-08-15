"""AutomL Pipeline — Evaluator & Metric Computation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

ERROR_METRICS = {"mse", "mae", "mape", "rmse", "log_loss"}


# ── Custom Scorer Functions ───────────────────────────────────────────────────


def mse_score(y_true: Any, y_pred: Any) -> float:
    """Tính Mean Squared Error (MSE) giữa giá trị thực và dự đoán."""
    return float(mean_squared_error(y_true, y_pred))


def mae_score(y_true: Any, y_pred: Any) -> float:
    """Tính Mean Absolute Error (MAE) giữa giá trị thực và dự đoán."""
    return float(mean_absolute_error(y_true, y_pred))


def mape_score(y_true: Any, y_pred: Any) -> float:
    """Tính Mean Absolute Percentage Error (MAPE), trả về giá trị phần trăm (%)."""
    epsilon = 1e-10
    return float(
        np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100
    )


def r2_score_sklearn(y_true: Any, y_pred: Any) -> float:
    """Wrapper cho sklearn r2_score, tính hệ số xác định R²."""
    return float(r2_score(y_true, y_pred))


# ── Normalization and Scoring Helpers ─────────────────────────────────────────


def _normalized_evaluation_metric(metric: str) -> str:
    if not isinstance(metric, str) or not metric.strip():
        raise ValueError("Metric đánh giá không hợp lệ")
    return metric.strip().lower().replace("-", "_").replace(" ", "_")


def _classification_score(
    metric: str,
    target: np.ndarray,
    predictions: np.ndarray,
    *,
    probabilities: np.ndarray | None = None,
    labels: np.ndarray | None = None,
) -> float:
    if metric == "accuracy":
        return float(accuracy_score(target, predictions))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(target, predictions))
    if metric == "log_loss":
        if probabilities is None or labels is None:
            raise ValueError("Estimator không cung cấp probability cho log_loss")
        return float(log_loss(target, probabilities, labels=labels))
    if metric in {"auc", "roc_auc"}:
        if probabilities is None or labels is None:
            raise ValueError("Estimator không cung cấp probability cho AUC")
        if probabilities.ndim != 2 or probabilities.shape[1] != len(labels):
            raise ValueError("Probability shape không khớp class labels")
        if len(labels) == 2:
            return float(roc_auc_score(target, probabilities[:, 1]))
        return float(
            roc_auc_score(
                target,
                probabilities,
                labels=labels,
                multi_class="ovr",
                average="weighted",
            )
        )
    average = "weighted"
    base_metric = metric
    for suffix in ("macro", "weighted", "micro"):
        marker = f"_{suffix}"
        if metric.endswith(marker):
            base_metric = metric[: -len(marker)]
            average = suffix
            break
    scoring_functions = {
        "f1": f1_score,
        "precision": precision_score,
        "recall": recall_score,
    }
    try:
        scorer = scoring_functions[base_metric]
    except KeyError as exc:
        raise ValueError(f"Metric classification chưa hỗ trợ: {metric}") from exc
    return float(scorer(target, predictions, average=average, zero_division=0))


def _regression_score(
    metric: str,
    target: np.ndarray,
    predictions: np.ndarray,
) -> float:
    scorers = {
        "mae": mean_absolute_error,
        "mape": mean_absolute_percentage_error,
        "mse": mean_squared_error,
        "r2": r2_score,
    }
    if metric == "rmse":
        return float(np.sqrt(mean_squared_error(target, predictions)))
    try:
        scorer = scorers[metric]
    except KeyError as exc:
        raise ValueError(f"Metric regression chưa hỗ trợ: {metric}") from exc
    return float(scorer(target, predictions))


def _finite_evaluation_number(value: Any, *, field_name: str) -> float:
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{field_name} không phải số")  # noqa: TRY004
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{field_name} không hữu hạn")
    return result


def safe_extract_score(metric_name: str, raw_score: Any) -> float | None:
    """Chuyển đổi điểm thô từ cross-validation thành giá trị an toàn.

    Xử lý NaN/Inf và lấy giá trị tuyệt đối cho error metrics
    (vì sklearn trả về giá trị âm cho greater_is_better=False).
    """
    if raw_score is None or np.isinf(raw_score) or np.isnan(raw_score):
        return None

    if metric_name in ERROR_METRICS:
        return abs(raw_score)

    return raw_score


# ── Evidence Construction ─────────────────────────────────────────────────────


def build_evaluation_evidence(
    *,
    estimator: Any,
    features: Any,
    target: Any,
    metric: str,
    cv_results: dict[str, Any],
    best_index: int,
    problem_type: str,
) -> dict[str, Any]:
    """Tạo summary từ CV và prediction thật; không nội suy fold score bị thiếu."""
    normalized_metric = _normalized_evaluation_metric(metric)
    mean_values = cv_results.get(f"mean_test_{normalized_metric}")
    std_values = cv_results.get(f"std_test_{normalized_metric}")
    if (
        not isinstance(mean_values, (list, tuple, np.ndarray))
        or not isinstance(std_values, (list, tuple, np.ndarray))
        or best_index < 0
        or best_index >= len(mean_values)
        or best_index >= len(std_values)
    ):
        raise ValueError("CV evidence không đầy đủ tại best index")
    cv_mean = _finite_evaluation_number(
        mean_values[best_index],
        field_name="cv_mean",
    )
    cv_std = _finite_evaluation_number(
        std_values[best_index],
        field_name="cv_std",
    )
    if normalized_metric in ERROR_METRICS:
        cv_mean = abs(cv_mean)

    target_array = np.asarray(target)
    predictions = np.asarray(estimator.predict(features))
    if len(target_array) == 0 or predictions.shape[0] != target_array.shape[0]:
        raise ValueError("Estimator prediction không khớp target")

    if problem_type == "classification":
        labels, counts = np.unique(target_array, return_counts=True)
        baseline_predictions = np.full(target_array.shape, labels[np.argmax(counts)])
        probabilities = None
        baseline_probabilities = None
        if normalized_metric in {"auc", "log_loss", "roc_auc"}:
            if not hasattr(estimator, "predict_proba"):
                raise ValueError("Estimator không hỗ trợ log_loss")
            probabilities = np.asarray(estimator.predict_proba(features))
            proportions = counts / counts.sum()
            baseline_probabilities = np.tile(proportions, (len(target_array), 1))
        train_metric = _classification_score(
            normalized_metric,
            target_array,
            predictions,
            probabilities=probabilities,
            labels=labels,
        )
        baseline_value = _classification_score(
            normalized_metric,
            target_array,
            baseline_predictions,
            probabilities=baseline_probabilities,
            labels=labels,
        )
    elif problem_type == "regression":
        numeric_target = target_array.astype(float)
        numeric_predictions = predictions.astype(float)
        baseline_predictions = np.full(
            numeric_target.shape,
            float(np.mean(numeric_target)),
        )
        train_metric = _regression_score(
            normalized_metric,
            numeric_target,
            numeric_predictions,
        )
        baseline_value = _regression_score(
            normalized_metric,
            numeric_target,
            baseline_predictions,
        )
    else:
        raise ValueError(f"Problem type chưa hỗ trợ: {problem_type}")

    evidence = {
        "metric": normalized_metric,
        "metric_value": cv_mean,
        "baseline_value": baseline_value,
        "train_metric": train_metric,
        "cv_mean": cv_mean,
        "cv_variance": cv_std**2,
        # Chưa có holdout calibration evidence nên giữ explicit unavailable.
        "calibration_error": None,
    }
    for field_name, value in evidence.items():
        if field_name not in {"metric", "calibration_error"}:
            evidence[field_name] = _finite_evaluation_number(
                value,
                field_name=field_name,
            )
    return evidence
