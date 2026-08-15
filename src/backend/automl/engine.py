"""
AutoML Engine — Slim Orchestrator & Facade.

Refactored from monolith (1193 lines) into modular pipeline components:
- automl.pipeline.preprocessor: Config loading, data parsing, and model registry
- automl.pipeline.trainer: Training loops, hyperparameter search, and inference
- automl.pipeline.evaluator: Metrics computation and evaluation evidence

This file provides a backward-compatible facade re-exporting all public APIs
for existing consumers (server/application.py, cluster/worker.py, infrastructure/messaging/kafka.py, tests).
"""

from __future__ import annotations

from automl.pipeline.evaluator import (
    ERROR_METRICS,
    _classification_score,
    _finite_evaluation_number,
    _normalized_evaluation_metric,
    _regression_score,
    build_evaluation_evidence,
    mae_score,
    mape_score,
    mse_score,
    r2_score_sklearn,
    safe_extract_score,
)
from automl.pipeline.preprocessor import (
    choose_model_version,
    get_config,
    get_dataset_and_user_info,
    get_model,
)
from automl.pipeline.trainer import (
    _check_global_time_budget,
    train_process,
    training,
    training_regression,
)

__all__ = [
    "ERROR_METRICS",
    "_check_global_time_budget",
    "_classification_score",
    "_finite_evaluation_number",
    "_normalized_evaluation_metric",
    "_regression_score",
    "build_evaluation_evidence",
    "choose_model_version",
    "get_config",
    "get_dataset_and_user_info",
    "get_model",
    "mae_score",
    "mape_score",
    "mse_score",
    "r2_score_sklearn",
    "safe_extract_score",
    "train_process",
    "training",
    "training_regression",
]
