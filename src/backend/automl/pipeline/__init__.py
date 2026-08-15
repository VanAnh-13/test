"""AutoML Pipeline Package.

Modules:
    preprocessor: Loading configuration, dataset info, and preprocessing
    trainer: Model training loops, CV optimization, inference, and job management
    evaluator: Metric computation, scoring functions, and evaluation evidence
"""

from __future__ import annotations

from automl.pipeline.evaluator import (
    ERROR_METRICS,
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
    train_process,
    training,
    training_regression,
)

__all__ = [
    "ERROR_METRICS",
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
