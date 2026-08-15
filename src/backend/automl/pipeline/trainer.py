"""AutomL Pipeline — Trainer & Model Fitting Logic."""

from __future__ import annotations

import logging
import random
import time
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold

from automl.pipeline.evaluator import (
    ERROR_METRICS,
    build_evaluation_evidence,
    mae_score,
    mape_score,
    mse_score,
    r2_score_sklearn,
    safe_extract_score,
)
from automl.search.factory import SearchStrategyFactory
from automl.search.strategy.base import SearchStrategy

np.random.seed(42)
random.seed(42)

logger = logging.getLogger(__name__)


def _check_global_time_budget(
    max_time: float | None,
    global_start: float,
    search_strategy: Any,
) -> tuple[bool, bool]:
    """Kiểm tra ngân sách thời gian toàn cục và cập nhật thời gian còn lại cho search strategy."""
    if max_time is None:
        return False, False

    elapsed = time.time() - global_start
    remaining = max_time - elapsed

    if remaining <= 0:
        logger.info("Hết thời gian toàn cục (%ss). Bỏ qua các model còn lại.", max_time)
        return True, True

    search_strategy.config["max_time"] = remaining
    return False, False


def training(
    models: dict[Any, dict[str, Any]],
    metric_list: list[str],
    metric_sort: str,
    X_train: Any,
    y_train: Any,
    search_algorithm: str = "grid_search",
    max_time: float | None = None,
) -> tuple[Any, Any, float, dict[str, Any], list[dict[str, Any]], bool]:
    """Huấn luyện các mô hình classification với tối ưu hóa siêu tham số."""
    best_model_id = None
    best_model = None
    best_score = -1.0
    best_params: dict[str, Any] = {}
    model_results: list[dict[str, Any]] = []
    any_time_limit_reached = False

    metric_sort = metric_sort.strip().lower().replace(" ", "_")

    def parse_metric(metric_str: str) -> tuple[str, str | None]:
        if metric_str in ("accuracy", "balanced_accuracy"):
            return metric_str, None
        if metric_str.endswith("_macro"):
            return metric_str[:-6], "macro"
        if metric_str.endswith("_weighted"):
            return metric_str[:-9], "weighted"
        return metric_str, None

    scoring: dict[str, Any] = {}
    for metric in metric_list:
        base_metric, avg_type = parse_metric(metric)
        if base_metric == "accuracy":
            scoring["accuracy"] = make_scorer(accuracy_score)
        elif base_metric == "balanced_accuracy":
            scoring["balanced_accuracy"] = make_scorer(balanced_accuracy_score)
        elif base_metric in ("precision", "recall", "f1"):
            score_func = {
                "precision": precision_score,
                "recall": recall_score,
                "f1": f1_score,
            }[base_metric]
            scoring[metric] = make_scorer(score_func, average=avg_type)
        else:
            score_func = globals().get(f"{base_metric}_score")
            if score_func:
                scoring[metric] = make_scorer(score_func, average=avg_type)
            else:
                raise ValueError(f"Metric không xác định: {metric}")

    base_metric_sort, _ = parse_metric(metric_sort)
    normalized_metric_sort = (
        base_metric_sort
        if base_metric_sort in ("accuracy", "balanced_accuracy")
        else metric_sort
    )

    strategy_config: dict[str, Any] = {
        "cv": 5,
        "scoring": scoring,
        "metric_sort": normalized_metric_sort,
        "error_score": "raise",
        "return_train_score": True,
    }
    if max_time is not None:
        strategy_config["max_time"] = max_time

    try:
        search_strategy = SearchStrategyFactory.create_strategy(
            search_algorithm, strategy_config
        )
    except ValueError as e:
        logger.warning("Cảnh báo: %s. Sử dụng tìm kiếm 'grid' mặc định.", e)
        search_strategy = SearchStrategyFactory.create_strategy("grid", strategy_config)

    global_start = time.time()

    for model_id in range(len(models)):
        should_stop, _ = _check_global_time_budget(
            max_time, global_start, search_strategy
        )
        if should_stop:
            any_time_limit_reached = True
            break

        model_info = models[model_id]
        model = model_info["model"]
        param_grid = model_info["params"]

        (
            best_params_model,
            best_score_model,
            _best_all_scores_model,
            cv_results,
            search_time_limit_reached,
        ) = search_strategy.search(
            model=model,
            param_grid=param_grid,
            X=X_train,
            y=y_train,
        )
        if search_time_limit_reached:
            any_time_limit_reached = True

        best_estimator = model.set_params(**best_params_model)
        best_estimator.fit(X_train, y_train)

        rank_key = f"rank_test_{normalized_metric_sort}"
        if rank_key in cv_results:
            rank_array = np.array(cv_results[rank_key])
        else:
            rank_array = np.array(cv_results.get("rank_test_score", []))

        best_idx = int(rank_array.argmin()) if len(rank_array) > 0 else 0

        scores_dict: dict[str, Any] = {}
        for metric in metric_list:
            key = f"mean_test_{metric}"
            if key in cv_results:
                scores_dict[metric] = cv_results[key][best_idx]

        results = {
            "model_id": model_id,
            "model_name": model.__class__.__name__,
            "best_params": best_params_model,
            "scores": scores_dict,
            "cv_results": cv_results,
            "evaluation": build_evaluation_evidence(
                estimator=best_estimator,
                features=X_train,
                target=y_train,
                metric=normalized_metric_sort,
                cv_results=cv_results,
                best_index=best_idx,
                problem_type="classification",
            ),
        }
        model_results.append(results)

        if best_score_model >= best_score:
            best_model_id = model_id
            best_model = best_estimator
            best_score = best_score_model
            best_params = best_params_model

    best_params = SearchStrategy.convert_numpy_types(best_params)
    best_score = SearchStrategy.convert_numpy_types(best_score)
    model_results = SearchStrategy.convert_numpy_types(model_results)

    return (
        best_model_id,
        best_model,
        best_score,
        best_params,
        model_results,
        any_time_limit_reached,
    )


def training_regression(
    models: dict[Any, dict[str, Any]],
    metric_list: list[str],
    metric_sort: str,
    X_train: Any,
    y_train: Any,
    search_algorithm: str = "grid_search",
    max_time: float | None = None,
) -> tuple[Any, Any, float, dict[str, Any], list[dict[str, Any]], bool]:
    """Huấn luyện các mô hình regression với tối ưu hóa siêu tham số."""
    metric_sort = metric_sort.strip().lower().replace(" ", "_")

    if metric_sort in ERROR_METRICS:
        global_best_score = np.inf
        find_min = True
    else:
        global_best_score = -np.inf
        find_min = False

    best_model_id = None
    best_model = None
    best_params: dict[str, Any] = {}
    model_results: list[dict[str, Any]] = []
    any_time_limit_reached = False

    scoring = {
        "mse": make_scorer(mse_score, greater_is_better=False),
        "mae": make_scorer(mae_score, greater_is_better=False),
        "mape": make_scorer(mape_score, greater_is_better=False),
        "r2": make_scorer(r2_score_sklearn, greater_is_better=True),
    }

    cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

    strategy_config: dict[str, Any] = {
        "cv": cv_strategy,
        "scoring": scoring,
        "metric_sort": metric_sort,
        "error_score": "raise",
        "return_train_score": False,
        "n_jobs": -1,
        "random_state": 42,
    }
    if max_time is not None:
        strategy_config["max_time"] = max_time

    try:
        search_strategy = SearchStrategyFactory.create_strategy(
            search_algorithm, strategy_config
        )
    except ValueError as e:
        logger.warning("Cảnh báo: %s. Sử dụng tìm kiếm 'grid_search' mặc định.", e)
        search_strategy = SearchStrategyFactory.create_strategy(
            "grid_search", strategy_config
        )

    global_start = time.time()

    for model_id in range(len(models)):
        should_stop, _ = _check_global_time_budget(
            max_time, global_start, search_strategy
        )
        if should_stop:
            any_time_limit_reached = True
            break

        model_info = models[model_id]
        model = clone(model_info["model"])
        if hasattr(model, "random_state"):
            model.set_params(random_state=42)

        param_grid = model_info.get("params") or [{}]

        (
            best_params_model,
            best_score_model,
            _best_all_scores_model,
            cv_results,
            search_time_limit_reached,
        ) = search_strategy.search(
            model=model,
            param_grid=param_grid,
            X=X_train,
            y=y_train,
        )
        if search_time_limit_reached:
            any_time_limit_reached = True

        best_params_model = SearchStrategy.convert_numpy_types(best_params_model)
        best_score_model = SearchStrategy.convert_numpy_types(best_score_model)
        cv_results = SearchStrategy.convert_numpy_types(cv_results)

        current_model_score = safe_extract_score(metric_sort, best_score_model)
        if current_model_score is None:
            continue

        best_estimator = clone(model).set_params(**best_params_model)
        if hasattr(best_estimator, "random_state"):
            best_estimator.set_params(random_state=42)
        best_estimator.fit(X_train, y_train)

        clean_scores: dict[str, Any] = {}
        rank_key = f"rank_test_{metric_sort}"
        if cv_results.get(rank_key):
            rank_array = np.array(cv_results[rank_key])
            best_idx = int(rank_array.argmin()) if len(rank_array) > 0 else 0
        elif cv_results.get("rank_test_score"):
            rank_array = np.array(cv_results["rank_test_score"])
            best_idx = int(rank_array.argmin()) if len(rank_array) > 0 else 0
        else:
            best_idx = 0

        for metric in metric_list:
            key = f"mean_test_{metric}"
            if key in cv_results and len(cv_results[key]) > best_idx:
                raw_val = cv_results[key][best_idx]
                clean_scores[metric] = safe_extract_score(metric, raw_val)
            else:
                clean_scores[metric] = None

        results = {
            "model_id": model_id,
            "model_name": model.__class__.__name__,
            "best_params": best_params_model,
            "scores": clean_scores,
            "cv_results": cv_results,
            "evaluation": build_evaluation_evidence(
                estimator=best_estimator,
                features=X_train,
                target=y_train,
                metric=metric_sort,
                cv_results=cv_results,
                best_index=best_idx,
                problem_type="regression",
            ),
        }
        model_results.append(results)

        is_better = (
            (current_model_score <= global_best_score)
            if find_min
            else (current_model_score >= global_best_score)
        )
        if is_better:
            global_best_score = current_model_score
            best_model_id = model_id
            best_model = best_estimator
            best_params = best_params_model

    return (
        best_model_id,
        best_model,
        global_best_score,
        best_params,
        model_results,
        any_time_limit_reached,
    )


def train_process(
    X_train: Any,
    y_train: Any,
    metric_list: list[str],
    metric_sort: str,
    models: dict[Any, dict[str, Any]],
    problem_type: str,
    search_algorithm: str = "grid_search",
    max_time: float | None = None,
) -> tuple[Any, Any, float, dict[str, Any], list[dict[str, Any]], bool]:
    """Điều phối quá trình huấn luyện dựa trên loại bài toán."""
    (
        best_model_id,
        best_model,
        best_score,
        best_params,
        model_scores,
        time_limit_reached,
    ) = (
        None,
        None,
        0.0,
        {},
        [],
        False,
    )

    if problem_type == "classification":
        (
            best_model_id,
            best_model,
            best_score,
            best_params,
            model_scores,
            time_limit_reached,
        ) = training(
            models,
            metric_list,
            metric_sort,
            X_train,
            y_train,
            search_algorithm,
            max_time,
        )
    elif problem_type == "regression":
        (
            best_model_id,
            best_model,
            best_score,
            best_params,
            model_scores,
            time_limit_reached,
        ) = training_regression(
            models,
            metric_list,
            metric_sort,
            X_train,
            y_train,
            search_algorithm,
            max_time,
        )

    return (
        best_model_id,
        best_model,
        best_score,
        best_params,
        model_scores,
        time_limit_reached,
    )

