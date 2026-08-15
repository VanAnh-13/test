"""
Chuyển kết quả công cụ và sự kiện plan thành các patch WorldState.

Việc điều phối công cụ ưu tiên bảng ánh xạ; các handler còn lại giữ tương thích
ngược với test của công cụ lõi.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from .schema import DatasetEntry, JobEntry, WorldState, utc_now

_MIN_LOG_PROB = -700.0  # Giá trị sàn an toàn để log(0) không gây tràn dưới.
_LOG_2PI = 1.8378770664093453  # math.log(2 * math.pi)


# ── Các hàm Bayes và không gian log ổn định số học ─────────


def log_sum_exp(values: Sequence[float]) -> float:
    """
    Tính log(sum(exp(v_i))) ổn định số học.

    Dùng thuật toán trừ giá trị lớn nhất bằng Python thuần để nhanh trên chuỗi nhỏ,
    đồng thời ngăn tràn trên và tràn dưới số học.
    """
    if not values:
        return float("-inf")

    finite_vals = [v for v in values if math.isfinite(v)]
    if not finite_vals:
        return float("-inf")
    max_val = max(finite_vals)
    if max_val == float("-inf"):
        return float("-inf")
    total_exp = sum(math.exp(v - max_val) for v in finite_vals)
    if total_exp <= 0.0:
        return float("-inf")
    return max_val + math.log(total_exp)


def safe_log(p: float, min_log: float = _MIN_LOG_PROB) -> float:
    """Tính log(p) an toàn với giá trị sàn để tránh tràn dưới thành -inf hoặc NaN."""
    if p <= 0.0 or not math.isfinite(p):
        return min_log
    val = math.log(p)
    return max(val, min_log)


def gaussian_log_likelihood(
    x: float, mean: float, std: float, min_std: float = 1e-6
) -> float:
    """
    Log likelihood khi quan sát giá trị vô hướng x theo Normal(mean, std^2).

    log N(x; mu, sigma) = -0.5 * log(2*pi) - log(sigma) - (x - mu)^2 / (2 * sigma^2)
    """
    std = max(float(std), min_std)
    diff = float(x) - float(mean)
    return -0.5 * _LOG_2PI - math.log(std) - (diff * diff) / (2.0 * std * std)


def bayesian_belief_update(
    priors: Sequence[float],
    log_likelihoods: Sequence[float],
) -> list[float]:
    """
    Tính phân phối hậu nghiệm P(H_i | D) trong không gian log.

    Tham số:
        priors: Xác suất tiên nghiệm P(H_i) >= 0.
        log_likelihoods: Log likelihood log P(D | H_i).

    Giá trị trả về:
        Xác suất hậu nghiệm đã chuẩn hóa với tổng bằng 1.0.
    """
    k = len(priors)
    if k == 0 or len(log_likelihoods) != k:
        raise ValueError(
            f"priors ({k}) and log_likelihoods ({len(log_likelihoods)}) must have matching non-zero length"
        )

    # Tính log hậu nghiệm chưa chuẩn hóa: log P(H_i) + log P(D | H_i).
    log_unnorm: list[float] = []
    for p, ll in zip(priors, log_likelihoods):
        if p <= 0.0 or not math.isfinite(p):
            log_unnorm.append(float("-inf"))
        else:
            log_p = safe_log(float(p))
            log_ll = float(ll) if math.isfinite(ll) else float("-inf")
            log_unnorm.append(log_p + log_ll)

    log_z = log_sum_exp(log_unnorm)
    if not math.isfinite(log_z) or log_z == float("-inf"):
        # Trường hợp suy biến thì dùng phân phối đều làm phương án dự phòng.
        return [1.0 / k] * k

    posteriors = [math.exp(v - log_z) if math.isfinite(v) else 0.0 for v in log_unnorm]
    total = sum(posteriors)
    if total > 0.0:
        return [p / total for p in posteriors]
    return [1.0 / k] * k


def bayesian_belief_update_linear(
    priors: Sequence[float],
    likelihoods: Sequence[float],
) -> list[float]:
    """
    Cập nhật Bayes từ likelihood tuyến tính rồi chuyển an toàn sang không gian log.
    """
    log_lls = [safe_log(float(l)) for l in likelihoods]
    return bayesian_belief_update(priors, log_lls)


def update_discrete_distribution(
    prior_dist: dict[str, float],
    log_likelihoods: dict[str, float],
) -> dict[str, float]:
    """
    Cập nhật phân phối niềm tin rời rạc trên các giả thuyết có tên.
    """
    keys = list(prior_dist.keys())
    if not keys:
        return {}
    priors = [float(prior_dist[k]) for k in keys]
    lls = [float(log_likelihoods.get(k, 0.0)) for k in keys]
    posteriors = bayesian_belief_update(priors, lls)
    return {k: post for k, post in zip(keys, posteriors)}


def bayesian_gaussian_update(
    prior_mean: float,
    prior_var: float,
    obs_mean: float,
    obs_var: float,
    min_var: float = 1e-12,
) -> tuple[float, float]:
    """
    Cập nhật Gaussian một chiều liên hợp theo bước lọc Kalman:
    Tính trung bình và phương sai hậu nghiệm từ tiên nghiệm N(mu_0, sigma_0^2)
    cùng observation N(mu_obs, sigma_obs^2).
    """
    prior_var = max(float(prior_var), min_var)
    obs_var = max(float(obs_var), min_var)

    # Hệ số Kalman: K = prior_var / (prior_var + obs_var).
    total_var = prior_var + obs_var
    k = prior_var / total_var
    post_mean = prior_mean + k * (obs_mean - prior_mean)
    post_var = max((1.0 - k) * prior_var, min_var)
    return float(post_mean), float(post_var)


def infer_distribution_type(
    metric_type: str | None = None,
    sample: Any = None,
) -> str:
    """
    Tự phát hiện loại phân phối từ tên metric hoặc dữ liệu mẫu.

    - accuracy / f1 / roc_auc / bounded in [0, 1] -> "beta"
    - vector of class probabilities / Dirichlet counts -> "dirichlet"
    - discrete choice distribution -> "categorical"
    - continuous unbounded / latent vectors -> "gaussian"
    """
    if metric_type:
        m = metric_type.lower().strip()
        if m in {
            "accuracy",
            "f1",
            "f1_score",
            "roc_auc",
            "auc",
            "precision",
            "recall",
            "balanced_accuracy",
            "r2",
            "r2_score",
        }:
            return "beta"

    if isinstance(sample, (list, tuple)) and all(
        isinstance(x, (int, float)) for x in sample
    ):
        if abs(sum(sample) - 1.0) < 1e-3 and all(0.0 <= x <= 1.0 for x in sample):
            return "dirichlet"
        return "gaussian"

    if (
        isinstance(sample, dict)
        and all(isinstance(v, (int, float)) for v in sample.values())
        and abs(sum(sample.values()) - 1.0) < 1e-3
    ):
        return "categorical"

    return "gaussian"


def bayesian_beta_update(
    prior_alpha: float,
    prior_beta: float,
    successes: float,
    failures: float,
    min_val: float = 1e-3,
) -> tuple[float, float]:
    """
    Conjugate Beta-Binomial update:
    alpha_post = alpha_prior + successes
    beta_post = beta_prior + failures
    """
    alpha_post = max(float(prior_alpha) + max(float(successes), 0.0), min_val)
    beta_post = max(float(prior_beta) + max(float(failures), 0.0), min_val)
    return alpha_post, beta_post


def bayesian_categorical_update(
    prior_probs: Sequence[float],
    counts_or_ll: Sequence[float],
) -> list[float]:
    """
    Bayesian update for categorical distribution.
    """
    log_lls = [safe_log(float(c)) if c >= 0.0 else float("-inf") for c in counts_or_ll]
    return bayesian_belief_update(prior_probs, log_lls)


def bayesian_dirichlet_update(
    prior_alphas: Sequence[float],
    observed_counts: Sequence[float],
    min_alpha: float = 1e-3,
) -> list[float]:
    """
    Conjugate Dirichlet-Multinomial update:
    alpha_{i, post} = alpha_{i, prior} + count_i
    """
    if len(prior_alphas) != len(observed_counts):
        raise ValueError(
            f"prior_alphas ({len(prior_alphas)}) and observed_counts ({len(observed_counts)}) must match in length"
        )
    return [
        max(float(a) + max(float(c), 0.0), min_alpha)
        for a, c in zip(prior_alphas, observed_counts)
    ]


def update_distribution(
    prior_spec: dict[str, Any],
    observation: Any,
) -> dict[str, Any]:
    """
    Unified Bayesian distribution update dispatcher.

    Supports: 'gaussian', 'beta', 'categorical', 'dirichlet'.
    """
    dist_type = str(prior_spec.get("dist_type", "gaussian")).lower()
    params = dict(prior_spec.get("params", {}))

    if dist_type == "beta":
        prior_a = float(params.get("alpha", 1.0))
        prior_b = float(params.get("beta", 1.0))
        if isinstance(observation, dict):
            s = float(observation.get("successes", 1.0))
            f = float(observation.get("failures", 0.0))
        elif isinstance(observation, (int, float)):
            # Scalar score in [0, 1] pseudo-counts (e.g. 10 effective trials)
            n_trials = float(params.get("n_trials", 10.0))
            s = float(observation) * n_trials
            f = (1.0 - float(observation)) * n_trials
        else:
            s, f = 1.0, 0.0
        post_a, post_b = bayesian_beta_update(prior_a, prior_b, s, f)
        return {
            "dist_type": "beta",
            "params": {"alpha": post_a, "beta": post_b},
            "mean": post_a / (post_a + post_b),
        }

    if dist_type == "dirichlet":
        prior_alphas = list(params.get("alphas", [1.0, 1.0]))
        if isinstance(observation, (list, tuple)):
            counts = [float(x) for x in observation]
        elif isinstance(observation, dict):
            counts = [float(v) for v in observation.values()]
        else:
            counts = [1.0] * len(prior_alphas)
        post_alphas = bayesian_dirichlet_update(prior_alphas, counts)
        total = sum(post_alphas)
        return {
            "dist_type": "dirichlet",
            "params": {"alphas": post_alphas},
            "mean": [a / total for a in post_alphas],
        }

    if dist_type == "categorical":
        prior_probs = list(params.get("probs", [0.5, 0.5]))
        if isinstance(observation, (list, tuple)):
            counts = [float(x) for x in observation]
        elif isinstance(observation, dict):
            counts = [float(v) for v in observation.values()]
        else:
            counts = [1.0] * len(prior_probs)
        post_probs = bayesian_categorical_update(prior_probs, counts)
        return {
            "dist_type": "categorical",
            "params": {"probs": post_probs},
            "mean": post_probs,
        }

    # Default Gaussian update
    p_mean = float(params.get("mean", 0.0))
    p_std = float(params.get("std", 1.0))
    if isinstance(observation, dict):
        o_mean = float(observation.get("mean", observation.get("value", 0.0)))
        o_std = float(observation.get("std", 1.0))
    elif isinstance(observation, (int, float)):
        o_mean = float(observation)
        o_std = float(params.get("obs_std", 0.1))
    else:
        o_mean, o_std = 0.0, 1.0
    post_mean, post_var = bayesian_gaussian_update(
        p_mean, p_std * p_std, o_mean, o_std * o_std
    )
    return {
        "dist_type": "gaussian",
        "params": {"mean": post_mean, "std": math.sqrt(post_var)},
        "mean": post_mean,
        "std": math.sqrt(post_var),
    }


# ── Điều phối sự kiện công cụ và plan ───────────────────────

# Ánh xạ tên công cụ sang handler và có thể mở rộng qua register_tool_handler.
_TOOL_HANDLERS: dict[str, str] = {
    "list_datasets": "list_datasets",
    "get_dataset_info": "get_dataset_info",
    "get_features": "get_features",
    "preview_data": "preview_data",
    "list_jobs": "list_jobs",
    "get_job_info": "get_job_info",
    "start_training": "start_training",
}


def _datasets_list_from_payload(payload: dict[str, Any]) -> list[dict]:
    if isinstance(payload.get("datasets"), list):
        return payload["datasets"]
    if isinstance(payload.get("data"), list):
        return payload["data"]
    if isinstance(payload, list):
        return payload
    return []


def _jobs_list_from_payload(payload: dict[str, Any]) -> list[dict]:
    if isinstance(payload.get("jobs"), list):
        return payload["jobs"]
    if isinstance(payload.get("data"), list):
        return payload["data"]
    return []


def _handle_list_datasets(state: WorldState, payload: dict[str, Any]) -> dict[str, Any]:
    now = utc_now()
    datasets_patch = dict(state.datasets)
    for ds in _datasets_list_from_payload(payload):
        if not isinstance(ds, dict):
            continue
        did = str(ds.get("id") or ds.get("_id") or "")
        if not did:
            continue
        datasets_patch[did] = DatasetEntry(
            id=did,
            name=ds.get("name") or ds.get("filename"),
            n_rows=ds.get("n_rows") or ds.get("row_count"),
            n_cols=ds.get("n_cols") or ds.get("col_count"),
            features=ds.get("features") or ds.get("columns"),
            target=ds.get("target"),
            problem_type_inferred=ds.get("problem_type")
            or ds.get("problem_type_inferred"),
            last_seen=now,
        )
    return {"datasets": datasets_patch}


def _handle_get_dataset_info(
    state: WorldState, payload: dict[str, Any]
) -> dict[str, Any]:
    now = utc_now()
    dataset_id = str(
        payload.get("id") or payload.get("_id") or payload.get("dataset_id") or ""
    )
    if not dataset_id:
        return {}
    datasets_patch = dict(state.datasets)
    prev = dict(datasets_patch.get(dataset_id) or {"id": dataset_id})
    prev.update(
        {
            k: v
            for k, v in {
                "id": dataset_id,
                "name": payload.get("name")
                or payload.get("filename")
                or prev.get("name"),
                "n_rows": payload.get("n_rows")
                or payload.get("row_count")
                or prev.get("n_rows"),
                "n_cols": payload.get("n_cols")
                or payload.get("col_count")
                or prev.get("n_cols"),
                "features": payload.get("features")
                or payload.get("columns")
                or prev.get("features"),
                "target": payload.get("target", prev.get("target")),
                "problem_type_inferred": payload.get("problem_type")
                or payload.get("problem_type_inferred")
                or prev.get("problem_type_inferred"),
                "last_seen": now,
            }.items()
            if v is not None
        }
    )
    datasets_patch[dataset_id] = prev  # type: ignore[assignment]
    patch: dict[str, Any] = {
        "datasets": datasets_patch,
        "active_dataset_id": dataset_id,
    }
    return patch


def _handle_get_features(state: WorldState, payload: dict[str, Any]) -> dict[str, Any]:
    dataset_id = str(
        payload.get("dataset_id") or payload.get("id") or state.active_dataset_id or ""
    )
    features = (
        payload.get("features") or payload.get("columns") or payload.get("list_feature")
    )
    if not dataset_id:
        return {}
    datasets_patch = dict(state.datasets)
    prev = dict(datasets_patch.get(dataset_id) or {"id": dataset_id})
    if features is not None:
        prev["features"] = features
    prev["last_seen"] = utc_now()
    datasets_patch[dataset_id] = prev  # type: ignore[assignment]
    return {"datasets": datasets_patch, "active_dataset_id": dataset_id}


def _handle_preview_data(state: WorldState, payload: dict[str, Any]) -> dict[str, Any]:
    # Preview mainly confirms dataset exists; optional n_rows from payload
    return _handle_get_dataset_info(state, payload)


def _handle_list_jobs(state: WorldState, payload: dict[str, Any]) -> dict[str, Any]:
    jobs_patch = dict(state.jobs)
    for job in _jobs_list_from_payload(payload):
        if not isinstance(job, dict):
            continue
        jid = str(job.get("id") or job.get("_id") or job.get("job_id") or "")
        if not jid:
            continue
        jobs_patch[jid] = JobEntry(
            id=jid,
            dataset_id=job.get("dataset_id") or job.get("id_data"),
            status=job.get("status"),
            config=job.get("config"),
            metrics=job.get("metrics"),
            best_model=job.get("best_model"),
            best_score=job.get("best_score"),
            started_at=job.get("started_at"),
            finished_at=job.get("finished_at"),
        )
    return {"jobs": jobs_patch}


def _handle_get_job_info(state: WorldState, payload: dict[str, Any]) -> dict[str, Any]:
    job_id = str(payload.get("id") or payload.get("job_id") or payload.get("_id") or "")
    if not job_id:
        return {}
    jobs_patch = dict(state.jobs)
    prev = dict(jobs_patch.get(job_id) or {"id": job_id})
    for key in (
        "dataset_id",
        "status",
        "config",
        "metrics",
        "best_model",
        "best_score",
        "started_at",
        "finished_at",
    ):
        if key in payload and payload[key] is not None:
            prev[key] = payload[key]
        # alternate keys
    if payload.get("id_data") and not prev.get("dataset_id"):
        prev["dataset_id"] = payload["id_data"]
    prev["id"] = job_id
    jobs_patch[job_id] = prev  # type: ignore[assignment]
    return {"jobs": jobs_patch, "active_job_id": job_id}


def _handle_start_training(
    state: WorldState, payload: dict[str, Any]
) -> dict[str, Any]:
    now = utc_now()
    job_id = str(payload.get("job_id") or payload.get("id") or "")
    if not job_id:
        return {}
    jobs_patch = dict(state.jobs)
    jobs_patch[job_id] = JobEntry(
        id=job_id,
        dataset_id=payload.get("dataset_id") or payload.get("id_data"),
        config=payload.get("config"),
        status=payload.get("status") or "starting",
        started_at=now,
    )
    return {
        "jobs": jobs_patch,
        "active_job_id": job_id,
        "phase": "train",
    }


_HANDLERS: dict[str, Callable[[WorldState, dict[str, Any]], dict[str, Any]]] = {
    "list_datasets": _handle_list_datasets,
    "get_dataset_info": _handle_get_dataset_info,
    "get_features": _handle_get_features,
    "preview_data": _handle_preview_data,
    "list_jobs": _handle_list_jobs,
    "get_job_info": _handle_get_job_info,
    "start_training": _handle_start_training,
}


def register_tool_handler(
    tool_name: str,
    handler: Callable[[WorldState, dict[str, Any]], dict[str, Any]],
) -> None:
    """Mở rộng updater mà không sửa logic rẽ nhánh lõi."""
    _HANDLERS[tool_name] = handler
    _TOOL_HANDLERS[tool_name] = tool_name


def apply_tool_output(
    state: WorldState, tool_name: str, payload: dict[str, Any]
) -> dict[str, Any]:
    """Phân tích kết quả công cụ và trả về patch cho world state."""
    if not isinstance(payload, dict):
        return {}
    if "error" in payload:
        return {}
    handler = _HANDLERS.get(tool_name)
    if handler is None:
        return {}
    return handler(state, payload)


def apply_plan_event(
    state: WorldState,
    event_type: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Chuyển sự kiện vòng đời plan thành patch world state.

    event_type: plan_created | plan_verified | plan_rejected |
                plan_selected | plan_revised | goal_updated | surprise_recorded
    """
    now = utc_now()
    patch: dict[str, Any] = {}

    if event_type in (
        "plan_created",
        "plan_verified",
        "plan_rejected",
        "plan_selected",
        "plan_revised",
    ):
        plan_id = str(payload.get("plan_id") or "")
        if not plan_id:
            return patch
        plans = dict(state.plans or {})
        prev = dict(plans.get(plan_id) or {"plan_id": plan_id})
        prev.update({k: v for k, v in payload.items() if v is not None})
        prev["plan_id"] = plan_id
        prev["updated_at"] = now
        if event_type == "plan_created":
            prev.setdefault("created_at", now)
            prev.setdefault("status", "draft")
        elif event_type == "plan_verified":
            prev["status"] = "verified"
            prev["verification"] = payload.get("verification") or {
                "pass": True,
                "reasons": [],
            }
        elif event_type == "plan_rejected":
            prev["status"] = "rejected"
            prev["verification"] = payload.get("verification") or {
                "pass": False,
                "reasons": payload.get("reasons") or [],
            }
        elif event_type == "plan_selected":
            prev["status"] = payload.get("status") or "executing"
            patch["active_plan_id"] = plan_id
        elif event_type == "plan_revised":
            prev["status"] = "draft"
        plans[plan_id] = prev  # type: ignore[assignment]
        patch["plans"] = plans
        if event_type in ("plan_verified", "plan_rejected"):
            patch["last_verification"] = prev.get("verification")

    elif event_type == "goal_updated":
        patch["active_goal"] = payload.get("goal") or payload
        if payload.get("goals") is not None:
            patch["goals"] = payload["goals"]

    elif event_type == "surprise_recorded":
        patch["last_surprise"] = payload.get("surprise") or payload

    elif event_type == "phase_updated":
        if payload.get("phase"):
            patch["phase"] = payload["phase"]

    return patch
