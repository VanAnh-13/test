"""Mock HAutoML tool environment for harness (deterministic)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from hagent.agent.harness.schema import AgentScenario

# Deterministic Student Performance regression metrics (aligns with mock_hautoml_server)
STUDENT_TRAINING_RESULTS: dict[str, dict[str, float]] = {
    "RandomForestRegressor": {"rmse": 1.82, "mae": 1.34, "r2": 0.87, "mse": 3.31},
    "XGBRegressor": {"rmse": 1.65, "mae": 1.21, "r2": 0.91, "mse": 2.72},
    "SVR": {"rmse": 2.15, "mae": 1.58, "r2": 0.82, "mse": 4.62},
    "LinearRegression": {"rmse": 2.45, "mae": 1.89, "r2": 0.76, "mse": 6.00},
}

STUDENT_DEFAULT_MODELS = [
    "RandomForestRegressor",
    "XGBRegressor",
    "SVR",
]


def _is_student_scenario(scenario: AgentScenario, ds: str | None) -> bool:
    if ds and "student" in str(ds).lower():
        return True
    if "student" in str(scenario.id).lower():
        return True
    if any("student" in str(t).lower() for t in (scenario.tags or [])):
        return True
    goal = scenario.goal or {}
    if str(goal.get("problem_type") or "").lower() == "regression" and (
        goal.get("target_column") == "G3" or goal.get("metric") == "rmse"
    ):
        return True
    return False


def _pick_models(params: dict[str, Any], scenario: AgentScenario) -> list[str]:
    for key in ("models", "model_names", "algorithms"):
        raw = params.get(key) or (scenario.goal or {}).get(key)
        if isinstance(raw, list) and raw:
            return [str(x) for x in raw]
        if isinstance(raw, str) and raw.strip():
            return [x.strip() for x in raw.split(",") if x.strip()]
    if _is_student_scenario(scenario, params.get("dataset_id")):
        return list(STUDENT_DEFAULT_MODELS)
    return ["model_0"]


def _build_student_job(
    jid: str,
    ds: str | None,
    models: list[str],
    *,
    metric: str = "rmse",
) -> dict[str, Any]:
    model_results = []
    best_model = None
    best_score = float("inf")
    for m in models:
        metrics = dict(
            STUDENT_TRAINING_RESULTS.get(m, {"rmse": 2.5, "mae": 2.0, "r2": 0.7})
        )
        model_results.append({"model": m, "metrics": metrics})
        score = float(metrics.get(metric, metrics.get("rmse", 999)))
        if score < best_score:
            best_score = score
            best_model = m
    return {
        "id": jid,
        "job_id": jid,
        "status": "completed",
        "dataset_id": ds,
        "problem_type": "regression",
        "target_column": "G3",
        "models_requested": models,
        "best_model": best_model,
        "best_score": round(best_score, 4),
        "model_results": model_results,
        "metrics": dict(STUDENT_TRAINING_RESULTS.get(best_model or "", {})),
    }


def make_mock_tool_invoker(
    scenario: AgentScenario,
    *,
    scores: list[float] | None = None,
) -> Callable:
    """
    Async invoker(action_type, params) -> dict

    Deterministic job ids and scores for offline/graph layers.
    Student Performance scenarios return multi-model RF/XGB/SVR results.
    """
    job_n = {"i": 0}
    jobs: dict[str, dict[str, Any]] = {}
    score_list = list(scores or [0.71, 0.88, 0.80, 0.76])
    wm = scenario.world_model or {}

    async def invoker(action_type: str, params: dict[str, Any]) -> dict[str, Any]:
        ds = (
            params.get("dataset_id")
            or (scenario.goal or {}).get("dataset_id")
            or wm.get("active_dataset_id")
        )
        wm_ds = (wm.get("datasets") or {}).get(ds or "", {})
        student = _is_student_scenario(scenario, ds)

        if action_type == "list_datasets":
            return {"datasets": list((wm.get("datasets") or {}).values())}

        if action_type in ("get_dataset_info", "preview_data", "get_features"):
            feats = wm_ds.get("features") or ["f1", "f2", "target"]
            return {
                "id": ds,
                "dataset_id": ds,
                "name": wm_ds.get("name", ds),
                "features": feats,
                "target": wm_ds.get("target")
                or (scenario.goal or {}).get("target_column"),
                "n_rows": wm_ds.get("n_rows", 100),
                "n_cols": wm_ds.get("n_cols", len(feats)),
                "problem_type": wm_ds.get("problem_type")
                or (scenario.goal or {}).get("problem_type"),
            }

        if action_type in ("get_available_models", "get_metrics"):
            ptype = (
                params.get("problem_type")
                or (scenario.goal or {}).get("problem_type")
                or ("regression" if student else "classification")
            )
            if str(ptype).lower() == "regression" or student:
                return {
                    "problem_type": "regression",
                    "models": list(STUDENT_DEFAULT_MODELS) + ["LinearRegression"],
                    "metrics": ["rmse", "mae", "r2", "mse"],
                    "default_metric": "rmse",
                }
            return {
                "problem_type": "classification",
                "models": ["rf", "lr", "xgb"],
                "metrics": ["accuracy", "f1", "rmse", "mae"],
            }

        if action_type == "start_training":
            job_n["i"] += 1
            jid = f"eval-job-{scenario.id}-{job_n['i']}"
            if student:
                models = _pick_models(params, scenario)
                metric = str((scenario.goal or {}).get("metric") or "rmse")
                job = _build_student_job(jid, ds, models, metric=metric)
                job["status"] = "starting"
                jobs[jid] = {**job, "status": "completed"}
                return {
                    "job_id": jid,
                    "status": "starting",
                    "dataset_id": ds,
                    "models_requested": models,
                }
            return {
                "job_id": jid,
                "status": "starting",
                "dataset_id": ds,
            }

        if action_type == "get_job_info":
            jid = params.get("job_id") or "eval-job"
            if jid in jobs:
                return dict(jobs[jid])
            if student:
                models = _pick_models(params, scenario)
                metric = str((scenario.goal or {}).get("metric") or "rmse")
                job = _build_student_job(str(jid), ds, models, metric=metric)
                jobs[str(jid)] = job
                return job
            try:
                idx = int(str(jid).rsplit("-", 1)[-1]) - 1
            except ValueError:
                idx = 0
            score = score_list[idx % len(score_list)]
            metric = (scenario.goal or {}).get("metric") or "f1"
            return {
                "id": jid,
                "job_id": jid,
                "status": "completed",
                "best_score": score,
                "best_model": f"model_{idx}",
                "metrics": {metric: score},
            }

        if action_type == "list_jobs":
            if jobs:
                return {"jobs": list(jobs.values()), "total": len(jobs)}
            return {"jobs": list((wm.get("jobs") or {}).values())}

        if action_type == "get_world_state":
            return dict(wm)

        if action_type == "check_system_health":
            return {"status": "ok"}

        if action_type == "cancel_job":
            return {"status": "cancelled", "job_id": params.get("job_id")}

        if action_type == "predict_batch":
            return {
                "status": "success",
                "job_id": params.get("job_id"),
                "n_predictions": 10,
            }

        return {"ok": True, "action": action_type}

    return invoker
