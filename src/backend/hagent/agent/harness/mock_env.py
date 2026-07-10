"""Mock HAutoML tool environment for harness (deterministic)."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from hagent.agent.harness.schema import AgentScenario


def make_mock_tool_invoker(
    scenario: AgentScenario,
    *,
    scores: Optional[list[float]] = None,
) -> Callable:
    """
    Async invoker(action_type, params) -> dict

    Deterministic job ids and scores for offline/graph layers.
    """
    job_n = {"i": 0}
    score_list = list(scores or [0.71, 0.88, 0.80, 0.76])
    wm = scenario.world_model or {}

    async def invoker(action_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        ds = (
            params.get("dataset_id")
            or (scenario.goal or {}).get("dataset_id")
            or wm.get("active_dataset_id")
        )
        wm_ds = (wm.get("datasets") or {}).get(ds or "", {})

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
            }

        if action_type in ("get_available_models", "get_metrics"):
            return {
                "problem_type": params.get("problem_type")
                or (scenario.goal or {}).get("problem_type")
                or "classification",
                "models": ["rf", "lr", "xgb"],
                "metrics": ["accuracy", "f1", "rmse", "mae"],
            }

        if action_type == "start_training":
            job_n["i"] += 1
            jid = f"eval-job-{scenario.id}-{job_n['i']}"
            return {
                "job_id": jid,
                "status": "starting",
                "dataset_id": ds,
            }

        if action_type == "get_job_info":
            jid = params.get("job_id") or "eval-job"
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
