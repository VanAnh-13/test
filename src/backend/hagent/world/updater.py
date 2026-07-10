"""
Apply tool outputs / plan events → WorldState patches.

Tool dispatch is table-driven where possible; residual handlers keep
backward-compatible tests for core tools.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .schema import DatasetEntry, JobEntry, PlanEntry, WorldState, utc_now

# tool_name → handler name (extensible via register_tool_handler)
_TOOL_HANDLERS: Dict[str, str] = {
    "list_datasets": "list_datasets",
    "get_dataset_info": "get_dataset_info",
    "get_features": "get_features",
    "preview_data": "preview_data",
    "list_jobs": "list_jobs",
    "get_job_info": "get_job_info",
    "start_training": "start_training",
}


def _datasets_list_from_payload(payload: Dict[str, Any]) -> List[dict]:
    if isinstance(payload.get("datasets"), list):
        return payload["datasets"]
    if isinstance(payload.get("data"), list):
        return payload["data"]
    if isinstance(payload, list):
        return payload
    return []


def _jobs_list_from_payload(payload: Dict[str, Any]) -> List[dict]:
    if isinstance(payload.get("jobs"), list):
        return payload["jobs"]
    if isinstance(payload.get("data"), list):
        return payload["data"]
    return []


def _handle_list_datasets(
    state: WorldState, payload: Dict[str, Any]
) -> Dict[str, Any]:
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
            problem_type_inferred=ds.get("problem_type") or ds.get("problem_type_inferred"),
            last_seen=now,
        )
    return {"datasets": datasets_patch}


def _handle_get_dataset_info(
    state: WorldState, payload: Dict[str, Any]
) -> Dict[str, Any]:
    now = utc_now()
    dataset_id = str(payload.get("id") or payload.get("_id") or payload.get("dataset_id") or "")
    if not dataset_id:
        return {}
    datasets_patch = dict(state.datasets)
    prev = dict(datasets_patch.get(dataset_id) or {"id": dataset_id})
    prev.update(
        {
            k: v
            for k, v in {
                "id": dataset_id,
                "name": payload.get("name") or payload.get("filename") or prev.get("name"),
                "n_rows": payload.get("n_rows") or payload.get("row_count") or prev.get("n_rows"),
                "n_cols": payload.get("n_cols") or payload.get("col_count") or prev.get("n_cols"),
                "features": payload.get("features") or payload.get("columns") or prev.get("features"),
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
    patch: Dict[str, Any] = {
        "datasets": datasets_patch,
        "active_dataset_id": dataset_id,
    }
    return patch


def _handle_get_features(
    state: WorldState, payload: Dict[str, Any]
) -> Dict[str, Any]:
    dataset_id = str(
        payload.get("dataset_id") or payload.get("id") or state.active_dataset_id or ""
    )
    features = payload.get("features") or payload.get("columns") or payload.get("list_feature")
    if not dataset_id:
        return {}
    datasets_patch = dict(state.datasets)
    prev = dict(datasets_patch.get(dataset_id) or {"id": dataset_id})
    if features is not None:
        prev["features"] = features
    prev["last_seen"] = utc_now()
    datasets_patch[dataset_id] = prev  # type: ignore[assignment]
    return {"datasets": datasets_patch, "active_dataset_id": dataset_id}


def _handle_preview_data(
    state: WorldState, payload: Dict[str, Any]
) -> Dict[str, Any]:
    # Preview mainly confirms dataset exists; optional n_rows from payload
    return _handle_get_dataset_info(state, payload)


def _handle_list_jobs(state: WorldState, payload: Dict[str, Any]) -> Dict[str, Any]:
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


def _handle_get_job_info(state: WorldState, payload: Dict[str, Any]) -> Dict[str, Any]:
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
    state: WorldState, payload: Dict[str, Any]
) -> Dict[str, Any]:
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


_HANDLERS: Dict[str, Callable[[WorldState, Dict[str, Any]], Dict[str, Any]]] = {
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
    handler: Callable[[WorldState, Dict[str, Any]], Dict[str, Any]],
) -> None:
    """Extend updater without editing core switch logic."""
    _HANDLERS[tool_name] = handler
    _TOOL_HANDLERS[tool_name] = tool_name


def apply_tool_output(
    state: WorldState, tool_name: str, payload: Dict[str, Any]
) -> Dict[str, Any]:
    """Parse tool output and return a patch for world state."""
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
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Plan lifecycle events → world state patch.

    event_type: plan_created | plan_verified | plan_rejected |
                plan_selected | plan_revised | goal_updated | surprise_recorded
    """
    now = utc_now()
    patch: Dict[str, Any] = {}

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
