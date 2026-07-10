"""
Structured world-model queries for planner / verifier / agents.

Prefer these helpers over ad-hoc dict key access.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from hagent.world.schema import (
    AutoMLObservation,
    DatasetEntry,
    GoalEntry,
    JobEntry,
    PlanEntry,
    WorldState,
)


def _as_mapping(wm: WorldState | AutoMLObservation | dict) -> dict:
    if isinstance(wm, WorldState):
        return wm.to_dict()
    if isinstance(wm, AutoMLObservation):
        return wm.to_dict()
    return dict(wm or {})


def get_dataset(
    wm: WorldState | AutoMLObservation | dict, dataset_id: str
) -> Optional[DatasetEntry]:
    data = _as_mapping(wm)
    datasets = data.get("datasets") or {}
    return datasets.get(dataset_id)


def list_dataset_ids(wm: WorldState | AutoMLObservation | dict) -> List[str]:
    data = _as_mapping(wm)
    return list((data.get("datasets") or {}).keys())


def features_of(
    wm: WorldState | AutoMLObservation | dict, dataset_id: str
) -> List[str]:
    ds = get_dataset(wm, dataset_id) or {}
    feats = ds.get("features") or []
    return list(feats) if isinstance(feats, list) else []


def past_best_jobs(
    wm: WorldState | AutoMLObservation | dict,
    *,
    problem_type: str | None = None,
    top_k: int = 5,
) -> List[JobEntry]:
    data = _as_mapping(wm)
    jobs = list((data.get("jobs") or {}).values())

    def score(j: dict) -> float:
        s = j.get("best_score")
        if s is not None:
            try:
                return float(s)
            except (TypeError, ValueError):
                pass
        metrics = j.get("metrics") or {}
        if isinstance(metrics, dict) and metrics:
            try:
                return float(max(metrics.values()))
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    filtered = []
    for j in jobs:
        if problem_type:
            cfg = j.get("config") or {}
            pt = cfg.get("problem_type") or j.get("problem_type")
            if pt and str(pt).lower() != str(problem_type).lower():
                continue
        if str(j.get("status", "")).lower() in ("completed", "done", "success"):
            filtered.append(j)
    filtered.sort(key=score, reverse=True)
    return filtered[:top_k]


def active_plan(wm: WorldState | dict) -> Optional[PlanEntry]:
    data = _as_mapping(wm)
    plan_id = data.get("active_plan_id")
    plans = data.get("plans") or {}
    if plan_id and plan_id in plans:
        return plans[plan_id]
    return None


def open_goals(wm: WorldState | dict) -> List[GoalEntry]:
    data = _as_mapping(wm)
    goals = data.get("goals") or []
    return [
        g
        for g in goals
        if isinstance(g, dict)
        and str(g.get("status", "open")).lower() in ("open", "in_progress")
    ]


def format_for_prompt(
    wm: WorldState | AutoMLObservation | dict,
    *,
    max_datasets: int = 10,
    max_jobs: int = 10,
) -> str:
    """Compact markdown summary for system prompts."""
    data = _as_mapping(wm)
    lines: List[str] = []
    datasets = data.get("datasets") or {}
    jobs = data.get("jobs") or {}

    if datasets:
        items = list(datasets.items())[:max_datasets]
        ds_lines = [
            f"- {did}: {d.get('name', '?')} "
            f"({d.get('n_rows', '?')}×{d.get('n_cols', '?')})"
            for did, d in items
        ]
        lines.append(f"**Datasets ({len(datasets)}):**\n" + "\n".join(ds_lines))
    else:
        lines.append("**Datasets:** none known yet")

    if jobs:
        items = list(jobs.items())[:max_jobs]
        job_lines = [
            f"- {jid}: status={j.get('status', '?')} "
            f"best={j.get('best_model', 'N/A')} "
            f"score={j.get('best_score', 'N/A')}"
            for jid, j in items
        ]
        lines.append(f"**Jobs ({len(jobs)}):**\n" + "\n".join(job_lines))
    else:
        lines.append("**Jobs:** none known yet")

    if data.get("active_plan_id"):
        lines.append(f"**Active plan:** {data['active_plan_id']}")
    if data.get("phase"):
        lines.append(f"**Phase:** {data['phase']}")
    if data.get("last_surprise"):
        s = data["last_surprise"]
        if isinstance(s, dict):
            lines.append(
                f"**Last surprise:** {s.get('level', '?')} ({s.get('value', '?')})"
            )

    return "\n".join(lines) if lines else "World model empty."
