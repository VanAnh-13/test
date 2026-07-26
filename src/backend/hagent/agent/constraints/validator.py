"""
Hard constraint validation before env tool calls.

Rules use world.query + goal/action — no free-text Pass/Fail only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hagent.world.query import features_of, get_dataset
from hagent.world.schema import AutoMLAction, AutoMLObservation, GoalSpec


@dataclass
class ValidationResult:
    ok: bool
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "pass": self.ok, "reasons": list(self.reasons)}


def validate_action(
    action: AutoMLAction,
    observation: AutoMLObservation,
    *,
    goal: GoalSpec | None = None,
    allowed_search: Optional[List[str]] = None,
    allowed_problem_types: Optional[List[str]] = None,
) -> ValidationResult:
    """Validate a single action against observation + goal."""
    reasons: List[str] = []
    goal = goal or observation.goal or {}
    if allowed_search is None:
        # Đọc từ campaign config để không drift với hagent.yaml; fallback
        # tĩnh đủ 5 thuật toán mà backend factory đã đăng ký
        try:
            from hagent.bridge.config import get_campaign_config

            allowed_search = list(
                get_campaign_config().get("search_algorithms") or []
            ) or None
        except Exception:
            allowed_search = None
    allowed_search = allowed_search or [
        "grid_search",
        "bayesian_search",
        "genetic_algorithm",
        "random_search",
        "successive_halving",
    ]
    allowed_problem_types = allowed_problem_types or [
        "classification",
        "regression",
    ]

    if action.type == "start_training":
        dataset_id = action.params.get("dataset_id") or goal.get("dataset_id")
        if not dataset_id:
            reasons.append("start_training requires dataset_id")
        else:
            ds = get_dataset(observation, str(dataset_id))
            if observation.datasets and ds is None:
                reasons.append(f"dataset_id={dataset_id} not in world model")
            feats = features_of(observation, str(dataset_id))
            target = action.params.get("target_column") or goal.get("target_column")
            # Accept target if it is a feature column OR the declared dataset target
            # (some fixtures keep label separate from feature list).
            declared_target = None
            if ds is not None:
                declared_target = getattr(ds, "target", None)
                if declared_target is None and isinstance(ds, dict):
                    declared_target = ds.get("target")
            if (
                target
                and feats
                and target not in feats
                and str(target) != str(declared_target or "")
            ):
                reasons.append(
                    f"target_column={target!r} not in dataset features"
                )
        ptype = action.params.get("problem_type") or goal.get("problem_type")
        if ptype and str(ptype).lower() not in allowed_problem_types:
            reasons.append(f"invalid problem_type={ptype!r}")
        search = action.params.get("search_algorithm")
        if search and str(search).lower() not in allowed_search:
            reasons.append(f"invalid search_algorithm={search!r}")
        time_limit = action.params.get("time_limit")
        if time_limit is not None:
            try:
                if int(time_limit) <= 0:
                    reasons.append("time_limit must be positive")
            except (TypeError, ValueError):
                reasons.append("time_limit must be int")

    if action.type == "get_job_info" and not action.params.get("job_id"):
        # Soft: may be filled later
        pass

    if action.type == "get_dataset_info" and not (
        action.params.get("dataset_id") or goal.get("dataset_id")
    ):
        if not observation.datasets:
            reasons.append("get_dataset_info needs dataset_id when WM has no datasets")

    return ValidationResult(ok=len(reasons) == 0, reasons=reasons)


def validate_plan_steps(
    steps: List[Dict[str, Any]] | List[Any],
    observation: AutoMLObservation,
    *,
    goal: GoalSpec | None = None,
) -> ValidationResult:
    """Validate all steps; aggregate reasons."""
    all_reasons: List[str] = []
    for i, step in enumerate(steps):
        if hasattr(step, "action"):
            action = step.action
        elif isinstance(step, dict):
            act = step.get("action") or step
            if isinstance(act, AutoMLAction):
                action = act
            else:
                action = AutoMLAction(
                    type=str(act.get("type") or act.get("tool") or ""),
                    params=dict(act.get("params") or {}),
                )
        else:
            continue
        if not action.type:
            all_reasons.append(f"step[{i}] missing action type")
            continue
        result = validate_action(action, observation, goal=goal)
        for r in result.reasons:
            all_reasons.append(f"step[{i}] {action.type}: {r}")
    return ValidationResult(ok=len(all_reasons) == 0, reasons=all_reasons)
