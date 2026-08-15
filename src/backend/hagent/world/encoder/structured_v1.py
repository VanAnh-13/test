"""
Structured encoder v1 — deterministic feature vector from AutoMLObservation.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from typing import Any

from hagent.world.schema import AutoMLObservation, GoalSpec, LatentState

# Nhãn phase mặc định khi cấu hình bỏ trống; cấu hình nên ghi đè các giá trị này.
_DEFAULT_PHASES = ("idle", "analyze", "select", "train", "evaluate", "respond")
_DEFAULT_GOAL_TYPES = (
    "train",
    "analyze",
    "evaluate",
    "list",
    "respond",
    "monitor",
    "unknown",
)


def _stable_unit(text: str) -> float:
    """Map arbitrary string → [0, 1) stably."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _clamp01(x: float) -> float:
    if math.isnan(x) or math.isinf(x):
        return 0.0
    return max(0.0, min(1.0, x))


# ── Named feature extractors (registry) ──────────────────


def _dataset_counts(obs: AutoMLObservation, _cfg: dict) -> list[float]:
    n = float(len(obs.datasets))
    return [_clamp01(n / 50.0), math.log1p(n) / 5.0]


def _job_status_histogram(obs: AutoMLObservation, cfg: dict) -> list[float]:
    statuses: list[str] = list(
        cfg.get(
            "job_statuses",
            ["pending", "starting", "running", "completed", "failed", "unknown"],
        )
    )
    counts = {s: 0 for s in statuses}
    for job in obs.jobs.values():
        st = str(job.get("status") or "unknown").lower()
        if st not in counts:
            st = "unknown"
            if "unknown" not in counts:
                counts["unknown"] = 0
        counts[st] = counts.get(st, 0) + 1
    total = max(1, len(obs.jobs))
    return [counts.get(s, 0) / total for s in statuses]


def _best_score_stats(obs: AutoMLObservation, _cfg: dict) -> list[float]:
    scores: list[float] = []
    for job in obs.jobs.values():
        s = job.get("best_score")
        if s is None:
            metrics = job.get("metrics") or {}
            if isinstance(metrics, dict) and metrics:
                try:
                    s = float(max(metrics.values()))
                except (TypeError, ValueError):
                    s = None
        if s is not None:
            try:
                scores.append(float(s))
            except (TypeError, ValueError):
                pass
    if not scores:
        return [0.0, 0.0, 0.0]
    return [
        _clamp01(min(scores)),
        _clamp01(sum(scores) / len(scores)),
        _clamp01(max(scores)),
    ]


def _phase_one_hot(obs: AutoMLObservation, cfg: dict) -> list[float]:
    phases: list[str] = list(cfg.get("phases", _DEFAULT_PHASES))
    phase = (obs.phase or "idle").lower()
    return [1.0 if phase == p else 0.0 for p in phases]


def _focus_flags(obs: AutoMLObservation, _cfg: dict) -> list[float]:
    focus = obs.focus or {}
    return [
        1.0 if focus.get("dataset_id") else 0.0,
        1.0 if focus.get("job_id") else 0.0,
        1.0 if focus.get("plan_id") else 0.0,
    ]


def _feature_coverage(obs: AutoMLObservation, _cfg: dict) -> list[float]:
    """How many datasets have features / target known."""
    if not obs.datasets:
        return [0.0, 0.0]
    with_feat = sum(1 for d in obs.datasets.values() if d.get("features"))
    with_target = sum(1 for d in obs.datasets.values() if d.get("target"))
    n = len(obs.datasets)
    return [with_feat / n, with_target / n]


def _goal_type_one_hot(obs: AutoMLObservation, cfg: dict) -> list[float]:
    types: list[str] = list(cfg.get("goal_types", _DEFAULT_GOAL_TYPES))
    gtype = "unknown"
    if obs.goal and obs.goal.get("goal_type"):
        gtype = str(obs.goal["goal_type"]).lower()
    if gtype not in types:
        gtype = "unknown"
    return [1.0 if gtype == t else 0.0 for t in types]


def _active_dataset_hash(obs: AutoMLObservation, _cfg: dict) -> list[float]:
    ds_id = (obs.focus or {}).get("dataset_id") or ""
    if not ds_id and obs.datasets:
        ds_id = next(iter(obs.datasets.keys()))
    return [_stable_unit(ds_id) if ds_id else 0.0]


EXTRACTORS: dict[str, Callable[[AutoMLObservation, dict], list[float]]] = {
    "dataset_counts": _dataset_counts,
    "job_status_histogram": _job_status_histogram,
    "best_score_stats": _best_score_stats,
    "phase_one_hot": _phase_one_hot,
    "focus_flags": _focus_flags,
    "feature_coverage": _feature_coverage,
    "goal_type_one_hot": _goal_type_one_hot,
    "active_dataset_hash": _active_dataset_hash,
}


class StructuredV1Encoder:
    """
    Deterministic encoder: concat configured extractors → pad/truncate to dim.
    """

    def __init__(self, config: dict | None = None):
        self.config = dict(config or {})
        self.dim = int(self.config.get("dim", 64))
        names = self.config.get("feature_extractors") or [
            "dataset_counts",
            "job_status_histogram",
            "best_score_stats",
            "phase_one_hot",
            "focus_flags",
            "feature_coverage",
            "goal_type_one_hot",
            "active_dataset_hash",
        ]
        unknown = [n for n in names if n not in EXTRACTORS]
        if unknown:
            raise ValueError(
                f"Unknown feature_extractors: {unknown}. Known: {sorted(EXTRACTORS)}"
            )
        self.extractor_names: list[str] = list(names)

    def _raw_features(self, observation: AutoMLObservation) -> list[float]:
        parts: list[float] = []
        for name in self.extractor_names:
            parts.extend(EXTRACTORS[name](observation, self.config))
        return parts

    def _to_latent(self, features: list[float], meta: dict[str, Any]) -> LatentState:
        vec = list(features)
        if len(vec) < self.dim:
            vec = vec + [0.0] * (self.dim - len(vec))
        elif len(vec) > self.dim:
            vec = vec[: self.dim]
        # Chuẩn hóa L2 để tránh co về vector không.
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vec = [v / norm for v in vec]
        meta = {**meta, "raw_feature_len": len(features), "l2_norm": norm}
        return LatentState(vector=vec, dim=self.dim, meta=meta)

    def encode(self, observation: AutoMLObservation) -> LatentState:
        feats = self._raw_features(observation)
        return self._to_latent(
            feats,
            {
                "encoder": "structured_v1",
                "n_datasets": len(observation.datasets),
                "n_jobs": len(observation.jobs),
                "phase": observation.phase,
            },
        )

    def encode_goal(
        self, goal: GoalSpec, observation: AutoMLObservation
    ) -> LatentState:
        # Mã hóa observation đã chèn goal để kích hoạt goal_type_one_hot.
        obs_with_goal = AutoMLObservation(
            user_id=observation.user_id,
            datasets=observation.datasets,
            jobs=observation.jobs,
            focus=observation.focus,
            phase=observation.phase,
            goal=goal,
            history_digest=observation.history_digest,
        )
        z = self.encode(obs_with_goal)
        # Điều chỉnh latent về phía mô tả goal để tính khoảng cách lập kế hoạch.
        gtype = str(goal.get("goal_type") or "unknown")
        metric = str(goal.get("metric") or "")
        target = float(goal.get("target_score") or 0.0)
        bias = [
            _stable_unit(gtype),
            _stable_unit(metric),
            _clamp01(target),
            1.0 if goal.get("dataset_id") else 0.0,
            1.0 if goal.get("target_column") else 0.0,
        ]
        vec = list(z.vector)
        for i, b in enumerate(bias):
            if i < len(vec):
                vec[i] = 0.7 * vec[i] + 0.3 * b
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vec = [v / norm for v in vec]
        return LatentState(
            vector=vec,
            dim=self.dim,
            meta={**z.meta, "goal_type": gtype, "is_goal": True},
        )
