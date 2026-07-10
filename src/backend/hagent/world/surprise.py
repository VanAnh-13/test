"""
Surprise / violation-of-expectation — ‖ẑ − z_actual‖.

Thresholds come from config (world_model.surprise), never hard-coded at call sites.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

from hagent.world.schema import LatentState, SurpriseResult


def latent_distance(
    predicted: LatentState,
    actual: LatentState,
    *,
    metric: str = "l2",
) -> float:
    """Distance between predicted and actual latents."""
    n = min(predicted.dim, actual.dim, len(predicted.vector), len(actual.vector))
    if n == 0:
        return 0.0
    a = predicted.vector[:n]
    b = actual.vector[:n]
    metric = (metric or "l2").lower()
    if metric == "l1":
        return float(sum(abs(x - y) for x, y in zip(a, b)))
    if metric == "cosine":
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return float(1.0 - max(-1.0, min(1.0, dot / (na * nb))))
    # default l2
    return float(math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))))


def classify_surprise(value: float, thresholds: Dict[str, float]) -> str:
    """Map scalar surprise → low|medium|high using config thresholds."""
    high = float(thresholds.get("high", 0.40))
    medium = float(thresholds.get("medium", 0.15))
    if value >= high:
        return "high"
    if value >= medium:
        return "medium"
    return "low"


def compute_surprise(
    predicted: LatentState,
    actual: LatentState,
    config: dict | None = None,
) -> SurpriseResult:
    cfg = dict(config or {})
    metric = str(cfg.get("metric") or "l2")
    thresholds = dict(cfg.get("thresholds") or {"medium": 0.15, "high": 0.40})
    value = latent_distance(predicted, actual, metric=metric)
    level = classify_surprise(value, thresholds)
    return SurpriseResult(
        value=value,
        level=level,
        predicted_dim=predicted.dim,
        actual_dim=actual.dim,
    )
