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


def compute_normalized_latent_surprise(
    predicted: LatentState,
    actual: LatentState,
    config: dict | None = None,
) -> SurpriseResult:
    """
    Surprise chuẩn hóa theo σ per-dim của predictor (dynamics ensemble):
    value = RMS của z-score từng chiều — đơn vị z, ngưỡng normalized_thresholds.
    """
    cfg = dict(config or {})
    thresholds = dict(
        cfg.get("normalized_thresholds") or {"medium": 1.5, "high": 3.0}
    )
    sigma_floor = float(cfg.get("sigma_floor", 1e-3))
    std = list((predicted.meta or {}).get("std") or [])
    n = min(predicted.dim, actual.dim, len(predicted.vector), len(actual.vector))
    if n == 0:
        return SurpriseResult(0.0, "low", predicted.dim, actual.dim)
    total = 0.0
    for i in range(n):
        s = std[i] if i < len(std) else 0.0
        s = max(float(s), sigma_floor)
        diff = (actual.vector[i] - predicted.vector[i]) / s
        total += diff * diff
    value = math.sqrt(total / n)
    return SurpriseResult(
        value=value,
        level=classify_surprise(value, thresholds),
        predicted_dim=predicted.dim,
        actual_dim=actual.dim,
    )


def compute_surprise(
    predicted: LatentState,
    actual: LatentState,
    config: dict | None = None,
) -> SurpriseResult:
    cfg = dict(config or {})
    # Predictor có uncertainty per-dim (dynamics ensemble) → tự chuyển sang
    # surprise chuẩn hóa; mọi call site (service, hooks) hưởng lợi không cần sửa
    if (predicted.meta or {}).get("std"):
        return compute_normalized_latent_surprise(predicted, actual, cfg)
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


# ── Outcome-space surprise ───────────────────────────────

_OUTCOME_SIGMA_FLOOR = 1e-6


def _pred_mean_std(predicted: Any) -> Tuple[float, float]:
    """Chấp nhận OutcomePrediction, dict {mean,std} hoặc tuple (mean, std)."""
    if hasattr(predicted, "mean") and hasattr(predicted, "std"):
        return float(predicted.mean), float(predicted.std)
    if isinstance(predicted, dict):
        return float(predicted["mean"]), float(predicted["std"])
    mean, std = predicted
    return float(mean), float(std)


def compute_outcome_surprise(
    predicted: Any,
    actual_score: float,
    config: dict | None = None,
) -> SurpriseResult:
    """
    Surprise trong không gian outcome: value = |y − μ| / σ (z-score).

    Ngưỡng lấy từ world_model.surprise.outcome_thresholds (đơn vị z-score,
    khác thang latent thresholds) — mặc định medium 1.5, high 3.0.
    """
    cfg = dict(config or {})
    thresholds = dict(cfg.get("outcome_thresholds") or {"medium": 1.5, "high": 3.0})
    mu, sigma = _pred_mean_std(predicted)
    sigma = max(sigma, _OUTCOME_SIGMA_FLOOR)
    value = abs(float(actual_score) - mu) / sigma
    level = classify_surprise(value, thresholds)
    return SurpriseResult(
        value=value,
        level=level,
        predicted_dim=1,
        actual_dim=1,
    )
