"""
Calibration metrics cho outcome predictions (Gaussian μ, σ).

- interval_coverage: tỉ lệ y nằm trong khoảng tin cậy trung tâm.
- expected_calibration_error: ECE trên PIT (probability integral transform).
- reliability_table: (nominal level, empirical coverage) để vẽ reliability plot.
- sharpness: σ trung bình (nhỏ hơn = dự đoán "sắc" hơn, chỉ có nghĩa khi calibrated).

Chỉ dùng statistics.NormalDist (stdlib) — không thêm dependency.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from statistics import NormalDist
from typing import Any

_NORMAL = NormalDist()


def _mean_std(pred: Any) -> tuple[float, float]:
    """Chấp nhận OutcomePrediction, dict {mean,std} hoặc tuple (mean, std)."""
    if hasattr(pred, "mean") and hasattr(pred, "std"):
        return float(pred.mean), float(pred.std)
    if isinstance(pred, dict):
        return float(pred["mean"]), float(pred["std"])
    mean, std = pred
    return float(mean), float(std)


def _pairs(
    predictions: Sequence[Any], targets: Sequence[float]
) -> list[tuple[float, float, float]]:
    if len(predictions) != len(targets):
        raise ValueError(
            f"predictions ({len(predictions)}) và targets ({len(targets)}) phải cùng độ dài"
        )
    out: list[tuple[float, float, float]] = []
    for pred, y in zip(predictions, targets):
        mu, sigma = _mean_std(pred)
        out.append((mu, max(sigma, 1e-12), float(y)))
    return out


def interval_coverage(
    predictions: Sequence[Any],
    targets: Sequence[float],
    *,
    confidence: float = 0.9,
) -> float:
    """Tỉ lệ target nằm trong khoảng tin cậy trung tâm μ ± z·σ."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence phải trong (0,1), nhận {confidence}")
    pairs = _pairs(predictions, targets)
    if not pairs:
        return 0.0
    z = _NORMAL.inv_cdf(0.5 + confidence / 2.0)
    hits = sum(1 for mu, sigma, y in pairs if abs(y - mu) <= z * sigma)
    return hits / len(pairs)


def pit_values(predictions: Sequence[Any], targets: Sequence[float]) -> list[float]:
    """uᵢ = Φ((yᵢ − μᵢ)/σᵢ) — nếu model calibrated thì u ~ Uniform(0,1)."""
    return [
        _NORMAL.cdf((y - mu) / sigma) for mu, sigma, y in _pairs(predictions, targets)
    ]


def expected_calibration_error(
    predictions: Sequence[Any],
    targets: Sequence[float],
    *,
    n_bins: int = 10,
) -> float:
    """
    ECE trên lưới quantile: với mỗi mức p, so |P(u ≤ p) − p|.
    0 = calibrated hoàn hảo; giá trị lớn = lệch nhiều.
    """
    us = pit_values(predictions, targets)
    if not us:
        return 0.0
    levels = [(i + 1) / (n_bins + 1) for i in range(n_bins)]
    total = 0.0
    for p in levels:
        empirical = sum(1 for u in us if u <= p) / len(us)
        total += abs(empirical - p)
    return total / len(levels)


def reliability_table(
    predictions: Sequence[Any],
    targets: Sequence[float],
    *,
    levels: Iterable[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
) -> list[dict]:
    """Bảng (nominal central coverage, empirical coverage) cho reliability plot."""
    rows = []
    for level in levels:
        rows.append(
            {
                "nominal": float(level),
                "empirical": interval_coverage(
                    predictions, targets, confidence=float(level)
                ),
            }
        )
    return rows


def sharpness(predictions: Sequence[Any]) -> float:
    """σ trung bình của các dự đoán."""
    if not predictions:
        return 0.0
    stds = [_mean_std(p)[1] for p in predictions]
    return sum(stds) / len(stds)
