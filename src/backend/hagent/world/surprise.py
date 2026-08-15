from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from hagent.world.schema import LatentState, SurpriseResult

DEFAULT_SURPRISE_THRESHOLDS: dict[str, float] = {"medium": 0.15, "high": 0.40}
DEFAULT_NORMALIZED_THRESHOLDS: dict[str, float] = {"medium": 1.5, "high": 3.0}
DEFAULT_OUTCOME_THRESHOLDS: dict[str, float] = {"medium": 1.5, "high": 3.0}
DEFAULT_PLAN_THRESHOLDS: dict[str, float] = {"medium": 0.50, "high": 1.00}
DEFAULT_SIGMA_FLOOR = 1e-3
DEFAULT_OUTCOME_SIGMA_FLOOR = 1e-6


def latent_distance(
    predicted: LatentState,
    actual: LatentState,
    *,
    metric: str = "l2",
) -> float:
    """Khoảng cách giữa latent dự đoán và latent thực tế."""
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


def classify_surprise(
    value: float,
    thresholds: dict[str, float] | None = None,
    default_thresholds: dict[str, float] | None = None,
) -> str:
    """Ánh xạ surprise vô hướng thành low, medium hoặc high theo ngưỡng cấu hình."""
    active_defaults = default_thresholds or DEFAULT_SURPRISE_THRESHOLDS
    t = dict(thresholds or active_defaults)
    high = float(t.get("high", active_defaults["high"]))
    medium = float(t.get("medium", active_defaults["medium"]))
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
    thresholds = dict(cfg.get("normalized_thresholds") or DEFAULT_NORMALIZED_THRESHOLDS)
    sigma_floor = float(cfg.get("sigma_floor", DEFAULT_SIGMA_FLOOR))
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
        level=classify_surprise(value, thresholds, DEFAULT_NORMALIZED_THRESHOLDS),
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
    # Surprise được chuẩn hóa để mọi nơi gọi như service và hook cùng hưởng lợi.
    if (predicted.meta or {}).get("std"):
        return compute_normalized_latent_surprise(predicted, actual, cfg)
    metric = str(cfg.get("metric") or "l2")
    thresholds = dict(cfg.get("thresholds") or DEFAULT_SURPRISE_THRESHOLDS)
    value = latent_distance(predicted, actual, metric=metric)
    level = classify_surprise(value, thresholds, DEFAULT_SURPRISE_THRESHOLDS)
    return SurpriseResult(
        value=value,
        level=level,
        predicted_dim=predicted.dim,
        actual_dim=actual.dim,
    )


# ── Outcome-space surprise ───────────────────────────────


def _pred_mean_std(predicted: Any) -> tuple[float, float]:
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
    thresholds = dict(cfg.get("outcome_thresholds") or DEFAULT_OUTCOME_THRESHOLDS)
    mu, sigma = _pred_mean_std(predicted)
    sigma = max(sigma, DEFAULT_OUTCOME_SIGMA_FLOOR)
    value = abs(float(actual_score) - mu) / sigma
    level = classify_surprise(value, thresholds, DEFAULT_OUTCOME_THRESHOLDS)
    return SurpriseResult(
        value=value,
        level=level,
        predicted_dim=1,
        actual_dim=1,
    )


# ── Multi-scale Plan-Level Surprise ────────────────────────


def compute_aggregate_plan_surprise(
    step_surprises: Sequence[float | SurpriseResult],
    config: dict | None = None,
    method: str = "mean",
) -> SurpriseResult:
    """
    Tính surprise tích lũy hoặc tổng hợp qua mọi bước thực thi plan.

    Tham số:
        step_surprises: Chuỗi giá trị vô hướng hoặc đối tượng SurpriseResult.
        config: Dict cấu hình chứa `plan_thresholds`.
        method: "mean" | "rms" | "max".

    Giá trị trả về:
        SurpriseResult chứa giá trị tổng hợp và mức đã phân loại.
    """
    if not step_surprises:
        return SurpriseResult(value=0.0, level="low", predicted_dim=0, actual_dim=0)

    raw_values: list[float] = []
    for s in step_surprises:
        if isinstance(s, SurpriseResult):
            raw_values.append(s.value)
        elif isinstance(s, (int, float)):
            raw_values.append(float(s))

    if not raw_values:
        return SurpriseResult(value=0.0, level="low", predicted_dim=0, actual_dim=0)

    if method == "max":
        agg_value = max(raw_values)
    elif method == "rms":
        agg_value = math.sqrt(sum(v * v for v in raw_values) / len(raw_values))
    else:  # default mean
        agg_value = sum(raw_values) / len(raw_values)

    cfg = dict(config or {})
    thresholds = dict(cfg.get("plan_thresholds") or DEFAULT_PLAN_THRESHOLDS)
    level = classify_surprise(agg_value, thresholds, DEFAULT_PLAN_THRESHOLDS)
    return SurpriseResult(
        value=agg_value,
        level=level,
        predicted_dim=0,
        actual_dim=0,
    )


def should_trigger_plan_revision(
    step_surprise: SurpriseResult | None,
    plan_surprise: SurpriseResult | None = None,
    config: dict | None = None,
) -> bool:
    """
    Xác định có cần lập lại plan theo surprise của bước và toàn plan hay không.
    """
    cfg = dict(config or {})
    min_level = str(cfg.get("min_revision_level", "high")).lower()

    return bool(
        (step_surprise and step_surprise.level == min_level)
        or (plan_surprise and plan_surprise.level == min_level)
    )


# ── Generalized Distribution Surprise (KL Divergence) ─────


def _digamma(x: float) -> float:
    """
    Tính hàm digamma psi(x) = d/dx ln Gamma(x).
    Dùng khai triển tiệm cận chính xác cho x > 0 với phép dịch truy hồi.
    """
    if x <= 0.0:
        return float("-inf")
    res = 0.0
    cur_x = float(x)
    while cur_x < 6.0:
        res -= 1.0 / cur_x
        cur_x += 1.0
    inv_x = 1.0 / cur_x
    inv_x2 = inv_x * inv_x
    res += (
        math.log(cur_x)
        - 0.5 * inv_x
        - inv_x2 * (1.0 / 12.0 - inv_x2 * (1.0 / 120.0 - inv_x2 * (1.0 / 252.0)))
    )
    return res


def kl_divergence_gaussian(mu1: float, std1: float, mu0: float, std0: float) -> float:
    """
    KL(N(mu1, std1^2) || N(mu0, std0^2)).
    """
    s1 = max(float(std1), 1e-12)
    s0 = max(float(std0), 1e-12)
    diff = float(mu1) - float(mu0)
    kl = math.log(s0 / s1) + (s1 * s1 + diff * diff) / (2.0 * s0 * s0) - 0.5
    return max(float(kl), 0.0)


def kl_divergence_beta(a1: float, b1: float, a0: float, b0: float) -> float:
    """
    KL(Beta(a1, b1) || Beta(a0, b0)).
    """
    a1 = max(float(a1), 1e-6)
    b1 = max(float(b1), 1e-6)
    a0 = max(float(a0), 1e-6)
    b0 = max(float(b0), 1e-6)

    ln_b0 = math.lgamma(a0) + math.lgamma(b0) - math.lgamma(a0 + b0)
    ln_b1 = math.lgamma(a1) + math.lgamma(b1) - math.lgamma(a1 + b1)

    psi_a1 = _digamma(a1)
    psi_b1 = _digamma(b1)
    psi_ab1 = _digamma(a1 + b1)

    kl = (
        (ln_b0 - ln_b1)
        + (a1 - a0) * psi_a1
        + (b1 - b0) * psi_b1
        + (a0 - a1 + b0 - b1) * psi_ab1
    )
    return max(float(kl), 0.0)


def kl_divergence_categorical(p1: Sequence[float], p0: Sequence[float]) -> float:
    """
    KL(P1 || P0) for discrete categorical distributions.
    """
    if len(p1) != len(p0) or not p1:
        return 0.0
    s1 = sum(p1) or 1.0
    s0 = sum(p0) or 1.0
    norm_p1 = [max(float(x) / s1, 1e-15) for x in p1]
    norm_p0 = [max(float(x) / s0, 1e-15) for x in p0]

    kl = sum(x * (math.log(x) - math.log(y)) for x, y in zip(norm_p1, norm_p0))
    return max(float(kl), 0.0)


def kl_divergence_dirichlet(
    alphas1: Sequence[float], alphas0: Sequence[float]
) -> float:
    """
    KL(Dir(alphas1) || Dir(alphas0)).
    """
    if len(alphas1) != len(alphas0) or not alphas1:
        return 0.0
    a1 = [max(float(x), 1e-6) for x in alphas1]
    a0 = [max(float(x), 1e-6) for x in alphas0]
    sum_a1 = sum(a1)
    sum_a0 = sum(a0)

    psi_sum_a1 = _digamma(sum_a1)
    kl = math.lgamma(sum_a1) - math.lgamma(sum_a0)
    for x1, x0 in zip(a1, a0):
        kl += math.lgamma(x0) - math.lgamma(x1)
        kl += (x1 - x0) * (_digamma(x1) - psi_sum_a1)
    return max(float(kl), 0.0)


def compute_distribution_surprise(
    predicted_dist: dict[str, Any],
    actual_dist_or_obs: Any,
    config: dict | None = None,
) -> SurpriseResult:
    """
    Tính surprise bằng độ phân kỳ KL đối xứng hoặc xuôi giữa phân phối dự đoán và thực tế.

    Hỗ trợ: 'gaussian', 'beta', 'categorical', 'dirichlet'.
    """
    cfg = dict(config or {})
    dist_type = str(predicted_dist.get("dist_type", "gaussian")).lower()
    p_params = dict(predicted_dist.get("params", {}))

    kl = 0.0
    if dist_type == "beta":
        a0 = float(p_params.get("alpha", 1.0))
        b0 = float(p_params.get("beta", 1.0))
        if isinstance(actual_dist_or_obs, dict):
            a1 = float(
                actual_dist_or_obs.get("params", {}).get(
                    "alpha", actual_dist_or_obs.get("alpha", 1.0)
                )
            )
            b1 = float(
                actual_dist_or_obs.get("params", {}).get(
                    "beta", actual_dist_or_obs.get("beta", 1.0)
                )
            )
        elif isinstance(actual_dist_or_obs, (int, float)):
            # Observation điểm nằm trong đoạn [0, 1].
            val = max(min(float(actual_dist_or_obs), 0.999), 0.001)
            a1 = val * 10.0
            b1 = (1.0 - val) * 10.0
        else:
            a1, b1 = a0, b0
        kl = kl_divergence_beta(a1, b1, a0, b0)

    elif dist_type == "dirichlet":
        alphas0 = list(p_params.get("alphas", [1.0, 1.0]))
        if isinstance(actual_dist_or_obs, dict):
            alphas1 = list(
                actual_dist_or_obs.get("params", {}).get(
                    "alphas", actual_dist_or_obs.get("alphas", alphas0)
                )
            )
        elif isinstance(actual_dist_or_obs, (list, tuple)):
            alphas1 = [max(float(x), 1e-3) for x in actual_dist_or_obs]
        else:
            alphas1 = list(alphas0)
        kl = kl_divergence_dirichlet(alphas1, alphas0)

    elif dist_type == "categorical":
        p0 = list(p_params.get("probs", [0.5, 0.5]))
        if isinstance(actual_dist_or_obs, dict):
            p1 = list(
                actual_dist_or_obs.get("params", {}).get(
                    "probs", actual_dist_or_obs.get("probs", p0)
                )
            )
        elif isinstance(actual_dist_or_obs, (list, tuple)):
            p1 = list(actual_dist_or_obs)
        else:
            p1 = list(p0)
        kl = kl_divergence_categorical(p1, p0)

    else:  # Gaussian
        mu0 = float(p_params.get("mean", 0.0))
        std0 = float(p_params.get("std", 1.0))
        if isinstance(actual_dist_or_obs, dict):
            mu1 = float(
                actual_dist_or_obs.get("params", {}).get(
                    "mean", actual_dist_or_obs.get("mean", mu0)
                )
            )
            std1 = float(
                actual_dist_or_obs.get("params", {}).get(
                    "std", actual_dist_or_obs.get("std", std0)
                )
            )
        elif isinstance(actual_dist_or_obs, (int, float)):
            mu1 = float(actual_dist_or_obs)
            std1 = std0
        else:
            mu1, std1 = mu0, std0
        kl = kl_divergence_gaussian(mu1, std1, mu0, std0)

    thresholds = dict(
        cfg.get("distribution_thresholds")
        or cfg.get("thresholds")
        or {"medium": 0.50, "high": 1.50}
    )
    level = classify_surprise(kl, thresholds, {"medium": 0.50, "high": 1.50})
    return SurpriseResult(
        value=kl,
        level=level,
        predicted_dim=1,
        actual_dim=1,
    )
