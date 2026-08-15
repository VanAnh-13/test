"""
Comprehensive unit tests for World Model surprise calculation and KL divergence (REFAC-014).
"""

from __future__ import annotations

import math

from hagent.world.schema import LatentState, SurpriseResult
from hagent.world.surprise import (
    _digamma,
    compute_aggregate_plan_surprise,
    compute_outcome_surprise,
    compute_surprise,
    latent_distance,
    should_trigger_plan_revision,
)


def test_digamma_properties() -> None:
    """_digamma thỏa mãn tính chất giải tích psi(x+1) - psi(x) = 1/x."""
    for x in (0.5, 1.0, 2.5, 5.0, 10.0):
        psi_x = _digamma(x)
        psi_x1 = _digamma(x + 1.0)
        assert abs((psi_x1 - psi_x) - (1.0 / x)) < 1e-6

    # Non-positive input
    assert _digamma(0.0) == float("-inf")
    assert _digamma(-1.0) == float("-inf")


def test_latent_distance_metrics() -> None:
    """latent_distance hỗ trợ nhiều khoảng cách: euclidean, cosine, manhattan."""
    z1 = LatentState(vector=[1.0, 0.0, 0.0], dim=3)
    z2 = LatentState(vector=[0.0, 1.0, 0.0], dim=3)

    # L2 distance: sqrt(1^2 + 1^2) = sqrt(2) ≈ 1.4142
    d_l2 = latent_distance(z1, z2, metric="l2")
    assert abs(d_l2 - math.sqrt(2)) < 1e-5

    # L1 distance: |1| + |1| = 2.0
    d_l1 = latent_distance(z1, z2, metric="l1")
    assert abs(d_l1 - 2.0) < 1e-5

    # Cosine: 1 - 0 = 1.0
    d_cos = latent_distance(z1, z2, metric="cosine")
    assert abs(d_cos - 1.0) < 1e-5

    # Normalized with meta std
    z_std = LatentState(vector=[0.0, 0.0, 0.0], dim=3, meta={"std": [0.5, 0.5, 0.5]})
    z_obs = LatentState(vector=[1.0, 0.0, 0.0], dim=3)
    d_norm = latent_distance(z_std, z_obs)
    # (1.0 / 0.5) / sqrt(3) ≈ 2.0 / 1.732 ≈ 1.1547
    assert d_norm > 0.0


def test_compute_surprise_identical_and_contradictory() -> None:
    """compute_surprise phân loại cấp độ ngạc nhiên chính xác."""
    z0 = LatentState(vector=[1.0, 0.0, 0.0], dim=3)
    z_same = LatentState(vector=[1.0, 0.0, 0.0], dim=3)
    z_far = LatentState(vector=[-10.0, -10.0, -10.0], dim=3)

    s_low = compute_surprise(z0, z_same)
    assert s_low.level == "low"
    assert s_low.value < 0.01

    s_high = compute_surprise(z0, z_far)
    assert s_high.level == "high"
    assert s_high.value > 1.0


def test_compute_outcome_surprise() -> None:
    """compute_outcome_surprise đo độ lệch z-score."""
    pred_obj = type("Pred", (), {"mean": 0.80, "std": 0.05})()

    # Outcome trùng mean -> surprise ~0 (low)
    s0 = compute_outcome_surprise(pred_obj, 0.80)
    assert s0.level == "low"
    assert s0.value == 0.0

    # Outcome lệch 4 sigma -> high surprise
    s_far = compute_outcome_surprise(pred_obj, 0.60)
    assert s_far.level == "high"
    assert abs(s_far.value - 4.0) < 1e-5


def test_compute_aggregate_plan_surprise_methods() -> None:
    """compute_aggregate_plan_surprise hỗ trợ mean, rms, max."""
    steps = [
        SurpriseResult(value=0.1, level="low", predicted_dim=1, actual_dim=1),
        SurpriseResult(value=0.5, level="medium", predicted_dim=1, actual_dim=1),
        SurpriseResult(value=0.9, level="high", predicted_dim=1, actual_dim=1),
    ]

    # Mean: (0.1 + 0.5 + 0.9) / 3 = 0.5
    s_mean = compute_aggregate_plan_surprise(steps, method="mean")
    assert abs(s_mean.value - 0.5) < 1e-6

    # Max: 0.9
    s_max = compute_aggregate_plan_surprise(steps, method="max")
    assert abs(s_max.value - 0.9) < 1e-6

    # RMS: sqrt((0.01 + 0.25 + 0.81)/3) = sqrt(0.35666) ≈ 0.5972
    s_rms = compute_aggregate_plan_surprise(steps, method="rms")
    assert abs(s_rms.value - math.sqrt((0.01 + 0.25 + 0.81) / 3)) < 1e-5

    # Empty sequence
    s_empty = compute_aggregate_plan_surprise([])
    assert s_empty.value == 0.0
    assert s_empty.level == "low"


def test_should_trigger_plan_revision() -> None:
    """should_trigger_plan_revision quyết định replanning chính xác."""
    s_low = SurpriseResult(value=0.1, level="low", predicted_dim=1, actual_dim=1)
    s_high = SurpriseResult(value=0.9, level="high", predicted_dim=1, actual_dim=1)

    assert should_trigger_plan_revision(s_low, s_low) is False
    assert should_trigger_plan_revision(s_high, s_low) is True
    assert should_trigger_plan_revision(s_low, s_high) is True
    assert should_trigger_plan_revision(None, None) is False
