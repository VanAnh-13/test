"""
Unit tests for configurable surprise thresholds and multi-scale plan surprise (REFAC-011).
"""

from __future__ import annotations

from hagent.world.schema import LatentState, SurpriseResult
from hagent.world.surprise import (
    classify_surprise,
    compute_aggregate_plan_surprise,
    compute_outcome_surprise,
    compute_surprise,
    should_trigger_plan_revision,
)


def test_classify_surprise_default_and_custom_thresholds() -> None:
    """classify_surprise sử dụng thresholds từ config và fallback an toàn sang defaults."""
    # Default thresholds (0.15, 0.40)
    assert classify_surprise(0.10) == "low"
    assert classify_surprise(0.20) == "medium"
    assert classify_surprise(0.50) == "high"

    # Custom strict thresholds (0.05, 0.10)
    strict_cfg = {"medium": 0.05, "high": 0.10}
    assert classify_surprise(0.08, strict_cfg) == "medium"
    assert classify_surprise(0.12, strict_cfg) == "high"

    # Custom loose thresholds (0.50, 0.90)
    loose_cfg = {"medium": 0.50, "high": 0.90}
    assert classify_surprise(0.40, loose_cfg) == "low"
    assert classify_surprise(0.60, loose_cfg) == "medium"
    assert classify_surprise(0.95, loose_cfg) == "high"


def test_compute_surprise_with_configured_thresholds() -> None:
    """compute_surprise nhận thresholds từ config dict."""
    z_pred = LatentState(vector=[0.0, 0.0], dim=2)
    z_actual = LatentState(vector=[0.3, 0.0], dim=2)  # distance = 0.3

    # Default config: 0.3 >= 0.15 (medium)
    res_default = compute_surprise(z_pred, z_actual, {})
    assert res_default.level == "medium"

    # Strict config where 0.3 is high (high=0.25)
    strict_res = compute_surprise(
        z_pred, z_actual, {"thresholds": {"medium": 0.10, "high": 0.25}}
    )
    assert strict_res.level == "high"

    # Tolerant config where 0.3 is low (medium=0.50)
    loose_res = compute_surprise(
        z_pred, z_actual, {"thresholds": {"medium": 0.50, "high": 1.00}}
    )
    assert loose_res.level == "low"


def test_compute_outcome_surprise_with_configured_thresholds() -> None:
    """compute_outcome_surprise nhận outcome_thresholds từ config."""
    pred = (0.80, 0.05)  # mean=0.80, std=0.05
    actual = 0.68  # diff = 0.12, z-score = 0.12 / 0.05 = 2.4

    # Default: medium=1.5, high=3.0 -> 2.4 is medium
    res = compute_outcome_surprise(pred, actual, {})
    assert res.level == "medium"

    # Config with strict threshold (high=2.0) -> 2.4 is high
    res_strict = compute_outcome_surprise(
        pred, actual, {"outcome_thresholds": {"medium": 1.0, "high": 2.0}}
    )
    assert res_strict.level == "high"


def test_compute_aggregate_plan_surprise() -> None:
    """compute_aggregate_plan_surprise tính tổng hợp surprise toàn bộ kế hoạch."""
    s1 = SurpriseResult(value=0.2, level="medium", predicted_dim=2, actual_dim=2)
    s2 = SurpriseResult(value=0.4, level="medium", predicted_dim=2, actual_dim=2)
    s3 = SurpriseResult(value=0.9, level="high", predicted_dim=2, actual_dim=2)

    # Mean: (0.2 + 0.4 + 0.9) / 3 = 0.50
    plan_surp_mean = compute_aggregate_plan_surprise([s1, s2, s3], method="mean")
    assert abs(plan_surp_mean.value - 0.50) < 1e-6
    assert plan_surp_mean.level == "medium"

    # Max: 0.90
    plan_surp_max = compute_aggregate_plan_surprise([s1, s2, s3], method="max")
    assert abs(plan_surp_max.value - 0.90) < 1e-6

    # Empty list
    empty_plan = compute_aggregate_plan_surprise([])
    assert empty_plan.value == 0.0
    assert empty_plan.level == "low"


def test_should_trigger_plan_revision_multi_scale() -> None:
    """should_trigger_plan_revision kích hoạt khi step hoặc plan surprise đạt ngưỡng."""
    step_low = SurpriseResult(value=0.1, level="low", predicted_dim=1, actual_dim=1)
    step_high = SurpriseResult(value=0.9, level="high", predicted_dim=1, actual_dim=1)
    plan_low = SurpriseResult(value=0.2, level="low", predicted_dim=0, actual_dim=0)
    plan_high = SurpriseResult(value=1.2, level="high", predicted_dim=0, actual_dim=0)

    # Cả hai low -> không revise
    assert not should_trigger_plan_revision(step_low, plan_low)

    # Step high -> revise
    assert should_trigger_plan_revision(step_high, plan_low)

    # Plan aggregate high (dù step hiện tại low) -> revise
    assert should_trigger_plan_revision(step_low, plan_high)
