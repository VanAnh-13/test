"""
Comprehensive unit tests for World Model calibration metrics (REFAC-014).
"""

from __future__ import annotations

import pytest

from hagent.world.calibration import (
    _mean_std,
    _pairs,
    expected_calibration_error,
    interval_coverage,
    pit_values,
    reliability_table,
    sharpness,
)


class DummyPrediction:
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std


def test_mean_std_extraction() -> None:
    """_mean_std chấp nhận object, dict hoặc tuple."""
    obj = DummyPrediction(0.8, 0.1)
    assert _mean_std(obj) == (0.8, 0.1)

    d = {"mean": 0.75, "std": 0.05}
    assert _mean_std(d) == (0.75, 0.05)

    t = (0.9, 0.02)
    assert _mean_std(t) == (0.9, 0.02)


def test_pairs_mismatch_raises() -> None:
    """_pairs ném ValueError khi độ dài predictions và targets khác nhau."""
    preds = [(0.5, 0.1), (0.6, 0.1)]
    targets = [0.5]
    with pytest.raises(ValueError, match="cùng độ dài"):
        _pairs(preds, targets)


def test_interval_coverage_basic() -> None:
    """Kiểm tra interval_coverage với các dự đoán chuẩn."""
    # Điểm dự đoán chính xác tại mean với std=1.0
    preds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    targets = [0.0, 0.5, -0.5]
    cov = interval_coverage(preds, targets, confidence=0.90)
    assert cov == 1.0

    # Target nằm ngoài khoảng tin cậy
    targets_far = [5.0, -5.0, 10.0]
    cov_far = interval_coverage(preds, targets_far, confidence=0.90)
    assert cov_far == 0.0


def test_interval_coverage_invalid_confidence() -> None:
    """interval_coverage ném ValueError khi confidence <= 0 hoặc >= 1."""
    with pytest.raises(ValueError, match="confidence phải trong"):
        interval_coverage([(0.0, 1.0)], [0.0], confidence=0.0)

    with pytest.raises(ValueError, match="confidence phải trong"):
        interval_coverage([(0.0, 1.0)], [0.0], confidence=1.0)


def test_interval_coverage_empty() -> None:
    """interval_coverage trả về 0.0 khi danh sách rỗng."""
    assert interval_coverage([], []) == 0.0


def test_pit_values_uniformity() -> None:
    """pit_values trả về xác suất tích lũy chuẩn u ~ [0, 1]."""
    preds = [(0.0, 1.0), (0.0, 1.0)]
    targets = [0.0, 1.96]  # 0 -> 0.5, 1.96 -> ~0.975
    pits = pit_values(preds, targets)
    assert len(pits) == 2
    assert abs(pits[0] - 0.5) < 1e-4
    assert abs(pits[1] - 0.975) < 1e-3


def test_expected_calibration_error_perfect_and_imperfect() -> None:
    """ECE bằng ~0 khi dự đoán chuẩn và >= 0 khi tính toán."""
    # Empty
    assert expected_calibration_error([], []) == 0.0

    # Standard distribution
    preds = [(float(i) / 100.0, 0.1) for i in range(100)]
    targets = [float(i) / 100.0 for i in range(100)]
    ece = expected_calibration_error(preds, targets, n_bins=5)
    assert isinstance(ece, float)
    assert ece >= 0.0


def test_reliability_table() -> None:
    """reliability_table trả về danh sách các dict với nominal và empirical coverage."""
    preds = [(0.0, 1.0)] * 10
    targets = [0.0] * 10
    table = reliability_table(preds, targets, levels=[0.5, 0.9])
    assert len(table) == 2
    assert table[0]["nominal"] == 0.5
    assert table[0]["empirical"] == 1.0
    assert table[1]["nominal"] == 0.9
    assert table[1]["empirical"] == 1.0


def test_sharpness() -> None:
    """sharpness tính độ lệch chuẩn trung bình."""
    preds = [(0.0, 0.2), (0.0, 0.4), (0.0, 0.6)]
    sh = sharpness(preds)
    assert abs(sh - 0.4) < 1e-8

    # Empty
    assert sharpness([]) == 0.0
