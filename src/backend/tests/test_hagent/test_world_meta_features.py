"""
Comprehensive unit tests for World Model meta-features extraction (REFAC-014).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hagent.world.meta_features import META_KEYS_V2, meta_features_from_frame


def test_meta_features_keys_present() -> None:
    """Đảm bảo mọi key trong META_KEYS_V2 đều xuất hiện trong output."""
    df = pd.DataFrame(
        {
            "feat_num": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feat_cat": ["a", "b", "a", "b", "a"],
            "target": [0, 1, 0, 1, 0],
        }
    )
    mf = meta_features_from_frame(df, target="target")
    for k in META_KEYS_V2:
        assert k in mf
    assert mf["n_rows"] == 5.0
    assert mf["n_cols"] == 2.0
    assert mf["n_classes"] == 2.0
    assert mf["frac_categorical"] == 0.5


def test_meta_features_missing_values() -> None:
    """Tính missing_frac chính xác khi có missing values."""
    df = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, 4.0],
            "b": [np.nan, np.nan, 1.0, 2.0],
            "target": [0, 1, 0, 1],
        }
    )
    # Total feature cells: 4 rows * 2 cols = 8 cells, 3 NaNs -> 3/8 = 0.375
    mf = meta_features_from_frame(df, target="target")
    assert abs(mf["missing_frac"] - 0.375) < 1e-6


def test_meta_features_imbalance() -> None:
    """Tính class_imbalance chính xác."""
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "target": [0, 0, 0, 0, 1],  # 4/5 = 0.8 dominant class
        }
    )
    mf = meta_features_from_frame(df, target="target")
    assert abs(mf["class_imbalance"] - 0.8) < 1e-6


def test_meta_features_no_target() -> None:
    """Khi DataFrame không có cột target, các trường target gán về 0."""
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
        }
    )
    mf = meta_features_from_frame(df, target="non_existent_target")
    assert mf["n_classes"] == 0.0
    assert mf["class_imbalance"] == 0.0
    assert mf["n_cols"] == 2.0


def test_meta_features_empty_dataframe() -> None:
    """Xử lý DataFrame rỗng không gây crash."""
    df = pd.DataFrame(columns=["feat1", "target"])
    mf = meta_features_from_frame(df, target="target")
    assert mf["n_rows"] == 0.0
    assert mf["missing_frac"] == 0.0
    assert mf["class_imbalance"] == 0.0
