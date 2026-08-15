"""
Tests cho automl/search/datasets_real.py — nạp dataset thật offline.
"""

from __future__ import annotations

import numpy as np
import pytest

from automl.search.datasets_real import (
    REGISTRY,
    available_datasets,
    load_dataset,
    load_real_datasets,
)

SMALL = {"iris", "wine", "breast_cancer", "digits", "glass", "online_shoppers"}


class TestRegistry:
    def test_small_datasets_registered(self):
        assert set(available_datasets(include_large=False)) == SMALL

    def test_large_dataset_registered_but_opt_in(self):
        """covtype (~250MB) phải có trong registry nhưng KHÔNG nằm trong
        mặc định — nếu không, mọi lần chạy benchmark đều kéo dài hàng chục phút."""
        assert "covtype" in available_datasets()
        assert "covtype" not in available_datasets(include_large=False)
        assert REGISTRY["covtype"]["large"] is True

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            load_dataset("nonexistent_dataset")


class TestSklearnBundled:
    @pytest.mark.parametrize(
        "name,rows,cols,classes",
        [
            ("iris", 150, 4, 3),
            ("wine", 178, 13, 3),
            ("breast_cancer", 569, 30, 2),
            ("digits", 1797, 64, 10),
        ],
    )
    def test_shapes(self, name, rows, cols, classes):
        d = load_dataset(name)
        assert d["n_rows"] == rows
        assert d["n_cols"] == cols
        assert d["n_classes"] == classes
        assert d["X"].shape == (rows, cols)
        assert len(d["y"]) == rows

    def test_no_network_needed(self):
        """sklearn bundled → nạp được kể cả khi OpenML sập."""
        d = load_dataset("iris")
        assert d["source"].startswith("sklearn.datasets.")


class TestRepoCsv:
    def test_glass(self):
        d = load_dataset("glass")
        assert d["n_rows"] == 214
        assert d["n_cols"] == 9  # 10 cột - target 'Type'
        assert d["n_classes"] == 6
        assert d["source"].endswith("glass.csv")

    def test_online_shoppers_encodes_categoricals(self):
        """Month/VisitorType/Weekend là phi số → one-hot, X phải toàn float."""
        d = load_dataset("online_shoppers")
        assert d["n_rows"] == 12330
        assert d["n_cols"] > 17  # 17 cột gốc + one-hot mở rộng
        assert d["n_classes"] == 2
        assert d["X"].dtype == np.float64
        assert np.isfinite(d["X"]).all()

    def test_target_is_integer_labels(self):
        d = load_dataset("online_shoppers")
        assert set(np.unique(d["y"])) == {0, 1}


class TestMetaFeatures:
    def test_meta_present_and_sane(self):
        d = load_dataset("glass")
        m = d["meta"]
        assert m["n_rows"] == 214
        assert m["n_classes"] == 6
        assert 0.0 < m["class_imbalance"] <= 1.0
        assert 0.0 <= m["missing_frac"] <= 1.0
        assert m["mean_abs_skew"] >= 0.0

    def test_imbalance_detects_skewed_target(self):
        """online_shoppers ~85% lớp âm → imbalance cao hơn iris cân bằng."""
        assert (
            load_dataset("online_shoppers")["meta"]["class_imbalance"]
            > load_dataset("iris")["meta"]["class_imbalance"]
        )


class TestBulkLoad:
    def test_load_all_small(self):
        ds = load_real_datasets(available_datasets(include_large=False))
        assert len(ds) == 6
        assert all(d["X"].shape[0] == len(d["y"]) for d in ds)

    def test_subset(self):
        ds = load_real_datasets(["iris", "wine"])
        assert [d["name"] for d in ds] == ["iris", "wine"]

    def test_bad_name_skipped_not_raised(self):
        ds = load_real_datasets(["iris", "does_not_exist"])
        assert [d["name"] for d in ds] == ["iris"]
