"""
Tests cho fetch_openml_datasets — registry + normalize_frame, không gọi mạng.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT = Path(__file__).parent.parent / "scripts" / "fetch_openml_datasets.py"
_spec = importlib.util.spec_from_file_location("fetch_openml_datasets", _SCRIPT)
mod = importlib.util.module_from_spec(_spec)
sys.modules["fetch_openml_datasets"] = mod
_spec.loader.exec_module(mod)


class TestRegistry:
    def test_eight_datasets_with_stable_ids(self):
        assert len(mod.DATASETS) == 8
        for name, spec in mod.DATASETS.items():
            assert isinstance(spec["data_id"], int) and spec["data_id"] > 0
            assert spec["problem_type"] in ("classification", "regression")

    def test_expected_names(self):
        assert set(mod.DATASETS) == {
            "credit-g",
            "diabetes",
            "vehicle",
            "blood-transfusion",
            "banknote",
            "kc1",
            "phoneme",
            "wine-quality-red",
        }


class TestNormalizeFrame:
    def test_renames_target_and_drops_na(self):
        df = pd.DataFrame(
            {"a": [1, 2, 3], "class": ["x", None, "y"]}
        )
        out = mod.normalize_frame(df, "class")
        assert "target" in out.columns
        assert "class" not in out.columns
        assert len(out) == 2

    def test_missing_target_raises(self):
        with pytest.raises(ValueError):
            mod.normalize_frame(pd.DataFrame({"a": [1]}), "nope")

    def test_index_reset(self):
        df = pd.DataFrame({"a": [1, 2, 3], "t": ["x", None, "y"]})
        out = mod.normalize_frame(df, "t")
        assert list(out.index) == [0, 1]
