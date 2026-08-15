"""
Tests cho world/meta_features.py và transfer LOO trong benchmark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hagent.agent.eval.benchmark import (
    generate_offline_samples,
    make_transfer_profiles,
    run_transfer_loo,
)
from hagent.world.meta_features import META_KEYS_V2, meta_features_from_frame
from hagent.world.predictor.outcome_head_v1 import train_outcome_head


class TestMetaFeaturesFromFrame:
    def _frame(self):
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            {
                "num1": rng.normal(0, 1, 100),
                "num2": rng.exponential(2, 100),  # skewed
                "cat1": ["a", "b"] * 50,
                "target": ["x"] * 70 + ["y"] * 30,
            }
        )

    def test_all_keys_present(self):
        meta = meta_features_from_frame(self._frame())
        assert set(META_KEYS_V2) <= set(meta)

    def test_values_correct(self):
        meta = meta_features_from_frame(self._frame())
        assert meta["n_rows"] == 100
        assert meta["n_cols"] == 3
        assert meta["n_classes"] == 2
        assert meta["class_imbalance"] == pytest.approx(0.7)
        assert meta["frac_categorical"] == pytest.approx(1 / 3)
        assert meta["missing_frac"] == 0.0
        assert meta["mean_abs_skew"] > 0

    def test_missing_values_counted(self):
        df = self._frame()
        df.loc[:9, "num1"] = np.nan
        meta = meta_features_from_frame(df)
        assert meta["missing_frac"] == pytest.approx(10 / 300)

    def test_no_target_column(self):
        df = self._frame().drop(columns=["target"])
        meta = meta_features_from_frame(df)
        assert meta["n_classes"] == 0
        assert meta["class_imbalance"] == 0.0

    def test_empty_frame(self):
        meta = meta_features_from_frame(pd.DataFrame())
        assert meta["n_rows"] == 0
        assert all(np.isfinite(v) for v in meta.values())


class TestTransferProfiles:
    def test_deterministic(self):
        p1 = make_transfer_profiles(6, seed=0)
        p2 = make_transfer_profiles(6, seed=0)
        assert [p.meta for p in p1] == [p.meta for p in p2]
        assert [p.algo_bonus for p in p1] == [p.algo_bonus for p in p2]

    def test_best_algo_depends_on_meta(self):
        profiles = make_transfer_profiles(12, seed=0)
        import math

        for p in profiles:
            big = math.log1p(p.meta["n_rows"]) / math.log(1e6) > 0.55
            best = max(p.algo_bonus, key=p.algo_bonus.get)
            if big:
                assert p.algo_bonus["bayesian_search"] > 0
            else:
                assert p.algo_bonus["grid_search"] > 0
            assert best in ("grid_search", "bayesian_search", "genetic_algorithm")

    def test_offline_samples_shape(self):
        prof = make_transfer_profiles(3, seed=1)[0]
        samples = generate_offline_samples(prof, m=20, seed=0)
        assert len(samples) == 20
        assert all("dataset_meta" in s and "best_score" in s for s in samples)


class TestCrossDatasetTransfer:
    def test_head_generalizes_to_unseen_profile(self):
        """Head train trên 5 profile phải dự đoán đúng THUẬT TOÁN tốt nhất
        trên profile chưa từng thấy (luật phụ thuộc meta)."""
        profiles = make_transfer_profiles(6, seed=0)
        held = profiles[0]
        train_samples = []
        for i, p in enumerate(profiles[1:], start=1):
            train_samples.extend(generate_offline_samples(p, m=60, seed=i))
        head = train_outcome_head(
            train_samples,
            config={"use_latent": False, "hidden_dim": 32},
            epochs=80,
            seed=0,
        )

        def pred_mean(algo):
            return head.predict(
                {
                    "search_algorithm": algo,
                    "problem_type": "classification",
                    "metric": "accuracy",
                    "time_limit": 600,
                },
                dict(held.meta),
            ).mean

        true_best = max(held.algo_bonus, key=held.algo_bonus.get)
        preds = {a: pred_mean(a) for a in held.algo_bonus}
        assert max(preds, key=preds.get) == true_best

    def test_loo_pretrained_at_least_scratch(self):
        """Pretrained có model từ job 0 → không thể tệ hơn scratch trên
        cùng seed/budget (luật meta rõ, nhiễu nhỏ)."""
        out = run_transfer_loo(k=6, heldout_index=0, budget_jobs=9, seed=0)
        assert out["pretrained"]["wm_trained_after_jobs"] == 0
        assert out["scratch"]["wm_trained_after_jobs"] not in (None, 0)
        assert out["pretrained"]["final_best"] >= out["scratch"]["final_best"] - 1e-9

    def test_loo_result_serializable(self):
        import json

        out = run_transfer_loo(k=4, heldout_index=1, budget_jobs=6, seed=1)
        json.dumps(out)
