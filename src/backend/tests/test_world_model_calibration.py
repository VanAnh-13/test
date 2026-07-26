"""
Tests cho OutcomeEnsemble (deep ensemble) và world/calibration.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from hagent.world.calibration import (
    expected_calibration_error,
    interval_coverage,
    pit_values,
    reliability_table,
    sharpness,
)
from hagent.world.predictor import (
    OutcomeEnsemble,
    create_outcome_ensemble,
    train_outcome_ensemble,
)
from hagent.world.predictor.outcome_head_v1 import OutcomePrediction

DATASET_META = {"n_rows": 1000, "n_cols": 12}
HEAD_CFG = {"use_latent": False, "hidden_dim": 24}


def _synthetic_samples(n=100, seed=0, noise=0.01):
    rng = np.random.default_rng(seed)
    algo_bonus = {"grid_search": 0.0, "bayesian_search": 0.08, "genetic_algorithm": 0.04}
    samples = []
    for _ in range(n):
        algo = str(rng.choice(list(algo_bonus)))
        t = int(rng.choice([60, 180, 600]))
        score = (
            0.70
            + algo_bonus[algo]
            + 0.05 * (np.log1p(t) / np.log1p(600))
            + rng.normal(0, noise)
        )
        samples.append(
            {
                "params": {
                    "search_algorithm": algo,
                    "problem_type": "classification",
                    "metric": "accuracy",
                    "time_limit": t,
                },
                "dataset_meta": DATASET_META,
                "best_score": float(score),
            }
        )
    return samples


PARAMS = {
    "search_algorithm": "bayesian_search",
    "problem_type": "classification",
    "metric": "accuracy",
    "time_limit": 600,
}


# ── Ensemble ─────────────────────────────────────────────


class TestOutcomeEnsemble:
    def test_empty_ensemble_not_ready(self):
        ens = OutcomeEnsemble({})
        assert not ens.is_ready
        assert ens.predict(PARAMS, DATASET_META) is None

    def test_train_and_predict(self):
        ens = train_outcome_ensemble(
            _synthetic_samples(), config=dict(HEAD_CFG), k=3, epochs=40, seed=0
        )
        assert ens.is_ready
        assert len(ens.members) == 3
        pred = ens.predict(PARAMS, DATASET_META)
        assert pred is not None
        assert np.isfinite(pred.mean)
        assert pred.std > 0
        assert pred.meta["n_members"] == 3
        assert pred.meta["epistemic_std"] >= 0
        assert len(pred.meta["member_means"]) == 3

    def test_members_differ_by_seed(self):
        ens = train_outcome_ensemble(
            _synthetic_samples(), config=dict(HEAD_CFG), k=3, epochs=20, seed=0
        )
        means = ens.predict(PARAMS, DATASET_META).meta["member_means"]
        # 3 seed khác nhau không thể cho ra dự đoán trùng khớp tuyệt đối
        assert len(set(round(m, 12) for m in means)) > 1

    def test_total_var_geq_epistemic(self):
        ens = train_outcome_ensemble(
            _synthetic_samples(), config=dict(HEAD_CFG), k=4, epochs=30, seed=1
        )
        pred = ens.predict(PARAMS, DATASET_META)
        assert pred.std >= pred.meta["epistemic_std"] - 1e-9

    def test_save_load_roundtrip(self, tmp_path):
        ens = train_outcome_ensemble(
            _synthetic_samples(), config=dict(HEAD_CFG), k=3, epochs=30, seed=0
        )
        p1 = ens.predict(PARAMS, DATASET_META)
        ckpt_dir = str(tmp_path / "ens")
        ens.save(ckpt_dir)

        ens2 = OutcomeEnsemble(dict(HEAD_CFG, checkpoint_dir=ckpt_dir))
        assert ens2.is_ready
        assert len(ens2.members) == 3
        p2 = ens2.predict(PARAMS, DATASET_META)
        assert p1.mean == pytest.approx(p2.mean)
        assert p1.std == pytest.approx(p2.std)

    def test_save_empty_raises(self, tmp_path):
        with pytest.raises(RuntimeError):
            OutcomeEnsemble({}).save(str(tmp_path / "x"))

    def test_factory(self):
        assert create_outcome_ensemble({"enabled": False}) is None
        ens = create_outcome_ensemble({"k": 2})
        assert isinstance(ens, OutcomeEnsemble)
        assert ens.k == 2

    def test_ensemble_mean_close_to_truth(self):
        ens = train_outcome_ensemble(
            _synthetic_samples(200), config=dict(HEAD_CFG), k=3, epochs=60, seed=0
        )
        pred = ens.predict(PARAMS, DATASET_META)
        truth = 0.70 + 0.08 + 0.05  # bayesian + time_limit=600
        assert abs(pred.mean - truth) < 0.1


# ── Calibration metrics ──────────────────────────────────


def _calibrated_preds(n=2000, sigma=0.1, seed=0):
    """Dự đoán calibrated hoàn hảo: y ~ N(μ, σ) đúng với σ khai báo."""
    rng = np.random.default_rng(seed)
    mus = rng.uniform(0, 1, size=n)
    ys = mus + rng.normal(0, sigma, size=n)
    preds = [OutcomePrediction(mean=float(m), std=sigma) for m in mus]
    return preds, ys.tolist()


class TestCalibrationMetrics:
    def test_perfectly_calibrated_coverage(self):
        preds, ys = _calibrated_preds()
        cov = interval_coverage(preds, ys, confidence=0.9)
        assert cov == pytest.approx(0.9, abs=0.03)

    def test_overconfident_low_coverage(self):
        preds, ys = _calibrated_preds(sigma=0.1)
        overconf = [OutcomePrediction(mean=p.mean, std=0.02) for p in preds]
        cov = interval_coverage(overconf, ys, confidence=0.9)
        assert cov < 0.5

    def test_underconfident_high_coverage(self):
        preds, ys = _calibrated_preds(sigma=0.1)
        underconf = [OutcomePrediction(mean=p.mean, std=0.5) for p in preds]
        cov = interval_coverage(underconf, ys, confidence=0.9)
        assert cov > 0.99

    def test_ece_small_when_calibrated(self):
        preds, ys = _calibrated_preds()
        assert expected_calibration_error(preds, ys) < 0.03

    def test_ece_large_when_miscalibrated(self):
        preds, ys = _calibrated_preds(sigma=0.1)
        overconf = [OutcomePrediction(mean=p.mean, std=0.02) for p in preds]
        assert expected_calibration_error(overconf, ys) > 0.15

    def test_pit_uniformish_when_calibrated(self):
        preds, ys = _calibrated_preds()
        us = pit_values(preds, ys)
        assert 0.45 < float(np.mean(us)) < 0.55

    def test_reliability_table_monotone_nominal(self):
        preds, ys = _calibrated_preds()
        rows = reliability_table(preds, ys)
        assert [r["nominal"] for r in rows] == sorted(r["nominal"] for r in rows)
        for r in rows:
            assert r["empirical"] == pytest.approx(r["nominal"], abs=0.05)

    def test_sharpness(self):
        preds, _ = _calibrated_preds(n=10, sigma=0.1)
        assert sharpness(preds) == pytest.approx(0.1)
        assert sharpness([]) == 0.0

    def test_accepts_dicts_and_tuples(self):
        preds = [{"mean": 0.5, "std": 0.1}, (0.7, 0.1)]
        ys = [0.5, 0.7]
        assert interval_coverage(preds, ys, confidence=0.9) == 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            interval_coverage([(0.5, 0.1)], [0.5, 0.6])

    def test_bad_confidence_raises(self):
        with pytest.raises(ValueError):
            interval_coverage([(0.5, 0.1)], [0.5], confidence=1.5)


# ── Ensemble + calibration tích hợp ──────────────────────


class TestEnsembleCalibrationIntegration:
    def test_trained_ensemble_reasonably_calibrated_on_train_dist(self):
        samples = _synthetic_samples(300, noise=0.02)
        ens = train_outcome_ensemble(
            samples, config=dict(HEAD_CFG), k=3, epochs=60, seed=0
        )
        holdout = _synthetic_samples(150, seed=99, noise=0.02)
        preds, ys = [], []
        for s in holdout:
            p = ens.predict(s["params"], s["dataset_meta"])
            if p is not None:
                preds.append(p)
                ys.append(s["best_score"])
        assert len(preds) == len(holdout)
        # Workshop-grade sanity: coverage@90 không sụp đổ và ECE không thảm họa
        cov = interval_coverage(preds, ys, confidence=0.9)
        assert cov > 0.5
        assert expected_calibration_error(preds, ys) < 0.35
