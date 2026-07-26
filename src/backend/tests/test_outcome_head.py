"""
Tests cho OutcomeHeadV1 — feature extraction, train NLL, save/load, ranking.
"""

from __future__ import annotations

import numpy as np
import pytest

from hagent.agent.campaign.schema import CampaignVariant
from hagent.world.predictor import create_outcome_head
from hagent.world.predictor.outcome_head_v1 import (
    OutcomeHeadV1,
    OutcomePrediction,
    extract_outcome_samples,
    outcome_feature_dim,
    outcome_features,
    rank_variants_by_outcome,
    train_outcome_head,
)

PARAMS_GRID = {
    "search_algorithm": "grid_search",
    "problem_type": "classification",
    "metric": "accuracy",
    "time_limit": 180,
    "models": ["random_forest", "svm"],
}
DATASET_META = {"n_rows": 1000, "n_cols": 12}


# ── Features ─────────────────────────────────────────────


class TestOutcomeFeatures:
    def test_fixed_dim_and_deterministic(self):
        x1 = outcome_features(PARAMS_GRID, DATASET_META, z=[0.1] * 64)
        x2 = outcome_features(PARAMS_GRID, DATASET_META, z=[0.1] * 64)
        assert x1.shape == x2.shape
        assert np.allclose(x1, x2)
        assert x1.shape[0] == outcome_feature_dim()

    def test_handles_missing_everything(self):
        x = outcome_features({}, None, None)
        assert x.shape[0] == outcome_feature_dim()
        assert np.all(np.isfinite(x))

    def test_unknown_values_hit_unknown_bucket(self):
        x_known = outcome_features({"search_algorithm": "grid_search"})
        x_unknown = outcome_features({"search_algorithm": "alien_search"})
        assert not np.allclose(x_known, x_unknown)
        assert np.all(np.isfinite(x_unknown))

    def test_no_latent_when_disabled(self):
        cfg = {"use_latent": False}
        dim_no_z = outcome_feature_dim(cfg)
        dim_z = outcome_feature_dim({"use_latent": True, "latent_dim": 64})
        assert dim_z == dim_no_z + 64

    def test_different_params_different_features(self):
        x1 = outcome_features(PARAMS_GRID, DATASET_META)
        p2 = dict(PARAMS_GRID, search_algorithm="bayesian_search", time_limit=600)
        x2 = outcome_features(p2, DATASET_META)
        assert not np.allclose(x1, x2)


# ── Head lifecycle ───────────────────────────────────────


class TestOutcomeHeadLifecycle:
    def test_untrained_predicts_none(self):
        head = OutcomeHeadV1({})
        assert not head.is_ready
        assert head.predict(PARAMS_GRID, DATASET_META) is None

    def test_init_random_predicts_finite(self):
        head = OutcomeHeadV1({})
        head.init_random(seed=0)
        pred = head.predict(PARAMS_GRID, DATASET_META, z=[0.0] * 64)
        assert isinstance(pred, OutcomePrediction)
        assert np.isfinite(pred.mean)
        assert pred.std > 0

    def test_save_load_roundtrip(self, tmp_path):
        head = OutcomeHeadV1({"hidden_dim": 32})
        head.init_random(seed=1)
        p1 = head.predict(PARAMS_GRID, DATASET_META)

        ckpt = str(tmp_path / "outcome.npz")
        head.save(ckpt)
        head2 = OutcomeHeadV1({"hidden_dim": 32, "checkpoint_path": ckpt})
        assert head2.is_ready
        p2 = head2.predict(PARAMS_GRID, DATASET_META)
        assert p1.mean == pytest.approx(p2.mean)
        assert p1.std == pytest.approx(p2.std)

    def test_missing_checkpoint_not_ready(self, tmp_path):
        head = OutcomeHeadV1({"checkpoint_path": str(tmp_path / "nope.npz")})
        assert not head.is_ready

    def test_factory_disabled_returns_none(self):
        assert create_outcome_head({"enabled": False}) is None

    def test_factory_creates_head(self):
        head = create_outcome_head({"hidden_dim": 16})
        assert isinstance(head, OutcomeHeadV1)

    def test_factory_rejects_unknown_backend(self):
        with pytest.raises(ValueError):
            create_outcome_head({"backend": "quantum_head"})


# ── Training ─────────────────────────────────────────────


def _synthetic_samples(n=120, seed=0):
    """best_score phụ thuộc algorithm + time_limit → head phải học được thứ tự."""
    rng = np.random.default_rng(seed)
    algo_bonus = {"grid_search": 0.0, "bayesian_search": 0.08, "genetic_algorithm": 0.04}
    samples = []
    for _ in range(n):
        algo = rng.choice(list(algo_bonus))
        t = int(rng.choice([60, 180, 600]))
        score = (
            0.70
            + algo_bonus[str(algo)]
            + 0.05 * (np.log1p(t) / np.log1p(600))
            + rng.normal(0, 0.01)
        )
        samples.append(
            {
                "params": {
                    "search_algorithm": str(algo),
                    "problem_type": "classification",
                    "metric": "accuracy",
                    "time_limit": t,
                },
                "dataset_meta": DATASET_META,
                "best_score": float(score),
            }
        )
    return samples


class TestTraining:
    def test_nll_decreases(self):
        head = train_outcome_head(
            _synthetic_samples(), config={"use_latent": False, "hidden_dim": 32},
            epochs=60, lr=0.01, seed=0,
        )
        hist = head.config.get("train_history")
        assert hist and hist[-1] < hist[0]

    def test_learned_ordering(self):
        head = train_outcome_head(
            _synthetic_samples(), config={"use_latent": False, "hidden_dim": 32},
            epochs=80, lr=0.01, seed=0,
        )
        best = head.predict(
            {"search_algorithm": "bayesian_search", "problem_type": "classification",
             "metric": "accuracy", "time_limit": 600},
            DATASET_META,
        )
        worst = head.predict(
            {"search_algorithm": "grid_search", "problem_type": "classification",
             "metric": "accuracy", "time_limit": 60},
            DATASET_META,
        )
        assert best.mean > worst.mean

    def test_empty_samples_returns_untrained_head(self):
        head = train_outcome_head([], config={"use_latent": False})
        # init_random đã chạy nên head vẫn ready, nhưng không có history
        assert head.config.get("train_history") is None

    def test_prediction_in_sane_range(self):
        head = train_outcome_head(
            _synthetic_samples(), config={"use_latent": False, "hidden_dim": 32},
            epochs=60, lr=0.01, seed=0,
        )
        pred = head.predict(
            {"search_algorithm": "genetic_algorithm", "problem_type": "classification",
             "metric": "accuracy", "time_limit": 180},
            DATASET_META,
        )
        assert 0.4 < pred.mean < 1.1
        assert 0 < pred.std < 0.5


# ── extract_outcome_samples ──────────────────────────────


def _traj_doc(job_id, status="completed", best_score=0.9, with_config=True):
    return {
        "next_observation": {
            "jobs": {
                job_id: {
                    "id": job_id,
                    "dataset_id": "ds1",
                    "status": status,
                    "best_score": best_score,
                    "config": (
                        {"search_algorithm": "grid_search", "time_limit": 180}
                        if with_config
                        else None
                    ),
                }
            },
            "datasets": {"ds1": {"n_rows": 500, "n_cols": 8}},
        },
        "z_next": {"vector": [0.1] * 64, "dim": 64},
    }


class TestExtractSamples:
    def test_extracts_completed_jobs(self):
        samples = extract_outcome_samples([_traj_doc("j1"), _traj_doc("j2")])
        assert len(samples) == 2
        s = samples[0]
        assert s["best_score"] == 0.9
        assert s["dataset_meta"]["n_rows"] == 500
        assert len(s["z"]) == 64

    def test_skips_incomplete_and_scoreless(self):
        docs = [
            _traj_doc("j1", status="running"),
            _traj_doc("j2", best_score=None),
            _traj_doc("j3", with_config=False),
        ]
        assert extract_outcome_samples(docs) == []

    def test_dedups_by_job_id_keeps_latest(self):
        d1 = _traj_doc("j1", best_score=0.5)
        d2 = _traj_doc("j1", best_score=0.95)
        samples = extract_outcome_samples([d1, d2])
        assert len(samples) == 1
        assert samples[0]["best_score"] == 0.95

    def test_empty_input(self):
        assert extract_outcome_samples([]) == []
        assert extract_outcome_samples(None) == []


# ── Ranking ──────────────────────────────────────────────


def _variants():
    return [
        CampaignVariant(
            variant_id=f"v{i}",
            label=f"variant_{i}",
            params={
                "search_algorithm": algo,
                "problem_type": "classification",
                "metric": "accuracy",
                "time_limit": t,
            },
        )
        for i, (algo, t) in enumerate(
            [("grid_search", 60), ("bayesian_search", 600), ("genetic_algorithm", 180)]
        )
    ]


class TestRanking:
    def test_no_head_preserves_order(self):
        variants = _variants()
        ranked = rank_variants_by_outcome(variants, head=None)
        assert [v.variant_id for v, _ in ranked] == [v.variant_id for v in variants]
        assert all(pred is None for _, pred in ranked)

    def test_untrained_head_preserves_order(self):
        variants = _variants()
        ranked = rank_variants_by_outcome(variants, head=OutcomeHeadV1({}))
        assert [v.variant_id for v, _ in ranked] == [v.variant_id for v in variants]

    def test_trained_head_ranks_by_mean(self):
        head = train_outcome_head(
            _synthetic_samples(), config={"use_latent": False, "hidden_dim": 32},
            epochs=80, lr=0.01, seed=0,
        )
        ranked = rank_variants_by_outcome(_variants(), head=head, dataset_meta=DATASET_META)
        means = [pred.mean for _, pred in ranked]
        assert means == sorted(means, reverse=True)
        # bayesian + 600s phải đứng đầu theo synthetic ground truth
        assert ranked[0][0].params["search_algorithm"] == "bayesian_search"

    def test_lower_is_better_flips_order(self):
        head = train_outcome_head(
            _synthetic_samples(), config={"use_latent": False, "hidden_dim": 32},
            epochs=40, lr=0.01, seed=0,
        )
        ranked = rank_variants_by_outcome(
            _variants(), head=head, dataset_meta=DATASET_META, higher_is_better=False
        )
        means = [pred.mean for _, pred in ranked]
        assert means == sorted(means)
