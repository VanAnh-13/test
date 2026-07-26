"""
Tests cho CemConfigV1Planner và tích hợp world model vào build_campaign.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from hagent.agent.campaign.builder import build_campaign
from hagent.world.planner import CemConfigV1Planner, create_campaign_planner
from hagent.world.predictor.outcome_head_v1 import (
    OutcomeHeadV1,
    OutcomePrediction,
    train_outcome_head,
)


def run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


HEAD_CFG = {"use_latent": False, "hidden_dim": 32}
GOAL = {
    "goal_type": "train",
    "dataset_id": "ds1",
    "problem_type": "classification",
    "metric": "accuracy",
    "target_column": "target",
}


class FakeOutcomeModel:
    """Score được định nghĩa trực tiếp — kiểm soát hoàn toàn ground truth."""

    is_ready = True

    def __init__(self, score_fn, std=0.05):
        self.score_fn = score_fn
        self.std = std
        self.calls = 0
        self.seen_dataset_meta: list = []

    def predict(self, params, dataset_meta=None, z=None):
        self.calls += 1
        self.seen_dataset_meta.append(dataset_meta)
        return OutcomePrediction(
            mean=float(self.score_fn(params)), std=self.std,
            meta={"predictor": "fake"},
        )


def _bayes600_best(params):
    score = 0.5
    if params.get("search_algorithm") == "bayesian_search":
        score += 0.2
    score += 0.1 * (np.log1p(params.get("time_limit") or 0) / np.log1p(600))
    return score


# ── Planner core ─────────────────────────────────────────


class TestCemConfigPlanner:
    def test_finds_best_config(self):
        planner = CemConfigV1Planner({"seed": 0})
        out = planner.plan_campaign_configs(
            base_params={"problem_type": "classification", "metric": "accuracy"},
            outcome_model=FakeOutcomeModel(_bayes600_best),
            n_return=3,
        )
        assert out[0] == {"search_algorithm": "bayesian_search", "time_limit": 600}

    def test_returns_distinct_configs(self):
        planner = CemConfigV1Planner({"seed": 0})
        out = planner.plan_campaign_configs(
            base_params={},
            outcome_model=FakeOutcomeModel(_bayes600_best),
            n_return=3,
        )
        sigs = [(c["search_algorithm"], c["time_limit"]) for c in out]
        assert len(sigs) == len(set(sigs)) == 3

    def test_deterministic_by_seed(self):
        for _ in range(2):
            outs = [
                CemConfigV1Planner({"seed": 7}).plan_campaign_configs(
                    base_params={},
                    outcome_model=FakeOutcomeModel(_bayes600_best),
                    n_return=3,
                )
                for _ in range(2)
            ]
            assert outs[0] == outs[1]

    def test_lower_is_better(self):
        def mae_score(params):
            # bayesian + 600 cho MAE thấp nhất (tốt nhất)
            return 1.0 - _bayes600_best(params)

        planner = CemConfigV1Planner({"seed": 0})
        out = planner.plan_campaign_configs(
            base_params={"metric": "mae"},
            outcome_model=FakeOutcomeModel(mae_score),
            n_return=1,
            higher_is_better=False,
        )
        assert out[0] == {"search_algorithm": "bayesian_search", "time_limit": 600}

    def test_fallback_without_model(self):
        planner = CemConfigV1Planner({"seed": 0})
        out = planner.plan_campaign_configs(
            base_params={}, outcome_model=None, n_return=3
        )
        assert len(out) == 3
        assert out[0]["search_algorithm"] == "grid_search"

    def test_fallback_with_unready_model(self):
        planner = CemConfigV1Planner({})
        out = planner.plan_campaign_configs(
            base_params={}, outcome_model=OutcomeHeadV1({}), n_return=2
        )
        assert len(out) == 2

    def test_exploration_weight_prefers_uncertain(self):
        """β lớn → config có σ cao thắng dù mean thấp hơn chút."""

        class VarStd(FakeOutcomeModel):
            def predict(self, params, dataset_meta=None, z=None):
                if params.get("search_algorithm") == "genetic_algorithm":
                    return OutcomePrediction(mean=0.78, std=0.3)
                return OutcomePrediction(mean=0.80, std=0.01)

        greedy = CemConfigV1Planner({"seed": 0, "exploration_weight": 0.0})
        explore = CemConfigV1Planner({"seed": 0, "exploration_weight": 0.5})
        m = VarStd(lambda p: 0)
        assert (
            greedy.plan_campaign_configs(base_params={}, outcome_model=m, n_return=1)[0][
                "search_algorithm"
            ]
            != "genetic_algorithm"
        )
        assert (
            explore.plan_campaign_configs(base_params={}, outcome_model=m, n_return=1)[
                0
            ]["search_algorithm"]
            == "genetic_algorithm"
        )

    def test_exploration_bonus_correct_sign_when_minimizing(self):
        """Khi minimize, σ cao vẫn phải là BONUS khám phá, không phải phạt."""

        class VarStd(FakeOutcomeModel):
            def predict(self, params, dataset_meta=None, z=None):
                if params.get("search_algorithm") == "genetic_algorithm":
                    return OutcomePrediction(mean=1.0, std=0.3)
                return OutcomePrediction(mean=1.0, std=0.01)

        explore = CemConfigV1Planner({"seed": 0, "exploration_weight": 0.5})
        out = explore.plan_campaign_configs(
            base_params={"metric": "mae"},
            outcome_model=VarStd(lambda p: 0),
            n_return=1,
            higher_is_better=False,
        )
        assert out[0]["search_algorithm"] == "genetic_algorithm"

    def test_respects_custom_space(self):
        planner = CemConfigV1Planner(
            {"search_algorithms": ["grid_search"], "time_limit_options": [42], "seed": 0}
        )
        out = planner.plan_campaign_configs(
            base_params={}, outcome_model=FakeOutcomeModel(_bayes600_best), n_return=2
        )
        assert out == [{"search_algorithm": "grid_search", "time_limit": 42}]

    def test_factory(self):
        assert create_campaign_planner({"enabled": False}) is None
        assert isinstance(create_campaign_planner({}), CemConfigV1Planner)
        with pytest.raises(ValueError):
            create_campaign_planner({"backend": "mcts"})


# ── Builder integration ──────────────────────────────────


class TestBuilderIntegration:
    def test_builder_without_model_unchanged(self):
        """Không có model → không xuất hiện variant wm_planner (hành vi cũ)."""
        camp = run(
            build_campaign(GOAL, user_id="u1", config={"n_job_candidates": 3})
        )
        assert len(camp.variants) == 3
        assert all(v.source != "wm_planner" for v in camp.variants)

    def test_builder_with_model_proposes_wm_variants(self):
        model = FakeOutcomeModel(_bayes600_best)
        camp = run(
            build_campaign(
                GOAL,
                user_id="u1",
                config={"n_job_candidates": 3},
                outcome_model=model,
            )
        )
        assert len(camp.variants) == 3
        sources = {v.source for v in camp.variants}
        assert "wm_planner" in sources
        assert model.calls > 0
        # Ranking: variant hứa hẹn nhất (bayesian/600) submit trước
        first = camp.variants[0].params
        assert first["search_algorithm"] == "bayesian_search"
        assert first["time_limit"] == 600

    def test_builder_gates_off(self):
        model = FakeOutcomeModel(_bayes600_best)
        camp = run(
            build_campaign(
                GOAL,
                user_id="u1",
                config={
                    "n_job_candidates": 3,
                    "wm_variant_proposal": False,
                    "wm_rank_variants": False,
                },
                outcome_model=model,
            )
        )
        assert all(v.source != "wm_planner" for v in camp.variants)

    def test_builder_variants_distinct(self):
        model = FakeOutcomeModel(_bayes600_best)
        camp = run(
            build_campaign(
                GOAL, user_id="u1", config={"n_job_candidates": 4},
                outcome_model=model,
            )
        )
        sigs = [
            (v.params.get("search_algorithm"), v.params.get("time_limit"))
            for v in camp.variants
        ]
        assert len(sigs) == len(set(sigs))

    def test_builder_passes_dataset_meta_from_world_model(self):
        """Chống train/serve skew: meta từ snapshot phải tới được model."""
        model = FakeOutcomeModel(_bayes600_best)
        meta = {"n_rows": 777, "n_cols": 9}
        camp = run(
            build_campaign(
                GOAL,
                user_id="u1",
                world_model={"datasets": {"ds1": meta}},
                config={"n_job_candidates": 3},
                outcome_model=model,
            )
        )
        assert len(camp.variants) == 3
        assert any(m == meta for m in model.seen_dataset_meta)
        assert all(m == meta for m in model.seen_dataset_meta if m is not None)

    def test_builder_explicit_none_disables_model(self):
        """outcome_model=None phải tắt hẳn — không rơi về checkpoint đĩa."""
        camp = run(
            build_campaign(
                GOAL, user_id="u1", config={"n_job_candidates": 3},
                outcome_model=None,
            )
        )
        assert all(v.source != "wm_planner" for v in camp.variants)

    def test_builder_with_trained_head(self):
        """Đường đi thật: head numpy train từ synthetic data."""
        rng = np.random.default_rng(0)
        samples = []
        for _ in range(120):
            algo = str(rng.choice(["grid_search", "bayesian_search", "genetic_algorithm"]))
            t = int(rng.choice([180, 300, 600]))
            samples.append(
                {
                    "params": {
                        "search_algorithm": algo,
                        "problem_type": "classification",
                        "metric": "accuracy",
                        "time_limit": t,
                    },
                    "best_score": _bayes600_best(
                        {"search_algorithm": algo, "time_limit": t}
                    )
                    + float(rng.normal(0, 0.01)),
                }
            )
        head = train_outcome_head(samples, config=dict(HEAD_CFG), epochs=80, seed=0)
        camp = run(
            build_campaign(
                GOAL, user_id="u1", config={"n_job_candidates": 3},
                outcome_model=head,
            )
        )
        assert camp.variants[0].params["search_algorithm"] == "bayesian_search"
