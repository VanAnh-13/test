"""
Tests cho T7 — budget campaign + plan_batch (MPC) trong vòng mở rộng.
"""

from __future__ import annotations

import asyncio

import numpy as np

from hagent.agent.campaign.builder import build_campaign, propose_extension_variants
from hagent.agent.campaign.schema import Campaign
from hagent.world.predictor.outcome_head_v1 import train_outcome_head

GOAL = {
    "goal_type": "train",
    "dataset_id": "ds1",
    "problem_type": "classification",
    "metric": "accuracy",
    "target_column": "target",
}
DATASET_META = {"n_rows": 1000, "n_cols": 12}
BUILD_CFG = {
    "n_job_candidates": 3,
    "warm_start_top_k": 0,
    "wm_variant_proposal": False,
    "wm_rank_variants": False,
}


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _head(seed=0):
    rng = np.random.default_rng(seed)
    algos = ["grid_search", "bayesian_search", "genetic_algorithm"]
    samples = [
        {
            "params": {
                "search_algorithm": str(rng.choice(algos)),
                "problem_type": "classification",
                "metric": "accuracy",
                "time_limit": int(rng.choice([180, 300, 600])),
            },
            "best_score": float(0.8 + rng.normal(0, 0.02)),
        }
        for _ in range(60)
    ]
    return train_outcome_head(
        samples, config={"use_latent": False, "hidden_dim": 24}, epochs=40, seed=seed
    )


class TestBudgetFields:
    def test_default_budget_from_n(self):
        camp = run(
            build_campaign(GOAL, user_id="t7", config=BUILD_CFG, outcome_model=None)
        )
        assert camp.total_budget == 6  # n=3 → fallback n×2

    def test_budget_from_goal_constraints(self):
        goal = dict(GOAL, constraints={"max_jobs": 10})
        camp = run(
            build_campaign(goal, user_id="t7", config=BUILD_CFG, outcome_model=None)
        )
        assert camp.total_budget == 10

    def test_roundtrip(self):
        camp = Campaign(
            campaign_id="c",
            goal=dict(GOAL),
            variants=[],
            total_budget=9,
            spent_budget=4,
        )
        back = Campaign.from_dict(camp.to_dict())
        assert back.total_budget == 9
        assert back.spent_budget == 4


class TestExtensionRespectsBudget:
    def _camp(self, total, spent, n_variants=3):
        camp = run(
            build_campaign(GOAL, user_id="t7", config=BUILD_CFG, outcome_model=None)
        )
        camp.total_budget = total
        camp.spent_budget = spent
        return camp

    def test_no_extension_when_budget_exhausted(self):
        camp = self._camp(total=3, spent=3)
        new = propose_extension_variants(
            camp, GOAL, dataset_meta=DATASET_META, outcome_model=_head(), n_extra=2
        )
        assert new == []

    def test_n_extra_capped_by_remaining(self):
        camp = self._camp(total=4, spent=3)  # còn đúng 1
        new = propose_extension_variants(
            camp, GOAL, dataset_meta=DATASET_META, outcome_model=_head(), n_extra=5
        )
        assert len(new) <= 1

    def test_full_remaining_allows_n_extra(self):
        camp = self._camp(total=10, spent=3)
        new = propose_extension_variants(
            camp, GOAL, dataset_meta=DATASET_META, outcome_model=_head(), n_extra=2
        )
        assert len(new) == 2


class TestMpcPlanBatchPath:
    def test_plan_batch_receives_budgets(self, monkeypatch):
        """Backend cem_mpc_v1 → extension đi qua plan_batch với đúng
        remaining/total (budget-annealed Thompson có đất trong prod)."""
        import hagent.bridge.config as bridge_config

        real_wm = bridge_config.get_world_model_config

        def patched_wm():
            cfg = dict(real_wm())
            planner_cfg = dict(cfg.get("campaign_planner") or {})
            planner_cfg["backend"] = "cem_mpc_v1"
            cfg["campaign_planner"] = planner_cfg
            return cfg

        monkeypatch.setattr(bridge_config, "get_world_model_config", patched_wm)

        seen = {}
        from hagent.world.planner.cem_mpc_v1 import CemMpcV1Planner

        real_plan_batch = CemMpcV1Planner.plan_batch

        def spy(self, **kwargs):
            seen.update(
                {k: kwargs.get(k) for k in ("n", "remaining_budget", "total_budget")}
            )
            return real_plan_batch(self, **kwargs)

        monkeypatch.setattr(CemMpcV1Planner, "plan_batch", spy)

        camp = run(
            build_campaign(GOAL, user_id="t7", config=BUILD_CFG, outcome_model=None)
        )
        camp.total_budget = 8
        camp.spent_budget = 3
        new = propose_extension_variants(
            camp, GOAL, dataset_meta=DATASET_META, outcome_model=_head(), n_extra=2
        )
        assert seen == {"n": 2, "remaining_budget": 5, "total_budget": 8}
        assert 1 <= len(new) <= 2
        assert all(v.source == "surprise_extension" for v in new)

    def test_default_backend_unchanged(self):
        """Không đổi config → vẫn cem_config_v1 (không plan_batch)."""
        from hagent.agent.campaign.builder import _campaign_config, _campaign_planner
        from hagent.world.planner.cem_config_v1 import CemConfigV1Planner

        planner = _campaign_planner(dict(_campaign_config()))
        assert isinstance(planner, CemConfigV1Planner)
