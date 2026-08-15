"""
Tests cho T4 — random_search + successive_halving khả dụng cho campaign,
vocab v2 đồng bộ, validator không drift.
"""

from __future__ import annotations

import asyncio

from hagent.agent.campaign.builder import build_campaign
from hagent.agent.constraints.validator import validate_action
from hagent.bridge.config import get_campaign_config, get_world_model_config
from hagent.world.schema import AutoMLAction, AutoMLObservation

FIVE = {
    "grid_search",
    "bayesian_search",
    "genetic_algorithm",
    "random_search",
    "successive_halving",
}

GOAL = {
    "goal_type": "train",
    "dataset_id": "ds1",
    "problem_type": "classification",
    "metric": "accuracy",
    "target_column": "target",
}


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestConfigConsistency:
    def test_campaign_pool_has_five(self):
        algos = set(get_campaign_config().get("search_algorithms") or [])
        assert FIVE <= algos

    def test_outcome_vocab_synced_head_and_ensemble(self):
        """Vocab head và ensemble PHẢI giống nhau — lệch là chiều feature
        member khác head, checkpoint dùng lẫn nhau sẽ vô nghĩa."""
        wm = get_world_model_config()
        head = list((wm.get("outcome_head") or {}).get("search_algorithms") or [])
        ens = list((wm.get("outcome_ensemble") or {}).get("search_algorithms") or [])
        assert set(head) == FIVE
        assert head == ens

    def test_checkpoint_paths_bumped_v2(self):
        """Đổi vocab = đổi chiều feature → path phải bump, không được để
        checkpoint v1 cũ bị nạp nhầm."""
        wm = get_world_model_config()
        assert "v2" in (wm.get("outcome_head") or {}).get("checkpoint_path", "")
        assert "v2" in (wm.get("outcome_ensemble") or {}).get("checkpoint_dir", "")

    def test_backend_factory_accepts_all_five(self):
        from automl.search.factory.search_strategy_factory import (
            SearchStrategyFactory,
        )

        for algo in FIVE:
            assert SearchStrategyFactory.is_strategy_available(algo), algo


class TestValidatorNoDrift:
    def _obs(self):
        return AutoMLObservation(
            user_id="u1",
            datasets={
                "ds1": {"id": "ds1", "features": ["a", "target"], "target": "target"}
            },
        )

    def test_accepts_successive_halving(self):
        action = AutoMLAction(
            type="start_training",
            params={
                "dataset_id": "ds1",
                "search_algorithm": "successive_halving",
                "problem_type": "classification",
                "target_column": "target",
            },
        )
        result = validate_action(action, self._obs(), goal=GOAL)
        assert not any("search_algorithm" in r for r in result.reasons)

    def test_rejects_unknown_algorithm(self):
        action = AutoMLAction(
            type="start_training",
            params={
                "dataset_id": "ds1",
                "search_algorithm": "quantum_annealing",
                "problem_type": "classification",
                "target_column": "target",
            },
        )
        result = validate_action(action, self._obs(), goal=GOAL)
        assert any("search_algorithm" in r for r in result.reasons)


class TestBuilderUsesNewAlgorithms:
    def test_round_robin_reaches_new_algos(self):
        """n=5 slot round-robin phải phủ đủ 5 thuật toán (warm-start tắt)."""
        camp = run(
            build_campaign(
                GOAL,
                user_id="t4_user_no_memory",
                config={
                    "n_job_candidates": 5,
                    "warm_start_top_k": 0,
                    "wm_variant_proposal": False,
                    "wm_rank_variants": False,
                },
                outcome_model=None,
            )
        )
        algos = {v.params.get("search_algorithm") for v in camp.variants}
        assert algos == FIVE
