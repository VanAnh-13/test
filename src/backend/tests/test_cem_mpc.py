"""
Tests cho CemMpcV1Planner — budget-annealed Thompson batch planning.
"""

from __future__ import annotations

from hagent.world.planner import CemMpcV1Planner, create_campaign_planner
from hagent.world.predictor.outcome_head_v1 import OutcomeHeadV1, OutcomePrediction


class FakeModel:
    is_ready = True

    def __init__(self, table):
        # table: {(algo, time): (mean, std)}
        self.table = table

    def predict(self, params, dataset_meta=None, z=None):
        key = (params.get("search_algorithm"), params.get("time_limit"))
        if key not in self.table:
            return OutcomePrediction(mean=0.5, std=0.05)
        mu, sig = self.table[key]
        return OutcomePrediction(mean=mu, std=sig)


# grid@180 chắc chắn thấp; bayesian@600 mean cao σ nhỏ; genetic@300 mean thấp
# hơn chút nhưng σ LỚN (ứng viên khám phá)
TABLE = {
    ("grid_search", 180): (0.60, 0.01),
    ("bayesian_search", 600): (0.80, 0.01),
    ("genetic_algorithm", 300): (0.78, 0.30),
}
CFG = {
    "seed": 0,
    "search_algorithms": ["grid_search", "bayesian_search", "genetic_algorithm"],
    "time_limit_options": [180, 300, 600],
}


class TestPlanBatch:
    def test_final_batch_pure_exploit(self):
        planner = CemMpcV1Planner(CFG)
        batch = planner.plan_batch(
            base_params={},
            outcome_model=FakeModel(TABLE),
            n=2,
            remaining_budget=2,  # remaining_after = 0 → exploit
            total_budget=12,
        )
        assert len(batch) == 2
        # slot đầu phải là argmax mean tuyệt đối
        assert batch[0]["search_algorithm"] == "bayesian_search"
        assert batch[0]["time_limit"] == 600

    def test_early_batch_explores_high_sigma(self):
        """Budget còn nhiều → ít nhất một seed chọn ứng viên σ cao trước
        ứng viên mean nhỉnh hơn nhưng σ nhỏ."""
        picked_genetic_first = 0
        for seed in range(10):
            planner = CemMpcV1Planner(dict(CFG, seed=seed))
            batch = planner.plan_batch(
                base_params={},
                outcome_model=FakeModel(TABLE),
                n=1,
                remaining_budget=12,
                total_budget=12,
            )
            if batch[0]["search_algorithm"] == "genetic_algorithm":
                picked_genetic_first += 1
        assert picked_genetic_first >= 2  # khám phá thật sự xảy ra

    def test_final_batch_never_gambles(self):
        """Batch cuối không bao giờ chọn σ cao khi mean thấp hơn."""
        for seed in range(10):
            planner = CemMpcV1Planner(dict(CFG, seed=seed))
            batch = planner.plan_batch(
                base_params={},
                outcome_model=FakeModel(TABLE),
                n=1,
                remaining_budget=1,
                total_budget=12,
            )
            assert batch[0]["search_algorithm"] == "bayesian_search"

    def test_batch_distinct_configs(self):
        planner = CemMpcV1Planner(CFG)
        batch = planner.plan_batch(
            base_params={},
            outcome_model=FakeModel(TABLE),
            n=3,
            remaining_budget=12,
            total_budget=12,
        )
        sigs = [
            (
                c.get("search_algorithm"),
                c.get("time_limit"),
                tuple(c.get("models") or []),
            )
            for c in batch
        ]
        assert len(sigs) == len(set(sigs))

    def test_deterministic_by_seed_and_budget(self):
        outs = [
            CemMpcV1Planner(dict(CFG, seed=5)).plan_batch(
                base_params={},
                outcome_model=FakeModel(TABLE),
                n=2,
                remaining_budget=8,
                total_budget=12,
            )
            for _ in range(2)
        ]
        assert outs[0] == outs[1]

    def test_fallback_without_model(self):
        planner = CemMpcV1Planner(CFG)
        batch = planner.plan_batch(
            base_params={},
            outcome_model=OutcomeHeadV1({}),  # not ready
            n=3,
            remaining_budget=12,
            total_budget=12,
        )
        assert len(batch) == 3

    def test_factory_backend(self):
        planner = create_campaign_planner({"backend": "cem_mpc_v1"})
        assert isinstance(planner, CemMpcV1Planner)
        # vẫn tương thích interface builder
        out = planner.plan_campaign_configs(
            base_params={}, outcome_model=FakeModel(TABLE), n_return=2
        )
        assert len(out) == 2
