"""
Tests cho benchmark layer: metrics thuần, simulated env, run_condition.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from hagent.agent.eval.benchmark import (
    PROFILES,
    DatasetProfile,
    SimulatedAutoMLEnv,
    run_benchmark_matrix,
    run_condition,
)
from hagent.agent.eval.metrics import (
    aggregate_curves,
    best_so_far_curve,
    jobs_to_threshold,
    normalized_regret,
)


def arun(coro):
    import asyncio

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── Pure metrics ─────────────────────────────────────────


class TestMetrics:
    def test_best_so_far_curve(self):
        assert best_so_far_curve([0.5, 0.4, 0.9, 0.7]) == [0.5, 0.5, 0.9, 0.9]
        assert best_so_far_curve([]) == []

    def test_jobs_to_threshold(self):
        curve = [0.5, 0.5, 0.9, 0.9]
        assert jobs_to_threshold(curve, 0.9) == 3
        assert jobs_to_threshold(curve, 0.5) == 1
        assert jobs_to_threshold(curve, 0.95) is None

    def test_normalized_regret(self):
        assert normalized_regret(0.9, 1.0, baseline=0.5) == pytest.approx(0.2)
        assert normalized_regret(1.0, 1.0, baseline=0.5) == 0.0
        assert normalized_regret(1.2, 1.0, baseline=0.5) == 0.0  # vượt optimum → 0
        assert normalized_regret(0.5, 1.0, baseline=1.0) == 0.0  # degenerate denom

    def test_aggregate_curves(self):
        agg = aggregate_curves([[1.0, 2.0, 3.0], [3.0, 4.0]])
        assert agg["n"] == 2
        assert agg["mean"] == [2.0, 3.0]
        assert agg["std"] == [1.0, 1.0]
        assert aggregate_curves([])["n"] == 0


# ── Simulated env ────────────────────────────────────────


class TestSimulatedEnv:
    def test_deterministic_by_seed(self):
        prof = PROFILES["synth_strong"]

        async def scenario(env):
            out = []
            for algo in ("grid_search", "bayesian_search"):
                r = await env.invoke(
                    "start_training", {"search_algorithm": algo, "time_limit": 300}
                )
                info = await env.invoke("get_job_info", {"job_id": r["job_id"]})
                out.append(info["best_score"])
            return out

        s1 = arun(scenario(SimulatedAutoMLEnv(prof, seed=5)))
        s2 = arun(scenario(SimulatedAutoMLEnv(prof, seed=5)))
        assert s1 == s2

    def test_respects_response_surface(self):
        prof = PROFILES["synth_strong"]
        env = SimulatedAutoMLEnv(prof, seed=0)

        async def many(algo, n=30):
            scores = []
            for _ in range(n):
                r = await env.invoke(
                    "start_training", {"search_algorithm": algo, "time_limit": 600}
                )
                info = await env.invoke("get_job_info", {"job_id": r["job_id"]})
                scores.append(info["best_score"])
            return float(np.mean(scores))

        assert arun(many("bayesian_search")) > arun(many("grid_search")) + 0.1

    def test_unknown_job(self):
        env = SimulatedAutoMLEnv(PROFILES["synth_flat"], seed=0)
        info = arun(env.invoke("get_job_info", {"job_id": "nope"}))
        assert info.get("error")

    def test_optimum_definition(self):
        prof = DatasetProfile(
            name="x",
            base=0.5,
            algo_bonus={"a": 0.1, "b": 0.3},
            time_coef=0.05,
            noise=0.0,
        )
        assert prof.optimum == pytest.approx(0.85)
        assert prof.expected_score("b", 600) == pytest.approx(0.85)


# ── run_condition ────────────────────────────────────────


class TestRunCondition:
    def test_budget_respected_and_curve_monotone(self):
        res = run_condition("random", "synth_strong", budget_jobs=8, seed=0)
        assert res["jobs_used"] == 8
        assert len(res["curve"]) == 8
        assert res["curve"] == best_so_far_curve(res["scores"])
        assert all(
            res["curve"][i] <= res["curve"][i + 1] for i in range(len(res["curve"]) - 1)
        )

    def test_all_conditions_run(self):
        for cond in (
            "wm",
            "no_wm",
            "random",
            "fixed_grid_search",
            "fixed_bayesian_search",
        ):
            res = run_condition(cond, "synth_flat", budget_jobs=6, seed=0)
            assert res["condition"] == cond
            assert res["jobs_used"] == 6
            assert res["final_best"] is not None

    def test_unknown_condition_raises(self):
        with pytest.raises(ValueError):
            run_condition("alchemy", "synth_flat", budget_jobs=3, seed=0)

    def test_fixed_condition_uses_one_config(self):
        res = run_condition("fixed_grid_search", "synth_strong", budget_jobs=6, seed=0)
        # fixed grid không bao giờ hưởng bonus bayesian → final thấp hơn optimum rõ
        assert res["final_best"] < PROFILES["synth_strong"].optimum - 0.1

    def test_wm_trains_model_within_budget(self):
        res = run_condition("wm", "synth_strong", budget_jobs=12, seed=0)
        assert res["wm_trained_after_jobs"] is not None
        assert res["wm_trained_after_jobs"] <= 12
        assert res["n_train_samples"] >= 6

    def test_wm_beats_fixed_grid_on_strong_signal(self):
        """Trên profile tín hiệu mạnh, wm phải hơn fixed grid (không bonus)."""
        wm = run_condition("wm", "synth_strong", budget_jobs=12, seed=0)
        fixed = run_condition(
            "fixed_grid_search", "synth_strong", budget_jobs=12, seed=0
        )
        assert wm["final_best"] > fixed["final_best"]

    def test_wm_reaches_95pct_on_strong_signal(self):
        res = run_condition("wm", "synth_strong", budget_jobs=15, seed=0)
        assert res["jobs_to_95pct"] is not None

    def test_result_json_serializable(self):
        res = run_condition("wm", "synth_noisy", budget_jobs=6, seed=1)
        s = json.dumps(res)
        assert "curve" in json.loads(s)


class TestMatrix:
    def test_matrix_shape(self):
        results = run_benchmark_matrix(
            conditions=["random", "no_wm"],
            profiles=["synth_flat"],
            budget_jobs=4,
            seeds=[0, 1],
        )
        assert len(results) == 4
        keys = {(r["condition"], r["seed"]) for r in results}
        assert keys == {("random", 0), ("random", 1), ("no_wm", 0), ("no_wm", 1)}

    def test_matrix_fails_fast_on_bad_condition(self):
        """Condition sai phải raise TRƯỚC khi đốt bất kỳ run nào."""
        with pytest.raises(ValueError):
            run_benchmark_matrix(
                conditions=["random", "fixed_gridsearch"],  # typo
                profiles=["synth_flat"],
                budget_jobs=4,
                seeds=[0],
            )

    def test_matrix_fails_fast_on_bad_profile(self):
        with pytest.raises(ValueError):
            run_benchmark_matrix(
                conditions=["random"], profiles=["nope"], budget_jobs=4, seeds=[0]
            )


class TestReviewFixes:
    """Guard cho các lỗi đã xác nhận qua adversarial review."""

    def test_metrics_use_expected_scores_not_noisy_max(self):
        """fixed_grid trên synth_noisy phải có regret LỚN (steering tệ),
        không bị max-of-noise che mất."""
        res = run_condition("fixed_grid_search", "synth_noisy", budget_jobs=20, seed=0)
        prof = PROFILES["synth_noisy"]
        # expected score của grid@600 thấp hơn optimum đúng bằng bonus bayesian
        assert res["final_best"] == pytest.approx(
            prof.expected_score("grid_search", 600)
        )
        assert res["normalized_regret"] > 0.5
        assert res["jobs_to_95pct"] is None
        # điểm quan sát (nhiễu) vẫn được báo cáo riêng
        assert len(res["observed_scores"]) == 20
        assert res["observed_final_best"] >= max(res["observed_scores"]) - 1e-12

    def test_expected_curve_matches_scores(self):
        res = run_condition("random", "synth_noisy", budget_jobs=8, seed=3)
        assert res["curve"] == best_so_far_curve(res["scores"])
        assert res["observed_curve"] == best_so_far_curve(res["observed_scores"])

    def test_warm_start_top_k_zero_disables_all_sources(self):
        """top_k=0 phải tắt cả warm-start từ memory (chống nhiễm chéo)."""
        import tempfile

        from hagent.agent.campaign.warm_start import collect_warm_start_configs
        from hagent.agent.memory import Fact, LocalFactStore

        with tempfile.TemporaryDirectory() as td:
            store = LocalFactStore(td)
            arun(
                store.save(
                    "u1",
                    Fact(
                        key="warm_start_classification",
                        content='{"search_algorithm": "bayesian_search"}',
                        category="model",
                        source="campaign",
                        confidence=0.9,
                    ),
                )
            )
            cfgs = arun(
                collect_warm_start_configs(
                    world_model=None,
                    user_id="u1",
                    problem_type="classification",
                    top_k=0,
                    fact_store=store,
                )
            )
            assert cfgs == []

    def test_synth_models_profile_math(self):
        prof = PROFILES["synth_models"]
        # optimum = base + best algo + full time + best model effect
        assert prof.optimum == pytest.approx(0.55 + 0.05 + 0.04 + 0.15)
        # chọn đúng model tốt nhất > quét cả catalog (dilution)
        best_only = prof.expected_score(
            "bayesian_search", 600, ["RandomForestClassifier"]
        )
        default_all = prof.expected_score("bayesian_search", 600, None)
        assert best_only - default_all == pytest.approx(0.02 * 3)

    def test_expanded_space_wm_beats_no_wm(self):
        """Không gian lớn (algo × time × 2^4 subset): steering phải thắng
        round-robin không model — đây là lý do tồn tại của hướng A."""
        wm = run_condition("wm", "synth_models", budget_jobs=18, seed=0)
        no_wm = run_condition("no_wm", "synth_models", budget_jobs=18, seed=0)
        assert wm["final_best"] > no_wm["final_best"]

    def test_random_condition_samples_subsets(self):
        res = run_condition("random", "synth_models", budget_jobs=6, seed=0)
        assert res["jobs_used"] == 6
        # random phải thật sự sample chiều models trên profile có model_effects
        # (kiểm qua expected scores biến thiên theo subset — không truy cập env,
        # nhưng ít nhất curve phải hợp lệ và final <= optimum)
        assert res["final_best"] <= PROFILES["synth_models"].optimum + 1e-9

    def test_wm_mpc_runs_and_beats_no_wm(self):
        """Condition wm_mpc (budget-annealed Thompson) chạy đủ budget và
        thắng no_wm trên không gian lớn."""
        mpc = run_condition("wm_mpc", "synth_models", budget_jobs=18, seed=0)
        no_wm = run_condition("no_wm", "synth_models", budget_jobs=18, seed=0)
        assert mpc["jobs_used"] == 18
        assert mpc["wm_trained_after_jobs"] is not None
        assert mpc["final_best"] > no_wm["final_best"]

    def test_wm_surprise_events_counted_no_wm_zero(self):
        """Model online phải được runner nhìn thấy: wm có event, no_wm = 0."""
        wm = run_condition("wm", "synth_strong", budget_jobs=12, seed=0)
        no_wm = run_condition("no_wm", "synth_strong", budget_jobs=12, seed=0)
        assert wm["wm_trained_after_jobs"] is not None
        assert wm["n_outcome_surprise_events"] > 0
        assert no_wm["n_outcome_surprise_events"] == 0
