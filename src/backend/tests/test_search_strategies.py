"""
Tests cho cải thiện thư viện HPO: RandomSearch, SuccessiveHalving,
dimension inference + fix hội tụ/scoring-None của BayesianSearch, factory.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from skopt.space import Categorical, Integer, Real

from automl.search.factory.search_strategy_factory import SearchStrategyFactory
from automl.search.strategy.bayesian_search import BayesianSearchStrategy
from automl.search.strategy.random_search import RandomSearchStrategy
from automl.search.strategy.successive_halving import SuccessiveHalvingStrategy

X, y = make_classification(
    n_samples=150, n_features=6, n_informative=4, n_classes=2, random_state=0
)
GRID = {"max_depth": [2, 4, 8], "min_samples_split": [2, 6, 12]}
FAST_CFG = dict(
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
    scoring={"accuracy": "accuracy"},
    metric_sort="accuracy",
    n_jobs=1,
    random_state=0,
    save_log=False,
    verbose=0,
)


def _unpack(result):
    best_params, best_score, best_all, cv_results, tl = result
    return best_params, best_score, best_all, cv_results, tl


# ── Random Search ────────────────────────────────────────


class TestRandomSearch:
    def test_contract_and_sanity(self):
        s = RandomSearchStrategy(**FAST_CFG, n_iter=6)
        bp, bs, ba, cv, tl = _unpack(
            s.search(DecisionTreeClassifier(random_state=0), GRID, X, y)
        )
        assert set(bp) <= {"max_depth", "min_samples_split"}
        assert 0.5 < bs <= 1.0
        assert ba["accuracy"] == pytest.approx(bs)
        assert len(cv["params"]) <= 6
        assert tl is False

    def test_enumerates_small_grid(self):
        s = RandomSearchStrategy(**FAST_CFG, n_iter=100)
        _, _, _, cv, _ = _unpack(
            s.search(DecisionTreeClassifier(random_state=0), GRID, X, y)
        )
        assert len(cv["params"]) == 9  # 3×3 — toàn bộ grid

    def test_no_duplicate_params(self):
        s = RandomSearchStrategy(**FAST_CFG, n_iter=6)
        _, _, _, cv, _ = _unpack(
            s.search(DecisionTreeClassifier(random_state=0), GRID, X, y)
        )
        sigs = [tuple(sorted(p.items())) for p in cv["params"]]
        assert len(sigs) == len(set(sigs))

    def test_reproducible_with_seed(self):
        outs = []
        for _ in range(2):
            s = RandomSearchStrategy(**FAST_CFG, n_iter=5)
            bp, bs, _, cv, _ = _unpack(
                s.search(DecisionTreeClassifier(random_state=0), GRID, X, y)
            )
            outs.append((bp, bs, [tuple(sorted(p.items())) for p in cv["params"]]))
        assert outs[0] == outs[1]

    def test_scoring_none_ok(self):
        cfg = dict(FAST_CFG)
        cfg["scoring"] = None
        s = RandomSearchStrategy(**cfg, n_iter=4)
        _, bs, _, _, _ = _unpack(
            s.search(DecisionTreeClassifier(random_state=0), GRID, X, y)
        )
        assert 0.5 < bs <= 1.0


# ── Successive Halving ───────────────────────────────────


class TestSuccessiveHalving:
    def test_contract_and_halving_happens(self):
        s = SuccessiveHalvingStrategy(
            **FAST_CFG, n_candidates=9, eta=3, min_resource_frac=1 / 9,
            min_subsample_rows=30,
        )
        bp, bs, ba, cv, tl = _unpack(
            s.search(DecisionTreeClassifier(random_state=0), GRID, X, y)
        )
        assert 0.5 < bs <= 1.0
        assert "resource_frac" in cv
        fracs = cv["resource_frac"]
        # Rung cuối full data và ÍT ứng viên hơn rung đầu (halving thật)
        n_first = sum(1 for f in fracs if f == fracs[0])
        n_full = sum(1 for f in fracs if f == 1.0)
        assert fracs[0] < 1.0
        assert 0 < n_full < n_first
        assert set(bp) <= {"max_depth", "min_samples_split"}

    def test_best_comes_from_full_fidelity(self):
        s = SuccessiveHalvingStrategy(
            **FAST_CFG, n_candidates=9, eta=3, min_resource_frac=1 / 9,
            min_subsample_rows=30,
        )
        bp, bs, _, cv, _ = _unpack(
            s.search(DecisionTreeClassifier(random_state=0), GRID, X, y)
        )
        full_scores = [
            s_ for s_, f in zip(cv["mean_test_score"], cv["resource_frac"]) if f == 1.0
        ]
        assert bs == pytest.approx(max(full_scores))

    def test_tiny_data_uses_full(self):
        """Dữ liệu quá nhỏ so với min_subsample_rows → mọi rung full data."""
        s = SuccessiveHalvingStrategy(
            **FAST_CFG, n_candidates=4, eta=2, min_resource_frac=0.25,
            min_subsample_rows=10_000,
        )
        _, bs, _, cv, _ = _unpack(
            s.search(DecisionTreeClassifier(random_state=0), GRID, X, y)
        )
        assert all(f == 1.0 or len(y) * f < 10_000 for f in cv["resource_frac"])
        assert bs > 0.5


# ── Tốc độ: song song theo ứng viên ──────────────────────


class TestParallelCandidates:
    def _run(self, cls, n_jobs, **extra):
        cfg = dict(FAST_CFG)
        cfg["n_jobs"] = n_jobs
        s = cls(**cfg, **extra)
        bp, bs, _, cv, _ = _unpack(
            s.search(DecisionTreeClassifier(random_state=0), GRID, X, y)
        )
        return bp, bs, [tuple(sorted(p.items())) for p in cv["params"]]

    def test_random_parallel_equals_serial(self):
        serial = self._run(RandomSearchStrategy, 1, n_iter=6)
        parallel = self._run(RandomSearchStrategy, 2, n_iter=6)
        assert serial == parallel

    def test_sh_parallel_equals_serial(self):
        kw = dict(n_candidates=9, eta=3, min_resource_frac=1 / 9, min_subsample_rows=30)
        serial = self._run(SuccessiveHalvingStrategy, 1, **kw)
        parallel = self._run(SuccessiveHalvingStrategy, 2, **kw)
        assert serial == parallel

    def test_bo_batch_path_parallel(self):
        """n_jobs=2 → batch ask/tell: đủ contract, số eval ≤ n_calls."""
        cfg = dict(FAST_CFG)
        cfg["n_jobs"] = 2
        s = BayesianSearchStrategy(**cfg, n_calls=8, n_initial_points=4)
        assert s._resolve_batch_size() == 2
        bp, bs, ba, cv, tl = _unpack(
            s.search(DecisionTreeClassifier(random_state=0), GRID, X, y)
        )
        assert 0.5 < bs <= 1.0
        assert len(cv["params"]) <= 8
        assert ba["accuracy"] == pytest.approx(bs)

    def test_bo_batch_size_resolution(self):
        s1 = BayesianSearchStrategy(**FAST_CFG)          # n_jobs=1
        assert s1._resolve_batch_size() == 1
        cfg = dict(FAST_CFG)
        cfg["n_jobs"] = 2
        s2 = BayesianSearchStrategy(**cfg, batch_size=4)  # override tường minh
        assert s2._resolve_batch_size() == 4

    def test_ga_parallel_path_contract(self):
        """GA n_jobs=2 (đường song song, inner cv=1): đúng contract, không nổ."""
        from automl.search.strategy.genetic_algorithm import GeneticAlgorithm

        cfg = dict(FAST_CFG)
        cfg["n_jobs"] = 2
        s = GeneticAlgorithm(
            **cfg, population_size=6, generation=2, elite_size=1,
        )
        bp, bs, _, _, tl = _unpack(
            s.search(DecisionTreeClassifier(random_state=0), GRID, X, y)
        )
        assert 0.5 < bs <= 1.0
        assert set(bp) <= {"max_depth", "min_samples_split"}

    def test_ga_respects_explicit_budget(self):
        """Regression: GA từng phình 4×2=8 thành 18×3=54 evals (3× grid vét cạn)
        trên không gian categorical nhỏ, bỏ qua budget người dùng."""
        from automl.search.strategy.genetic_algorithm import GeneticAlgorithm

        s = GeneticAlgorithm(
            **FAST_CFG, population_size=4, generation=2, elite_size=1,
        )
        _, _, _, cv, _ = _unpack(
            s.search(DecisionTreeClassifier(random_state=0), GRID, X, y)
        )
        # GRID = 3×3 = 9 tổ hợp; budget người dùng 4×2 = 8 đánh giá
        assert s.config["generation"] == 2, "generation bị auto-adjust ghi đè"
        assert len(cv["params"]) <= 9, (
            f"GA chạy {len(cv['params'])} đánh giá — vượt cả grid vét cạn (9)"
        )

    def test_ga_auto_adjust_never_exceeds_grid(self):
        """Không đặt budget → auto-adjust được phép, nhưng không quá số tổ hợp."""
        from automl.search.strategy.genetic_algorithm import GeneticAlgorithm

        s = GeneticAlgorithm(**FAST_CFG, elite_size=1)
        _, _, _, cv, _ = _unpack(
            s.search(DecisionTreeClassifier(random_state=0), GRID, X, y)
        )
        assert len(cv["params"]) <= 9

    def test_bo_batch_reuses_configured_pool(self):
        """Regression: BO từng tạo Parallel(n_jobs=batch_size) khác n_jobs của
        config → 2 pool loky chồng nhau (24 tiến trình/16 lõi), stall 2231s."""
        import inspect

        src = inspect.getsource(
            BayesianSearchStrategy._search_single_grid_batch
        )
        assert "Parallel(n_jobs=b)" not in src
        assert "self.config.get('n_jobs')" in src

    def test_grid_backend_prefers_loky(self):
        from automl.search.strategy.grid_search import GridSearchStrategy

        g = GridSearchStrategy(**FAST_CFG)
        # workload trung bình từng bị chọn 'threading' (GIL-bound) — giờ loky
        assert g._select_optimal_backend(n_combinations=9, data_size=12_000) == "loky"
        assert g._select_optimal_backend(n_combinations=2, data_size=12_000) == "threading"

    def test_batch_evaluator_handles_failures(self):
        """Ứng viên lỗi → None, không sập cả batch."""
        s = RandomSearchStrategy(**FAST_CFG)
        results = s._evaluate_batch(
            DecisionTreeClassifier(random_state=0),
            [{"max_depth": 4}, {"max_depth": -99}],  # -99 không hợp lệ
            X,
            y,
        )
        assert results[0] is not None
        assert results[1] is None


# ── Bayesian fixes ───────────────────────────────────────


class TestBayesianDimensionInference:
    def test_int_list_to_integer(self):
        dim = BayesianSearchStrategy._infer_dimension("n", [3, 5, 7, 9])
        assert isinstance(dim, Integer)
        assert (dim.low, dim.high) == (3, 9)

    def test_float_list_log_scale(self):
        dim = BayesianSearchStrategy._infer_dimension("C", [0.001, 0.01, 0.1, 1])
        assert isinstance(dim, Real)
        assert dim.prior == "log-uniform"

    def test_float_small_span_uniform(self):
        dim = BayesianSearchStrategy._infer_dimension("x", [0.1, 0.2, 0.5])
        assert isinstance(dim, Real)
        assert dim.prior == "uniform"

    def test_mixed_stays_categorical(self):
        dim = BayesianSearchStrategy._infer_dimension(
            "max_features", ["sqrt", "log2", 0.5, 1]
        )
        assert isinstance(dim, Categorical)

    def test_bools_stay_categorical(self):
        dim = BayesianSearchStrategy._infer_dimension("flag", [True, False])
        assert isinstance(dim, Categorical)

    def test_single_value_categorical(self):
        dim = BayesianSearchStrategy._infer_dimension("k", ["rbf"])
        assert isinstance(dim, Categorical)


class TestBayesianConvergence:
    def test_single_plateau_step_is_not_convergence(self):
        """Bug cũ: một bước không cải thiện sau patience → dừng. Phải là False."""
        history = [0.5, 0.6, 0.7, 0.8, 0.9, 0.9]  # chỉ MỘT bước phẳng cuối
        assert not BayesianSearchStrategy._converged(history, patience=5, threshold=0.001)

    def test_full_plateau_is_convergence(self):
        history = [0.5, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        assert BayesianSearchStrategy._converged(history, patience=5, threshold=0.001)

    def test_short_history_not_converged(self):
        assert not BayesianSearchStrategy._converged([0.9, 0.9], patience=5, threshold=0.001)

    def test_small_improvements_count_as_plateau(self):
        history = [0.5, 0.9] + [0.9 + i * 1e-5 for i in range(1, 6)]
        assert BayesianSearchStrategy._converged(history, patience=5, threshold=0.001)

    def test_bo_end_to_end_with_scoring_none(self):
        """scoring=None không crash (bug cũ: for key in None)."""
        cfg = dict(FAST_CFG)
        cfg["scoring"] = None
        s = BayesianSearchStrategy(**cfg, n_calls=8, n_initial_points=4)
        bp, bs, _, _, _ = _unpack(
            s.search(DecisionTreeClassifier(random_state=0), GRID, X, y)
        )
        assert 0.5 < bs <= 1.0
        assert set(bp) <= {"max_depth", "min_samples_split"}

    def test_bo_infers_integer_params(self):
        """Với inference bật, BO có thể đề xuất giá trị NGOÀI list gốc."""
        s = BayesianSearchStrategy(**FAST_CFG, n_calls=10, n_initial_points=5)
        bp, bs, _, cv, _ = _unpack(
            s.search(DecisionTreeClassifier(random_state=0), GRID, X, y)
        )
        assert 0.5 < bs <= 1.0
        for p in cv["params"]:
            if "max_depth" in p:
                assert 2 <= p["max_depth"] <= 8  # trong range Integer(2,8)


# ── Factory ──────────────────────────────────────────────


class TestFactory:
    def test_new_strategies_registered(self):
        assert isinstance(
            SearchStrategyFactory.create_strategy("random_search"),
            RandomSearchStrategy,
        )
        assert isinstance(
            SearchStrategyFactory.create_strategy("successive_halving"),
            SuccessiveHalvingStrategy,
        )

    def test_aliases(self):
        assert isinstance(
            SearchStrategyFactory.create_strategy("random"), RandomSearchStrategy
        )
        assert isinstance(
            SearchStrategyFactory.create_strategy("sh"), SuccessiveHalvingStrategy
        )
        assert isinstance(
            SearchStrategyFactory.create_strategy("halving"),
            SuccessiveHalvingStrategy,
        )

    def test_availability_flags(self):
        assert SearchStrategyFactory.is_strategy_available("random_search")
        assert SearchStrategyFactory.is_strategy_available("successive_halving")
        assert not SearchStrategyFactory.is_strategy_available("quantum")

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            SearchStrategyFactory.create_strategy("quantum_annealing")
