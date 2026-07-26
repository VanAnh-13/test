"""
Random Search — baseline chuẩn cho HPO (Bergstra & Bengio, 2012).

Sample uniform từ param_grid (list giá trị mỗi tham số), dedup, tôn trọng
max_time; khi tổng tổ hợp ≤ n_iter thì liệt kê toàn bộ (tương đương grid).
Trả về đúng contract chung: (best_params, best_score, best_all_scores,
cv_results_, time_limit_reached).
"""

import logging
import time
from itertools import product
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_validate

from automl.search.strategy.base import SearchStrategy, normalize_param_grid

logger = logging.getLogger(__name__)


class RandomSearchStrategy(SearchStrategy):
    """Tìm kiếm ngẫu nhiên uniform trên không gian tham số rời rạc."""

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        base_config = SearchStrategy.get_default_config()
        yaml_config = SearchStrategy._load_yaml_config('random_search')
        base_config.update(
            {
                'n_iter': 30,
                'max_duplicate_attempts': 50,
            }
        )
        if yaml_config:
            base_config.update(yaml_config)
        return base_config

    # ── Sampling ─────────────────────────────────────────

    @staticmethod
    def _n_combinations(param_grid_list: List[Dict[str, list]]) -> int:
        total = 0
        for grid in param_grid_list:
            n = 1
            for values in grid.values():
                n *= max(1, len(values))
            total += n
        return total

    @staticmethod
    def _enumerate_all(param_grid_list: List[Dict[str, list]]) -> List[Dict[str, Any]]:
        combos: List[Dict[str, Any]] = []
        for grid in param_grid_list:
            if not grid:
                combos.append({})
                continue
            keys = list(grid.keys())
            for values in product(*(grid[k] for k in keys)):
                combos.append(dict(zip(keys, values)))
        return combos

    def _sample_combos(
        self, param_grid_list: List[Dict[str, list]], n_iter: int, rng: np.random.Generator
    ) -> List[Dict[str, Any]]:
        """Sample dedup; grid nhỏ hơn n_iter → liệt kê toàn bộ."""
        if self._n_combinations(param_grid_list) <= n_iter:
            return self._enumerate_all(param_grid_list)

        max_attempts = int(self.config.get('max_duplicate_attempts', 50))
        seen = set()
        combos: List[Dict[str, Any]] = []
        attempts = 0
        while len(combos) < n_iter and attempts < n_iter * max_attempts:
            attempts += 1
            grid = param_grid_list[int(rng.integers(0, len(param_grid_list)))]
            params = {
                k: values[int(rng.integers(0, len(values)))]
                for k, values in grid.items()
            }
            key = tuple(sorted((k, str(v)) for k, v in params.items()))
            if key in seen:
                continue
            seen.add(key)
            combos.append(params)
        return combos

    # ── Evaluation ───────────────────────────────────────

    def _evaluate(
        self,
        model: BaseEstimator,
        params: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        cv_n_jobs: int | None = None,
    ) -> Dict[str, float]:
        est = clone(model)
        est.set_params(**params)
        scoring = self.config.get('scoring')
        cv_out = cross_validate(
            estimator=est,
            X=X,
            y=y,
            cv=self.config['cv'],
            n_jobs=cv_n_jobs if cv_n_jobs is not None else self.config['n_jobs'],
            scoring=scoring,
            error_score=self.config['error_score'],
            return_train_score=False,
        )
        all_scores: Dict[str, float] = {}
        if scoring:
            for key in scoring:
                test_key = f'test_{key}'
                if test_key in cv_out:
                    all_scores[key] = float(np.mean(cv_out[test_key]))
        elif 'test_score' in cv_out:
            primary = self.config.get('metric_sort', 'accuracy')
            all_scores[primary] = float(np.mean(cv_out['test_score']))
        return all_scores

    def _evaluate_batch(
        self,
        model: BaseEstimator,
        params_list: List[Dict[str, Any]],
        X: np.ndarray,
        y: np.ndarray,
    ) -> List[Dict[str, float] | None]:
        """
        Đánh giá song song THEO ỨNG VIÊN (outer) — nhanh hơn hẳn chỉ song song
        theo fold khi số fold < số core. Inner cross_validate chạy n_jobs=1
        để tránh oversubscription. Lỗi một ứng viên → None (không sập batch).
        """
        n_jobs = int(self.config.get('n_jobs') or 1)
        if n_jobs == 1 or len(params_list) <= 1:
            out: List[Dict[str, float] | None] = []
            for params in params_list:
                try:
                    out.append(self._evaluate(model, params, X, y))
                except Exception as exc:
                    logger.warning("Bỏ qua %s: %s", params, exc)
                    out.append(None)
            return out

        from joblib import Parallel, delayed

        def _one(params):
            try:
                return self._evaluate(model, params, X, y, cv_n_jobs=1)
            except Exception as exc:  # pragma: no cover - phụ thuộc estimator
                logger.warning("Bỏ qua %s: %s", params, exc)
                return None

        return Parallel(n_jobs=n_jobs)(delayed(_one)(p) for p in params_list)

    # ── Search ───────────────────────────────────────────

    def search(
        self,
        model: BaseEstimator,
        param_grid: List[Dict[str, Any]],
        X: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> Tuple[Dict, float, Dict, Dict, bool]:
        self.set_config(**kwargs)
        self._start_timer()
        self._init_search_log()
        log_file = self.create_log_file_path(model, 'random_search')

        param_grid_list = normalize_param_grid(param_grid)
        primary_metric = self.config.get('metric_sort', 'accuracy')
        rng = np.random.default_rng(self.config.get('random_state'))
        n_iter = max(1, int(self.config.get('n_iter', 30)))

        combos = self._sample_combos(param_grid_list, n_iter, rng)

        scoring = self.config.get('scoring') or {}
        metric_names = list(scoring.keys()) if scoring else [primary_metric]
        cv_results_: Dict[str, Any] = {
            'params': [],
            'mean_test_score': [],
            'std_test_score': [],
        }
        for metric in metric_names:
            cv_results_[f'mean_test_{metric}'] = []
            cv_results_[f'std_test_{metric}'] = []

        best_params: Dict[str, Any] = {}
        best_score = float('-inf')
        best_all_scores: Dict[str, float] = {}

        # Đánh giá theo lô song song (outer parallelism) — đơn vị thời gian
        # để proactive-stop là MỘT LÔ
        from joblib import effective_n_jobs

        chunk_size = max(1, effective_n_jobs(self.config.get('n_jobs') or 1))
        i = 0
        while i < len(combos):
            batch = combos[i:i + chunk_size]
            t0 = time.time()
            results = self._evaluate_batch(model, batch, X, y)
            duration = time.time() - t0

            for offset, (params, all_scores) in enumerate(zip(batch, results)):
                if all_scores is None:
                    continue
                score = all_scores.get(primary_metric)
                if score is None and all_scores:
                    score = max(all_scores.values())
                score = float(score) if score is not None else 0.0

                cv_results_['params'].append(dict(params))
                cv_results_['mean_test_score'].append(score)
                cv_results_['std_test_score'].append(0.0)
                for metric in metric_names:
                    cv_results_[f'mean_test_{metric}'].append(all_scores.get(metric, 0.0))
                    cv_results_[f'std_test_{metric}'].append(0.0)

                self._log_evaluation(
                    model.__class__.__name__, 'random_search', params, all_scores,
                    iteration=i + offset + 1, total=len(combos),
                )

                if score > best_score:
                    best_score = score
                    best_params = dict(params)
                    best_all_scores = dict(all_scores)

            i += len(batch)
            if i < len(combos) and not self._should_start_next_iteration(
                iteration_duration=duration
            ):
                logger.info("Random search dừng do time limit tại %d/%d", i, len(combos))
                break

        if cv_results_['mean_test_score']:
            scores_arr = np.array(cv_results_['mean_test_score'])
            cv_results_['rank_test_score'] = (
                np.argsort(np.argsort(-scores_arr)) + 1
            ).tolist()

        if best_score == float('-inf'):
            best_score = 0.0

        self._save_search_log(log_file)
        return self._finalize_results(best_params, best_score, best_all_scores, cv_results_)
