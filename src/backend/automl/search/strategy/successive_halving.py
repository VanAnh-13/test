"""
Successive Halving — HPO multi-fidelity (Jamieson & Talwalkar, 2016).

Fidelity = tỉ lệ dữ liệu huấn luyện (stratified subsample). Bắt đầu với nhiều
ứng viên trên ít dữ liệu, mỗi rung giữ top 1/eta và tăng dữ liệu ×eta —
phần lớn budget dồn cho ứng viên hứa hẹn. Ứng viên khởi tạo sample như
random search (dedup, enumerate-all khi grid nhỏ).

cv_results_ ghi thêm 'resource_frac' cho từng đánh giá. Điểm/best lấy từ
rung cao nhất mà ứng viên đạt được.
"""

import logging
import math
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from automl.search.strategy.base import SearchStrategy, normalize_param_grid
from automl.search.strategy.random_search import RandomSearchStrategy

logger = logging.getLogger(__name__)


class SuccessiveHalvingStrategy(RandomSearchStrategy):
    """Halving trên fraction dữ liệu; kế thừa sampling/evaluate từ RandomSearch."""

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        base_config = SearchStrategy.get_default_config()
        yaml_config = SearchStrategy._load_yaml_config('successive_halving')
        base_config.update(
            {
                'eta': 3,
                'n_candidates': 27,
                'min_resource_frac': 1.0 / 9.0,
                'min_subsample_rows': 60,
                'max_duplicate_attempts': 50,
            }
        )
        if yaml_config:
            base_config.update(yaml_config)
        return base_config

    def _subsample(
        self, X: np.ndarray, y: np.ndarray, frac: float, rung: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stratified subsample theo frac; dữ liệu quá nhỏ → dùng toàn bộ."""
        n = len(y)
        min_rows = int(self.config.get('min_subsample_rows', 60))
        if frac >= 1.0 or n * frac < min_rows:
            return X, y
        seed = self.config.get('random_state')
        try:
            X_sub, _, y_sub, _ = train_test_split(
                X,
                y,
                train_size=frac,
                stratify=y,
                random_state=(seed + rung) if seed is not None else None,
            )
            return X_sub, y_sub
        except ValueError:
            # stratify fail (lớp quá hiếm ở frac nhỏ) → dùng toàn bộ
            return X, y

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
        log_file = self.create_log_file_path(model, 'successive_halving')

        param_grid_list = normalize_param_grid(param_grid)
        primary_metric = self.config.get('metric_sort', 'accuracy')
        rng = np.random.default_rng(self.config.get('random_state'))
        eta = max(2, int(self.config.get('eta', 3)))
        n_candidates = max(2, int(self.config.get('n_candidates', 27)))
        min_frac = min(1.0, max(1e-3, float(self.config.get('min_resource_frac', 1 / 9))))

        candidates = self._sample_combos(param_grid_list, n_candidates, rng)
        if not candidates:
            return {}, 0.0, {}, {}, False

        # Số rung theo cả budget dữ liệu lẫn số ứng viên khả dụng
        n_rungs = 1 + max(
            0,
            min(
                int(math.floor(math.log(1.0 / min_frac, eta))),
                int(math.floor(math.log(max(1, len(candidates)), eta))),
            ),
        )
        fracs = [min(1.0, min_frac * (eta ** r)) for r in range(n_rungs)]
        fracs[-1] = 1.0  # rung cuối luôn full data

        scoring = self.config.get('scoring') or {}
        metric_names = list(scoring.keys()) if scoring else [primary_metric]
        cv_results_: Dict[str, Any] = {
            'params': [],
            'mean_test_score': [],
            'std_test_score': [],
            'resource_frac': [],
        }
        for metric in metric_names:
            cv_results_[f'mean_test_{metric}'] = []
            cv_results_[f'std_test_{metric}'] = []

        best_params: Dict[str, Any] = {}
        best_score = float('-inf')
        best_all_scores: Dict[str, float] = {}
        stopped = False

        survivors = list(candidates)
        for rung, frac in enumerate(fracs):
            X_r, y_r = self._subsample(X, y, frac, rung)
            if self.config.get('verbose', 0) > 0:
                logger.info(
                    "SH rung %d/%d: %d ứng viên × frac=%.3f (%d hàng)",
                    rung + 1, len(fracs), len(survivors), frac, len(y_r),
                )

            # Mỗi rung là embarrassingly parallel — đánh giá theo lô song song
            from joblib import effective_n_jobs

            chunk_size = max(1, effective_n_jobs(self.config.get('n_jobs') or 1))
            rung_scores: List[float] = []
            evaluated: List[Dict[str, Any]] = []
            j = 0
            while j < len(survivors):
                batch = survivors[j:j + chunk_size]
                t0 = time.time()
                results = self._evaluate_batch(model, batch, X_r, y_r)
                duration = time.time() - t0

                for params, all_scores in zip(batch, results):
                    if all_scores is None:
                        continue
                    score = all_scores.get(primary_metric)
                    if score is None and all_scores:
                        score = max(all_scores.values())
                    score = float(score) if score is not None else 0.0

                    rung_scores.append(score)
                    evaluated.append(params)
                    cv_results_['params'].append(dict(params))
                    cv_results_['mean_test_score'].append(score)
                    cv_results_['std_test_score'].append(0.0)
                    cv_results_['resource_frac'].append(frac)
                    for metric in metric_names:
                        cv_results_[f'mean_test_{metric}'].append(all_scores.get(metric, 0.0))
                        cv_results_[f'std_test_{metric}'].append(0.0)

                    self._log_evaluation(
                        model.__class__.__name__, 'successive_halving', params,
                        all_scores,
                    )

                    # Best chỉ tính ở rung cao nhất (fidelity cao nhất)
                    if frac >= fracs[-1] or rung == len(fracs) - 1:
                        if score > best_score:
                            best_score = score
                            best_params = dict(params)
                            best_all_scores = dict(all_scores)

                j += len(batch)
                if j < len(survivors) and not self._should_start_next_iteration(
                    iteration_duration=duration
                ):
                    logger.info("SH dừng do time limit tại rung %d", rung + 1)
                    stopped = True
                    break

            if not evaluated:
                break

            # Fallback best khi time limit chặn trước rung cuối
            if best_score == float('-inf'):
                top = int(np.argmax(rung_scores))
                best_score = rung_scores[top]
                best_params = dict(evaluated[top])

            if stopped or rung == len(fracs) - 1:
                break

            # Giữ top 1/eta cho rung sau
            keep = max(1, int(math.ceil(len(evaluated) / eta)))
            order = np.argsort(-np.array(rung_scores))[:keep]
            survivors = [evaluated[i] for i in order]

        if cv_results_['mean_test_score']:
            scores_arr = np.array(cv_results_['mean_test_score'])
            cv_results_['rank_test_score'] = (
                np.argsort(np.argsort(-scores_arr)) + 1
            ).tolist()

        if best_score == float('-inf'):
            best_score = 0.0

        self._save_search_log(log_file)
        return self._finalize_results(best_params, best_score, best_all_scores, cv_results_)
