"""
CEM thật trên không gian config campaign — {search_algorithm × time_limit}.

Khác cem_lite (liệt kê skeleton hành động), planner này chạy Cross-Entropy
Method đúng nghĩa trên phân phối categorical từng chiều config:
  lặp: sample n_candidates → score bằng outcome model (μ + β·σ)
       → chọn elite theo elite_fraction → refit phân phối có smoothing.

Outcome model chưa sẵn sàng → fallback round-robin deterministic (trùng logic
diversify của builder) để hành vi hệ thống không đổi khi chưa train model.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_ALGOS = ["grid_search", "bayesian_search", "genetic_algorithm"]
_DEFAULT_TIME_OPTIONS = [180, 300, 600]


def _config_signature(params: Dict[str, Any]) -> tuple:
    return (
        params.get("search_algorithm"),
        params.get("time_limit"),
        tuple(params.get("models") or []),
    )


class CemConfigV1Planner:
    """CEM trên các chiều categorical của config một training job."""

    def __init__(self, config: dict | None = None):
        self.config = dict(config or {})
        self.algorithms: List[str] = list(
            self.config.get("search_algorithms") or _DEFAULT_ALGOS
        )
        self.time_options: List[int] = [
            int(t) for t in (self.config.get("time_limit_options") or _DEFAULT_TIME_OPTIONS)
        ]
        self.n_candidates = max(4, int(self.config.get("n_candidates", 32)))
        self.n_iterations = max(1, int(self.config.get("n_iterations", 8)))
        self.elite_fraction = min(
            0.9, max(0.05, float(self.config.get("elite_fraction", 0.25)))
        )
        self.smoothing = min(1.0, max(0.0, float(self.config.get("smoothing", 0.25))))
        self.exploration_weight = float(self.config.get("exploration_weight", 0.1))
        self.seed = int(self.config.get("seed", 0))

    # ── Fallback (không có outcome model) ────────────────

    def _fallback_configs(self, n_return: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for i in range(n_return):
            out.append(
                {
                    "search_algorithm": self.algorithms[i % len(self.algorithms)],
                    "time_limit": self.time_options[i % len(self.time_options)],
                }
            )
        return out

    # ── CEM core ─────────────────────────────────────────

    def plan_campaign_configs(
        self,
        *,
        base_params: Dict[str, Any],
        dataset_meta: Dict[str, Any] | None = None,
        z: Sequence[float] | None = None,
        outcome_model: Any | None = None,
        n_return: int = 3,
        higher_is_better: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Trả về tối đa n_return dict {search_algorithm, time_limit} distinct,
        xếp theo score dự đoán (tốt nhất trước).
        """
        n_return = max(1, int(n_return))
        if outcome_model is None or not getattr(outcome_model, "is_ready", False):
            return self._fallback_configs(n_return)

        rng = np.random.default_rng(self.seed)
        n_algo = len(self.algorithms)
        n_time = len(self.time_options)
        p_algo = np.full(n_algo, 1.0 / n_algo)
        p_time = np.full(n_time, 1.0 / n_time)

        sign = 1.0 if higher_is_better else -1.0
        scored: Dict[tuple, float] = {}

        def score_config(algo: str, t: int) -> Optional[float]:
            params = dict(base_params)
            params["search_algorithm"] = algo
            params["time_limit"] = t
            key = (algo, t)
            if key in scored:
                return scored[key]
            pred = outcome_model.predict(params, dataset_meta, z)
            if pred is None:
                return None
            val = sign * (pred.mean + self.exploration_weight * pred.std)
            scored[key] = val
            return val

        n_elite = max(1, math.ceil(self.elite_fraction * self.n_candidates))
        for _ in range(self.n_iterations):
            idx_algo = rng.choice(n_algo, size=self.n_candidates, p=p_algo)
            idx_time = rng.choice(n_time, size=self.n_candidates, p=p_time)
            batch = []
            for ia, it in zip(idx_algo, idx_time):
                val = score_config(self.algorithms[ia], self.time_options[it])
                if val is not None:
                    batch.append((val, ia, it))
            if not batch:
                return self._fallback_configs(n_return)

            batch.sort(key=lambda item: item[0], reverse=True)
            elite = batch[:n_elite]

            new_algo = np.zeros(n_algo)
            new_time = np.zeros(n_time)
            for _, ia, it in elite:
                new_algo[ia] += 1.0
                new_time[it] += 1.0
            new_algo = (new_algo + self.smoothing) / (new_algo.sum() + self.smoothing * n_algo)
            new_time = (new_time + self.smoothing) / (new_time.sum() + self.smoothing * n_time)
            p_algo, p_time = new_algo, new_time

        # Xếp mọi config đã chấm điểm, lấy distinct top n_return
        ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
        out: List[Dict[str, Any]] = []
        for (algo, t), _ in ranked:
            out.append({"search_algorithm": algo, "time_limit": t})
            if len(out) >= n_return:
                break
        if not out:
            return self._fallback_configs(n_return)
        return out
