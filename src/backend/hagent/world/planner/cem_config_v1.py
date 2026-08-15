"""
CEM thật trên không gian config campaign.

Các chiều tìm kiếm:
  - search_algorithm, time_limit: categorical (phân phối refit từ elite).
  - models: subset của model_options — Bernoulli độc lập từng model,
    xác suất kèm refit từ elite (chuẩn cross-entropy cho biến nhị phân).
  - categorical_dims (config): các chiều categorical bổ sung tùy nền tảng
    (vd. cv_folds) — không cần sửa code khi thêm chiều mới.

Outcome model chưa sẵn sàng → fallback round-robin deterministic (trùng logic
diversify của builder) để hành vi hệ thống không đổi khi chưa train model.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

_DEFAULT_ALGOS = ["grid_search", "bayesian_search", "genetic_algorithm"]
_DEFAULT_TIME_OPTIONS = [180, 300, 600]


class CemConfigV1Planner:
    """CEM trên các chiều categorical + subset của config một training job."""

    def __init__(self, config: dict | None = None):
        self.config = dict(config or {})
        self.algorithms: list[str] = list(
            self.config.get("search_algorithms") or _DEFAULT_ALGOS
        )
        self.time_options: list[int] = [
            int(t) for t in (self.config.get("time_limit_options") or _DEFAULT_TIME_OPTIONS)
        ]
        self.model_options: list[str] = list(self.config.get("model_options") or [])
        self.min_models = max(1, int(self.config.get("min_models", 1)))
        self.categorical_dims: dict[str, list[Any]] = {
            str(k): list(v)
            for k, v in dict(self.config.get("categorical_dims") or {}).items()
            if v
        }
        self.n_candidates = max(4, int(self.config.get("n_candidates", 32)))
        self.n_iterations = max(1, int(self.config.get("n_iterations", 8)))
        self.elite_fraction = min(
            0.9, max(0.05, float(self.config.get("elite_fraction", 0.25)))
        )
        self.smoothing = min(1.0, max(0.0, float(self.config.get("smoothing", 0.25))))
        self.exploration_weight = float(self.config.get("exploration_weight", 0.1))
        self.seed = int(self.config.get("seed", 0))

    # ── Fallback (không có outcome model) ────────────────

    def _fallback_configs(self, n_return: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for i in range(n_return):
            out.append(
                {
                    "search_algorithm": self.algorithms[i % len(self.algorithms)],
                    "time_limit": self.time_options[i % len(self.time_options)],
                }
            )
        return out

    # ── Sampling helpers ─────────────────────────────────

    def _sample_models(
        self, rng: np.random.Generator, p_model: np.ndarray
    ) -> tuple[str, ...]:
        mask = rng.random(len(self.model_options)) < p_model
        if mask.sum() < self.min_models:
            # Ép đủ min_models model có xác suất cao nhất
            order = np.argsort(-p_model)
            mask[:] = False
            mask[order[: self.min_models]] = True
        return tuple(
            sorted(m for m, keep in zip(self.model_options, mask) if keep)
        )

    # ── CEM core ─────────────────────────────────────────

    def plan_campaign_configs(
        self,
        *,
        base_params: dict[str, Any],
        dataset_meta: dict[str, Any] | None = None,
        z: Sequence[float] | None = None,
        outcome_model: Any | None = None,
        n_return: int = 3,
        higher_is_better: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Trả về tối đa n_return dict config distinct (search_algorithm,
        time_limit, models?, extra dims?), xếp theo score dự đoán.
        """
        n_return = max(1, int(n_return))
        if outcome_model is None or not getattr(outcome_model, "is_ready", False):
            return self._fallback_configs(n_return)

        rng = np.random.default_rng(self.seed)
        n_algo = len(self.algorithms)
        n_time = len(self.time_options)
        p_algo = np.full(n_algo, 1.0 / n_algo)
        p_time = np.full(n_time, 1.0 / n_time)
        p_model = np.full(len(self.model_options), 0.5) if self.model_options else None
        extra_names = list(self.categorical_dims)
        p_extra = {
            name: np.full(len(opts), 1.0 / len(opts))
            for name, opts in self.categorical_dims.items()
        }

        sign = 1.0 if higher_is_better else -1.0
        scored: dict[tuple, float] = {}
        proposals: dict[tuple, dict[str, Any]] = {}

        def score_config(
            algo: str,
            t: int,
            models: tuple[str, ...] | None,
            extra: dict[str, Any],
        ) -> float | None:
            params = dict(base_params)
            params["search_algorithm"] = algo
            params["time_limit"] = t
            if models is not None:
                params["models"] = list(models)
            params.update(extra)
            key = (algo, t, models or (), tuple(sorted(extra.items())))
            if key in scored:
                return scored[key]
            pred = outcome_model.predict(params, dataset_meta, z)
            if pred is None:
                return None
            # Optimism-in-face-of-uncertainty đúng cả hai chiều: bonus σ luôn
            # cộng vào phía "tốt" (sign*(μ+βσ) sẽ thành PHẠT σ khi minimize).
            val = sign * pred.mean + self.exploration_weight * pred.std
            scored[key] = val
            prop: dict[str, Any] = {"search_algorithm": algo, "time_limit": t}
            if models is not None:
                prop["models"] = list(models)
            prop.update(extra)
            proposals[key] = prop
            return val

        n_elite = max(1, math.ceil(self.elite_fraction * self.n_candidates))
        for _ in range(self.n_iterations):
            batch = []
            for _ in range(self.n_candidates):
                ia = int(rng.choice(n_algo, p=p_algo))
                it = int(rng.choice(n_time, p=p_time))
                models = (
                    self._sample_models(rng, p_model) if p_model is not None else None
                )
                extra = {
                    name: self.categorical_dims[name][
                        int(rng.choice(len(self.categorical_dims[name]), p=p_extra[name]))
                    ]
                    for name in extra_names
                }
                val = score_config(self.algorithms[ia], self.time_options[it], models, extra)
                if val is not None:
                    batch.append((val, ia, it, models, extra))
            if not batch:
                return self._fallback_configs(n_return)

            batch.sort(key=lambda item: item[0], reverse=True)
            elite = batch[:n_elite]

            new_algo = np.zeros(n_algo)
            new_time = np.zeros(n_time)
            new_model = (
                np.zeros(len(self.model_options)) if p_model is not None else None
            )
            new_extra = {name: np.zeros(len(opts)) for name, opts in self.categorical_dims.items()}
            for _, ia, it, models, extra in elite:
                new_algo[ia] += 1.0
                new_time[it] += 1.0
                if new_model is not None and models is not None:
                    for m in models:
                        new_model[self.model_options.index(m)] += 1.0
                for name in extra_names:
                    new_extra[name][self.categorical_dims[name].index(extra[name])] += 1.0

            p_algo = (new_algo + self.smoothing) / (new_algo.sum() + self.smoothing * n_algo)
            p_time = (new_time + self.smoothing) / (new_time.sum() + self.smoothing * n_time)
            if new_model is not None:
                # Bernoulli refit: tần suất xuất hiện trong elite, có smoothing
                p_model = (new_model + self.smoothing) / (len(elite) + 2.0 * self.smoothing)
                p_model = np.clip(p_model, 0.02, 0.98)
            for name in extra_names:
                counts = new_extra[name]
                p_extra[name] = (counts + self.smoothing) / (
                    counts.sum() + self.smoothing * len(self.categorical_dims[name])
                )

        ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
        out: list[dict[str, Any]] = []
        for key, _ in ranked:
            out.append(proposals[key])
            if len(out) >= n_return:
                break
        if not out:
            return self._fallback_configs(n_return)
        return out
