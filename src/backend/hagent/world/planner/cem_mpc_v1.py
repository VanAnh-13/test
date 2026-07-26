"""
CEM-MPC v1 — batch planner budget-aware, receding horizon.

Mỗi campaign là một "bước MPC": re-plan với model mới nhất, chọn batch bằng
Thompson sampling trên dự đoán (μ, σ) của outcome model, với exploration
tự giảm theo budget còn lại:

    σ_eff = σ · sqrt(remaining_after_batch / total_budget)

Batch cuối (remaining_after = 0) → σ_eff = 0 → exploit thuần (argmax μ).
Pool ứng viên lấy từ CemConfigV1Planner (đã gồm model subset + extra dims).

Đây là MPC theo nghĩa receding-horizon + budget-annealed exploration; KHÔNG
mô phỏng retrain trong imagination (chi phí không đáng cho model numpy nhỏ,
và re-plan mỗi campaign đã cập nhật model thật).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from hagent.world.planner.cem_config_v1 import CemConfigV1Planner

logger = logging.getLogger(__name__)


class CemMpcV1Planner:
    """Budget-aware batch planner; bọc CemConfigV1Planner làm pool ứng viên."""

    def __init__(self, config: dict | None = None):
        self.config = dict(config or {})
        self.pool_size = max(4, int(self.config.get("pool_size", 12)))
        self.min_sigma = float(self.config.get("min_sigma", 1e-6))
        self.seed = int(self.config.get("seed", 0))
        self._pool_planner = CemConfigV1Planner(self.config)

    # Tương thích interface builder (dùng như cem_config_v1 khi không có budget)
    def plan_campaign_configs(self, **kwargs) -> List[Dict[str, Any]]:
        return self._pool_planner.plan_campaign_configs(**kwargs)

    def plan_batch(
        self,
        *,
        base_params: Dict[str, Any],
        dataset_meta: Dict[str, Any] | None = None,
        z: Sequence[float] | None = None,
        outcome_model: Any | None = None,
        n: int,
        remaining_budget: int,
        total_budget: int,
        higher_is_better: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Chọn n config cho campaign kế tiếp, biết còn remaining_budget job
        trong tổng total_budget.
        """
        n = max(1, int(n))
        if outcome_model is None or not getattr(outcome_model, "is_ready", False):
            return self._pool_planner.plan_campaign_configs(
                base_params=base_params,
                dataset_meta=dataset_meta,
                z=z,
                outcome_model=outcome_model,
                n_return=n,
                higher_is_better=higher_is_better,
            )

        # Pool ứng viên distinct, đã xếp theo score CEM
        pool = self._pool_planner.plan_campaign_configs(
            base_params=base_params,
            dataset_meta=dataset_meta,
            z=z,
            outcome_model=outcome_model,
            n_return=max(self.pool_size, n),
            higher_is_better=higher_is_better,
        )
        if not pool:
            return []

        preds = []
        for prop in pool:
            params = dict(base_params)
            params.update(prop)
            pred = outcome_model.predict(params, dataset_meta, z)
            if pred is not None:
                preds.append((prop, float(pred.mean), max(float(pred.std), self.min_sigma)))
        if not preds:
            return pool[:n]

        sign = 1.0 if higher_is_better else -1.0
        remaining_after = max(0, int(remaining_budget) - n)
        total = max(1, int(total_budget))
        anneal = math.sqrt(remaining_after / total)

        if anneal <= 0.0:
            # Batch cuối: exploit thuần
            preds.sort(key=lambda item: sign * item[1], reverse=True)
            return [prop for prop, _, _ in preds[:n]]

        # Thompson sampling: mỗi slot một lượt draw mới trên các ứng viên còn lại
        rng = np.random.default_rng(self.seed + remaining_budget)
        chosen: List[Dict[str, Any]] = []
        available = list(preds)
        for _ in range(min(n, len(available))):
            draws = [
                sign * (mu + anneal * sig * float(rng.standard_normal()))
                for _, mu, sig in available
            ]
            idx = int(np.argmax(draws))
            chosen.append(available[idx][0])
            available.pop(idx)
        return chosen
