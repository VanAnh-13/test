"""
Deep ensemble của K OutcomeHeadV1 (Lakshminarayanan et al., 2017).

Mỗi member train cùng data với seed khác nhau. Kết hợp mixture moments:
  μ* = mean(μᵢ)
  σ*² = mean(σᵢ² + μᵢ²) − μ*²   (tổng aleatoric + epistemic)

Checkpoint: thư mục chứa member_{i}.npz. is_ready khi ≥1 member sẵn sàng.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import structlog

from hagent.world.predictor.outcome_head_v1 import (
    OutcomeHeadV1,
    OutcomePrediction,
    train_outcome_head,
)

logger = structlog.get_logger(__name__)

_MEMBER_PREFIX = "member_"


class OutcomeEnsemble:
    """Tập hợp K outcome heads; API predict giống OutcomeHeadV1."""

    def __init__(self, config: dict | None = None):
        self.config = dict(config or {})
        self.k = max(1, int(self.config.get("k", 5)))
        self.checkpoint_dir = self.config.get("checkpoint_dir")
        self.members: list[OutcomeHeadV1] = []

        if self.checkpoint_dir:
            self._try_load(str(self.checkpoint_dir))

    # ── State ────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return any(m.is_ready for m in self.members)

    def member_config(self) -> dict:
        cfg = dict(self.config)
        cfg.pop("checkpoint_dir", None)
        cfg.pop("checkpoint_path", None)
        cfg.pop("k", None)
        return cfg

    def _try_load(self, directory: str) -> None:
        d = Path(directory)
        if not d.is_dir():
            logger.info("Outcome ensemble checkpoint dir missing: %s", directory)
            return
        loaded: list[OutcomeHeadV1] = []
        for p in sorted(d.glob(f"{_MEMBER_PREFIX}*.npz")):
            cfg = self.member_config()
            cfg["checkpoint_path"] = str(p)
            head = OutcomeHeadV1(cfg)
            if head.is_ready:
                loaded.append(head)
        if loaded:
            self.members = loaded
            logger.info(
                "Loaded outcome ensemble: %d members from %s", len(loaded), directory
            )

    def save(self, directory: str) -> None:
        if not self.members:
            raise RuntimeError("No ensemble members to save")
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(self.members):
            m.save(str(d / f"{_MEMBER_PREFIX}{i}.npz"))

    # ── Inference ────────────────────────────────────────

    def predict(
        self,
        params: dict[str, Any],
        dataset_meta: dict[str, Any] | None = None,
        z: Sequence[float] | None = None,
    ) -> OutcomePrediction | None:
        preds = [
            p
            for m in self.members
            if (p := m.predict(params, dataset_meta, z)) is not None
        ]
        if not preds:
            return None

        n = len(preds)
        mu_star = sum(p.mean for p in preds) / n
        second_moment = sum(p.std * p.std + p.mean * p.mean for p in preds) / n
        total_var = max(0.0, second_moment - mu_star * mu_star)
        mean_of_means_sq = sum((p.mean - mu_star) ** 2 for p in preds) / n
        return OutcomePrediction(
            mean=mu_star,
            std=math.sqrt(total_var) or 1e-6,
            meta={
                "predictor": "outcome_ensemble",
                "n_members": n,
                "epistemic_std": math.sqrt(mean_of_means_sq),
                "aleatoric_std": math.sqrt(
                    max(0.0, sum(p.std * p.std for p in preds) / n)
                ),
                "member_means": [p.mean for p in preds],
            },
        )


def train_outcome_ensemble(
    samples: list[dict[str, Any]],
    *,
    config: dict | None = None,
    k: int | None = None,
    epochs: int = 200,
    lr: float = 0.01,
    seed: int = 0,
) -> OutcomeEnsemble:
    """Train K members trên cùng samples với seed lệch nhau."""
    ens = OutcomeEnsemble(dict(config or {}))
    n_members = max(1, int(k if k is not None else ens.k))
    member_cfg = ens.member_config()
    ens.members = [
        train_outcome_head(
            samples, config=dict(member_cfg), epochs=epochs, lr=lr, seed=seed + i
        )
        for i in range(n_members)
    ]
    return ens
