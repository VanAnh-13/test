"""
Dynamics ensemble — K NeuralJepaV1 (seed lệch) cho transition ẑ' = f(z, a).

Predict trả LatentState với meta["std"] = độ lệch chuẩn per-dim giữa các
member (epistemic). Surprise chuẩn hóa (world/surprise.py) tự nhận meta.std
để tính z-score thay vì khoảng cách thô với ngưỡng cứng.

Tương thích WorldPredictor protocol — cắm được vào WorldModelService qua
factory backend "dynamics_ensemble".
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from hagent.world.predictor.neural_jepa_v1 import (
    NeuralJepaV1Predictor,
    train_neural_jepa,
)
from hagent.world.schema import AutoMLAction, LatentState

logger = logging.getLogger(__name__)

_MEMBER_PREFIX = "member_"


class DynamicsEnsemble:
    """Tập hợp K neural predictors; mean latent + per-dim std."""

    def __init__(self, config: dict | None = None):
        self.config = dict(config or {})
        self.k = max(1, int(self.config.get("k", 5)))
        self.checkpoint_dir = self.config.get("checkpoint_dir")
        self.members: List[NeuralJepaV1Predictor] = []

        if self.checkpoint_dir:
            self._try_load(str(self.checkpoint_dir))

    @property
    def is_ready(self) -> bool:
        return any(m._loaded and m._W1 is not None for m in self.members)

    def _member_config(self) -> dict:
        cfg = dict(self.config)
        for key in ("checkpoint_dir", "k"):
            cfg.pop(key, None)
        cfg.pop("checkpoint_path", None)
        return cfg

    def _try_load(self, directory: str) -> None:
        d = Path(directory)
        if not d.is_dir():
            logger.info("Dynamics ensemble checkpoint dir missing: %s", directory)
            return
        loaded: List[NeuralJepaV1Predictor] = []
        for p in sorted(d.glob(f"{_MEMBER_PREFIX}*.npz")):
            cfg = self._member_config()
            cfg["checkpoint_path"] = str(p)
            member = NeuralJepaV1Predictor(cfg)
            if member._loaded and member._W1 is not None:
                loaded.append(member)
        if loaded:
            self.members = loaded
            logger.info(
                "Loaded dynamics ensemble: %d members from %s", len(loaded), directory
            )

    def save(self, directory: str) -> None:
        ready = [m for m in self.members if m._loaded and m._W1 is not None]
        if not ready:
            raise RuntimeError("No dynamics ensemble members to save")
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(ready):
            m.save(str(d / f"{_MEMBER_PREFIX}{i}.npz"), latent_dim=m._latent_dim or 0)

    def predict(self, z: LatentState, action: AutoMLAction) -> LatentState:
        ready = [m for m in self.members if m._loaded and m._W1 is not None]
        if not ready:
            # Chưa train → identity kèm cờ mode (không bịa uncertainty)
            return LatentState(
                vector=list(z.vector),
                dim=z.dim,
                meta={"predictor": "dynamics_ensemble", "mode": "identity"},
            )

        outs = np.array(
            [m.predict(z, action).vector for m in ready], dtype=np.float64
        )
        mean = outs.mean(axis=0)
        std = outs.std(axis=0)
        norm = float(np.linalg.norm(mean)) or 1.0
        return LatentState(
            vector=(mean / norm).tolist(),
            dim=z.dim,
            meta={
                "predictor": "dynamics_ensemble",
                "mode": "ensemble",
                "n_members": len(ready),
                "std": std.tolist(),
                "std_mean": float(std.mean()),
            },
        )


def train_dynamics_ensemble(
    trajectories: List[Dict[str, Any]],
    *,
    latent_dim: int,
    k: int = 5,
    hidden_dim: int = 128,
    action_space: Sequence[str] | None = None,
    epochs: int = 50,
    lr: float = 0.01,
    seed: int = 0,
) -> DynamicsEnsemble:
    """Train K members trên cùng trajectories với seed lệch nhau."""
    ens = DynamicsEnsemble({"k": k, "hidden_dim": hidden_dim})
    ens.members = [
        train_neural_jepa(
            trajectories,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            action_space=action_space,
            epochs=epochs,
            lr=lr,
            seed=seed + i,
        )
        for i in range(max(1, k))
    ]
    return ens
