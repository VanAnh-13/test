"""
Neural JEPA-lite predictor — small MLP: ẑ' = f_θ(z, a).

Numpy-only checkpoint (.npz). Falls back to tabular if no weights / load fail.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from hagent.world.predictor.base import load_mlp_weights
from hagent.world.schema import AutoMLAction, LatentState

logger = structlog.get_logger(__name__)

_DEFAULT_ACTIONS = [
    "list_datasets",
    "get_dataset_info",
    "get_features",
    "preview_data",
    "get_available_models",
    "get_metrics",
    "start_training",
    "get_job_info",
    "list_jobs",
    "check_system_health",
    "get_world_state",
    "cancel_job",
    "predict_batch",
]


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _action_one_hot(action_type: str, action_space: Sequence[str]) -> np.ndarray:
    n = len(action_space)
    v = np.zeros(n, dtype=np.float64)
    try:
        idx = list(action_space).index(action_type)
        v[idx] = 1.0
    except ValueError:
        # Unknown action: hash into a bucket
        if n:
            v[hash(action_type) % n] = 0.5
    return v


class NeuralJepaV1Predictor:
    """
    One-hidden-layer MLP:
      h = tanh(W1 @ [z; a_oh] + b1)
      z' = normalize(W2 @ h + b2)
    """

    def __init__(self, config: dict | None = None):
        self.config = dict(config or {})
        self.hidden_dim = int(self.config.get("hidden_dim", 128))
        self.action_space: list[str] = list(
            self.config.get("action_space") or _DEFAULT_ACTIONS
        )
        self.checkpoint_path = self.config.get("checkpoint_path")
        self.fallback_backend = str(
            self.config.get("fallback") or "tabular_transition_v1"
        )
        self._fallback = None
        self.W1: np.ndarray | None = None
        self._b1: np.ndarray | None = None
        self._W2: np.ndarray | None = None
        self._b2: np.ndarray | None = None
        self._latent_dim: int | None = None
        self.loaded = False

        path = self.checkpoint_path
        if path:
            self._try_load(str(path))
        if not self.loaded:
            self._init_fallback()

    def _init_fallback(self) -> None:
        # Direct tabular import — avoid factory circular import with neural_jepa_v1
        try:
            from hagent.world.predictor.tabular_transition_v1 import (
                TabularTransitionV1Predictor,
            )

            fb_cfg = dict(self.config.get("fallback_config") or {})
            self._fallback = TabularTransitionV1Predictor(fb_cfg)
        except Exception as exc:
            logger.debug("Neural fallback init failed: %s", exc)
            self._fallback = None

    def _try_load(self, path: str) -> None:
        p = Path(path)
        if not p.is_file():
            logger.info("Neural JEPA checkpoint missing: %s", path)
            return
        try:
            data = np.load(str(p), allow_pickle=True)
            self.W1, self._b1, self._W2, self._b2 = load_mlp_weights(data)
            self._latent_dim = int(data["latent_dim"]) if "latent_dim" in data else None
            if "action_space" in data:
                raw = data["action_space"]
                if isinstance(raw, np.ndarray):
                    self.action_space = [str(x) for x in raw.tolist()]
                else:
                    self.action_space = [str(x) for x in list(raw)]
            self.loaded = True
            logger.info("Loaded neural JEPA checkpoint from %s", path)
        except Exception as exc:
            logger.warning("Failed to load neural JEPA checkpoint: %s", exc)
            self.loaded = False

    def save(self, path: str, *, latent_dim: int) -> None:
        if self.W1 is None:
            raise RuntimeError("No weights to save")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            W1=self.W1,
            b1=self._b1,
            W2=self._W2,
            b2=self._b2,
            latent_dim=latent_dim,
            action_space=np.array(self.action_space, dtype=object),
            hidden_dim=self.hidden_dim,
        )

    def init_random(self, latent_dim: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        a_dim = len(self.action_space)
        in_dim = latent_dim + a_dim
        h = self.hidden_dim
        scale1 = 1.0 / math.sqrt(in_dim)
        scale2 = 1.0 / math.sqrt(h)
        self.W1 = rng.normal(0, scale1, size=(h, in_dim))
        self._b1 = np.zeros(h)
        self._W2 = rng.normal(0, scale2, size=(latent_dim, h))
        self._b2 = np.zeros(latent_dim)
        self._latent_dim = latent_dim
        self.loaded = True

    def _forward(self, z_vec: np.ndarray, a_oh: np.ndarray) -> np.ndarray:
        assert self.W1 is not None and self._b1 is not None
        assert self._W2 is not None and self._b2 is not None
        x = np.concatenate([z_vec, a_oh])
        h = np.tanh(self.W1 @ x + self._b1)
        out = self._W2 @ h + self._b2
        n = np.linalg.norm(out) or 1.0
        return out / n

    def predict(self, z: LatentState, action: AutoMLAction) -> LatentState:
        if not self.loaded or self.W1 is None:
            if self._fallback is not None:
                return self._fallback.predict(z, action)
            vec = _l2_normalize(list(z.vector))
            return LatentState(
                vector=vec,
                dim=z.dim,
                meta={"predictor": "neural_jepa_v1", "mode": "identity"},
            )

        dim = z.dim
        if self._latent_dim and self._latent_dim != dim:
            # Dimension mismatch — fall back
            if self._fallback is not None:
                return self._fallback.predict(z, action)

        z_vec = np.asarray(list(z.vector)[:dim], dtype=np.float64)
        if z_vec.shape[0] < dim:
            z_vec = np.pad(z_vec, (0, dim - z_vec.shape[0]))
        a_oh = _action_one_hot(action.type, self.action_space)

        # Resize input weights if action space length differs slightly
        expected_in = self.W1.shape[1]
        x = np.concatenate([z_vec, a_oh])
        if x.shape[0] != expected_in:
            if self._fallback is not None:
                return self._fallback.predict(z, action)
            x = x[:expected_in] if x.shape[0] > expected_in else np.pad(
                x, (0, expected_in - x.shape[0])
            )
            h = np.tanh(self.W1 @ x + self._b1)
            out = self._W2 @ h + self._b2
            n = float(np.linalg.norm(out) or 1.0)
            vec = (out / n).tolist()
        else:
            vec = self._forward(z_vec, a_oh).tolist()

        if len(vec) < dim:
            vec = vec + [0.0] * (dim - len(vec))
        vec = _l2_normalize(vec[:dim])
        return LatentState(
            vector=vec,
            dim=dim,
            meta={
                "predictor": "neural_jepa_v1",
                "action_type": action.type,
                "mode": "neural",
            },
        )

    @property
    def latent_dim(self):
        return self._latent_dim


def train_neural_jepa(
    trajectories: list[dict[str, Any]],
    *,
    latent_dim: int,
    hidden_dim: int = 128,
    action_space: Sequence[str] | None = None,
    epochs: int = 50,
    lr: float = 0.01,
    seed: int = 0,
) -> NeuralJepaV1Predictor:
    """
    Offline SGD on ‖f(z,a) − z_next‖² from trajectory docs.
    """
    space = list(action_space or _DEFAULT_ACTIONS)
    pred = NeuralJepaV1Predictor(
        {
            "hidden_dim": hidden_dim,
            "action_space": space,
            "fallback": "tabular_transition_v1",
        }
    )
    pred.init_random(latent_dim, seed=seed)

    samples = []
    for doc in trajectories:
        try:
            z = np.asarray(doc["z"]["vector"], dtype=np.float64)[:latent_dim]
            z_next = np.asarray(doc["z_next"]["vector"], dtype=np.float64)[:latent_dim]
            a_type = str((doc.get("action") or {}).get("type") or "")
            if z.shape[0] < latent_dim:
                z = np.pad(z, (0, latent_dim - z.shape[0]))
            if z_next.shape[0] < latent_dim:
                z_next = np.pad(z_next, (0, latent_dim - z_next.shape[0]))
            samples.append((z, a_type, z_next))
        except Exception:
            continue

    if not samples:
        logger.warning("No valid trajectory samples for neural train")
        return pred

    assert pred.W1 is not None
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        rng.shuffle(samples)
        total = 0.0
        for z, a_type, z_next in samples:
            a_oh = _action_one_hot(a_type, space)
            x = np.concatenate([z, a_oh])
            h = np.tanh(pred.W1 @ x + pred._b1)
            raw = pred._W2 @ h + pred._b2
            nrm = float(np.linalg.norm(raw) or 1.0)
            y_hat = raw / nrm
            err = y_hat - z_next
            loss = float(np.dot(err, err))
            total += loss

            # Gradient through normalize (approx): d(raw/n)/draw ≈ (I - yy^T)/n
            # Simplified: treat as MSE on raw then re-normalize weights lightly
            d_y = 2.0 * err
            # Chain rule approx via unnormalized raw
            d_raw = d_y / nrm
            d_h = pred._W2.T @ d_raw
            d_h *= 1.0 - h * h  # tanh'
            pred._W2 -= lr * np.outer(d_raw, h)
            pred._b2 -= lr * d_raw
            pred.W1 -= lr * np.outer(d_h, x)
            pred._b1 -= lr * d_h

        if epoch % max(1, epochs // 5) == 0:
            logger.info(
                "neural_jepa epoch %d mean_loss=%.6f",
                epoch,
                total / max(1, len(samples)),
            )

    return pred
