"""World predictor protocol — ẑ' = pred(z, a) — và helper checkpoint dùng chung."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from hagent.world.schema import AutoMLAction, LatentState


@runtime_checkable
class WorldPredictor(Protocol):
    def predict(self, z: LatentState, action: AutoMLAction) -> LatentState:
        """Predict the next latent under action a_t."""
        ...


def load_mlp_weights(
    data: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Đọc bộ trọng số MLP (W1, b1, W2, b2) từ checkpoint .npz đã mở.

    Dùng chung cho mọi predictor MLP một lớp ẩn (neural_jepa_v1,
    outcome_head_v1) để tránh lặp lại khối np.load/np.asarray.
    """
    return (
        np.asarray(data["W1"], dtype=np.float64),
        np.asarray(data["b1"], dtype=np.float64),
        np.asarray(data["W2"], dtype=np.float64),
        np.asarray(data["b2"], dtype=np.float64),
    )
