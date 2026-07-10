"""World predictor protocol — ẑ' = pred(z, a)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from hagent.world.schema import AutoMLAction, LatentState


@runtime_checkable
class WorldPredictor(Protocol):
    def predict(self, z: LatentState, action: AutoMLAction) -> LatentState:
        """Predict next latent under action a_t."""
        ...
