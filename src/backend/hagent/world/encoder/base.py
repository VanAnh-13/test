"""World encoder protocol — z = enc(o)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from hagent.world.schema import AutoMLObservation, GoalSpec, LatentState


@runtime_checkable
class WorldEncoder(Protocol):
    def encode(self, observation: AutoMLObservation) -> LatentState:
        """Map observation o_t → latent z_t."""
        ...

    def encode_goal(
        self, goal: GoalSpec, observation: AutoMLObservation
    ) -> LatentState:
        """Map goal (+ context observation) → goal latent z_g."""
        ...
