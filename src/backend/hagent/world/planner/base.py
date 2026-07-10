"""Latent planner protocol."""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from hagent.world.schema import AutoMLAction, GoalSpec, LatentState, PlanResult


@runtime_checkable
class WorldPlanner(Protocol):
    def plan(
        self,
        z0: LatentState,
        z_goal: LatentState,
        *,
        goal: GoalSpec,
        action_space: List[str],
        observation_context: dict | None = None,
    ) -> List[PlanResult]:
        """Return top plan candidates ranked by latent cost."""
        ...
