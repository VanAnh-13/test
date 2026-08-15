"""Latent planner protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from hagent.world.schema import GoalSpec, LatentState, PlanResult


@runtime_checkable
class WorldPlanner(Protocol):
    def plan(
        self,
        z0: LatentState,
        z_goal: LatentState,
        *,
        goal: GoalSpec,
        action_space: list[str],
        observation_context: dict | None = None,
    ) -> list[PlanResult]:
        """Return top plan candidates ranked by latent cost."""
        ...
