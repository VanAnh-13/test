"""
HAgent World Model package (LeWM-inspired).

Public API — prefer WorldModelService for agents.
"""

from hagent.world.schema import (
    AutoMLAction,
    AutoMLObservation,
    GoalSpec,
    LatentState,
    PlanResult,
    SurpriseResult,
    WorldState,
    utc_now,
)
from hagent.world.service import WorldModelService
from hagent.world.state_store import WorldStateStore, create_world_state_store
from hagent.world.updater import apply_plan_event, apply_tool_output

__all__ = [
    "AutoMLAction",
    "AutoMLObservation",
    "GoalSpec",
    "LatentState",
    "PlanResult",
    "SurpriseResult",
    "WorldState",
    "WorldModelService",
    "WorldStateStore",
    "create_world_state_store",
    "apply_tool_output",
    "apply_plan_event",
    "utc_now",
]
