"""
World Model schemas — structured observation / action / latent / persisted state.

LeWM mapping (arXiv:2603.19312):
  o_t  -> AutoMLObservation
  a_t  -> AutoMLAction
  z_t  -> LatentState
  WorldState is the durable document behind observations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, TypedDict


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _json_ready(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if value.__class__.__name__ == "ObjectId":
        return str(value)
    return value


# ── Persisted entries ────────────────────────────────────


class DatasetEntry(TypedDict, total=False):
    id: str
    name: str
    n_rows: int
    n_cols: int
    features: List[str]
    target: Optional[str]
    problem_type_inferred: Optional[str]
    last_seen: datetime


class JobEntry(TypedDict, total=False):
    id: str
    dataset_id: str
    config: Dict[str, Any]
    status: str
    metrics: Dict[str, float]
    best_model: Optional[str]
    best_score: Optional[float]
    started_at: Optional[datetime]
    finished_at: Optional[datetime]


class PlanEntry(TypedDict, total=False):
    plan_id: str
    title: str
    status: str
    requirements: Dict[str, Any]
    steps: List[Dict[str, Any]]
    constraints: Dict[str, Any]
    score_estimate: Optional[float]
    verification: Optional[Dict[str, Any]]
    world_refs: Dict[str, Any]
    created_at: Any
    updated_at: Any


class GoalEntry(TypedDict, total=False):
    goal_id: str
    description: str
    status: str
    goal_type: str
    metric: Optional[str]
    problem_type: Optional[str]
    linked_plan_id: Optional[str]
    linked_job_ids: List[str]


# ── LeWM-style runtime types ─────────────────────────────


class FocusSpec(TypedDict, total=False):
    dataset_id: Optional[str]
    job_id: Optional[str]
    plan_id: Optional[str]


class GoalSpec(TypedDict, total=False):
    """Goal for latent planning (encode_goal target)."""
    goal_type: str  # e.g. train | analyze | evaluate | list | respond
    description: str
    metric: Optional[str]
    target_score: Optional[float]
    problem_type: Optional[str]
    dataset_id: Optional[str]
    target_column: Optional[str]
    constraints: Dict[str, Any]


@dataclass
class AutoMLObservation:
    """Structured observation o_t — encoder input only (not chat text)."""

    user_id: str
    datasets: Dict[str, DatasetEntry] = field(default_factory=dict)
    jobs: Dict[str, JobEntry] = field(default_factory=dict)
    focus: FocusSpec = field(default_factory=dict)
    phase: str = "idle"
    goal: Optional[GoalSpec] = None
    history_digest: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _json_ready(asdict(self))

    @classmethod
    def from_world_state(
        cls,
        state: "WorldState",
        *,
        phase: str | None = None,
        goal: GoalSpec | None = None,
        history_digest: str | None = None,
    ) -> "AutoMLObservation":
        focus: FocusSpec = {}
        if state.active_dataset_id:
            focus["dataset_id"] = state.active_dataset_id
        if state.active_job_id:
            focus["job_id"] = state.active_job_id
        if state.active_plan_id:
            focus["plan_id"] = state.active_plan_id
        return cls(
            user_id=state.user_id,
            datasets=dict(state.datasets or {}),
            jobs=dict(state.jobs or {}),
            focus=focus,
            phase=phase or state.phase or "idle",
            goal=goal or state.active_goal,
            history_digest=history_digest,
        )


@dataclass
class AutoMLAction:
    """Closed action space — maps 1:1 to registered tools."""

    type: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "params": dict(self.params)}


@dataclass
class LatentState:
    """Compact latent z_t."""

    vector: List[float]
    dim: int
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"vector": list(self.vector), "dim": self.dim, "meta": dict(self.meta)}


@dataclass
class SurpriseResult:
    value: float
    level: str  # low | medium | high
    predicted_dim: int
    actual_dim: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlanStep:
    action: AutoMLAction
    agent: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "agent": self.agent,
        }


@dataclass
class PlanResult:
    """Output of latent planner (CEM-lite)."""

    plan_id: str
    steps: List[PlanStep]
    cost: float
    score_estimate: Optional[float] = None
    title: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "title": self.title,
            "cost": self.cost,
            "score_estimate": self.score_estimate,
            "steps": [s.to_dict() for s in self.steps],
            "meta": dict(self.meta),
        }


# ── Durable world document ───────────────────────────────


@dataclass
class WorldState:
    user_id: str
    datasets: Dict[str, DatasetEntry] = field(default_factory=dict)
    jobs: Dict[str, JobEntry] = field(default_factory=dict)
    goals: List[Dict[str, Any]] = field(default_factory=list)
    plans: Dict[str, PlanEntry] = field(default_factory=dict)
    active_plan_id: Optional[str] = None
    active_dataset_id: Optional[str] = None
    active_job_id: Optional[str] = None
    active_goal: Optional[GoalSpec] = None
    phase: str = "idle"
    last_verification: Optional[Dict[str, Any]] = None
    last_surprise: Optional[Dict[str, Any]] = None
    cost_metrics: Dict[str, Any] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=utc_now)
    created_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return _json_ready(asdict(self))

    def to_observation(
        self,
        *,
        phase: str | None = None,
        goal: GoalSpec | None = None,
    ) -> AutoMLObservation:
        return AutoMLObservation.from_world_state(
            self, phase=phase, goal=goal
        )
