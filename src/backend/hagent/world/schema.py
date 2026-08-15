"""
Schema World Model cho observation, action, latent và trạng thái lưu bền vững.

LeWM mapping (arXiv:2603.19312):
  o_t  -> AutoMLObservation
  a_t  -> AutoMLAction
  z_t  -> LatentState
  WorldState là tài liệu bền vững đứng sau các observation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, TypedDict


def utc_now() -> datetime:
    return datetime.now(UTC)


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


# ── Các bản ghi lưu bền vững ─────────────────────────────


class DatasetEntry(TypedDict, total=False):
    id: str
    name: str
    n_rows: int
    n_cols: int
    features: list[str]
    target: str | None
    problem_type_inferred: str | None
    last_seen: datetime


class JobEntry(TypedDict, total=False):
    id: str
    dataset_id: str
    config: dict[str, Any]
    status: str
    metrics: dict[str, float]
    best_model: str | None
    best_score: float | None
    started_at: datetime | None
    finished_at: datetime | None


class PlanEntry(TypedDict, total=False):
    plan_id: str
    title: str
    status: str
    requirements: dict[str, Any]
    steps: list[dict[str, Any]]
    constraints: dict[str, Any]
    score_estimate: float | None
    verification: dict[str, Any] | None
    world_refs: dict[str, Any]
    created_at: Any
    updated_at: Any


class GoalEntry(TypedDict, total=False):
    goal_id: str
    description: str
    status: str
    goal_type: str
    metric: str | None
    problem_type: str | None
    linked_plan_id: str | None
    linked_job_ids: list[str]


CURRENT_SCHEMA_VERSION = "1.0"


# ── Các kiểu runtime theo LeWM ───────────────────────────


class FocusSpec(TypedDict, total=False):
    dataset_id: str | None
    job_id: str | None
    plan_id: str | None


class GoalSpec(TypedDict, total=False):
    """Mục tiêu cho quá trình lập kế hoạch latent, dùng làm đích của encode_goal."""

    goal_type: str  # ví dụ: train | analyze | evaluate | list | respond
    description: str
    metric: str | None
    target_score: float | None
    problem_type: str | None
    dataset_id: str | None
    target_column: str | None
    constraints: dict[str, Any]


@dataclass
class AutoMLObservation:
    """Observation o_t có cấu trúc, chỉ dùng làm đầu vào encoder, không phải nội dung chat."""

    user_id: str
    datasets: dict[str, DatasetEntry] = field(default_factory=dict)
    jobs: dict[str, JobEntry] = field(default_factory=dict)
    focus: FocusSpec = field(default_factory=dict)
    phase: str = "idle"
    goal: GoalSpec | None = None
    history_digest: str | None = None
    schema_version: str = CURRENT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))

    @classmethod
    def from_world_state(
        cls,
        state: WorldState,
        *,
        phase: str | None = None,
        goal: GoalSpec | None = None,
        history_digest: str | None = None,
    ) -> AutoMLObservation:
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
            schema_version=state.schema_version,
        )


@dataclass
class AutoMLAction:
    """Không gian action đóng, ánh xạ 1:1 với các công cụ đã đăng ký."""

    type: str
    params: dict[str, Any] = field(default_factory=dict)
    schema_version: str = CURRENT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "params": dict(self.params),
            "schema_version": self.schema_version,
        }


@dataclass
class LatentState:
    """Biểu diễn latent z_t gọn."""

    vector: list[float]
    dim: int
    meta: dict[str, Any] = field(default_factory=dict)
    schema_version: str = CURRENT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "vector": list(self.vector),
            "dim": self.dim,
            "meta": dict(self.meta),
            "schema_version": self.schema_version,
        }


@dataclass
class SurpriseResult:
    value: float
    level: str  # low | medium | high
    predicted_dim: int
    actual_dim: int
    schema_version: str = CURRENT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DistributionType(str, Enum):
    GAUSSIAN = "gaussian"
    BETA = "beta"
    CATEGORICAL = "categorical"
    DIRICHLET = "dirichlet"


@dataclass
class DistributionSpec:
    """Đặc tả phân phối tham số cho kết quả và niềm tin của model."""

    dist_type: str  # "gaussian" | "beta" | "categorical" | "dirichlet"
    params: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    schema_version: str = CURRENT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "dist_type": self.dist_type,
            "params": dict(self.params),
            "meta": dict(self.meta),
            "schema_version": self.schema_version,
        }


@dataclass
class PlanStep:
    action: AutoMLAction
    agent: str | None = None
    schema_version: str = CURRENT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "agent": self.agent,
            "schema_version": self.schema_version,
        }


@dataclass
class PlanResult:
    """Kết quả của bộ lập kế hoạch latent CEM-lite."""

    plan_id: str
    steps: list[PlanStep]
    cost: float
    score_estimate: float | None = None
    title: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    schema_version: str = CURRENT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "title": self.title,
            "cost": self.cost,
            "score_estimate": self.score_estimate,
            "steps": [s.to_dict() for s in self.steps],
            "meta": dict(self.meta),
            "schema_version": self.schema_version,
        }


# ── Tài liệu world bền vững ──────────────────────────────


@dataclass
class WorldState:
    user_id: str
    datasets: dict[str, DatasetEntry] = field(default_factory=dict)
    jobs: dict[str, JobEntry] = field(default_factory=dict)
    goals: list[dict[str, Any]] = field(default_factory=list)
    plans: dict[str, PlanEntry] = field(default_factory=dict)
    active_plan_id: str | None = None
    active_dataset_id: str | None = None
    active_job_id: str | None = None
    active_goal: GoalSpec | None = None
    phase: str = "idle"
    last_verification: dict[str, Any] | None = None
    last_surprise: dict[str, Any] | None = None
    cost_metrics: dict[str, Any] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=utc_now)
    created_at: datetime = field(default_factory=utc_now)
    schema_version: str = CURRENT_SCHEMA_VERSION
    update_frequency: float = 1.0
    surprise_momentum: float = 0.0

    @classmethod
    def from_execution_snapshot(
        cls,
        snapshot: Mapping[str, Any],
        *,
        user_id: Any = None,
    ) -> WorldState:
        """Khôi phục các trường dùng khi cập nhật world trong lúc thực thi."""
        return cls(
            user_id=str(user_id or snapshot.get("user_id") or ""),
            datasets=dict(snapshot.get("datasets") or {}),
            jobs=dict(snapshot.get("jobs") or {}),
            goals=list(snapshot.get("goals") or []),
            plans=dict(snapshot.get("plans") or {}),
            active_plan_id=snapshot.get("active_plan_id"),
            active_dataset_id=snapshot.get("active_dataset_id"),
            active_job_id=snapshot.get("active_job_id"),
            active_goal=snapshot.get("active_goal"),
            phase=str(snapshot.get("phase") or "idle"),
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))

    def to_observation(
        self,
        *,
        phase: str | None = None,
        goal: GoalSpec | None = None,
    ) -> AutoMLObservation:
        return AutoMLObservation.from_world_state(self, phase=phase, goal=goal)
