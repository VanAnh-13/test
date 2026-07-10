"""Agent harness scenario + result schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExpectSpec:
    """Declarative expectations for a scenario run."""

    success: bool = True
    goal_type: Optional[str] = None
    route_in: List[str] = field(default_factory=list)
    tools_called_min: int = 0
    tools_called_max: Optional[int] = None
    tools_include: List[str] = field(default_factory=list)
    tools_order: List[str] = field(default_factory=list)
    has_job: Optional[bool] = None
    hierarchy_status: Optional[str] = None
    campaign_status: Optional[str] = None
    plan_status: Optional[str] = None
    wm_has_job: Optional[bool] = None
    event_types_include: List[str] = field(default_factory=list)
    max_elapsed_seconds: Optional[float] = None
    max_tools: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "ExpectSpec":
        data = dict(data or {})
        # aliases
        if "max_tools" in data and "tools_called_max" not in data:
            data["tools_called_max"] = data.get("max_tools")
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class AgentScenario:
    """One harness scenario (YAML-loadable)."""

    id: str
    name: str
    message: str
    tags: List[str] = field(default_factory=list)
    user_id: str = "eval_user"
    world_model: Dict[str, Any] = field(default_factory=dict)
    goal: Dict[str, Any] = field(default_factory=dict)
    turns: List[Dict[str, str]] = field(default_factory=list)
    expect: ExpectSpec = field(default_factory=ExpectSpec)
    # Offline mode hints (legacy Phase 7)
    expect_goal_type: Optional[str] = None
    expect_min_tools: int = 0
    expect_has_job: bool = False
    expect_metric: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["expect"] = self.expect.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentScenario":
        raw = dict(data)
        expect = ExpectSpec.from_dict(raw.pop("expect", None))
        # Bridge legacy fields into expect when not set
        if raw.get("expect_goal_type") and not expect.goal_type:
            expect.goal_type = raw["expect_goal_type"]
        if raw.get("expect_min_tools") and not expect.tools_called_min:
            expect.tools_called_min = int(raw["expect_min_tools"])
        if raw.get("expect_has_job") is not None and expect.has_job is None:
            expect.has_job = bool(raw["expect_has_job"])
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        obj = cls(
            **{k: v for k, v in raw.items() if k in known and k != "expect"},
            expect=expect,
        )
        return obj


@dataclass
class AgentRunResult:
    """Normalized result of one scenario × layer/mode."""

    scenario_id: str
    layer: str  # offline | graph | api
    mode: str
    success: bool
    reasons: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    tools_called: int = 0
    tool_names: List[str] = field(default_factory=list)
    steps_executed: int = 0
    revisions: int = 0
    campaign_variants: int = 0
    campaign_completed: int = 0
    best_score: Optional[float] = None
    best_job_id: Optional[str] = None
    plan_status: Optional[str] = None
    campaign_status: Optional[str] = None
    hierarchy_status: Optional[str] = None
    hierarchy_depth: int = 0
    route: Optional[str] = None
    event_types: List[str] = field(default_factory=list)
    cost_metrics: Dict[str, Any] = field(default_factory=dict)
    response: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
