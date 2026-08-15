"""
Các định nghĩa kiểu dùng chung cho toàn bộ hệ thống HAgent.

Module này là lớp thấp nhất, không được import bất kỳ module nội bộ nào của
hagent để tránh phụ thuộc vòng.

Quy tắc thiết kế:
  - Chỉ dùng thư viện chuẩn và pydantic nếu cần.
  - Không import từ hagent.world, hagent.agent, hagent.bridge
  - Các module khác import từ đây, không ngược lại
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

try:
    # Dành cho Python 3.11 trở lên.
    from typing import NotRequired, TypedDict
except ImportError:  # Dành cho Python 3.10.
    from typing import NotRequired  # type: ignore[assignment]

    from typing_extensions import TypedDict

# ── Enums ────────────────────────────────────────────────────────────────────


class RouteType(str, Enum):
    """Loại routing trong graph điều phối.

    Thay thế các chuỗi literal rải rác như 'train', 'evaluate' trong graph.py
    để bảo đảm an toàn kiểu và hỗ trợ IDE.
    """

    TRAIN = "train"
    EVALUATE = "evaluate"
    ANALYZE = "analyze"
    RESPOND = "respond"
    CAMPAIGN = "campaign"
    HIERARCHY = "hierarchy"
    PLAN = "plan"
    REVISE = "revise"
    PLAN_EXECUTOR = "plan_executor"
    COORDINATOR_TOOLS = "coordinator_tools"
    SUB_TOOLS = "sub_tools"
    SYNTHESIZE = "synthesize"
    TOOLS = "tools"
    END = "end"


class PlanStatus(str, Enum):
    """Trạng thái vòng đời của một plan.

    Ánh xạ với trường plan_status trong AutoMLState.
    """

    READY = "ready"
    EXECUTING = "executing"
    NEED_REVISE = "need_revise"
    DONE = "done"
    FAILED = "failed"
    ABORTED = "aborted"


class SurpriseLevel(str, Enum):
    """Mức độ surprise từ World Model."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AgentPhase(str, Enum):
    """Giai đoạn trong vòng đời agent."""

    IDLE = "idle"
    ANALYZE = "analyze"
    SELECT = "select"
    TRAIN = "train"
    EVALUATE = "evaluate"
    RESPOND = "respond"


# ── Cấu trúc dữ liệu dùng chung ──────────────────────────────────────────────

from pydantic import BaseModel, Field, field_validator


class PlanAction(BaseModel):
    """Định nghĩa action trong một bước của plan."""

    type: str
    params: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "1.0"

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("action type cannot be empty")
        return v.strip()


class PlanStep(BaseModel):
    """Một bước có kiểu trong plan thực thi."""

    action: PlanAction | dict[str, Any]
    agent: str | None = None
    step_id: str | None = None
    description: str | None = None
    schema_version: str = "1.0"

    @field_validator("action")
    @classmethod
    def validate_action(
        cls, v: PlanAction | dict[str, Any]
    ) -> PlanAction | dict[str, Any]:
        if isinstance(v, dict) and not v.get("type"):
            raise ValueError("Plan step action must contain a non-empty 'type'")
        return v

    def get_action_type(self) -> str:
        if isinstance(self.action, PlanAction):
            return self.action.type
        if isinstance(self.action, dict):
            return str(self.action.get("type", ""))
        return ""

    def get_action_params(self) -> dict[str, Any]:
        if isinstance(self.action, PlanAction):
            return dict(self.action.params)
        if isinstance(self.action, dict):
            return dict(self.action.get("params") or {})
        return {}


class Plan(BaseModel):
    """Đối tượng plan thực thi có kiểu."""

    plan_id: str
    steps: list[PlanStep] = Field(default_factory=list)
    title: str = ""
    cost: float = 0.0
    score_estimate: float | None = None
    status: PlanStatus = PlanStatus.READY
    meta: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "1.0"

    @field_validator("plan_id")
    @classmethod
    def validate_plan_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("plan_id cannot be empty")
        return v.strip()

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# ── Trạng thái agent graph ───────────────────────────────────────────────────


def merge_agent_messages(left: list[Any] | None, right: Any) -> list[Any]:
    """Giữ đúng reducer message của LangGraph mà không buộc core import sớm."""
    try:
        from langgraph.graph.message import add_messages
    except ImportError:  # pragma: no cover - môi trường chỉ kiểm tra contract
        current = list(left or [])
        if right is None:
            return current
        return current + (right if isinstance(right, list) else [right])
    return list(add_messages(left or [], right))


class AgentState(TypedDict, total=False):
    """
    TypedDict chuẩn cho LangGraph StateGraph của HAgent.

    Đây là kiểu trung tâm dùng trong chú thích kiểu của:
      - các hàm định tuyến như coordinator_route và subagent_route;
      - các vị từ hỗ trợ như _should_run_hierarchy và _should_run_campaign;
      - các hàm node như synthesizer_node.

    Lý do tách khỏi AutoMLState (orchestration/state.py):
      - AutoMLState dùng ``Annotated[list, add_messages]`` dành riêng cho LangGraph;
        AgentState dùng ``list`` thuần để core không phụ thuộc LangGraph.
      - Module core/types.py là lớp thấp nhất nên không được import hagent.agent.
      - Graph runtime vẫn dùng AutoMLState cho StateGraph để giữ bộ gộp message.

    Tương thích ngược:
      - ``AutoMLState`` ⊆ ``AgentState`` về mặt khóa; mọi trường của AgentState
        đều có mặt trong AutoMLState; LangGraph truyền state như dict nên tương thích.
    """

    # ── Core ──────────────────────────────────────────────
    messages: Annotated[list[Any], merge_agent_messages]
    """LangGraph message list. AutoMLState override bằng Annotated[list, add_messages]."""

    # ── Định tuyến và điều phối ──────────────────────────
    next_agent: NotRequired[str | None]
    current_phase: NotRequired[str | None]

    # ── Mục tiêu và lập kế hoạch ─────────────────────────
    goal: NotRequired[dict[str, Any] | None]
    plans: NotRequired[list | None]
    selected_plan: NotRequired[dict[str, Any] | None]
    plan_status: NotRequired[str | None]
    plan_step_index: NotRequired[int | None]
    plan_verification: NotRequired[dict[str, Any] | None]
    revision_count: NotRequired[int | None]
    last_step_error: NotRequired[str | None]
    execution_log: NotRequired[list | None]
    execution_events: NotRequired[list | None]

    # ── World Model và latent ────────────────────────────
    world_model: NotRequired[dict[str, Any] | None]
    latent: NotRequired[dict[str, Any] | None]
    surprise: NotRequired[dict[str, Any] | None]
    cost_metrics: NotRequired[dict[str, Any] | None]

    # ── Campaign, giai đoạn 6 ────────────────────────────
    campaign: NotRequired[dict[str, Any] | None]
    campaign_status: NotRequired[str | None]
    campaign_tick: NotRequired[int | None]

    # ── Hierarchy, giai đoạn 7 ───────────────────────────
    hierarchy: NotRequired[dict[str, Any] | None]
    hierarchy_status: NotRequired[str | None]
    hierarchy_train_active: NotRequired[bool | None]
    _hierarchy_train_active: NotRequired[bool | None]

    # ── Context middleware chỉ dùng lúc chạy ─────────────
    # Các khóa này bị loại trước khi state đi vào graph nhưng cần có kiểu tại boundary.
    _world_store: NotRequired[Any | None]
    _wm_service: NotRequired[Any | None]
    user_token: NotRequired[str | None]

    # ── Context AutoML ───────────────────────────────────
    dataset_context: NotRequired[dict[str, Any] | None]
    job_context: NotRequired[dict[str, Any] | None]
    active_jobs: NotRequired[list | None]
    evaluation: NotRequired[dict[str, Any] | None]
    user_requirements: NotRequired[dict[str, Any] | None]

    # ── Bộ nhớ và người dùng ─────────────────────────────
    memory_context: NotRequired[str | None]
    user_id: NotRequired[str | None]


# ── Response có kiểu của công cụ (REFAC-019) ─────────────────────────────────


class ToolResponse(BaseModel):
    """Response cơ sở có kiểu cho mọi công cụ HAgent."""

    success: bool = True
    data: Any = None
    error: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
