"""
hagent.core chứa các hợp đồng và kiểu dùng chung cho toàn bộ hệ thống HAgent.

Module này là lớp thấp nhất trong đồ thị phụ thuộc của HAgent.
Không import bất kỳ module nội bộ nào (hagent.world, hagent.agent, v.v.)

API công khai:
    Kiểu:     RouteType, PlanStatus, SurpriseLevel, AgentPhase
              AgentState, PlanStep
    Lỗi:      HAgentError, PlanningError, ExecutionError,
              WorldModelError, LLMError, ToolError
    Hợp đồng: AgentMessage, MessageType
    Sự kiện:  HAgentEvent, PlanCreated, PlanRevised,
              StepExecuted, SurpriseDetected, WorldModelUpdated,
              CampaignStarted, CampaignCompleted
"""

from hagent.core.errors import (
    ExecutionError,
    HAgentError,
    LLMError,
    PlanningError,
    ToolError,
    WorldModelError,
)
from hagent.core.events import (
    CampaignCompleted,
    CampaignStarted,
    HAgentEvent,
    PlanCreated,
    PlanRevised,
    StepExecuted,
    SurpriseDetected,
    WorldModelUpdated,
)
from hagent.core.protocols import (
    AgentMessage,
    MessageType,
)
from hagent.core.types import (
    AgentPhase,
    AgentState,
    PlanStatus,
    PlanStep,
    RouteType,
    SurpriseLevel,
)

__all__ = [
    "AgentMessage",
    "AgentPhase",
    "AgentState",
    "CampaignCompleted",
    "CampaignStarted",
    "ExecutionError",
    "HAgentError",
    "HAgentEvent",
    "LLMError",
    "MessageType",
    "PlanCreated",
    "PlanRevised",
    "PlanStatus",
    "PlanStep",
    "PlanningError",
    "RouteType",
    "StepExecuted",
    "SurpriseDetected",
    "SurpriseLevel",
    "ToolError",
    "WorldModelError",
    "WorldModelUpdated",
]
