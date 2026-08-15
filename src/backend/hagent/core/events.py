"""
Event types cho HAgent system.

Events là các immutable value objects mô tả điều gì đã xảy ra
trong hệ thống. Dùng cho:
  - Structured logging (thay vì free-text log messages)
  - SSE streaming tới frontend
  - Audit trail / trajectory recording
  - Inter-module communication (publish/subscribe)

Tất cả events đều là dataclasses frozen=True để đảm bảo immutability.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _new_event_id() -> str:
    return uuid.uuid4().hex[:12]


# ── Base event ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HAgentEvent:
    """Base class cho tất cả HAgent events.

    Không instantiate trực tiếp — dùng các subclass cụ thể.
    """

    event_id: str = field(default_factory=_new_event_id)
    timestamp: str = field(default_factory=_utc_now_iso)
    user_id: str | None = None
    conversation_id: str | None = None

    @property
    def event_type(self) -> str:
        return type(self).__name__

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
        }


# ── Planning events ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PlanCreated(HAgentEvent):
    """Phát ra khi agent tạo xong một plan mới.

    Dùng để log, stream tới frontend, và bắt đầu execution loop.
    """

    plan_id: str = ""
    title: str = ""
    n_steps: int = 0
    score_estimate: float | None = None
    cost_estimate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "plan_id": self.plan_id,
                "title": self.title,
                "n_steps": self.n_steps,
                "score_estimate": self.score_estimate,
                "cost_estimate": self.cost_estimate,
            }
        )
        return d


@dataclass(frozen=True)
class PlanRevised(HAgentEvent):
    """Phát ra khi agent phải replan do surprise quá cao."""

    plan_id: str = ""
    revision_count: int = 0
    trigger_surprise: float | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "plan_id": self.plan_id,
                "revision_count": self.revision_count,
                "trigger_surprise": self.trigger_surprise,
                "reason": self.reason,
            }
        )
        return d


# ── Execution events ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StepExecuted(HAgentEvent):
    """Phát ra sau mỗi bước execution trong plan."""

    plan_id: str = ""
    step_index: int = 0
    tool_name: str = ""
    success: bool = True
    elapsed_ms: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "plan_id": self.plan_id,
                "step_index": self.step_index,
                "tool_name": self.tool_name,
                "success": self.success,
                "elapsed_ms": self.elapsed_ms,
                "error": self.error,
            }
        )
        return d


# ── World Model events ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SurpriseDetected(HAgentEvent):
    """Phát ra khi World Model phát hiện surprise vượt threshold.

    Đây là event quan trọng nhất trong LeWM paradigm —
    triggers replanning và world model recalibration.
    """

    surprise_value: float = 0.0
    surprise_level: str = "low"  # low | medium | high
    epistemic: float | None = None
    aleatoric: float | None = None
    step_index: int | None = None
    plan_id: str | None = None
    triggered_replan: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "surprise_value": self.surprise_value,
                "surprise_level": self.surprise_level,
                "epistemic": self.epistemic,
                "aleatoric": self.aleatoric,
                "step_index": self.step_index,
                "plan_id": self.plan_id,
                "triggered_replan": self.triggered_replan,
            }
        )
        return d


@dataclass(frozen=True)
class WorldModelUpdated(HAgentEvent):
    """Phát ra sau khi World Model cập nhật beliefs."""

    n_trajectories: int = 0
    update_type: str = ""  # bayesian | batch | forced
    distribution_types: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "n_trajectories": self.n_trajectories,
                "update_type": self.update_type,
                "distribution_types": list(self.distribution_types),
            }
        )
        return d


# ── Campaign events ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CampaignStarted(HAgentEvent):
    """Phát ra khi bắt đầu một campaign mới."""

    campaign_id: str = ""
    n_candidates: int = 0
    budget: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "campaign_id": self.campaign_id,
                "n_candidates": self.n_candidates,
                "budget": self.budget,
            }
        )
        return d


@dataclass(frozen=True)
class CampaignCompleted(HAgentEvent):
    """Phát ra khi campaign hoàn thành (thành công hoặc early stopped)."""

    campaign_id: str = ""
    n_completed: int = 0
    best_score: float | None = None
    early_stopped: bool = False
    stop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "campaign_id": self.campaign_id,
                "n_completed": self.n_completed,
                "best_score": self.best_score,
                "early_stopped": self.early_stopped,
                "stop_reason": self.stop_reason,
            }
        )
        return d
