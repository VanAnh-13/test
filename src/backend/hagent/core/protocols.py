"""
Message protocol có kiểu cho giao tiếp giữa các agent trong hệ thống HAgent.

Module này là lớp thấp nhất, không import bất kỳ module nội bộ nào của hagent.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, ClassVar

import uuid

import structlog
from pydantic import BaseModel, Field, field_validator

logger = structlog.get_logger("hagent.core.protocols")


# ── Protocol message giữa các agent ──────────────────────────────────────────


class MessageType(str, Enum):
    """Loại message trong giao tiếp giữa các agent."""

    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    ERROR = "error"


class AgentMessage(BaseModel):
    """Định dạng message có kiểu cho giao tiếp giữa các agent.

    Thay thế dict không kiểu trong subagents/protocol.py.
    Có phiên bản để hỗ trợ tương thích ngược và tương thích xuôi.

    Các trường:
        id: UUID của message.
        version: Phiên bản protocol, hiện là "1.0".
        sender: ID của agent gửi.
        recipient: ID của agent nhận hoặc "broadcast".
        type: REQUEST | RESPONSE | EVENT | ERROR.
        payload: Dữ liệu message theo domain.
        timestamp: Dấu thời gian UTC theo ISO 8601.
        correlation_id: ID liên kết request với response.
        meta: Metadata bổ sung.
    """

    PROTOCOL_VERSION: ClassVar[str] = "1.0"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version: str = "1.0"
    sender: str
    recipient: str
    type: MessageType
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    correlation_id: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        if v != cls.PROTOCOL_VERSION:
            logger.warning(
                "AgentMessage protocol version mismatch: expected %s, got %s",
                cls.PROTOCOL_VERSION,
                v,
            )
        return v

    def to_dict(self) -> dict[str, Any]:
        """Xuất dict theo định dạng tương thích ngược."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentMessage:
        """Nhập dict theo định dạng tương thích ngược."""
        return cls.model_validate(data)
