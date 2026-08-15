"""
Inter-Subagent Messaging Protocol (REFAC-021).

Tiện ích giao tiếp chuẩn hóa giữa các sub-agents dựa trên `AgentMessage` Pydantic model.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from hagent.core.protocols import AgentMessage, MessageType

logger = structlog.get_logger(__name__)


def create_request(
    sender: str,
    recipient: str,
    payload: dict[str, Any],
    *,
    correlation_id: str | None = None,
    meta: dict[str, Any] | None = None,
) -> AgentMessage:
    """Tạo một request message từ agent này tới agent khác."""
    return AgentMessage(
        sender=sender,
        recipient=recipient,
        type=MessageType.REQUEST,
        payload=payload,
        correlation_id=correlation_id,
        meta=dict(meta or {}),
    )


def create_response(
    request_message: AgentMessage,
    sender: str,
    payload: dict[str, Any],
    *,
    meta: dict[str, Any] | None = None,
) -> AgentMessage:
    """Tạo một response message để hồi đáp một request message đã nhận."""
    return AgentMessage(
        sender=sender,
        recipient=request_message.sender,
        type=MessageType.RESPONSE,
        payload=payload,
        correlation_id=request_message.correlation_id or request_message.id,
        meta=dict(meta or {}),
    )


def create_event(
    sender: str,
    recipient: str = "broadcast",
    payload: dict[str, Any] | None = None,
    *,
    meta: dict[str, Any] | None = None,
) -> AgentMessage:
    """Tạo một event notification broadcast hoặc gửi tới recipient cụ thể."""
    return AgentMessage(
        sender=sender,
        recipient=recipient,
        type=MessageType.EVENT,
        payload=dict(payload or {}),
        meta=dict(meta or {}),
    )


def create_error(
    sender: str,
    recipient: str,
    error_message: str,
    *,
    correlation_id: str | None = None,
    error_details: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> AgentMessage:
    """Tạo một error message thông báo sự cố trong quá trình xử lý."""
    return AgentMessage(
        sender=sender,
        recipient=recipient,
        type=MessageType.ERROR,
        payload={"error": error_message, "details": error_details or {}},
        correlation_id=correlation_id,
        meta=dict(meta or {}),
    )


def serialize_message(message: AgentMessage) -> str:
    """Serialize AgentMessage sang JSON string."""
    return message.model_dump_json()


def deserialize_message(data: str | dict[str, Any]) -> AgentMessage:
    """Deserialize JSON string hoặc dict sang AgentMessage với version validation."""
    if isinstance(data, str):
        raw = json.loads(data)
    else:
        raw = data
    return AgentMessage.model_validate(raw)


__all__ = [
    "AgentMessage",
    "MessageType",
    "create_error",
    "create_event",
    "create_request",
    "create_response",
    "deserialize_message",
    "serialize_message",
]
