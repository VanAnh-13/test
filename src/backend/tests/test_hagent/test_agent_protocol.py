"""
Unit tests for Inter-Agent Protocol (REFAC-021).
"""

from __future__ import annotations

import logging

import pytest

from hagent.agent.subagents.manager import SubagentManager
from hagent.agent.subagents.protocol import (
    MessageType,
    create_error,
    create_event,
    create_request,
    create_response,
    deserialize_message,
    serialize_message,
)


def test_agent_message_roundtrip() -> None:
    """AgentMessage serialize và deserialize bảo toàn toàn bộ dữ liệu."""
    msg = create_request(
        sender="data_analyst",
        recipient="model_selector",
        payload={
            "dataset_id": "ds_iris",
            "features": ["f1", "f2"],
            "problem_type": "classification",
        },
        correlation_id="corr_12345",
        meta={"priority": "high"},
    )

    encoded = serialize_message(msg)
    decoded = deserialize_message(encoded)

    assert decoded.id == msg.id
    assert decoded.sender == "data_analyst"
    assert decoded.recipient == "model_selector"
    assert decoded.type == MessageType.REQUEST
    assert decoded.payload["dataset_id"] == "ds_iris"
    assert decoded.correlation_id == "corr_12345"
    assert decoded.meta["priority"] == "high"
    assert decoded.version == "1.0"


def test_protocol_version_mismatch_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Khi nhận message có version khác PROTOCOL_VERSION, hệ thống ghi log warning."""
    with caplog.at_level(logging.WARNING):
        raw_dict = {
            "id": "msg_future_999",
            "version": "2.5",  # Future protocol version
            "sender": "external_agent",
            "recipient": "coordinator",
            "type": "event",
            "payload": {"status": "ready"},
        }
        msg = deserialize_message(raw_dict)
        assert msg.version == "2.5"
        assert any(
            "protocol version mismatch" in record.message.lower()
            for record in caplog.records
        )


def test_protocol_message_factories() -> None:
    """Các hàm helper create_request, create_response, create_event, create_error hoạt động chính xác."""
    req = create_request(
        sender="agent_a",
        recipient="agent_b",
        payload={"query": "analyze"},
        correlation_id="req_001",
    )
    assert req.type == MessageType.REQUEST

    resp = create_response(
        request_message=req,
        sender="agent_b",
        payload={"result": "ok"},
    )
    assert resp.type == MessageType.RESPONSE
    assert resp.recipient == "agent_a"
    assert resp.correlation_id == "req_001"

    evt = create_event(
        sender="training_monitor",
        payload={"job_id": "job_1", "progress": 0.8},
    )
    assert evt.type == MessageType.EVENT
    assert evt.recipient == "broadcast"

    err = create_error(
        sender="model_selector",
        recipient="coordinator",
        error_message="Dataset not found",
        correlation_id="req_001",
    )
    assert err.type == MessageType.ERROR
    assert err.payload["error"] == "Dataset not found"


def test_subagent_manager_routing() -> None:
    """SubagentManager định tuyến tin nhắn đúng đích và hỗ trợ broadcast."""
    manager = SubagentManager()
    manager.register_agent("analyst")
    manager.register_agent("selector")
    manager.register_agent("trainer")

    # 1. Direct message
    req = create_request(
        sender="analyst", recipient="selector", payload={"task": "select_models"}
    )
    manager.send_message(req)

    assert len(manager.receive_messages("analyst")) == 0
    assert len(manager.receive_messages("trainer")) == 0
    selector_msgs = manager.receive_messages("selector")
    assert len(selector_msgs) == 1
    assert selector_msgs[0].payload["task"] == "select_models"

    # 2. Broadcast message
    evt = create_event(sender="trainer", payload={"job_status": "completed"})
    manager.send_message(evt)

    # Broadcast gửi tới analyst và selector nhưng không gửi lại trainer
    assert len(manager.receive_messages("trainer")) == 0
    assert len(manager.receive_messages("analyst")) == 1
    assert len(manager.receive_messages("selector")) == 1
