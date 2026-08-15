"""Kiểm tra vòng đời adapter Kafka trong package infrastructure."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from infrastructure.messaging import kafka


@pytest.fixture(autouse=True)
def _reset_producer_instance():
    previous = kafka.producer_instance
    kafka.producer_instance = None
    yield
    kafka.producer_instance = previous


def test_kafka_module_khong_con_nam_o_root_backend():
    backend_root = Path(__file__).parent.parent
    assert not (backend_root / "kafka_consumer.py").exists()


def test_get_producer_fail_closed_khi_chua_khoi_dong():
    with pytest.raises(RuntimeError, match="chưa được khởi động"):
        kafka.get_producer()


@pytest.mark.asyncio
async def test_start_va_stop_producer_quan_ly_dung_singleton(monkeypatch):
    producer = AsyncMock()
    producer_factory = Mock(return_value=producer)
    ensure_topic = AsyncMock()
    monkeypatch.setattr(kafka, "AIOKafkaProducer", producer_factory)
    monkeypatch.setattr(kafka, "_ensure_topic", ensure_topic)
    monkeypatch.setenv("KAFKA_SERVER", "kafka.test:9092")
    monkeypatch.setenv("KAFKA_TOPIC", "training-test")

    await kafka.start_producer()

    assert kafka.get_producer() is producer
    ensure_topic.assert_awaited_once_with("kafka.test:9092", "training-test")
    producer.start.assert_awaited_once()

    await kafka.stop_producer()

    producer.stop.assert_awaited_once()
    with pytest.raises(RuntimeError, match="chưa được khởi động"):
        kafka.get_producer()
