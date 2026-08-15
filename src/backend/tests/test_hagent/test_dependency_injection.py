"""
Kiểm thử đơn vị cho Các Provider Dependency Injection (REFAC-026).
"""

# ruff: noqa: B008

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

os.environ.setdefault("MINIO_ENDPOINT", "localhost:9000")
os.environ.setdefault("MINIO_ACCESS_KEY", "minioadmin")
os.environ.setdefault("MINIO_SECRET_KEY", "minioadmin")
os.environ.setdefault("DATABASE_URI", "mongodb://localhost:27017")
os.environ.setdefault("SECRET_KEY", "testsecretkey1234567890123456789012")

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from config.providers import (
    get_app_settings,
    get_db,
    get_kafka_producer,
    get_minio_client,
    get_mongo_client,
)
from config.settings import AppSettings


def test_get_app_settings() -> None:
    """Provider get_app_settings trả về đối tượng AppSettings hợp lệ."""
    settings = get_app_settings()
    assert isinstance(settings, AppSettings)
    assert settings.database_name == "AutoML"
    assert settings.port == 8080


def test_get_mongo_client_from_app_state() -> None:
    """get_mongo_client lấy AsyncMongoClient từ application state hoặc ném 503 khi chưa khởi tạo."""
    mock_request = MagicMock()
    mock_client = MagicMock()
    mock_request.app.state.client = mock_client

    assert get_mongo_client(mock_request) is mock_client

    mock_request_none = MagicMock()
    mock_request_none.app.state.client = None
    with pytest.raises(HTTPException) as exc_info:
        get_mongo_client(mock_request_none)
    assert exc_info.value.status_code == 503


def test_get_db_from_app_state() -> None:
    """get_db lấy AsyncDatabase từ state.db hoặc state.client[dbname] hoặc ném 503."""
    mock_request = MagicMock()
    mock_db = MagicMock()
    mock_request.app.state.db = mock_db

    assert get_db(mock_request) is mock_db

    # Test fallback to client
    mock_request_fallback = MagicMock()
    mock_request_fallback.app.state.db = None
    mock_client = MagicMock()
    mock_request_fallback.app.state.client = mock_client
    db_instance = get_db(mock_request_fallback)
    assert db_instance is not None

    # Test uninitialized raises 503
    mock_request_empty = MagicMock()
    mock_request_empty.app.state.db = None
    mock_request_empty.app.state.client = None
    with pytest.raises(HTTPException) as exc_info:
        get_db(mock_request_empty)
    assert exc_info.value.status_code == 503


def test_get_minio_and_kafka_providers() -> None:
    """get_minio_client và get_kafka_producer truy xuất state hoặc fallback an toàn."""
    mock_request = MagicMock()
    mock_minio = MagicMock()
    mock_request.app.state.minio = mock_minio
    assert get_minio_client(mock_request) is mock_minio

    mock_kafka = MagicMock()
    mock_request.app.state.kafka_producer = mock_kafka
    assert get_kafka_producer(mock_request) is mock_kafka


def test_fastapi_dependency_overrides_with_test_client() -> None:
    """FastAPI app.dependency_overrides hoạt động liền mạch với DI container."""
    test_app = FastAPI()

    @test_app.get("/test-di-endpoint")
    async def sample_endpoint(db: MagicMock = Depends(get_db)):
        # Simulate query using injected db
        doc = await db.find_one({"key": "test"})
        return {"data": doc}

    mock_test_db = AsyncMock()
    mock_test_db.find_one.return_value = {"key": "test", "val": 42}

    test_app.dependency_overrides[get_db] = lambda: mock_test_db

    with TestClient(test_app) as client:
        resp = client.get("/test-di-endpoint")
        assert resp.status_code == 200
        assert resp.json() == {"data": {"key": "test", "val": 42}}

    test_app.dependency_overrides.clear()
