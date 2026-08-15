"""
Kiểm thử đơn vị cho Kiến trúc API Phân tầng và Modular (REFAC-025).
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

os.environ.setdefault("MINIO_ENDPOINT", "localhost:9000")
os.environ.setdefault("MINIO_ACCESS_KEY", "minioadmin")
os.environ.setdefault("MINIO_SECRET_KEY", "minioadmin")
os.environ.setdefault("DATABASE_URI", "mongodb://localhost:27017")
os.environ.setdefault("SECRET_KEY", "testsecretkey1234567890123456789012")

from fastapi.testclient import TestClient

from server.application import app


def test_app_route_mounting() -> None:
    """Tất cả các route v1 từ các module phân tách được mount đầy đủ vào app chính."""
    routes = [route.path for route in app.routes if hasattr(route, "path")]

    # Core root endpoints
    assert "/" in routes
    assert "/home" in routes

    # Users endpoints
    assert "/users" in routes
    assert "/users/" in routes
    assert "/delete/{username}" in routes
    assert "/update/{username}" in routes
    assert "/change-password" in routes
    assert "/update_avatar" in routes
    assert "/get_avatar/{username}" in routes
    assert "/contact" in routes

    # Datasets endpoints
    assert "/get-list-data-by-userid" in routes
    assert "/get-data-info" in routes
    assert "/upload-dataset" in routes
    assert "/update-dataset/{dataset_id}" in routes
    assert "/delete-dataset/{dataset_id}" in routes
    assert "/get-data-from-uci" in routes

    # Training endpoints
    assert "/get-list-job-by-userId" in routes
    assert "/get-job-info" in routes
    assert "/training-file-local" in routes
    assert "/train-from-requestbody-json/" in routes

    # Models endpoints
    assert "/api/v1/available-models/{problem_type}" in routes

    # Admin endpoints
    assert "/get-list-data-user" in routes


def test_root_and_home_endpoints() -> None:
    """Root và home endpoints phản hồi thành công và trả về thông tin hợp lệ."""
    mock_db = AsyncMock()
    mock_client = AsyncMock()
    with (
        patch("server.application.connection", new_callable=AsyncMock) as mock_conn,
        patch("server.application.chat_store.ensure_indexes", new_callable=AsyncMock),
        patch("server.application.start_producer", new_callable=AsyncMock),
        patch("server.application.kafka_consumer_process", new_callable=AsyncMock),
        patch("server.application.monitor_tasks", new_callable=AsyncMock),
    ):
        mock_conn.return_value = (mock_db, mock_client)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp_root = client.get("/")
            assert resp_root.status_code == 200
            assert "HAutoML" in resp_root.json()

            resp_home = client.get("/home")
            assert resp_home.status_code == 200
            assert resp_home.json()["AutoML"] == "version 1.0"


def test_available_models_endpoint() -> None:
    """Endpoint /api/v1/available-models/{problem_type} trả về danh sách thuật toán hoặc 400 nếu sai loại."""
    mock_db = AsyncMock()
    mock_client = AsyncMock()
    with (
        patch("server.application.connection", new_callable=AsyncMock) as mock_conn,
        patch("server.application.chat_store.ensure_indexes", new_callable=AsyncMock),
        patch("server.application.start_producer", new_callable=AsyncMock),
        patch("server.application.kafka_consumer_process", new_callable=AsyncMock),
        patch("server.application.monitor_tasks", new_callable=AsyncMock),
    ):
        mock_conn.return_value = (mock_db, mock_client)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp_bad = client.get("/api/v1/available-models/invalid_problem")
            assert resp_bad.status_code == 400
            assert "Loại bài toán không hợp lệ" in resp_bad.json()["detail"]

            resp_cls = client.get("/api/v1/available-models/classification")
            if resp_cls.status_code == 200:
                data = resp_cls.json()
                assert data["problem_type"] == "classification"
                assert "models" in data
                assert isinstance(data["models"], list)
