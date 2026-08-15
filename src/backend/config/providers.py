"""
Provider Dependency Injection cho FastAPI (REFAC-026).
"""

# ruff: noqa: BLE001, S110

from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException, Request, status
from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase

from config.settings import AppSettings, get_settings

logger = logging.getLogger(__name__)


def get_app_settings() -> AppSettings:
    """Cung cấp cấu hình ứng dụng AppSettings."""
    return get_settings()


def get_mongo_client(request: Request) -> AsyncMongoClient:
    """Cung cấp kết nối MongoDB AsyncMongoClient từ trạng thái ứng dụng (app.state)."""
    client = getattr(request.app.state, "client", None)
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Client kết nối MongoDB chưa được khởi tạo hoặc không khả dụng.",
        )
    return client


def get_db(request: Request) -> AsyncDatabase:
    """Cung cấp cơ sở dữ liệu MongoDB AsyncDatabase từ trạng thái ứng dụng (app.state)."""
    db = getattr(request.app.state, "db", None)
    if db is not None:
        return db

    # Dự phòng sang client[database_name] nếu client tồn tại nhưng db chưa được gán trực tiếp
    client = getattr(request.app.state, "client", None)
    if client is not None:
        settings = get_settings()
        return client[settings.database_name]

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Kết nối cơ sở dữ liệu chưa được khởi tạo hoặc không khả dụng.",
    )


def get_minio_client(request: Request) -> Any:
    """Cung cấp client lưu trữ MinIO từ app.state hoặc singleton của module."""
    minio = getattr(request.app.state, "minio", None)
    if minio is not None:
        return minio

    # Dự phòng sang module singleton minIOStorage nếu khả dụng.
    # Bắt rộng có chủ đích: đây là fallback tùy chọn, MinIO có thể chưa cấu hình.
    try:
        from automl.v2.minio import minIOStorage

        return minIOStorage
    except Exception:
        logger.debug("MinIO singleton fallback không khả dụng", exc_info=True)

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Dịch vụ lưu trữ MinIO chưa được cấu hình hoặc không khả dụng.",
    )


def get_kafka_producer(request: Request) -> Any:
    """Cung cấp Kafka Producer client từ app.state hoặc singleton của module."""
    producer = getattr(
        request.app.state,
        "kafka_producer",
        getattr(request.app.state, "producer", None),
    )
    if producer is not None:
        return producer

    # P2-FIX: Xóa phantom import 'database.database.producer' không tồn tại.
    # Kafka producer được khởi tạo trong app.lifespan và lưu vào app.state.
    # Nếu không có trong state, trả None (endpoint sẽ tự xử lý trường hợp này).
    return None
