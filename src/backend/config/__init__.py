"""
Gói cấu hình và quản lý dependency injection cho backend (REFAC-026).
"""

from __future__ import annotations

from config.providers import (
    get_app_settings,
    get_db,
    get_kafka_producer,
    get_minio_client,
    get_mongo_client,
)
from config.settings import AppSettings, get_settings

__all__ = [
    "AppSettings",
    "get_app_settings",
    "get_db",
    "get_kafka_producer",
    "get_minio_client",
    "get_mongo_client",
    "get_settings",
]
