"""Helper kết nối MongoDB dùng chung cho các store sync của HAgent.

Module này là lớp thấp nhất (chỉ phụ thuộc pymongo), không import bất kỳ
module nội bộ nào của hagent để tránh phụ thuộc vòng.
"""

from __future__ import annotations

from pymongo import MongoClient
from pymongo.errors import PyMongoError


def connect_mongo_client(
    mongodb_uri: str,
    *,
    error_type: type[Exception],
    unavailable_message: str,
    server_selection_timeout_ms: int = 2000,
) -> MongoClient:
    """Tạo MongoClient đã ping, đóng client khi lỗi và gói lỗi vào error_type.

    Args:
        mongodb_uri: URI kết nối MongoDB (bắt buộc, không rỗng).
        error_type: Exception domain của store gọi (vd: RuntimeLedgerUnavailable).
        unavailable_message: Thông điệp khi không kết nối được.
        server_selection_timeout_ms: Timeout chọn server (phải >= 1).

    Raises:
        error_type: URI rỗng hoặc không kết nối/ping được.
        ValueError: server_selection_timeout_ms < 1.
    """
    if not isinstance(mongodb_uri, str) or not mongodb_uri.strip():
        raise error_type("MongoDB URI is required")
    if server_selection_timeout_ms < 1:
        raise ValueError("server_selection_timeout_ms must be positive")
    client: MongoClient | None = None
    try:
        client = MongoClient(
            mongodb_uri,
            serverSelectionTimeoutMS=server_selection_timeout_ms,
            tz_aware=True,
        )
        client.admin.command("ping")
        return client
    except (PyMongoError, TypeError, ValueError):
        if client is not None:
            client.close()
        raise error_type(unavailable_message) from None
