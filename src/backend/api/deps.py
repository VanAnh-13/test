"""
Các dependency dùng chung cho tầng API (REFAC-025, REFAC-026).
"""

# ruff: noqa: B008

from __future__ import annotations

from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import Depends, HTTPException, Request, status
from pymongo.asynchronous.database import AsyncDatabase

from config.providers import (
    get_app_settings,
    get_db,
    get_kafka_producer,
    get_minio_client,
    get_mongo_client,
)
from users.engine import check_exits_username
from users.utils.authentication import jwt_service

# AUDIT-002: chỉ token có type này mới được dùng để xác thực Bearer request.
# Ngăn refresh/verification/password_reset token bị tái sử dụng làm access token.
_BEARER_ALLOWED_TOKEN_TYPE = "access"


async def _resolve_user_by_id(user_id: str, db: AsyncDatabase) -> dict[str, Any] | None:
    """Tra cứu người dùng theo `_id` (ObjectId).

    AUDIT-002 FIX: `sub` trong JWT do users/routers.py phát hành (login, refresh,
    verification, password-reset) LUÔN LÀ `str(user['_id'])`, không phải username.
    Tra theo username (như trước đây) khiến mọi Bearer token hợp lệ đều bị từ chối.
    """
    try:
        object_id = ObjectId(user_id)
    except (InvalidId, TypeError):
        return None
    return await db.tbl_User.find_one({"_id": object_id}, {"password": 0})


async def _extract_user_from_bearer_token(
    request: Request, db: AsyncDatabase
) -> dict[str, Any] | None:
    """Xác thực JWT Bearer token và trả về document người dùng tương ứng.

    Trả `None` nếu không có header Bearer, token không hợp lệ/hết hạn, sai `type`,
    hoặc người dùng không còn tồn tại.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None

    token = auth_header[len("Bearer ") :]
    payload = jwt_service.verify_token(token)
    if not payload:
        return None
    # AUDIT-002 FIX (JWT type confusion): từ chối token không phải loại 'access'.
    if payload.get("type") != _BEARER_ALLOWED_TOKEN_TYPE:
        return None

    user_id = payload.get("sub")
    if not user_id:
        return None
    return await _resolve_user_by_id(user_id, db)


async def get_current_user(
    request: Request,
    db: AsyncDatabase = Depends(get_db),
) -> dict[str, Any]:
    """Dependency trích xuất và xác thực người dùng hiện tại.

    Hỗ trợ cả JWT Bearer token (Authorization header, ưu tiên, tra theo `_id`)
    và session cookie (legacy, tra theo username).
    """
    user = await _extract_user_from_bearer_token(request, db)
    if user is not None:
        return user

    # Dự phòng: session cookie (legacy, hiện không có nơi nào set `session['user']`).
    username = request.session.get("user")
    if username:
        session_user = await check_exits_username(username, db)
        if session_user:
            return session_user

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Yêu cầu xác thực tài khoản hợp lệ",
        headers={"WWW-Authenticate": "Bearer"},
    )


__all__ = [
    "get_app_settings",
    "get_current_user",
    "get_db",
    "get_kafka_producer",
    "get_minio_client",
    "get_mongo_client",
]
