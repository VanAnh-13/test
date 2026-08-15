"""
Kiểm thử đơn vị cho api/deps.py::get_current_user (AUDIT-002).

Bối cảnh lỗi: users/routers.py phát hành mọi JWT (access, refresh,
verification, password_reset) với `sub = str(user['_id'])`. Trước AUDIT-002,
api/deps.py lại tra cứu `sub` như một USERNAME (check_exits_username), khiến
MỌI Bearer token hợp lệ tới toàn bộ tầng api/v1/* đều bị từ chối 401.
AUDIT-002 sửa lại để tra theo `_id` và chỉ chấp nhận token type == 'access'.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

os.environ.setdefault("MINIO_ENDPOINT", "localhost:9000")
os.environ.setdefault("MINIO_ACCESS_KEY", "minioadmin")
os.environ.setdefault("MINIO_SECRET_KEY", "minioadmin")
os.environ.setdefault("DATABASE_URI", "mongodb://localhost:27017")
os.environ.setdefault("SECRET_KEY", "testsecretkey1234567890123456789012")

import pytest
from bson import ObjectId
from fastapi import HTTPException

from api.deps import get_current_user
from users.utils.authentication import jwt_service

USER_OBJECT_ID = ObjectId("64b64b64b64b64b64b64b640")
USER_DOC = {
    "_id": USER_OBJECT_ID,
    "username": "alice",
    "email": "alice@example.com",
    "role": "user",
}


def _make_request(*, authorization: str | None = None, session_user: str | None = None):
    """Tạo mock Request tối thiểu cho get_current_user."""
    request = MagicMock()
    headers = {}
    if authorization is not None:
        headers["Authorization"] = authorization
    request.headers.get = lambda key, default="": headers.get(key, default)
    request.session = {"user": session_user} if session_user else {}
    return request


def _make_db(*, find_one_result=None):
    db = MagicMock()
    db.tbl_User = MagicMock()
    db.tbl_User.find_one = AsyncMock(return_value=find_one_result)
    return db


async def test_get_current_user_accepts_valid_access_token_resolved_by_id() -> None:
    """Access token với sub=_id hợp lệ phải tra đúng người dùng theo ObjectId."""
    token = jwt_service.create_access_token(
        {"sub": str(USER_OBJECT_ID), "role": "user"}
    )
    request = _make_request(authorization=f"Bearer {token}")
    db = _make_db(find_one_result=USER_DOC)

    user = await get_current_user(request, db)

    assert user == USER_DOC
    db.tbl_User.find_one.assert_awaited_once()
    called_filter, called_projection = db.tbl_User.find_one.await_args.args
    assert called_filter == {"_id": USER_OBJECT_ID}
    assert called_projection == {"password": 0}


@pytest.mark.parametrize(
    "token_factory",
    [
        lambda: jwt_service.create_refresh_token({"sub": str(USER_OBJECT_ID)}),
        lambda: jwt_service.create_verification_token(
            {"sub": str(USER_OBJECT_ID), "email": "alice@example.com"}
        ),
        lambda: jwt_service.create_password_reset_token(
            {"sub": str(USER_OBJECT_ID), "email": "alice@example.com"}
        ),
    ],
)
async def test_get_current_user_rejects_non_access_token_types(token_factory) -> None:
    """JWT type confusion: refresh/verification/password_reset không được dùng làm access token."""
    token = token_factory()
    request = _make_request(authorization=f"Bearer {token}")
    db = _make_db(find_one_result=USER_DOC)

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(request, db)

    assert exc_info.value.status_code == 401
    db.tbl_User.find_one.assert_not_awaited()


async def test_get_current_user_rejects_invalid_token() -> None:
    """Token không giải mã được (sai chữ ký/hỏng định dạng) phải bị từ chối."""
    request = _make_request(authorization="Bearer not-a-valid-jwt")
    db = _make_db(find_one_result=USER_DOC)

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(request, db)

    assert exc_info.value.status_code == 401


async def test_get_current_user_rejects_when_user_no_longer_exists() -> None:
    """Token hợp lệ nhưng user đã bị xoá khỏi DB thì vẫn phải từ chối."""
    token = jwt_service.create_access_token({"sub": str(USER_OBJECT_ID)})
    request = _make_request(authorization=f"Bearer {token}")
    db = _make_db(find_one_result=None)

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(request, db)

    assert exc_info.value.status_code == 401


async def test_get_current_user_rejects_when_no_credentials() -> None:
    """Không có Authorization header và không có session hợp lệ → 401."""
    request = _make_request()
    db = _make_db(find_one_result=None)

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(request, db)

    assert exc_info.value.status_code == 401


async def test_get_current_user_session_fallback_uses_username_lookup() -> None:
    """Dự phòng session cookie (legacy) vẫn tra theo username, không theo _id."""
    request = _make_request(session_user="alice")
    db = _make_db(find_one_result=USER_DOC)

    user = await get_current_user(request, db)

    assert user == USER_DOC
    db.tbl_User.find_one.assert_awaited_once()
    called_filter = db.tbl_User.find_one.await_args.args[0]
    assert called_filter == {"username": "alice"}
