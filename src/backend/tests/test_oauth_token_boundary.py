from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from urllib.parse import parse_qs, urlsplit

import pytest
from bson import ObjectId
from fastapi import HTTPException, Response
from fastapi.security import HTTPAuthorizationCredentials
from jwt import PyJWTError

from hagent.bridge import auth as bridge_auth
from users import routers, schema


@pytest.mark.asyncio
async def test_google_callback_chi_dua_code_mot_lan_vao_url(monkeypatch):
    user_id = ObjectId()
    users = SimpleNamespace(
        find_one=AsyncMock(
            return_value={
                "_id": user_id,
                "email": "oauth@example.com",
                "role": "user",
                "is_verified": True,
            }
        ),
        update_one=AsyncMock(),
    )
    linked_accounts = SimpleNamespace(
        find_one=AsyncMock(return_value={"provider": "google"}),
        insert_one=AsyncMock(),
    )
    login_codes = SimpleNamespace(insert_one=AsyncMock())
    db = SimpleNamespace(
        tbl_User=users,
        linked_accounts=linked_accounts,
        oauth_login_codes=login_codes,
    )
    google = SimpleNamespace(
        authorize_access_token=AsyncMock(
            return_value={
                "userinfo": {
                    "email": "oauth@example.com",
                    "sub": "google-user-id",
                    "name": "OAuth User",
                    "picture": "https://example.test/avatar.png",
                }
            }
        )
    )
    monkeypatch.setattr(routers.oauth, "google", google, raising=False)
    monkeypatch.setenv("FRONTEND_URL", "https://frontend.example.test")
    create_access_token = Mock(return_value="access-token-khong-duoc-vao-url")
    create_refresh_token = Mock(return_value="refresh-token-khong-duoc-vao-url")
    monkeypatch.setattr(routers.jwt_service, "create_access_token", create_access_token)
    monkeypatch.setattr(
        routers.jwt_service, "create_refresh_token", create_refresh_token
    )

    result = await routers.google_callback(Mock(), Response(), db)

    location = result.headers["location"]
    query = parse_qs(urlsplit(location).query)
    assert set(query) == {"code"}
    assert "access-token-khong-duoc-vao-url" not in location
    assert "refresh-token-khong-duoc-vao-url" not in location
    stored = login_codes.insert_one.await_args.args[0]
    assert stored["code_hash"] != query["code"][0]
    assert stored["user_id"] == user_id
    create_access_token.assert_not_called()
    create_refresh_token.assert_not_called()


@pytest.mark.asyncio
async def test_google_callback_fail_closed_khi_thieu_frontend_url(monkeypatch):
    authorize_access_token = AsyncMock()
    monkeypatch.setattr(
        routers.oauth,
        "google",
        SimpleNamespace(authorize_access_token=authorize_access_token),
        raising=False,
    )
    monkeypatch.delenv("FRONTEND_URL", raising=False)

    with pytest.raises(HTTPException) as error:
        await routers.google_callback(Mock(), Response(), SimpleNamespace())

    assert error.value.status_code == 500
    authorize_access_token.assert_not_awaited()


@pytest.mark.asyncio
async def test_oauth_code_chi_trao_doi_duoc_mot_lan(monkeypatch):
    user_id = ObjectId()
    login_codes = SimpleNamespace(
        find_one_and_delete=AsyncMock(side_effect=[{"user_id": user_id}, None])
    )
    users = SimpleNamespace(
        find_one=AsyncMock(
            return_value={
                "_id": user_id,
                "email": "oauth@example.com",
                "role": "user",
            }
        )
    )
    db = SimpleNamespace(tbl_User=users, oauth_login_codes=login_codes)
    monkeypatch.setattr(
        routers.jwt_service, "create_access_token", Mock(return_value="access")
    )
    monkeypatch.setattr(
        routers.jwt_service, "create_refresh_token", Mock(return_value="refresh")
    )
    request = schema.OAuthCodeExchangeRequest(
        code="opaque-code-du-32-ky-tu-an-toan-001"
    )

    token = await routers.exchange_oauth_code(request, db)

    assert token.access_token == "access"
    assert token.refresh_token == "refresh"
    with pytest.raises(HTTPException) as replay_error:
        await routers.exchange_oauth_code(request, db)
    assert replay_error.value.status_code == 400
    assert login_codes.find_one_and_delete.await_count == 2


def test_jwt_loi_khong_phan_chieu_chi_tiet_hoac_token(monkeypatch, caplog):
    token = "jwt-bi-mat-khong-duoc-ghi-log"
    monkeypatch.setattr(
        bridge_auth,
        "get_auth_config",
        lambda: {"secret_key": "secret", "algorithm": "HS256"},
    )
    monkeypatch.setattr(
        bridge_auth,
        "decode",
        Mock(side_effect=PyJWTError("chi-tiet-thu-vien-nhay-cam")),
    )

    with caplog.at_level(logging.DEBUG), pytest.raises(HTTPException) as error:
        bridge_auth.verify_jwt_token(token)

    assert error.value.status_code == 401
    assert error.value.detail == "Token không hợp lệ"
    assert token not in caplog.text
    assert "chi-tiet-thu-vien-nhay-cam" not in caplog.text


@pytest.mark.asyncio
async def test_dependency_bridge_khong_log_authorization_header(monkeypatch, caplog):
    token = "authorization-khong-duoc-ghi-log"
    expected = SimpleNamespace(user_id="user-1")
    monkeypatch.setattr(bridge_auth, "verify_jwt_token", Mock(return_value=expected))
    request = SimpleNamespace(headers={"authorization": f"Bearer {token}"})
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    with caplog.at_level(logging.DEBUG):
        result = await bridge_auth.get_current_user(request, credentials)

    assert result is expected
    assert token not in caplog.text


def test_frontend_khong_nhan_hoac_phat_tan_refresh_token_cho_browser():
    frontend_root = Path(__file__).resolve().parents[2] / "frontend" / "src"
    nextauth_source = (
        frontend_root / "pages" / "api" / "auth" / "[...nextauth].ts"
    ).read_text(encoding="utf-8")
    google_source = (
        frontend_root / "app" / "(auth)" / "google" / "page.tsx"
    ).read_text(encoding="utf-8")
    login_source = (
        frontend_root / "app" / "(auth)" / "login" / "LoginForm.tsx"
    ).read_text(encoding="utf-8")

    assert "session.user.refresh_token" not in nextauth_source
    assert 'access_token: { label: "Access Token"' not in nextauth_source
    assert 'refresh_token: { label: "Refresh Token"' not in nextauth_source
    assert "console.log" not in nextauth_source
    assert (
        'authorization_code: { label: "Authorization Code", type: "hidden" }'
        in nextauth_source
    )
    assert 'searchParams?.get("code")' in google_source
    assert 'searchParams.get("access_token")' not in google_source
    assert google_source.index("replaceState") < google_source.index(
        'signIn("credentials"'
    )
    assert "data.access_token" not in login_source
