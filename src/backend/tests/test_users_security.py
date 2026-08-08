from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from bson import ObjectId
from fastapi import BackgroundTasks, HTTPException, Response
from pydantic import ValidationError

from users import engine, routers, schema
from users.schema import UserLoginRequest, UserRegisterRequest, VerifyOtp
from users.utils.authentication import JWTService
from users.utils.security import HashHelper


def _database(**collections):
    return SimpleNamespace(**collections)


@pytest.mark.asyncio
async def test_register_chi_luu_password_hash(monkeypatch):
    user_id = ObjectId()
    users = SimpleNamespace(
        find_one=AsyncMock(
            side_effect=[
                None,
                {
                    "_id": user_id,
                    "username": "secure-user",
                    "email": "secure@example.com",
                    "role": "user",
                    "is_verified": True,
                },
            ]
        ),
        insert_one=AsyncMock(return_value=SimpleNamespace(inserted_id=user_id)),
    )
    linked_accounts = SimpleNamespace(insert_one=AsyncMock())
    db = _database(tbl_User=users, linked_accounts=linked_accounts)
    monkeypatch.setattr(routers, "_is_skip_email_verification_enabled", lambda: True)

    await routers.register(
        UserRegisterRequest(
            username="secure-user",
            email="secure@example.com",
            gender="other",
            date="2000-01-01",
            number="0123456789",
            fullName="Secure User",
            password="correct-horse-battery-staple",
        ),
        BackgroundTasks(),
        db,
    )

    account = linked_accounts.insert_one.await_args.args[0]
    assert account["password"] != "correct-horse-battery-staple"
    assert HashHelper.verify_password(
        "correct-horse-battery-staple", account["password"]
    )


@pytest.mark.asyncio
async def test_login_nang_cap_password_plaintext_cu(monkeypatch):
    user_id = ObjectId()
    users = SimpleNamespace(
        find_one=AsyncMock(
            return_value={
                "_id": user_id,
                "username": "legacy-user",
                "email": "legacy@example.com",
                "role": "user",
                "is_verified": True,
            }
        )
    )
    linked_accounts = SimpleNamespace(
        find_one=AsyncMock(
            return_value={
                "_id": ObjectId(),
                "user_id": user_id,
                "provider": "local",
                "password": "legacy-password",
            }
        ),
        update_one=AsyncMock(),
    )
    db = _database(tbl_User=users, linked_accounts=linked_accounts)
    monkeypatch.setattr(routers.jwt_service, "create_access_token", lambda _: "access")
    monkeypatch.setattr(routers.jwt_service, "create_refresh_token", lambda _: "refresh")

    token = await routers.login(
        UserLoginRequest(username="legacy-user", password="legacy-password"),
        Response(),
        db,
    )

    assert token.access_token == "access"
    password_hash = linked_accounts.update_one.await_args.args[1]["$set"]["password"]
    assert password_hash != "legacy-password"
    assert HashHelper.verify_password("legacy-password", password_hash)


@pytest.mark.asyncio
async def test_login_nang_cap_ca_hai_ban_ghi_plaintext_cu(monkeypatch):
    user_id = ObjectId()
    users = SimpleNamespace(
        find_one=AsyncMock(
            return_value={
                "_id": user_id,
                "username": "dual-legacy-user",
                "email": "dual-legacy@example.com",
                "role": "user",
                "is_verified": True,
                "password": "legacy-password",
            }
        ),
        update_one=AsyncMock(),
    )
    linked_accounts = SimpleNamespace(update_one=AsyncMock())
    db = _database(tbl_User=users, linked_accounts=linked_accounts)
    monkeypatch.setattr(routers.jwt_service, "create_access_token", lambda _: "access")
    monkeypatch.setattr(routers.jwt_service, "create_refresh_token", lambda _: "refresh")

    await routers.login(
        UserLoginRequest(username="dual-legacy-user", password="legacy-password"),
        Response(),
        db,
    )

    user_hash = users.update_one.await_args.args[1]["$set"]["password"]
    linked_hash = linked_accounts.update_one.await_args.args[1]["$set"]["password"]
    assert user_hash == linked_hash
    assert HashHelper.verify_password("legacy-password", user_hash)


@pytest.mark.asyncio
async def test_login_don_ban_ghi_linked_plaintext_sau_migration_do(monkeypatch):
    user_id = ObjectId()
    canonical_hash = HashHelper.get_password_hash("legacy-password")
    users = SimpleNamespace(
        find_one=AsyncMock(
            return_value={
                "_id": user_id,
                "username": "partially-migrated-user",
                "email": "partially-migrated@example.com",
                "role": "user",
                "is_verified": True,
                "password": canonical_hash,
            }
        ),
        update_one=AsyncMock(),
    )
    linked_accounts = SimpleNamespace(update_one=AsyncMock())
    db = _database(tbl_User=users, linked_accounts=linked_accounts)
    monkeypatch.setattr(routers.jwt_service, "create_access_token", lambda _: "access")
    monkeypatch.setattr(routers.jwt_service, "create_refresh_token", lambda _: "refresh")

    await routers.login(
        UserLoginRequest(
            username="partially-migrated-user", password="legacy-password"
        ),
        Response(),
        db,
    )

    users.update_one.assert_not_awaited()
    linked_hash = linked_accounts.update_one.await_args.args[1]["$set"]["password"]
    assert linked_hash == canonical_hash


@pytest.mark.asyncio
async def test_verify_otp_cap_token_mot_lan_thay_vi_tra_password(monkeypatch):
    user_id = ObjectId()
    users = SimpleNamespace(
        find_one_and_update=AsyncMock(
            side_effect=[
                {
                    "_id": user_id,
                    "email": "reset@example.com",
                },
                None,
            ]
        ),
    )
    db = _database(tbl_User=users, linked_accounts=SimpleNamespace())
    monkeypatch.setattr(
        routers.jwt_service,
        "create_password_reset_token",
        lambda _: "short-lived-reset-token",
        raising=False,
    )

    result = await routers.verify_reset_otp(
        VerifyOtp(email="reset@example.com", otp="123456"), db
    )

    assert result == {
        "reset_token": "short-lived-reset-token",
        "expires_in": 300,
    }
    consume_filter = users.find_one_and_update.await_args_list[0].args[0]
    assert consume_filter["email"] == "reset@example.com"
    assert consume_filter["otp"] == "123456"
    assert "$gte" in consume_filter["createAtOTP"]
    consume_update = users.find_one_and_update.await_args_list[0].args[1]
    assert "password_reset_nonce_hash" in consume_update["$set"]
    assert consume_update["$unset"] == {"otp": "", "createAtOTP": ""}

    with pytest.raises(HTTPException) as replay_error:
        await routers.verify_reset_otp(
            VerifyOtp(email="reset@example.com", otp="123456"), db
        )
    assert replay_error.value.status_code == 400


def test_password_register_login_co_gioi_han_do_dai():
    register_data = {
        "username": "secure-user",
        "email": "secure@example.com",
        "gender": "other",
        "date": "2000-01-01",
        "number": "0123456789",
        "fullName": "Secure User",
        "password": "x" * 129,
    }
    with pytest.raises(ValidationError):
        UserRegisterRequest(**register_data)
    with pytest.raises(ValidationError):
        UserLoginRequest(username="secure-user", password="x" * 129)


@pytest.mark.asyncio
async def test_verify_otp_khong_tra_password_legacy(monkeypatch):
    user_id = ObjectId()
    users = SimpleNamespace(
        find_one_and_update=AsyncMock(
            return_value={
                "_id": user_id,
                "email": "reset@example.com",
                "password": "khong-duoc-tra-ve",
            }
        ),
    )
    db = _database(tbl_User=users, linked_accounts=SimpleNamespace())
    monkeypatch.setattr(
        routers.jwt_service,
        "create_password_reset_token",
        lambda _: "short-lived-reset-token",
        raising=False,
    )

    result = await routers.verify_reset_otp(
        VerifyOtp(email="reset@example.com", otp="123456"), db
    )

    assert "password" not in result
    assert "khong-duoc-tra-ve" not in repr(result)


@pytest.mark.asyncio
async def test_reset_password_hash_va_tu_choi_replay(monkeypatch):
    user_id = ObjectId()
    consumed_user = {
        "_id": user_id,
        "email": "reset@example.com",
    }
    users = SimpleNamespace(
        find_one_and_update=AsyncMock(side_effect=[consumed_user, None])
    )
    linked_accounts = SimpleNamespace(update_one=AsyncMock())
    db = _database(tbl_User=users, linked_accounts=linked_accounts)
    monkeypatch.setattr(
        routers.jwt_service,
        "verify_password_reset_token",
        lambda _: {
            "sub": str(user_id),
            "email": "reset@example.com",
            "nonce": "one-time-nonce",
            "type": "password_reset",
        },
        raising=False,
    )
    request_type = getattr(schema, "PasswordResetRequest")
    payload = request_type(
        reset_token="short-lived-reset-token",
        new_password="new-password-123",
        confirm_password="new-password-123",
    )

    result = await routers.reset_password(payload, db)

    assert result["status"] == "success"
    password_hash = users.find_one_and_update.await_args_list[0].args[1]["$set"][
        "password"
    ]
    assert password_hash != "new-password-123"
    assert HashHelper.verify_password("new-password-123", password_hash)
    linked_hash = linked_accounts.update_one.await_args.args[1]["$set"]["password"]
    assert linked_hash == password_hash

    with pytest.raises(HTTPException) as replay_error:
        await routers.reset_password(payload, db)
    assert replay_error.value.status_code == 401


@pytest.mark.asyncio
@pytest.mark.parametrize("stored_password", ["legacy-password", None])
async def test_doi_password_xac_thuc_va_ghi_hash(stored_password):
    user_id = ObjectId()
    linked_password = "legacy-password" if stored_password is None else None
    users = SimpleNamespace(update_one=AsyncMock())
    linked_accounts = SimpleNamespace(
        find_one=AsyncMock(
            return_value={
                "_id": ObjectId(),
                "user_id": user_id,
                "provider": "local",
                "password": linked_password,
            }
        ),
        update_one=AsyncMock(),
    )
    db = _database(tbl_User=users, linked_accounts=linked_accounts)
    user = {"_id": user_id}
    if stored_password is not None:
        user["password"] = stored_password

    await engine.handle_change_password(
        user,
        current_password="legacy-password",
        new_password="new-password-123",
        db=db,
    )

    target = users.update_one if stored_password is not None else linked_accounts.update_one
    password_hash = target.await_args.args[1]["$set"]["password"]
    assert password_hash != "new-password-123"
    assert HashHelper.verify_password("new-password-123", password_hash)
    if stored_password is not None:
        linked_hash = linked_accounts.update_one.await_args.args[1]["$set"]["password"]
        assert linked_hash == password_hash


def test_password_reset_token_dung_loai_va_het_han(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "test-secret-only")
    monkeypatch.setenv("ALGORITHM", "HS256")
    service = JWTService()

    reset_token = service.create_password_reset_token(
        {"sub": "user-1", "email": "reset@example.com", "nonce": "nonce"}
    )
    verification_token = service.create_verification_token(
        {"sub": "user-1", "email": "reset@example.com"}
    )

    assert service.verify_password_reset_token(reset_token)["type"] == "password_reset"
    assert service.verify_password_reset_token(verification_token) is None

    monkeypatch.setenv("PASSWORD_RESET_EXPIRE_MINUTES", "0")
    expired_service = JWTService()
    expired_token = expired_service.create_password_reset_token(
        {"sub": "user-1", "email": "reset@example.com", "nonce": "nonce"}
    )
    assert expired_service.verify_password_reset_token(expired_token) is None


@pytest.mark.parametrize("stored_password", [None, 123, "$argon2id$corrupted"])
def test_password_hong_hoac_sai_kieu_phai_fail_closed(stored_password):
    assert engine.verify_stored_password(str(stored_password), stored_password) == (
        False,
        False,
    )


@pytest.mark.asyncio
async def test_contact_fail_closed_khi_thieu_cau_hinh_smtp(monkeypatch):
    for name in (
        "CONTACT_SMTP_HOST",
        "CONTACT_SMTP_PORT",
        "CONTACT_SMTP_USERNAME",
        "CONTACT_SMTP_PASSWORD",
        "CONTACT_RECEIVER_EMAIL",
    ):
        monkeypatch.delenv(name, raising=False)
    smtp = Mock(side_effect=AssertionError("Không được kết nối khi thiếu cấu hình"))
    monkeypatch.setattr(engine.smtplib, "SMTP", smtp)
    db = _database(tbl_contacts=SimpleNamespace(insert_one=AsyncMock()))

    with pytest.raises(HTTPException) as error:
        await engine.handle_contact("User", "user@example.com", "Help", db)

    assert error.value.status_code == 503
    assert error.value.detail == "Contact email service is not configured"
    smtp.assert_not_called()


def test_source_khong_chua_literal_credential_smtp():
    source = Path(engine.__file__).read_text(encoding="utf-8")
    assert "CONTACT_SMTP_PASSWORD" in source
    assert 'sender_password = "' not in source
