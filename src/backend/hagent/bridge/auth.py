"""
HAgent Bridge — Xác thực JWT

Sử dụng chung SECRET_KEY và ALGORITHM với HAutoML backend.
Cấu hình tải từ hagent.yaml.
"""

from datetime import UTC, datetime

import structlog
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWTError, decode

from .config import get_auth_config

logger = structlog.get_logger(__name__)

security = HTTPBearer(auto_error=False)


class TokenPayload:
    """Dữ liệu đã giải mã từ JWT token."""

    def __init__(self, payload: dict, raw_token: str = ""):
        self.user_id: str = payload.get("sub", payload.get("user_id", ""))
        self.email: str = payload.get("email", "")
        self.exp: float = payload.get("exp", 0)
        self.token_type: str = payload.get("type", "access")
        self.raw: dict = payload
        self.raw_token: str = raw_token

    @property
    def is_expired(self) -> bool:
        """Kiểm tra token đã hết hạn chưa."""
        return datetime.now(UTC).timestamp() > self.exp


def verify_jwt_token(token: str) -> TokenPayload:
    """
    Xác thực JWT token dùng secret từ hagent.yaml.
    Ném HTTPException nếu token không hợp lệ hoặc đã hết hạn.
    """
    auth_cfg = get_auth_config()

    if not auth_cfg["secret_key"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="SECRET_KEY chưa được cấu hình trong hagent.yaml hoặc biến môi trường",
        )

    try:
        payload = decode(
            token,
            auth_cfg["secret_key"],
            algorithms=[auth_cfg["algorithm"]],
        )
    except PyJWTError:
        logger.warning("Xác thực JWT thất bại")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token không hợp lệ",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None

    token_payload = TokenPayload(payload, token)

    if token_payload.is_expired:
        logger.debug("JWT đã hết hạn")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token đã hết hạn",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if token_payload.token_type != "access":
        logger.debug("JWT không phải access token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Loại token không hợp lệ — yêu cầu access token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return token_payload


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> TokenPayload:
    if credentials is None:
        logger.debug("Thiếu thông tin xác thực")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Thiếu header Authorization",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return verify_jwt_token(credentials.credentials)


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> TokenPayload | None:
    """
    FastAPI dependency: trích xuất JWT tùy chọn (không bắt buộc).
    Trả về None nếu không có token (dùng cho endpoint công khai).
    """
    if credentials is None:
        return None
    try:
        return verify_jwt_token(credentials.credentials)
    except HTTPException:
        return None
