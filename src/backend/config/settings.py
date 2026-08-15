"""
Quản lý Cấu hình và Cài đặt Ứng dụng (REFAC-026).
"""

from __future__ import annotations

import os
import secrets
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field

DeployMode = Literal["development", "test", "private", "public"]

# Tập hợp các chuỗi bị coi là placeholder chưa đổi
_PLACEHOLDER_SECRETS: frozenset[str] = frozenset(
    {
        "minioadmin",
        "change-me",
        "change-me-to-a-long-random-string",
        "change-this-in-production-session-secret-key",
        "changeme",
        "replaceme",
        "secret",
        "password",
        "",
    }
)

# Chế độ deploy yêu cầu secret thật (khác placeholder)
_STRICT_MODES: frozenset[str] = frozenset({"private", "public"})


def _require_secret(value: str, env_var: str, deploy_mode: str) -> str:
    """Kiểm tra secret không phải placeholder khi ở chế độ production.

    Raises:
        ValueError: Nếu đang chạy ở strict mode mà secret vẫn là placeholder.
    """
    if deploy_mode in _STRICT_MODES and value in _PLACEHOLDER_SECRETS:
        raise ValueError(
            f"[P1-3] Biến môi trường '{env_var}' chưa được cấu hình. "
            f"Trong chế độ '{deploy_mode}', không được dùng giá trị mặc định. "
            f"Hãy đặt '{env_var}' trong file .env hoặc biến môi trường hệ thống."
        )
    return value



class AppSettings(BaseModel):
    """Mô hình cấu hình ứng dụng với kiểu dữ liệu tường minh (type-safe).

    Các trường có hậu tố _secret_key, session_secret chỉ giữ placeholder rõ ràng.
    Giá trị thật phải được cung cấp qua biến môi trường.
    """

    deploy_mode: DeployMode = "development"
    database_uri: str = "mongodb://localhost:27017"
    database_name: str = "AutoML"
    database_server_selection_timeout_ms: int = 2000

    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    # Giá trị mặc định là placeholder — chỉ dùng được ở development/test
    minio_secret_key: str = "minioadmin"
    minio_secure: bool = False

    kafka_server: str = "localhost:9092"
    kafka_topic: str = "example-topic"

    # Giá trị mặc định là placeholder — chỉ dùng được ở development/test
    session_secret: str = Field(default="change-this-in-production-session-secret-key")
    session_https_only: bool = False
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8080"]
    )

    host: str = "0.0.0.0"
    port: int = 8080
    reload: bool = False


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Khởi tạo và cache AppSettings từ biến môi trường.

    Trong chế độ 'private' hoặc 'public', sẽ raise ValueError ngay khi
    khởi động nếu bất kỳ secret nào vẫn là giá trị placeholder.
    """
    deploy_mode = os.getenv("DEPLOY_MODE", "development").strip().lower()
    if deploy_mode not in {"development", "test", "private", "public"}:
        deploy_mode = "development"

    # Đọc các secret từ env trước khi validate
    raw_minio_secret = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    raw_session_secret = os.getenv(
        "SECRET_KEY",
        os.getenv("SESSION_SECRET", "change-this-in-production-session-secret-key"),
    )

    # P1-3 FIX: Fail fast nếu đang ở production mà secret vẫn là placeholder
    _require_secret(raw_minio_secret, "MINIO_SECRET_KEY", deploy_mode)
    _require_secret(raw_session_secret, "SECRET_KEY", deploy_mode)

    return AppSettings(
        deploy_mode=deploy_mode,  # type: ignore[arg-type]
        database_uri=os.getenv(
            "MONGODB_CONNECT", os.getenv("DATABASE_URI", "mongodb://localhost:27017")
        ),
        database_name=os.getenv("MONGODB_DB_NAME", "AutoML"),
        database_server_selection_timeout_ms=int(
            os.getenv("MONGODB_TIMEOUT_MS", "2000")
        ),
        minio_endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        minio_access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        minio_secret_key=raw_minio_secret,
        minio_secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
        kafka_server=os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS",
            os.getenv("KAFKA_SERVER", "localhost:9092"),
        ),
        kafka_topic=os.getenv("KAFKA_TOPIC", "example-topic"),
        session_secret=raw_session_secret,
        session_https_only=os.getenv("SESSION_HTTPS_ONLY", "false").lower() == "true",
        cors_origins=[
            o.strip()
            for o in os.getenv(
                "CORS_ORIGINS", "http://localhost:3000,http://localhost:8080"
            ).split(",")
            if o.strip()
        ],
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8080")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )


def generate_secret_key(length: int = 64) -> str:
    """Sinh secret key ngẫu nhiên an toàn mật mã (dùng khi setup lần đầu).

    Ví dụ::

        python -c "from config.settings import generate_secret_key; print(generate_secret_key())"
    """
    return secrets.token_hex(length)
