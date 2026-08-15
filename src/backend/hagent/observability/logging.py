"""Cấu hình logging có cấu trúc và ngữ cảnh request cho HAgent."""

from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any, TextIO
from uuid import uuid4

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars, merge_contextvars

CORRELATION_ID_HEADER = "X-Correlation-ID"
_CORRELATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_PRODUCTION_ENVIRONMENTS = frozenset({"prod", "production"})
_SENSITIVE_KEY_PARTS = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "cookie",
        "email",
        "mongodb_uri",
        "password",
        "principal_id",
        "refresh_token",
        "secret",
        "token",
        "user_id",
    }
)
_BEARER_PATTERN = re.compile(r"(?i)(bearer\s+)[^\s,;]+")
_EMAIL_PATTERN = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_MONGODB_PATTERN = re.compile(r"(?i)mongodb(?:\+srv)?://\S+")
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret)=([^&\s]+)"
)
_REDACTED = "[DA_AN]"


def _is_production(environment: str | None) -> bool:
    selected = environment or os.getenv("HAGENT_ENVIRONMENT", "development")
    return selected.strip().lower() in _PRODUCTION_ENVIRONMENTS


def _is_sensitive_key(key: object) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _redact_text(value: str) -> str:
    value = _BEARER_PATTERN.sub(rf"\1{_REDACTED}", value)
    value = _EMAIL_PATTERN.sub(_REDACTED, value)
    value = _MONGODB_PATTERN.sub(_REDACTED, value)
    return _SECRET_ASSIGNMENT_PATTERN.sub(rf"\1={_REDACTED}", value)


def _redact_value(value: Any) -> Any:
    if isinstance(value, MutableMapping):
        return {
            key: _REDACTED if _is_sensitive_key(key) else _redact_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item) for item in value)
    if isinstance(value, str):
        return _redact_text(value)
    return value


def redact_sensitive_data(
    _logger: Any,
    _method_name: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Ẩn bí mật và PII trước khi renderer ghi dữ liệu ra ngoài."""
    return _redact_value(event_dict)


def configure_logging(
    environment: str | None = None,
    *,
    stream: TextIO | None = None,
) -> None:
    """Cấu hình JSON cho production và console dễ đọc cho development."""
    output = stream or sys.stdout
    production = _is_production(environment)
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer(ensure_ascii=False)
        if production
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    timestamp = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared_processors: list[structlog.types.Processor] = [
        merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        timestamp,
        redact_sensitive_data,
    ]
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    handler = logging.StreamHandler(output)
    handler.setFormatter(formatter)
    root_logger = logging.root
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(os.getenv("HAGENT_LOG_LEVEL", "INFO").upper())

    structlog.configure(
        processors=[
            merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            timestamp,
            redact_sensitive_data,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )


def normalize_correlation_id(candidate: str | None) -> str:
    """Giữ ID hợp lệ từ client hoặc sinh ID an toàn cho request mới."""
    value = (candidate or "").strip()
    return value if _CORRELATION_ID_PATTERN.fullmatch(value) else uuid4().hex


async def correlation_id_middleware(
    request: Any,
    call_next: Callable[[Any], Awaitable[Any]],
) -> Any:
    """Bind correlation ID vào context bất đồng bộ và phản hồi HTTP."""
    clear_contextvars()
    correlation_id = normalize_correlation_id(
        request.headers.get(CORRELATION_ID_HEADER)
    )
    bind_contextvars(correlation_id=correlation_id)
    request.state.correlation_id = correlation_id
    try:
        response = await call_next(request)
        response.headers[CORRELATION_ID_HEADER] = correlation_id
        return response
    finally:
        clear_contextvars()
