"""Cấu hình bảo mật và vòng đời runtime của backend."""

from __future__ import annotations

import ipaddress
import math
import os
import re
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, cast
from urllib.parse import SplitResult, unquote, urlsplit

DeployMode = Literal["development", "test", "private", "public"]
AgentRuntimeMode = Literal["legacy", "shadow", "journey"]
CheckpointBackend = Literal["memory", "mongodb"]

_SERVER_MODES = frozenset({"private", "public"})
_SUPPORTED_MODES = frozenset({"development", "test", *_SERVER_MODES})
_PUBLIC_HOST_LABEL = re.compile(
    r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$", re.IGNORECASE
)
_SECRET_PLACEHOLDER_MARKERS = (
    "changeme",
    "replaceme",
    "exampleonly",
    "placeholder",
    "secretkey",
)
_DYNAMIC_DNS_DOMAINS = ("sslip.io", "nip.io")
_RESERVED_EXAMPLE_DOMAINS = (
    "invalid",
    "test",
    "example",
    "localhost",
    "example.com",
    "example.net",
    "example.org",
)
_DEVELOPMENT_SESSION_SECRET = secrets.token_urlsafe(48)
_AGENT_RUNTIME_MODES = frozenset({"legacy", "shadow", "journey"})
_CHECKPOINT_BACKENDS = frozenset({"memory", "mongodb"})
_SAFE_MONGO_DB_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$")
_DEFAULT_CHECKPOINT_TTL_SECONDS = 30 * 24 * 60 * 60
_DEFAULT_EVENT_RETENTION_DAYS = 30
_DEFAULT_ARTIFACT_RETENTION_DAYS = 180
_DEFAULT_SERVER_SELECTION_TIMEOUT_MS = 2000
_DEFAULT_READINESS_TIMEOUT_SECONDS = 3.0
_INVALID_PERCENT_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")


class ServerRuntimeConfigError(ValueError):
    """Báo cấu hình runtime không an toàn mà không kèm giá trị nhạy cảm."""


@dataclass(frozen=True, slots=True)
class CookieRuntimePolicy:
    """Policy tối thiểu để quyết định thuộc tính Secure của session cookie."""

    session_https_only: bool = False


@dataclass(frozen=True, slots=True)
class AgentRuntimeServerConfig:
    """Cấu hình composition của Agent Runtime, không hiển thị Mongo credential."""

    mode: AgentRuntimeMode
    persistence_mode: CheckpointBackend
    mongodb_uri: str | None = field(repr=False)
    db_name: str
    checkpoint_ttl_seconds: int
    event_retention_days: int
    artifact_retention_days: int
    server_selection_timeout_ms: int
    allow_memory: bool


@dataclass(frozen=True, slots=True)
class ServerRuntimeConfig:
    """Cấu hình đã kiểm tra để wiring HTTP middleware và tiến trình server."""

    deploy_mode: DeployMode
    app_origin: str | None
    cors_origins: tuple[str, ...]
    session_secret: str = field(repr=False)
    session_https_only: bool = False
    reload_enabled: bool = False
    agent_runtime: AgentRuntimeServerConfig = field(
        default_factory=lambda: AgentRuntimeServerConfig(
            mode="legacy",
            persistence_mode="memory",
            mongodb_uri=None,
            db_name="hagent_journey",
            checkpoint_ttl_seconds=_DEFAULT_CHECKPOINT_TTL_SECONDS,
            event_retention_days=_DEFAULT_EVENT_RETENTION_DAYS,
            artifact_retention_days=_DEFAULT_ARTIFACT_RETENTION_DAYS,
            server_selection_timeout_ms=_DEFAULT_SERVER_SELECTION_TIMEOUT_MS,
            allow_memory=True,
        )
    )
    readiness_timeout_seconds: float = _DEFAULT_READINESS_TIMEOUT_SECONDS

    @property
    def server_mode(self) -> bool:
        return self.deploy_mode in _SERVER_MODES


def _parse_mode(environment: Mapping[str, str]) -> DeployMode:
    value = environment.get("DEPLOY_MODE", "development").strip().lower()
    if value not in _SUPPORTED_MODES:
        raise ServerRuntimeConfigError("DEPLOY_MODE không được hỗ trợ.")
    return cast(DeployMode, value)


def _parse_boolean(
    environment: Mapping[str, str],
    key: str,
    *,
    default: bool | None,
) -> bool:
    raw = environment.get(key)
    if raw is None or not raw.strip():
        if default is None:
            raise ServerRuntimeConfigError(f"{key} là bắt buộc trong server mode.")
        return default
    normalized = raw.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ServerRuntimeConfigError(f"{key} chỉ nhận true hoặc false.")


def _parse_origin(raw_origin: str) -> tuple[str, SplitResult]:
    origin = raw_origin.strip()
    if not origin or origin != raw_origin or "*" in origin:
        raise ServerRuntimeConfigError(
            "APP_ORIGIN phải là một origin chính xác, không wildcard."
        )
    try:
        parsed = urlsplit(origin)
        _ = parsed.port
    except ValueError as exc:
        raise ServerRuntimeConfigError("APP_ORIGIN có port không hợp lệ.") from exc
    if (
        parsed.netloc.endswith(":")
        or parsed.port == 0
        or "?" in origin
        or "#" in origin
        or any(ord(character) < 32 or ord(character) == 127 for character in origin)
    ):
        raise ServerRuntimeConfigError(
            "APP_ORIGIN có delimiter hoặc ký tự không hợp lệ."
        )
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ServerRuntimeConfigError(
            "APP_ORIGIN phải dùng HTTP hoặc HTTPS và có hostname."
        )
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise ServerRuntimeConfigError(
            "APP_ORIGIN phải là origin, không chứa userinfo/path/query/fragment."
        )
    return origin, parsed


def _validate_private_origin(parsed: SplitResult) -> None:
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme != "http":
        raise ServerRuntimeConfigError("DEPLOY_MODE private phải dùng APP_ORIGIN HTTP.")
    if hostname == "localhost":
        return
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError as exc:
        raise ServerRuntimeConfigError(
            "DEPLOY_MODE private chỉ cho loopback origin qua SSH tunnel."
        ) from exc
    if not address.is_loopback:
        raise ServerRuntimeConfigError(
            "DEPLOY_MODE private chỉ cho loopback origin qua SSH tunnel."
        )


def _validate_public_origin(origin: str, parsed: SplitResult) -> None:
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme != "https":
        raise ServerRuntimeConfigError("DEPLOY_MODE public phải dùng APP_ORIGIN HTTPS.")
    if parsed.port is not None:
        raise ServerRuntimeConfigError(
            "APP_ORIGIN public không nhận port tường minh, kể cả port 443."
        )
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        raise ServerRuntimeConfigError(
            "APP_ORIGIN public phải dùng FQDN, không dùng địa chỉ IP."
        )
    labels = hostname.split(".")
    if (
        len(labels) < 2
        or len(hostname) > 253
        or all(label.isdigit() for label in labels)
        or not any(character.isalpha() for character in labels[-1])
        or any(not _PUBLIC_HOST_LABEL.fullmatch(label) for label in labels)
    ):
        raise ServerRuntimeConfigError("APP_ORIGIN public phải là FQDN hợp lệ.")
    if any(
        _is_domain_or_subdomain(hostname, domain) for domain in _DYNAMIC_DNS_DOMAINS
    ):
        raise ServerRuntimeConfigError("APP_ORIGIN public phải là FQDN hợp lệ.")
    if any(
        _is_domain_or_subdomain(hostname, domain)
        for domain in _RESERVED_EXAMPLE_DOMAINS
    ) or "placeholder" in re.sub(r"[^a-z0-9]", "", hostname):
        raise ServerRuntimeConfigError(
            "APP_ORIGIN public không được dùng domain placeholder."
        )
    if origin != f"https://{hostname}":
        raise ServerRuntimeConfigError(
            "APP_ORIGIN public phải ở dạng canonical HTTPS origin."
        )


def _is_domain_or_subdomain(hostname: str, domain: str) -> bool:
    return hostname == domain or hostname.endswith(f".{domain}")


def _load_origin(
    environment: Mapping[str, str], deploy_mode: DeployMode
) -> tuple[str | None, tuple[str, ...]]:
    raw_origin = environment.get("APP_ORIGIN")
    if raw_origin is None or not raw_origin:
        if deploy_mode in _SERVER_MODES:
            raise ServerRuntimeConfigError("APP_ORIGIN là bắt buộc trong server mode.")
        return None, ()
    origin, parsed = _parse_origin(raw_origin)
    if deploy_mode == "private":
        _validate_private_origin(parsed)
    elif deploy_mode == "public":
        _validate_public_origin(origin, parsed)
    return origin, (origin,)


def _is_weak_secret(value: str) -> bool:
    normalized = value.strip().lower()
    compact = re.sub(r"[^a-z0-9]", "", normalized)
    return (
        value != value.strip()
        or len(value) < 32
        or len(set(value)) < 8
        or compact.isdigit()
        or _is_repeated_pattern(value)
        or any(marker in compact for marker in _SECRET_PLACEHOLDER_MARKERS)
    )


def _is_repeated_pattern(value: str) -> bool:
    return any(
        len(value) % width == 0 and value == value[:width] * (len(value) // width)
        for width in range(1, len(value) // 2 + 1)
    )


def _load_session_secret(
    environment: Mapping[str, str], deploy_mode: DeployMode
) -> str:
    value = environment.get("SUPER_SECRET_KEY", "")
    if not value:
        if deploy_mode in _SERVER_MODES:
            raise ServerRuntimeConfigError(
                "SUPER_SECRET_KEY là bắt buộc trong server mode."
            )
        return _DEVELOPMENT_SESSION_SECRET
    if _is_weak_secret(value):
        raise ServerRuntimeConfigError("SUPER_SECRET_KEY không đạt yêu cầu độ mạnh.")
    return value


def _parse_choice(
    environment: Mapping[str, str],
    key: str,
    *,
    allowed: frozenset[str],
    default: str | None,
) -> str:
    raw = environment.get(key)
    if raw is None or not raw.strip():
        if default is None:
            raise ServerRuntimeConfigError(f"{key} là bắt buộc trong server mode.")
        value = default
    else:
        value = raw.strip().lower()
    if value not in allowed:
        raise ServerRuntimeConfigError(f"{key} có giá trị không hợp lệ.")
    return value


def _parse_integer(
    environment: Mapping[str, str],
    key: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw = environment.get(key)
    try:
        value = default if raw is None or not raw.strip() else int(raw)
    except (TypeError, ValueError):
        raise ServerRuntimeConfigError(f"{key} phải là số nguyên hợp lệ.") from None
    if value < minimum or value > maximum:
        raise ServerRuntimeConfigError(f"{key} nằm ngoài giới hạn an toàn.")
    return value


def _parse_timeout_seconds(environment: Mapping[str, str]) -> float:
    raw = environment.get("SERVER_READINESS_TIMEOUT_SECONDS")
    try:
        value = (
            _DEFAULT_READINESS_TIMEOUT_SECONDS
            if raw is None or not raw.strip()
            else float(raw)
        )
    except (TypeError, ValueError):
        raise ServerRuntimeConfigError(
            "SERVER_READINESS_TIMEOUT_SECONDS phải là số hợp lệ."
        ) from None
    if not math.isfinite(value) or value < 0.01 or value > 30:
        raise ServerRuntimeConfigError(
            "SERVER_READINESS_TIMEOUT_SECONDS nằm ngoài giới hạn an toàn."
        )
    return value


def _load_mongodb_uri(
    environment: Mapping[str, str],
    *,
    required: bool,
    require_auth: bool,
) -> str | None:
    raw = environment.get("MONGODB_CONNECT")
    if raw is None or not raw.strip():
        if raw is not None:
            raise ServerRuntimeConfigError("MONGODB_CONNECT không hợp lệ.")
        if required:
            raise ServerRuntimeConfigError(
                "MONGODB_CONNECT là bắt buộc cho Mongo runtime."
            )
        return None
    if raw != raw.strip():
        raise ServerRuntimeConfigError("MONGODB_CONNECT không hợp lệ.")
    try:
        parsed = urlsplit(raw)
        _ = parsed.port
    except ValueError:
        raise ServerRuntimeConfigError("MONGODB_CONNECT không hợp lệ.") from None
    if (
        any(ord(character) < 32 or ord(character) == 127 for character in raw)
        or _INVALID_PERCENT_ESCAPE.search(raw)
        or parsed.scheme not in {"mongodb", "mongodb+srv"}
        or not parsed.hostname
        or parsed.fragment
        or (parsed.scheme == "mongodb+srv" and parsed.port is not None)
    ):
        raise ServerRuntimeConfigError("MONGODB_CONNECT không hợp lệ.")
    username = unquote(parsed.username or "")
    password = unquote(parsed.password or "")
    decoded_credentials = (username, password)
    if any(
        "\ufffd" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        for value in decoded_credentials
    ):
        raise ServerRuntimeConfigError("MONGODB_CONNECT không hợp lệ.")
    if require_auth and (not username.strip() or not password.strip()):
        raise ServerRuntimeConfigError("MONGODB_CONNECT server mode phải có xác thực.")
    return raw


def _load_agent_runtime_config(
    environment: Mapping[str, str],
    deploy_mode: DeployMode,
) -> AgentRuntimeServerConfig:
    server_mode = deploy_mode in _SERVER_MODES
    mode = cast(
        AgentRuntimeMode,
        _parse_choice(
            environment,
            "HAGENT_RUNTIME_MODE",
            allowed=_AGENT_RUNTIME_MODES,
            default="legacy",
        ),
    )
    persistence_mode = cast(
        CheckpointBackend,
        _parse_choice(
            environment,
            "HAGENT_CHECKPOINT_BACKEND",
            allowed=_CHECKPOINT_BACKENDS,
            default="mongodb" if server_mode else "memory",
        ),
    )
    if server_mode and persistence_mode != "mongodb":
        raise ServerRuntimeConfigError(
            "HAGENT_CHECKPOINT_BACKEND phải là mongodb trong server mode."
        )

    mongodb_uri = _load_mongodb_uri(
        environment,
        required=server_mode and persistence_mode == "mongodb",
        require_auth=server_mode,
    )
    db_name = environment.get("HAGENT_RUNTIME_DB_NAME", "hagent_journey")
    if db_name != db_name.strip() or not _SAFE_MONGO_DB_NAME.fullmatch(db_name):
        raise ServerRuntimeConfigError("HAGENT_RUNTIME_DB_NAME không hợp lệ.")
    artifact_retention_raw = environment.get("HAGENT_ARTIFACT_RETENTION_DAYS")
    if artifact_retention_raw is not None and not artifact_retention_raw.strip():
        raise ServerRuntimeConfigError(
            "HAGENT_ARTIFACT_RETENTION_DAYS phải là số nguyên hợp lệ."
        )

    return AgentRuntimeServerConfig(
        mode=mode,
        persistence_mode=persistence_mode,
        mongodb_uri=mongodb_uri,
        db_name=db_name,
        checkpoint_ttl_seconds=_parse_integer(
            environment,
            "HAGENT_CHECKPOINT_TTL_SECONDS",
            default=_DEFAULT_CHECKPOINT_TTL_SECONDS,
            minimum=3600,
            maximum=90 * 24 * 60 * 60,
        ),
        event_retention_days=_parse_integer(
            environment,
            "HAGENT_EVENT_RETENTION_DAYS",
            default=_DEFAULT_EVENT_RETENTION_DAYS,
            minimum=1,
            maximum=90,
        ),
        artifact_retention_days=_parse_integer(
            environment,
            "HAGENT_ARTIFACT_RETENTION_DAYS",
            default=_DEFAULT_ARTIFACT_RETENTION_DAYS,
            minimum=1,
            maximum=3650,
        ),
        server_selection_timeout_ms=_parse_integer(
            environment,
            "HAGENT_RUNTIME_SERVER_SELECTION_TIMEOUT_MS",
            default=_DEFAULT_SERVER_SELECTION_TIMEOUT_MS,
            minimum=1,
            maximum=30_000,
        ),
        allow_memory=not server_mode and persistence_mode == "memory",
    )


def _load_session_https_only(
    environment: Mapping[str, str], deploy_mode: DeployMode
) -> bool:
    session_https_only = _parse_boolean(
        environment,
        "SESSION_HTTPS_ONLY",
        default=None if deploy_mode in _SERVER_MODES else False,
    )
    if deploy_mode == "private" and session_https_only:
        raise ServerRuntimeConfigError(
            "SESSION_HTTPS_ONLY phải false trong DEPLOY_MODE private HTTP."
        )
    if deploy_mode == "public" and not session_https_only:
        raise ServerRuntimeConfigError(
            "SESSION_HTTPS_ONLY phải true trong DEPLOY_MODE public."
        )
    return session_https_only


def load_cookie_runtime_policy(
    environment: Mapping[str, str] | None = None,
) -> CookieRuntimePolicy:
    """Đọc policy cookie mà không dựng secret hoặc dependency của server."""

    source = os.environ if environment is None else environment
    deploy_mode = _parse_mode(source)

    return CookieRuntimePolicy(
        session_https_only=_load_session_https_only(source, deploy_mode),
    )


def load_server_runtime_config(
    environment: Mapping[str, str] | None = None,
) -> ServerRuntimeConfig:
    """Đọc full server config và luôn kiểm tra dependency durable bắt buộc."""

    source = os.environ if environment is None else environment
    deploy_mode = _parse_mode(source)
    app_origin, cors_origins = _load_origin(source, deploy_mode)
    session_secret = _load_session_secret(source, deploy_mode)
    session_https_only = _load_session_https_only(source, deploy_mode)
    reload_enabled = _parse_boolean(source, "BACKEND_RELOAD", default=False)
    if deploy_mode in _SERVER_MODES and reload_enabled:
        raise ServerRuntimeConfigError("BACKEND_RELOAD phải false trong server mode.")
    agent_runtime = _load_agent_runtime_config(source, deploy_mode)
    readiness_timeout_seconds = _parse_timeout_seconds(source)

    return ServerRuntimeConfig(
        deploy_mode=deploy_mode,
        app_origin=app_origin,
        cors_origins=cors_origins,
        session_secret=session_secret,
        session_https_only=session_https_only,
        reload_enabled=reload_enabled,
        agent_runtime=agent_runtime,
        readiness_timeout_seconds=readiness_timeout_seconds,
    )
