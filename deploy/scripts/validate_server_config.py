"""Kiểm tra fail-closed file cấu hình triển khai private/public."""

from __future__ import annotations

import argparse
import ipaddress
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from urllib.parse import SplitResult, parse_qs, urlsplit

_KEY_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")
_REFERENCE_PATTERN = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")
_RELEASE_TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HOST_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
_DB_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$")
_KAFKA_TOPIC_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,249}$")
_PLACEHOLDER_MARKERS = ("CHANGE_ME", "REPLACE_ME", "EXAMPLE_ONLY")
_REQUIRED_KEYS = {
    "DEPLOY_MODE",
    "COMPOSE_PROFILES",
    "GATEWAY_BIND_IP",
    "GATEWAY_HTTP_PORT",
    "GATEWAY_HTTPS_PORT",
    "SITE_ADDRESS",
    "APP_ORIGIN",
    "NEXTAUTH_URL",
    "FRONTEND_URL",
    "REDIRECT_URI",
    "SESSION_HTTPS_ONLY",
    "SKIP_EMAIL_VERIFICATION",
    "AUTH_API_BASE_URL",
    "HAGENT_INTERNAL_URL",
    "HAUTOML_BASE_URL",
    "HAGENT_RUN_API_URL",
    "RELEASE_TAG",
    "TOOLKIT_IMAGE",
    "BRIDGE_IMAGE",
    "WORKER_IMAGE",
    "FRONTEND_IMAGE",
    "SECRET_KEY",
    "SUPER_SECRET_KEY",
    "NEXTAUTH_SECRET",
    "ALGORITHM",
    "ACCESS_EXPIRE",
    "REFRESH_EXPIRE",
    "PASSWORD_RESET_EXPIRE_MINUTES",
    "MONGO_ROOT_USERNAME",
    "MONGO_ROOT_PASSWORD",
    "MONGODB_CONNECT",
    "MONGODB_DB_NAME",
    "HAGENT_RUNTIME_DB_NAME",
    "HAGENT_CHECKPOINT_BACKEND",
    "HAGENT_CHECKPOINT_TTL_SECONDS",
    "HAGENT_EVENT_RETENTION_DAYS",
    "HAGENT_ARTIFACT_RETENTION_DAYS",
    "HAGENT_RUNTIME_MODE",
    "MINIO_ENDPOINT",
    "MINIO_ACCESS_KEY",
    "MINIO_SECRET_KEY",
    "MINIO_SECURE",
    "KAFKA_SERVER",
    "KAFKA_TOPIC",
    "MAIL_USERNAME",
    "MAIL_PASSWORD",
    "LOGO",
    "LLM_DEFAULT_MODEL",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OLLAMA_BASE_URL",
    "LOCAL_BASE_URL",
    "LOCAL_MODEL_NAME",
    "LOCAL_API_KEY",
}
_INTERNAL_URLS = {
    "AUTH_API_BASE_URL": "http://toolkit:8585",
    "HAGENT_INTERNAL_URL": "http://hagent_bridge:9900",
    "HAUTOML_BASE_URL": "http://toolkit:8585",
    "HAGENT_RUN_API_URL": "http://toolkit:8585/api/v1/runs",
}
_IMAGE_NAMES = {
    "TOOLKIT_IMAGE": "hagent-toolkit",
    "BRIDGE_IMAGE": "hagent-bridge",
    "WORKER_IMAGE": "hagent-worker",
    "FRONTEND_IMAGE": "hagent-frontend",
}


class ConfigValidationError(ValueError):
    """Lỗi cấu hình đã khử giá trị nhạy cảm."""

    def __init__(self, issues: str | Sequence[str]) -> None:
        normalized = (issues,) if isinstance(issues, str) else tuple(issues)
        self.issues = normalized
        super().__init__("; ".join(normalized))


def load_env_file(path: Path) -> dict[str, str]:
    """Đọc tập con dotenv không có ngữ nghĩa phụ thuộc shell/Compose."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ConfigValidationError("Không đọc được file cấu hình") from exc

    values: dict[str, str] = {}
    issues: list[str] = []
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if raw_line != raw_line.strip():
            issues.append(f"Dòng {line_number}: không cho phép khoảng trắng bao quanh")
            continue
        if "=" not in raw_line:
            issues.append(f"Dòng {line_number}: thiếu dấu '='")
            continue
        key, value = raw_line.split("=", 1)
        if not _KEY_PATTERN.fullmatch(key):
            issues.append(f"Dòng {line_number}: tên biến không hợp lệ")
            continue
        if key in values:
            issues.append(f"{key}: khai báo trùng")
            continue
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            issues.append(f"{key}: chứa ký tự điều khiển")
            continue
        if any(character.isspace() for character in value):
            issues.append(f"{key}: không cho phép khoảng trắng trong giá trị")
            continue
        if any(character in value for character in ('"', "'", "#", "\\")):
            issues.append(f"{key}: chứa cú pháp dotenv không được hỗ trợ")
            continue
        if "$" in _REFERENCE_PATTERN.sub("", value):
            issues.append(f"{key}: cú pháp tham chiếu không được hỗ trợ")
            continue
        values[key] = value
    if issues:
        raise ConfigValidationError(issues)
    return values


def resolve_config(raw: Mapping[str, str]) -> dict[str, str]:
    """Mở rộng tham chiếu ``${KEY}`` mà không dùng shell evaluation."""
    resolved: dict[str, str] = {}
    resolving: list[str] = []

    def resolve_key(key: str) -> str:
        if key in resolved:
            return resolved[key]
        if key in resolving:
            cycle = " -> ".join((*resolving, key))
            raise ConfigValidationError(f"Phát hiện vòng lặp tham chiếu: {cycle}")
        if key not in raw:
            owner = resolving[-1] if resolving else key
            raise ConfigValidationError(
                f"{owner}: tham chiếu biến chưa khai báo ({key})"
            )
        resolving.append(key)
        value = raw[key]

        def replace_reference(match: re.Match[str]) -> str:
            return resolve_key(match.group(1))

        value = _REFERENCE_PATTERN.sub(replace_reference, value)
        resolving.pop()
        if "$" in value:
            raise ConfigValidationError(f"{key}: cú pháp tham chiếu không được hỗ trợ")
        resolved[key] = value
        return value

    for config_key in raw:
        resolve_key(config_key)
    return resolved


def _is_placeholder(value: str) -> bool:
    upper_value = value.upper()
    return any(marker in upper_value for marker in _PLACEHOLDER_MARKERS)


def _require_exact(
    config: Mapping[str, str],
    key: str,
    expected: str,
    issues: list[str],
) -> None:
    if config.get(key) != expected:
        issues.append(f"{key}: không khớp contract triển khai")


def _validate_integer(
    config: Mapping[str, str],
    key: str,
    minimum: int,
    maximum: int,
    issues: list[str],
) -> None:
    try:
        value = int(config.get(key, ""))
    except ValueError:
        issues.append(f"{key}: phải là số nguyên")
        return
    if not minimum <= value <= maximum:
        issues.append(f"{key}: nằm ngoài giới hạn cho phép")


def _parse_http_url(
    config: Mapping[str, str],
    key: str,
    issues: list[str],
    *,
    allow_path: bool,
) -> SplitResult | None:
    value = config.get(key, "")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        issues.append(f"{key}: URL không hợp lệ")
        return None
    valid = True
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        issues.append(f"{key}: phải là URL HTTP(S) tuyệt đối")
        return None
    if (
        parsed.username is not None
        or parsed.password is not None
        or "@" in parsed.netloc
        or parsed.query
        or parsed.fragment
        or "?" in value
        or "#" in value
    ):
        issues.append(f"{key}: không được chứa credential, query hoặc fragment")
        valid = False
    if parsed.netloc.rsplit("@", 1)[-1].endswith(":"):
        issues.append(f"{key}: port không hợp lệ")
        valid = False
    if not allow_path and parsed.path:
        issues.append(f"{key}: không được chứa path")
        valid = False
    hostname = parsed.hostname
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        labels = hostname.split(".")
        if len(hostname) > 253 or any(
            not _HOST_LABEL_PATTERN.fullmatch(label) for label in labels
        ):
            issues.append(f"{key}: hostname không hợp lệ")
            valid = False
    if port is not None and not 1 <= port <= 65535:
        issues.append(f"{key}: port nằm ngoài giới hạn")
        valid = False
    return parsed if valid else None


def _validate_mode(
    config: Mapping[str, str], issues: list[str], template: bool
) -> None:
    mode = config.get("DEPLOY_MODE")
    if mode not in {"private", "public"}:
        issues.append("DEPLOY_MODE: chỉ nhận private hoặc public")
        return
    _require_exact(config, "COMPOSE_PROFILES", mode, issues)
    _require_exact(config, "SKIP_EMAIL_VERIFICATION", "false", issues)
    origin = config.get("APP_ORIGIN", "")

    if mode == "private":
        _require_exact(config, "GATEWAY_BIND_IP", "127.0.0.1", issues)
        _require_exact(config, "GATEWAY_HTTP_PORT", "8080", issues)
        _require_exact(config, "GATEWAY_HTTPS_PORT", "", issues)
        _require_exact(config, "SITE_ADDRESS", "http://", issues)
        _require_exact(config, "APP_ORIGIN", "http://localhost:8080", issues)
        _require_exact(config, "SESSION_HTTPS_ONLY", "false", issues)
    else:
        _require_exact(config, "GATEWAY_BIND_IP", "0.0.0.0", issues)
        _require_exact(config, "GATEWAY_HTTP_PORT", "80", issues)
        _require_exact(config, "GATEWAY_HTTPS_PORT", "443", issues)
        _require_exact(config, "SESSION_HTTPS_ONLY", "true", issues)
        parsed = _parse_http_url(config, "APP_ORIGIN", issues, allow_path=False)
        if parsed is not None:
            hostname = parsed.hostname or ""
            if parsed.scheme != "https":
                issues.append("APP_ORIGIN: public bắt buộc HTTPS")
            if parsed.port is not None:
                issues.append("APP_ORIGIN: public không nhận port tường minh")
            try:
                ipaddress.ip_address(hostname)
            except ValueError:
                if all(label.isdigit() for label in hostname.split(".")):
                    issues.append("APP_ORIGIN: public không nhận raw IP rút gọn")
            else:
                issues.append("APP_ORIGIN: public không nhận raw IP")
            labels = hostname.split(".")
            if (
                len(labels) < 2
                or any(not _HOST_LABEL_PATTERN.fullmatch(label) for label in labels)
                or not any(character.isalpha() for character in labels[-1])
            ):
                issues.append("APP_ORIGIN: hostname không phải FQDN hợp lệ")
            if hostname.endswith((".sslip.io", ".nip.io")):
                issues.append("APP_ORIGIN: không được dùng wildcard-IP DNS")
            if hostname.endswith(".invalid") and not template:
                issues.append("APP_ORIGIN: production không nhận domain mẫu")
            if config.get("SITE_ADDRESS") != hostname:
                issues.append("SITE_ADDRESS: phải trùng hostname public")

    _require_exact(config, "NEXTAUTH_URL", origin, issues)
    _require_exact(config, "FRONTEND_URL", origin, issues)
    _require_exact(config, "REDIRECT_URI", f"{origin}/api/backend", issues)
    _require_exact(config, "LOGO", f"{origin}/image.png", issues)


def _validate_secrets(
    config: Mapping[str, str], issues: list[str], template: bool
) -> None:
    minimum_lengths = {
        "SECRET_KEY": 32,
        "SUPER_SECRET_KEY": 32,
        "NEXTAUTH_SECRET": 32,
        "MONGO_ROOT_PASSWORD": 24,
        "MINIO_SECRET_KEY": 24,
        "MAIL_PASSWORD": 24,
    }
    for key, minimum in minimum_lengths.items():
        value = config.get(key, "")
        if not value:
            issues.append(f"{key}: bắt buộc")
        elif not template and (_is_placeholder(value) or len(value) < minimum):
            issues.append(f"{key}: placeholder hoặc secret quá yếu")

    if not template:
        session_secrets = {
            config.get("SECRET_KEY", ""),
            config.get("SUPER_SECRET_KEY", ""),
            config.get("NEXTAUTH_SECRET", ""),
        }
        if len(session_secrets) != 3:
            issues.append("SECRET_KEY/SUPER_SECRET_KEY/NEXTAUTH_SECRET: phải khác nhau")
        for key in ("MONGO_ROOT_USERNAME", "MINIO_ACCESS_KEY", "MAIL_USERNAME"):
            value = config.get(key, "")
            if not value or _is_placeholder(value):
                issues.append(f"{key}: thiếu giá trị production")


def _validate_storage(
    config: Mapping[str, str], issues: list[str], template: bool
) -> None:
    _require_exact(config, "HAGENT_CHECKPOINT_BACKEND", "mongodb", issues)
    if config.get("HAGENT_RUNTIME_MODE") not in {"legacy", "shadow", "journey"}:
        issues.append("HAGENT_RUNTIME_MODE: giá trị không hợp lệ")
    _require_exact(config, "MINIO_ENDPOINT", "minio:9000", issues)
    _require_exact(config, "MINIO_SECURE", "false", issues)
    _require_exact(config, "KAFKA_SERVER", "kafka:9092", issues)
    kafka_topic = config.get("KAFKA_TOPIC", "")
    if (
        not _KAFKA_TOPIC_PATTERN.fullmatch(kafka_topic)
        or kafka_topic in {".", ".."}
        or (not template and _is_placeholder(kafka_topic))
    ):
        issues.append("KAFKA_TOPIC: tên topic không hợp lệ")

    for key in ("MONGODB_DB_NAME", "HAGENT_RUNTIME_DB_NAME"):
        database_name = config.get(key, "")
        if not _DB_NAME_PATTERN.fullmatch(database_name) or (
            not template and _is_placeholder(database_name)
        ):
            issues.append(f"{key}: tên database Mongo không hợp lệ")

    try:
        mongo_url = urlsplit(config.get("MONGODB_CONNECT", ""))
        mongo_port = mongo_url.port
    except ValueError:
        mongo_url = None
        mongo_port = None
    if (
        mongo_url is None
        or mongo_url.scheme != "mongodb"
        or mongo_url.hostname != "mongo"
        or mongo_port != 27017
        or mongo_url.username != config.get("MONGO_ROOT_USERNAME")
        or mongo_url.password != config.get("MONGO_ROOT_PASSWORD")
        or parse_qs(mongo_url.query).get("authSource") != ["admin"]
    ):
        issues.append("MONGODB_CONNECT: phải dùng Mongo nội bộ có xác thực")


def _validate_release(
    config: Mapping[str, str], issues: list[str], template: bool
) -> None:
    release_tag = config.get("RELEASE_TAG", "")
    if not _RELEASE_TAG_PATTERN.fullmatch(release_tag):
        issues.append("RELEASE_TAG: định dạng không hợp lệ")
    if release_tag.lower() == "latest" or (
        not template and _is_placeholder(release_tag)
    ):
        issues.append(
            "RELEASE_TAG: phải là tag bất biến, không dùng placeholder/latest"
        )
    for key, image_name in _IMAGE_NAMES.items():
        if config.get(key) != f"{image_name}:{release_tag}":
            issues.append(f"{key}: phải dùng RELEASE_TAG chung")


def _provider_key_configured(
    config: Mapping[str, str], key: str, issues: list[str], template: bool
) -> bool:
    value = config.get(key, "")
    if not value:
        return False
    if template:
        return True
    if _is_placeholder(value) or len(value) < 16:
        issues.append(f"{key}: placeholder hoặc API key quá yếu")
        return False
    return True


def _provider_url_configured(
    config: Mapping[str, str],
    key: str,
    issues: list[str],
    template: bool,
    *,
    allow_path: bool,
) -> bool:
    value = config.get(key, "")
    parsed = _parse_http_url(config, key, issues, allow_path=allow_path)
    if parsed is None:
        return False
    hostname = parsed.hostname or ""
    if not template and (_is_placeholder(value) or hostname.endswith(".invalid")):
        issues.append(f"{key}: không được dùng URL placeholder")
        return False
    return True


def _validate_providers(
    config: Mapping[str, str], issues: list[str], template: bool
) -> None:
    openai = _provider_key_configured(config, "OPENAI_API_KEY", issues, template)
    anthropic = _provider_key_configured(config, "ANTHROPIC_API_KEY", issues, template)
    ollama_url = config.get("OLLAMA_BASE_URL", "")
    local_url = config.get("LOCAL_BASE_URL", "")
    local_model = config.get("LOCAL_MODEL_NAME", "")
    local_api_key = config.get("LOCAL_API_KEY", "")
    ollama = False
    local = False

    if ollama_url:
        ollama = _provider_url_configured(
            config,
            "OLLAMA_BASE_URL",
            issues,
            template,
            allow_path=False,
        )
    if bool(local_url) != bool(local_model):
        issues.append("LOCAL_BASE_URL/LOCAL_MODEL_NAME: phải cấu hình cùng nhau")
    local_model_valid = bool(local_model)
    if local_model and not template and _is_placeholder(local_model):
        issues.append("LOCAL_MODEL_NAME: không được dùng placeholder")
        local_model_valid = False
    local_key_valid = True
    if local_api_key:
        local_key_valid = _provider_key_configured(
            config, "LOCAL_API_KEY", issues, template
        )
    if local_url:
        local_url_valid = _provider_url_configured(
            config,
            "LOCAL_BASE_URL",
            issues,
            template,
            allow_path=True,
        )
        local = local_url_valid and local_model_valid and local_key_valid
    if not any((openai, anthropic, ollama, local)):
        issues.append("LLM provider: cần ít nhất một provider hợp lệ")

    required_provider = {
        "openai-gpt4o-mini": openai,
        "openai-gpt4o": openai,
        "anthropic-sonnet": anthropic,
        "ollama-llama": ollama,
        "ollama-ci": ollama,
        "local-compatible": local,
    }.get(config.get("LLM_DEFAULT_MODEL", ""))
    if required_provider is not True:
        issues.append("LLM_DEFAULT_MODEL: provider mặc định chưa được cấu hình")


def validate_config(
    raw: Mapping[str, str], *, template: bool = False
) -> dict[str, str]:
    """Kiểm tra toàn bộ hợp đồng và trả cấu hình đã mở rộng tham chiếu."""
    missing = sorted(_REQUIRED_KEYS.difference(raw))
    unknown = sorted(set(raw).difference(_REQUIRED_KEYS))
    structural_issues = [f"{key}: thiếu biến bắt buộc" for key in missing]
    structural_issues.extend(f"{key}: biến không được hỗ trợ" for key in unknown)
    if structural_issues:
        raise ConfigValidationError(structural_issues)

    config = resolve_config(raw)
    issues: list[str] = []
    if any("sslip.io" in value.lower() for value in config.values()):
        issues.append("Cấu hình không được chứa sslip.io")
    for key, expected in _INTERNAL_URLS.items():
        _require_exact(config, key, expected, issues)
    _validate_mode(config, issues, template)
    _validate_secrets(config, issues, template)
    _validate_storage(config, issues, template)
    _validate_release(config, issues, template)
    _validate_providers(config, issues, template)
    _require_exact(config, "ALGORITHM", "HS256", issues)
    _validate_integer(config, "ACCESS_EXPIRE", 1, 1440, issues)
    _validate_integer(config, "REFRESH_EXPIRE", 1, 90, issues)
    _validate_integer(config, "PASSWORD_RESET_EXPIRE_MINUTES", 1, 30, issues)
    _validate_integer(config, "HAGENT_CHECKPOINT_TTL_SECONDS", 3600, 7776000, issues)
    _validate_integer(config, "HAGENT_EVENT_RETENTION_DAYS", 1, 90, issues)
    _validate_integer(config, "HAGENT_ARTIFACT_RETENTION_DAYS", 1, 3650, issues)
    if issues:
        raise ConfigValidationError(issues)
    return config


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template",
        action="store_true",
        help="Cho phép placeholder nhưng vẫn kiểm tra cấu trúc và policy.",
    )
    parser.add_argument("env_file", type=Path)
    args = parser.parse_args(argv)
    try:
        raw = load_env_file(args.env_file)
        resolved = validate_config(raw, template=args.template)
    except ConfigValidationError as exc:
        print("Cấu hình server không hợp lệ:", file=sys.stderr)
        for issue in exc.issues:
            print(f"- {issue}", file=sys.stderr)
        return 1
    print(f"OK: cấu hình server {resolved['DEPLOY_MODE']} hợp lệ.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
