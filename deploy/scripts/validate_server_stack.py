"""Kiểm tra cấu trúc production Compose theo contract Azure private-first."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from validate_server_config import ConfigValidationError, load_env_file, validate_config

_APP_SERVICES = frozenset(
    {
        "frontend",
        "toolkit",
        "worker",
        "hagent_bridge",
        "credential_init",
        "mongo",
        "mongo_provision",
        "kafka",
        "minio",
        "minio_provision",
    }
)
_GATEWAY_BY_MODE = {"private": "caddy_private", "public": "caddy_public"}
_EXPECTED_VOLUMES = frozenset(
    {
        "backend_data",
        "bridge_data",
        "caddy_config",
        "caddy_data",
        "kafka_data",
        "minio_bootstrap_secrets",
        "minio_data",
        "minio_provision_state",
        "mongo_bootstrap_secrets",
        "mongo_data",
    }
)
_EXPECTED_VOLUME_DEFINITIONS = {
    volume_name: {"name": f"hagent-server_{volume_name}"}
    for volume_name in _EXPECTED_VOLUMES
}
_EXPECTED_NETWORK_DEFINITIONS = {
    "app": {"name": "hagent-server_app", "driver": "bridge", "ipam": {}},
    "data": {
        "name": "hagent-server_data",
        "driver": "bridge",
        "ipam": {},
        "internal": True,
    },
    "edge": {"name": "hagent-server_edge", "driver": "bridge", "ipam": {}},
}
_ALLOWED_STACK_FIELDS = frozenset(
    {
        "name",
        "networks",
        "services",
        "volumes",
        "x-app-security",
        "x-caddy-service",
        "x-logging",
    }
)
_BASE_SERVICE_FIELDS = frozenset(
    {
        "command",
        "entrypoint",
        "image",
        "init",
        "logging",
        "restart",
        "security_opt",
        "stop_grace_period",
    }
)
_EXPECTED_SERVICE_FIELDS = {
    "caddy_private": _BASE_SERVICE_FIELDS
    | {
        "cap_add",
        "cap_drop",
        "depends_on",
        "environment",
        "healthcheck",
        "networks",
        "ports",
        "profiles",
        "read_only",
        "tmpfs",
        "volumes",
    },
    "caddy_public": _BASE_SERVICE_FIELDS
    | {
        "cap_add",
        "cap_drop",
        "depends_on",
        "environment",
        "healthcheck",
        "networks",
        "ports",
        "profiles",
        "read_only",
        "tmpfs",
        "volumes",
    },
    "credential_init": _BASE_SERVICE_FIELDS
    | {"cap_drop", "network_mode", "read_only", "tmpfs", "volumes"},
    "frontend": _BASE_SERVICE_FIELDS
    | {
        "cap_drop",
        "depends_on",
        "environment",
        "expose",
        "healthcheck",
        "networks",
        "pull_policy",
        "read_only",
        "tmpfs",
    },
    "toolkit": _BASE_SERVICE_FIELDS
    | {
        "cap_drop",
        "depends_on",
        "environment",
        "expose",
        "healthcheck",
        "networks",
        "pull_policy",
        "read_only",
        "tmpfs",
        "volumes",
    },
    "worker": _BASE_SERVICE_FIELDS
    | {
        "cap_drop",
        "depends_on",
        "environment",
        "expose",
        "healthcheck",
        "networks",
        "pull_policy",
        "read_only",
        "tmpfs",
        "volumes",
    },
    "hagent_bridge": _BASE_SERVICE_FIELDS
    | {
        "cap_drop",
        "depends_on",
        "environment",
        "expose",
        "healthcheck",
        "networks",
        "pull_policy",
        "read_only",
        "tmpfs",
        "volumes",
    },
    "mongo": _BASE_SERVICE_FIELDS
    | {"depends_on", "expose", "healthcheck", "networks", "tmpfs", "volumes"},
    "mongo_provision": _BASE_SERVICE_FIELDS
    | {
        "cap_drop",
        "depends_on",
        "environment",
        "networks",
        "read_only",
        "tmpfs",
        "volumes",
    },
    "kafka": _BASE_SERVICE_FIELDS
    | {"environment", "expose", "healthcheck", "networks", "volumes"},
    "minio": _BASE_SERVICE_FIELDS
    | {"depends_on", "expose", "healthcheck", "networks", "volumes"},
    "minio_provision": _BASE_SERVICE_FIELDS
    | {
        "cap_drop",
        "depends_on",
        "environment",
        "networks",
        "read_only",
        "tmpfs",
        "volumes",
    },
}
_SCRIPTED_SERVICES = frozenset(
    {"credential_init", "minio", "minio_provision", "mongo", "mongo_provision"}
)
_EXPECTED_RAW_SERVICE_FIELDS = {
    service_name: (
        (set(fields) - {"command", "entrypoint"})
        | ({"command", "entrypoint"} if service_name in _SCRIPTED_SERVICES else set())
        | ({"command"} if service_name == "worker" else set())
    )
    for service_name, fields in _EXPECTED_SERVICE_FIELDS.items()
}
_EXPECTED_RAW_GATEWAY_PORTS = {
    "caddy_private": [
        {
            "target": 80,
            "published": "${GATEWAY_HTTP_PORT:?GATEWAY_HTTP_PORT is required}",
            "host_ip": "${GATEWAY_BIND_IP:?GATEWAY_BIND_IP is required}",
            "protocol": "tcp",
        }
    ],
    "caddy_public": [
        {
            "target": 80,
            "published": "${GATEWAY_HTTP_PORT:?GATEWAY_HTTP_PORT is required}",
            "host_ip": "${GATEWAY_BIND_IP:?GATEWAY_BIND_IP is required}",
            "protocol": "tcp",
        },
        {
            "target": 443,
            "published": "443",
            "host_ip": "${GATEWAY_BIND_IP:?GATEWAY_BIND_IP is required}",
            "protocol": "tcp",
        },
    ],
}
_INFRA_IMAGES = {
    "caddy_private": "caddy:2.11.4-alpine@sha256:5f5c8640aae01df9654968d946d8f1a56c497f1dd5c5cda4cf95ab7c14d58648",
    "caddy_public": "caddy:2.11.4-alpine@sha256:5f5c8640aae01df9654968d946d8f1a56c497f1dd5c5cda4cf95ab7c14d58648",
    "credential_init": "mongo:7.0.16@sha256:c630c59342c1493d50345136df2af14a76b9e827dd5316bfabee07a0880a5f3a",
    "mongo": "mongo:7.0.16@sha256:c630c59342c1493d50345136df2af14a76b9e827dd5316bfabee07a0880a5f3a",
    "mongo_provision": "mongo:7.0.16@sha256:c630c59342c1493d50345136df2af14a76b9e827dd5316bfabee07a0880a5f3a",
    "kafka": "apache/kafka:3.9.2@sha256:05b4616e0702ef2729327705d54ad6b50ea70b271c4b730fabd2320789fb7b02",
    "minio": "minio/minio:RELEASE.2025-09-07T16-13-09Z@sha256:14cea493d9a34af32f524e538b8346cf79f3321eff8e708c1e2960462bd8936e",
    "minio_provision": "minio/mc:RELEASE.2025-08-13T08-35-41Z@sha256:a7fe349ef4bd8521fb8497f55c6042871b2ae640607cf99d9bede5e9bdf11727",
}
_IMAGE_KEYS = {
    "frontend": "FRONTEND_IMAGE",
    "toolkit": "TOOLKIT_IMAGE",
    "worker": "WORKER_IMAGE",
    "hagent_bridge": "BRIDGE_IMAGE",
}
_EXPECTED_NETWORK_MEMBERSHIP = {
    "frontend": {"app", "edge"},
    "toolkit": {"app", "data", "edge"},
    "worker": {"app", "data"},
    "hagent_bridge": {"app", "data"},
    "credential_init": set(),
    "mongo": {"data"},
    "mongo_provision": {"data"},
    "kafka": {"data"},
    "minio": {"data"},
    "minio_provision": {"data"},
    "caddy_private": {"edge"},
    "caddy_public": {"edge"},
}
_EXPECTED_ENV_KEYS = {
    "frontend": {
        "AUTH_API_BASE_URL",
        "HAGENT_INTERNAL_URL",
        "NEXTAUTH_SECRET",
        "NEXTAUTH_URL",
    },
    "toolkit": {
        "ACCESS_EXPIRE",
        "ALGORITHM",
        "ANTHROPIC_API_KEY",
        "APP_ORIGIN",
        "BACKEND_RELOAD",
        "DEPLOY_MODE",
        "FRONTEND_URL",
        "HAGENT_ARTIFACT_RETENTION_DAYS",
        "HAGENT_CHECKPOINT_BACKEND",
        "HAGENT_CHECKPOINT_TTL_SECONDS",
        "HAGENT_EVENT_RETENTION_DAYS",
        "HAGENT_RUNTIME_DB_NAME",
        "HAGENT_RUNTIME_MODE",
        "KAFKA_SERVER",
        "KAFKA_TOPIC",
        "LLM_DEFAULT_MODEL",
        "LOCAL_API_KEY",
        "LOCAL_BASE_URL",
        "LOCAL_MODEL_NAME",
        "LOGO",
        "MAIL_PASSWORD",
        "MAIL_USERNAME",
        "MINIO_ACCESS_KEY",
        "MINIO_ENDPOINT",
        "MINIO_SECRET_KEY",
        "MINIO_SECURE",
        "MONGODB_CONNECT",
        "MONGODB_DB_NAME",
        "NUMBER_WORKERS",
        "OLLAMA_BASE_URL",
        "OPENAI_API_KEY",
        "PASSWORD_RESET_EXPIRE_MINUTES",
        "REDIRECT_URI",
        "REFRESH_EXPIRE",
        "SECRET_KEY",
        "SESSION_HTTPS_ONLY",
        "SKIP_EMAIL_VERIFICATION",
        "SUPER_SECRET_KEY",
        "WORKER_LIST",
    },
    "worker": {
        "HOST_BACK_END",
        "KAFKA_SERVER",
        "KAFKA_TOPIC",
        "MINIO_ACCESS_KEY",
        "MINIO_ENDPOINT",
        "MINIO_SECRET_KEY",
        "MINIO_SECURE",
        "MONGODB_CONNECT",
        "MONGODB_DB_NAME",
        "PORT_BACK_END",
        "WORKER_HOST",
        "WORKER_PORT",
    },
    "hagent_bridge": {
        "ALGORITHM",
        "ANTHROPIC_API_KEY",
        "HAGENT_RUN_API_URL",
        "HAUTOML_BASE_URL",
        "LLM_DEFAULT_MODEL",
        "LOCAL_API_KEY",
        "LOCAL_BASE_URL",
        "LOCAL_MODEL_NAME",
        "MONGODB_CONNECT",
        "MONGODB_DB_NAME",
        "OLLAMA_BASE_URL",
        "OPENAI_API_KEY",
        "SECRET_KEY",
    },
    "credential_init": set(),
    "mongo": set(),
    "mongo_provision": {
        "HAGENT_RUNTIME_DB_NAME",
        "MONGO_APP_PASSWORD",
        "MONGO_APP_USERNAME",
        "MONGODB_DB_NAME",
    },
    "kafka": {
        "KAFKA_ADVERTISED_LISTENERS",
        "KAFKA_CONTROLLER_LISTENER_NAMES",
        "KAFKA_CONTROLLER_QUORUM_VOTERS",
        "KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS",
        "KAFKA_LISTENERS",
        "KAFKA_LISTENER_SECURITY_PROTOCOL_MAP",
        "KAFKA_NODE_ID",
        "KAFKA_NUM_PARTITIONS",
        "KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR",
        "KAFKA_PROCESS_ROLES",
        "KAFKA_TRANSACTION_STATE_LOG_MIN_ISR",
        "KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR",
    },
    "minio": set(),
    "minio_provision": {
        "MC_CONFIG_DIR",
        "MINIO_APP_ACCESS_KEY",
        "MINIO_APP_SECRET_KEY",
    },
    "caddy_private": {"SITE_ADDRESS"},
    "caddy_public": {"SITE_ADDRESS"},
}
_PROFILE_ENV_DIFFS = {
    "frontend": {"NEXTAUTH_URL"},
    "toolkit": {
        "APP_ORIGIN",
        "DEPLOY_MODE",
        "FRONTEND_URL",
        "LOGO",
        "REDIRECT_URI",
        "SESSION_HTTPS_ONLY",
    },
}
_CONFIG_ENV_ALIASES = {
    ("mongo_provision", "MONGO_APP_USERNAME"): "MONGO_ROOT_USERNAME",
    ("mongo_provision", "MONGO_APP_PASSWORD"): "MONGO_ROOT_PASSWORD",
    ("minio_provision", "MINIO_APP_ACCESS_KEY"): "MINIO_ACCESS_KEY",
    ("minio_provision", "MINIO_APP_SECRET_KEY"): "MINIO_SECRET_KEY",
}
_FIXED_ENV_VALUES = {
    "minio_provision": {"MC_CONFIG_DIR": "/tmp/mc"},
    "toolkit": {
        "BACKEND_RELOAD": "false",
        "NUMBER_WORKERS": "1",
        "WORKER_LIST": "http://worker:5101",
    },
    "worker": {
        "HOST_BACK_END": "toolkit",
        "PORT_BACK_END": "8585",
        "WORKER_HOST": "worker",
        "WORKER_PORT": "5101",
    },
    "kafka": {
        "KAFKA_ADVERTISED_LISTENERS": "PLAINTEXT://kafka:9092",
        "KAFKA_CONTROLLER_LISTENER_NAMES": "CONTROLLER",
        "KAFKA_CONTROLLER_QUORUM_VOTERS": "1@kafka:9093",
        "KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS": "0",
        "KAFKA_LISTENERS": "PLAINTEXT://:9092,CONTROLLER://:9093",
        "KAFKA_LISTENER_SECURITY_PROTOCOL_MAP": (
            "CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT"
        ),
        "KAFKA_NODE_ID": "1",
        "KAFKA_NUM_PARTITIONS": "1",
        "KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR": "1",
        "KAFKA_PROCESS_ROLES": "broker,controller",
        "KAFKA_TRANSACTION_STATE_LOG_MIN_ISR": "1",
        "KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR": "1",
    },
}
_DEFAULT_LOGGING = {
    "driver": "json-file",
    "options": {"max-file": "3", "max-size": "10m"},
}
_ONE_SHOT_SERVICES = frozenset(
    {"credential_init", "minio_provision", "mongo_provision"}
)
_EXPECTED_STOP_GRACE = {
    "caddy_private": "30s",
    "caddy_public": "30s",
    "credential_init": "30s",
    "frontend": "30s",
    "toolkit": "3m0s",
    "worker": "3m0s",
    "hagent_bridge": "1m0s",
    "mongo": "1m30s",
    "mongo_provision": "30s",
    "kafka": "1m30s",
    "minio": "1m0s",
    "minio_provision": "30s",
}
_EXPECTED_MOUNTS = {
    "caddy_private": {
        ("bind", "Caddyfile", "/etc/caddy/Caddyfile", True),
        ("volume", "caddy_data", "/data", False),
        ("volume", "caddy_config", "/config", False),
    },
    "caddy_public": {
        ("bind", "Caddyfile", "/etc/caddy/Caddyfile", True),
        ("volume", "caddy_data", "/data", False),
        ("volume", "caddy_config", "/config", False),
    },
    "credential_init": {
        ("volume", "mongo_bootstrap_secrets", "/run/mongo-bootstrap", False),
        ("volume", "minio_bootstrap_secrets", "/run/minio-bootstrap", False),
    },
    "frontend": set(),
    "toolkit": {("volume", "backend_data", "/var/lib/hagent", False)},
    "worker": {("volume", "backend_data", "/var/lib/hagent", False)},
    "hagent_bridge": {("volume", "bridge_data", "/var/lib/hagent", False)},
    "mongo": {
        ("volume", "mongo_data", "/data/db", False),
        ("volume", "mongo_bootstrap_secrets", "/run/mongo-bootstrap", True),
    },
    "mongo_provision": {
        ("volume", "mongo_bootstrap_secrets", "/run/mongo-bootstrap", True)
    },
    "kafka": {("volume", "kafka_data", "/var/lib/kafka/data", False)},
    "minio": {
        ("volume", "minio_data", "/data", False),
        ("volume", "minio_bootstrap_secrets", "/run/minio-bootstrap", True),
    },
    "minio_provision": {
        ("volume", "minio_bootstrap_secrets", "/run/minio-bootstrap", True),
        ("volume", "minio_provision_state", "/var/lib/hagent-provision", False),
    },
}
_EXPECTED_HEALTH_TESTS = {
    "caddy_private": [
        "CMD",
        "caddy",
        "validate",
        "--config",
        "/etc/caddy/Caddyfile",
        "--adapter",
        "caddyfile",
    ],
    "caddy_public": [
        "CMD",
        "caddy",
        "validate",
        "--config",
        "/etc/caddy/Caddyfile",
        "--adapter",
        "caddyfile",
    ],
    "frontend": [
        "CMD",
        "node",
        "-e",
        (
            "fetch('http://127.0.0.1:3000/hagent').then((response) => { "
            "if (!response.ok) process.exit(1); }).catch(() => process.exit(1));"
        ),
    ],
    "toolkit": [
        "CMD",
        "python",
        "-c",
        (
            "import urllib.request; "
            "urllib.request.urlopen('http://127.0.0.1:8585/ready', timeout=3).close()"
        ),
    ],
    "worker": [
        "CMD",
        "python",
        "-c",
        (
            "import urllib.request; "
            "urllib.request.urlopen('http://127.0.0.1:5101/health', timeout=3).close()"
        ),
    ],
    "hagent_bridge": [
        "CMD",
        "python",
        "-c",
        (
            "import urllib.request; urllib.request.urlopen( "
            "'http://127.0.0.1:9900/api/v1/ready', timeout=3).close()"
        ),
    ],
    "mongo": [
        "CMD-SHELL",
        (
            'mongosh --quiet --host 127.0.0.1 --username "$$(cat '
            '/run/mongo-bootstrap/root-username)" --password "$$(cat '
            '/run/mongo-bootstrap/root-password)" --authenticationDatabase admin '
            '--eval "quit(db.adminCommand({ ping: 1 }).ok ? 0 : 2)"'
        ),
    ],
    "kafka": [
        "CMD-SHELL",
        (
            "/opt/kafka/bin/kafka-topics.sh --bootstrap-server 127.0.0.1:9092 "
            "--list >/dev/null 2>&1"
        ),
    ],
    "minio": [
        "CMD-SHELL",
        "curl --fail --silent --show-error http://127.0.0.1:9000/minio/health/ready",
    ],
}
_EXPECTED_HEALTH_TIMING = {
    "caddy_private": ("30s", "5s", 3, "10s"),
    "caddy_public": ("30s", "5s", 3, "10s"),
    "frontend": ("30s", "5s", 3, "30s"),
    "toolkit": ("30s", "5s", 5, "1m0s"),
    "worker": ("30s", "5s", 3, "30s"),
    "hagent_bridge": ("30s", "5s", 3, "30s"),
    "mongo": ("10s", "5s", 10, "20s"),
    "kafka": ("10s", "10s", 12, "30s"),
    "minio": ("10s", "5s", 10, "20s"),
}
_EXPECTED_COMMAND_SHA256 = {
    "credential_init": "95075df0422d40600a436bbd4dd2bcdfc774f67ab6044add639e021d48253bcf",
    "minio": "de5798ac42f88d9696c477d752ba9dd597270004dd6bf301e86687ccccd424df",
    "minio_provision": "5ef505c5e74a55da257dcc92961a6d24fdb14c721f498e89f6d0c809760c5814",
    "mongo": "82092f7b0d66408869811ae33b37953545db02ed793637641336e05616beb9b2",
    "mongo_provision": "ee48ed32e97aa157fbb3748dc073fedc7a954f8fa1ff84a931179a5345c0439f",
}
_EXPECTED_ENTRYPOINTS = {
    "caddy_private": None,
    "caddy_public": None,
    "credential_init": ["/bin/bash", "-euc"],
    "frontend": None,
    "hagent_bridge": None,
    "kafka": None,
    "minio": ["/bin/sh", "-euc"],
    "minio_provision": ["/bin/sh", "-euc"],
    "mongo": ["/bin/bash", "-euc"],
    "mongo_provision": ["/bin/bash", "-euc"],
    "toolkit": None,
    "worker": None,
}
_EXPECTED_TMPFS = {
    "caddy_private": ["/tmp:rw,nosuid,nodev,size=32m"],
    "caddy_public": ["/tmp:rw,nosuid,nodev,size=32m"],
    "credential_init": ["/tmp:rw,nosuid,nodev,size=16m"],
    "frontend": [
        "/tmp:rw,nosuid,nodev,size=64m",
        "/app/.next/cache:rw,nosuid,nodev,size=128m,uid=1000,gid=1000",
    ],
    "hagent_bridge": ["/tmp:rw,nosuid,nodev,size=128m,uid=10002,gid=10002"],
    "kafka": None,
    "minio": None,
    "minio_provision": ["/tmp:rw,nosuid,nodev,size=32m"],
    "mongo": ["/tmp:rw,nosuid,nodev,size=32m"],
    "mongo_provision": ["/tmp:rw,nosuid,nodev,size=32m"],
    "toolkit": ["/tmp:rw,nosuid,nodev,size=512m,uid=10001,gid=10001"],
    "worker": ["/tmp:rw,nosuid,nodev,size=2g,uid=10001,gid=10001"],
}
_OPTIONAL_INTERPOLATION_KEYS = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "LOCAL_API_KEY",
        "LOCAL_BASE_URL",
        "LOCAL_MODEL_NAME",
        "OLLAMA_BASE_URL",
        "OPENAI_API_KEY",
    }
)
_FORBIDDEN_COMMAND_PATTERN = re.compile(
    r"(?:^|\s)(?:--reload|next\s+dev|npm\s+run\s+dev)(?:\s|$)",
    re.IGNORECASE,
)


class StackValidationError(ValueError):
    """Lỗi stack đã khử giá trị cấu hình và secret."""

    def __init__(self, issues: str | Sequence[str]) -> None:
        normalized = (issues,) if isinstance(issues, str) else tuple(issues)
        self.issues = normalized
        super().__init__("; ".join(normalized))


def _as_mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def render_compose(*, compose_file: Path, env_file: Path) -> dict[str, Any]:
    """Render Compose bằng CLI thật nhưng không kết nối Docker Engine."""

    command = [
        "docker",
        "compose",
        "--env-file",
        str(env_file.resolve()),
        "-f",
        str(compose_file.resolve()),
        "config",
        "--format",
        "json",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=compose_file.parent,
            capture_output=True,
            check=False,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise StackValidationError("Không chạy được Docker Compose CLI") from exc
    if completed.returncode != 0:
        raise StackValidationError("Docker Compose không render được stack")
    try:
        rendered = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise StackValidationError("Docker Compose trả cấu hình không hợp lệ") from exc
    if not isinstance(rendered, dict):
        raise StackValidationError("Docker Compose trả cấu hình không hợp lệ")
    return rendered


def _has_pinned_tag(image: object) -> bool:
    if not isinstance(image, str) or not image:
        return False
    leaf = image.rsplit("/", 1)[-1]
    if leaf.lower() == "latest" or leaf.lower().startswith("latest@"):
        return False
    if ":latest" in leaf.lower():
        return False
    return ":" in leaf or "@sha256:" in leaf


def _validate_gateway(
    services: Mapping[str, Any], deploy_mode: str, issues: list[str]
) -> None:
    gateway_name = _GATEWAY_BY_MODE[deploy_mode]
    gateway = _as_mapping(services.get(gateway_name))
    expected_ports = (
        [
            {
                "mode": "ingress",
                "protocol": "tcp",
                "target": 80,
                "published": "8080",
                "host_ip": "127.0.0.1",
            }
        ]
        if deploy_mode == "private"
        else [
            {
                "mode": "ingress",
                "protocol": "tcp",
                "target": 80,
                "published": "80",
                "host_ip": "0.0.0.0",
            },
            {
                "mode": "ingress",
                "protocol": "tcp",
                "target": 443,
                "published": "443",
                "host_ip": "0.0.0.0",
            },
        ]
    )
    if gateway.get("ports") != expected_ports:
        issues.append(f"{gateway_name}: host port không khớp profile")
    for service_name, service_value in services.items():
        service = _as_mapping(service_value)
        if service_name != gateway_name and service.get("ports"):
            issues.append("Chỉ gateway Caddy được publish host port")


def _validate_service_baseline(
    service_name: str, service: Mapping[str, Any], issues: list[str]
) -> None:
    if set(service) != _EXPECTED_SERVICE_FIELDS[service_name]:
        issues.append(f"{service_name}: field Compose không đúng contract")
    process_kind = "provision" if service_name in _ONE_SHOT_SERVICES else "runtime"
    if service.get("init") is not True:
        issues.append(f"{service_name}: init process không đúng contract")
    if service.get("entrypoint") != _EXPECTED_ENTRYPOINTS[service_name]:
        issues.append(f"{service_name}: entrypoint {process_kind} không đúng contract")
    if service.get("tmpfs") != _EXPECTED_TMPFS[service_name]:
        issues.append(f"{service_name}: tmpfs không đúng contract")
    command = service.get("command")
    expected_digest = _EXPECTED_COMMAND_SHA256.get(service_name)
    if expected_digest is not None:
        command_items = command if isinstance(command, list) else [str(command)]
        command_digest = hashlib.sha256(
            "\0".join(str(item) for item in command_items).encode("utf-8")
        ).hexdigest()
        if command_digest != expected_digest:
            issues.append(f"{service_name}: command {process_kind} không đúng contract")
    elif service_name == "worker":
        if command != [
            "uvicorn",
            "cluster.worker:app",
            "--host",
            "0.0.0.0",
            "--port",
            "5101",
        ]:
            issues.append("worker: command runtime không đúng contract")
    elif command is not None:
        issues.append(f"{service_name}: không được override image command")
    if "build" in service:
        issues.append(f"{service_name}: không được build image tại server")
    image = service.get("image")
    if not _has_pinned_tag(image):
        issues.append(f"{service_name}: image phải pin tag hoặc digest")
    if isinstance(image, str) and ":latest" in image.lower():
        issues.append(f"{service_name}: không được dùng latest")
    expected_restart = (
        "on-failure:3" if service_name in _ONE_SHOT_SERVICES else "unless-stopped"
    )
    if service.get("restart") != expected_restart:
        issues.append(f"{service_name}: restart policy không hợp lệ")
    if service.get("stop_grace_period") != _EXPECTED_STOP_GRACE[service_name]:
        issues.append(f"{service_name}: stop grace period không hợp lệ")
    if service.get("logging") != _DEFAULT_LOGGING:
        issues.append(f"{service_name}: logging chưa có giới hạn")
    healthcheck = _as_mapping(service.get("healthcheck"))
    if service_name in _ONE_SHOT_SERVICES:
        if healthcheck:
            issues.append(f"{service_name}: init job không dùng healthcheck giả")
    else:
        test = healthcheck.get("test")
        test_items = test if isinstance(test, list) else []
        expected_timing = _EXPECTED_HEALTH_TIMING[service_name]
        actual_timing = (
            healthcheck.get("interval"),
            healthcheck.get("timeout"),
            healthcheck.get("retries"),
            healthcheck.get("start_period"),
        )
        if (
            test_items != _EXPECTED_HEALTH_TESTS[service_name]
            or actual_timing != expected_timing
            or healthcheck.get("disable") is True
        ):
            issues.append(f"{service_name}: phải có healthcheck đúng contract")
    if service.get("privileged") is True:
        issues.append(f"{service_name}: không được privileged")
    if service.get("user") is not None:
        issues.append(f"{service_name}: không được override image user")
    if set(service.get("security_opt") or []) != {"no-new-privileges:true"}:
        issues.append(f"{service_name}: security_opt không đúng allowlist")
    if service.get("volumes_from"):
        issues.append(f"{service_name}: không được kế thừa volume từ service khác")
    if service.get("configs") or service.get("secrets") or service.get("env_file"):
        issues.append(f"{service_name}: không được thêm config/secret/env_file mount")
    if service.get("use_api_socket"):
        issues.append(f"{service_name}: không được dùng Docker API socket")
    if service.get("devices") or service.get("device_cgroup_rules"):
        issues.append(f"{service_name}: không được cấp device host")
    namespace_fields = ("pid", "ipc", "uts", "cgroup", "userns_mode", "cgroup_parent")
    if any(service.get(field) for field in namespace_fields):
        issues.append(f"{service_name}: không được dùng namespace host/tùy chỉnh")
    if service_name == "credential_init" and service.get("network_mode") != "none":
        issues.append("credential_init: network_mode phải là none")
    elif service_name != "credential_init" and service.get("network_mode") is not None:
        issues.append(f"{service_name}: không được override network_mode")
    if service_name not in _GATEWAY_BY_MODE.values() and service.get("cap_add"):
        issues.append(f"{service_name}: không được thêm Linux capability")
    if service.get("container_name"):
        issues.append(f"{service_name}: không được khóa container_name")
    if service.get("extra_hosts"):
        issues.append(f"{service_name}: không được thêm host mapping")
    security_options = set(service.get("security_opt") or [])
    if "no-new-privileges:true" not in security_options:
        issues.append(f"{service_name}: thiếu no-new-privileges")
    command = service.get("command", [])
    command_text = " ".join(command) if isinstance(command, list) else str(command)
    if _FORBIDDEN_COMMAND_PATTERN.search(command_text):
        issues.append(f"{service_name}: không được chạy dev/reload")
    if service_name in _IMAGE_KEYS and service.get("pull_policy") != "never":
        issues.append(f"{service_name}: app image phải dùng pull_policy never")


def _validate_mounts(
    service_name: str, service: Mapping[str, Any], issues: list[str]
) -> None:
    actual_mounts: list[tuple[str, str, str, bool]] = []
    for mount_value in service.get("volumes") or []:
        mount = _as_mapping(mount_value)
        source = str(mount.get("source", "")).replace("\\", "/").lower()
        target = str(mount.get("target", "")).replace("\\", "/").lower()
        normalized_source = (
            "Caddyfile" if source.endswith("/deploy/caddyfile") else source
        )
        actual_mounts.append(
            (
                str(mount.get("type", "")),
                normalized_source,
                str(mount.get("target", "")),
                mount.get("read_only") is True,
            )
        )
        if "docker.sock" in source or "docker.sock" in target:
            issues.append(f"{service_name}: không được mount Docker socket")
        if mount.get("type") != "bind":
            continue
        allowed_caddyfile = (
            service_name in _GATEWAY_BY_MODE.values()
            and source.endswith("/deploy/caddyfile")
            and target == "/etc/caddy/caddyfile"
            and mount.get("read_only") is True
        )
        if not allowed_caddyfile:
            issues.append(f"{service_name}: không được bind-mount source")
    expected_mounts = _EXPECTED_MOUNTS[service_name]
    if (
        len(actual_mounts) != len(expected_mounts)
        or set(actual_mounts) != expected_mounts
    ):
        issues.append(f"{service_name}: durable mount contract không hợp lệ")


def _validate_networks_and_environment(
    service_name: str,
    service: Mapping[str, Any],
    expected_config: Mapping[str, str] | None,
    issues: list[str],
) -> None:
    networks = set(_as_mapping(service.get("networks")))
    if networks != _EXPECTED_NETWORK_MEMBERSHIP[service_name]:
        issues.append(f"{service_name}: network membership không hợp lệ")
    environment = _as_mapping(service.get("environment"))
    if set(environment) != _EXPECTED_ENV_KEYS[service_name]:
        issues.append(f"{service_name}: environment key không khớp allowlist")
        return
    if expected_config is None:
        return
    fixed_values = _FIXED_ENV_VALUES.get(service_name, {})
    for environment_key, value in environment.items():
        source_key = _CONFIG_ENV_ALIASES.get(
            (service_name, environment_key), environment_key
        )
        if source_key in expected_config:
            if value != expected_config[source_key]:
                issues.append(
                    f"{service_name}.{environment_key}: không lấy đúng từ env contract"
                )
        elif fixed_values.get(environment_key) != value:
            issues.append(
                f"{service_name}.{environment_key}: giá trị cố định không hợp lệ"
            )


def _validate_images(
    services: Mapping[str, Any],
    expected_config: Mapping[str, str] | None,
    issues: list[str],
) -> None:
    for service_name, expected_image in _INFRA_IMAGES.items():
        if (
            service_name in services
            and _as_mapping(services[service_name]).get("image") != expected_image
        ):
            issues.append(f"{service_name}: infra image không đúng bản pin")
    if expected_config is None:
        return
    for service_name, image_key in _IMAGE_KEYS.items():
        if _as_mapping(services.get(service_name)).get("image") != expected_config.get(
            image_key
        ):
            issues.append(f"{service_name}: app image không khớp release config")


def _validate_app_security(
    services: Mapping[str, Any], gateway_name: str, issues: list[str]
) -> None:
    for service_name in ("frontend", "toolkit", "worker", "hagent_bridge"):
        service = _as_mapping(services.get(service_name))
        if service.get("read_only") is not True:
            issues.append(f"{service_name}: root filesystem phải read-only")
        if set(service.get("cap_drop") or []) != {"ALL"} or service.get("cap_add"):
            issues.append(f"{service_name}: capability allowlist không hợp lệ")
    for service_name in _ONE_SHOT_SERVICES:
        service = _as_mapping(services.get(service_name))
        if (
            service.get("read_only") is not True
            or set(service.get("cap_drop") or []) != {"ALL"}
            or service.get("cap_add")
        ):
            issues.append(f"{service_name}: init job chưa đạt quyền tối thiểu")
    if _as_mapping(services.get(gateway_name)).get("read_only") is not True:
        issues.append(f"{gateway_name}: root filesystem phải read-only")
    gateway = _as_mapping(services.get(gateway_name))
    if set(gateway.get("cap_drop") or []) != {"ALL"} or set(
        gateway.get("cap_add") or []
    ) != {"NET_BIND_SERVICE"}:
        issues.append(f"{gateway_name}: capability allowlist không hợp lệ")


def _validate_dependencies(services: Mapping[str, Any], issues: list[str]) -> None:
    required = {
        "mongo": {"credential_init"},
        "mongo_provision": {"mongo"},
        "minio": {"credential_init"},
        "minio_provision": {"credential_init", "minio"},
        "toolkit": {"kafka", "minio_provision", "mongo_provision"},
        "worker": {"toolkit"},
        "hagent_bridge": {"mongo_provision", "toolkit"},
        "frontend": {"hagent_bridge", "toolkit"},
        "caddy_private": {"frontend", "toolkit"},
        "caddy_public": {"frontend", "toolkit"},
    }
    for service_name, dependencies in required.items():
        if service_name not in services:
            continue
        actual = _as_mapping(_as_mapping(services[service_name]).get("depends_on"))
        conditions = {
            dependency: (
                "service_completed_successfully"
                if dependency in _ONE_SHOT_SERVICES
                else "service_healthy"
            )
            for dependency in dependencies
        }
        if set(actual) != dependencies or any(
            _as_mapping(actual[name]).get("condition") != condition
            or _as_mapping(actual[name]).get("required") is not True
            for name, condition in conditions.items()
        ):
            issues.append(f"{service_name}: dependency readiness không hợp lệ")


def validate_stack(
    stack: Mapping[str, Any],
    *,
    deploy_mode: str,
    expected_config: Mapping[str, str] | None = None,
) -> None:
    """Kiểm tra một stack đã được Docker Compose render."""

    if deploy_mode not in _GATEWAY_BY_MODE:
        raise StackValidationError("DEPLOY_MODE không hợp lệ cho server stack")
    services = _as_mapping(stack.get("services"))
    gateway_name = _GATEWAY_BY_MODE[deploy_mode]
    issues: list[str] = []
    if set(stack) != _ALLOWED_STACK_FIELDS:
        issues.append("Stack có top-level field ngoài allowlist")
    if set(services) != _APP_SERVICES | {gateway_name}:
        issues.append("Danh sách service không khớp profile")
    volumes = _as_mapping(stack.get("volumes"))
    if volumes != _EXPECTED_VOLUME_DEFINITIONS:
        issues.append("Định nghĩa named volume không hợp lệ")
    networks = _as_mapping(stack.get("networks"))
    if networks != _EXPECTED_NETWORK_DEFINITIONS:
        issues.append("Định nghĩa network không hợp lệ")
    if "configs" in stack or "secrets" in stack:
        issues.append("Stack không được khai báo top-level config/secret mount")

    if set(services) == _APP_SERVICES | {gateway_name}:
        for service_name, service_value in services.items():
            service = _as_mapping(service_value)
            _validate_service_baseline(service_name, service, issues)
            _validate_mounts(service_name, service, issues)
            _validate_networks_and_environment(
                service_name, service, expected_config, issues
            )
        _validate_gateway(services, deploy_mode, issues)
        _validate_images(services, expected_config, issues)
        _validate_app_security(services, gateway_name, issues)
        _validate_dependencies(services, issues)
        toolkit = _as_mapping(services["toolkit"])
        replicas = _as_mapping(toolkit.get("deploy")).get("replicas", 1)
        if replicas != 1 or toolkit.get("scale", 1) != 1:
            issues.append("toolkit: v1 chỉ cho một replica")
        if _as_mapping(toolkit.get("environment")).get("NUMBER_WORKERS") != "1":
            issues.append("toolkit: NUMBER_WORKERS phải là 1 trong v1")
    if issues:
        raise StackValidationError(issues)


def _normalized_service_for_pair(
    service_name: str, service: Mapping[str, Any]
) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(service))
    environment = dict(_as_mapping(normalized.get("environment")))
    for key in _PROFILE_ENV_DIFFS.get(service_name, set()):
        environment.pop(key, None)
    normalized["environment"] = environment
    return normalized


def validate_profile_pair(
    private_stack: Mapping[str, Any], public_stack: Mapping[str, Any]
) -> None:
    """Chứng minh chuyển public không thay app image, volume hoặc graph nội bộ."""

    issues: list[str] = []
    private_services = _as_mapping(private_stack.get("services"))
    public_services = _as_mapping(public_stack.get("services"))
    for service_name in _APP_SERVICES:
        private_service = _as_mapping(private_services.get(service_name))
        public_service = _as_mapping(public_services.get(service_name))
        if _normalized_service_for_pair(
            service_name, private_service
        ) != _normalized_service_for_pair(service_name, public_service):
            issues.append(f"{service_name}: thay đổi ngoài runtime edge config")
    if private_stack.get("volumes") != public_stack.get("volumes"):
        issues.append("Private/public không dùng cùng named volume")
    if private_stack.get("networks") != public_stack.get("networks"):
        issues.append("Private/public không dùng cùng network graph")
    private_gateway = copy.deepcopy(
        dict(_as_mapping(private_services.get("caddy_private")))
    )
    public_gateway = copy.deepcopy(
        dict(_as_mapping(public_services.get("caddy_public")))
    )
    for gateway in (private_gateway, public_gateway):
        gateway.pop("ports", None)
        gateway.pop("profiles", None)
        environment = dict(_as_mapping(gateway.get("environment")))
        environment.pop("SITE_ADDRESS", None)
        gateway["environment"] = environment
    if private_gateway != public_gateway:
        issues.append("Private/public không dùng cùng Caddy image/config/volume")
    if issues:
        raise StackValidationError(issues)


def validate_compose_source_text(source_text: str, *, config_keys: set[str]) -> None:
    """Khóa env/image trong nội dung YAML vào interpolation thay vì literal."""
    try:
        source = yaml.safe_load(source_text)
    except yaml.YAMLError as exc:
        raise StackValidationError("Production Compose không phải YAML hợp lệ") from exc
    services = _as_mapping(_as_mapping(source).get("services"))
    issues: list[str] = []
    if set(_as_mapping(source)) != _ALLOWED_STACK_FIELDS:
        issues.append("Raw stack có top-level field ngoài allowlist")
    for service_name, service_value in services.items():
        expected_fields = _EXPECTED_RAW_SERVICE_FIELDS.get(service_name)
        if (
            expected_fields is None
            or set(_as_mapping(service_value)) != expected_fields
        ):
            issues.append(f"{service_name}: raw service field không đúng contract")
    if "configs" in source or "secrets" in source:
        issues.append("Raw stack không được khai báo top-level config/secret mount")
    for service_name, service_value in services.items():
        service = _as_mapping(service_value)
        if service.get("configs") or service.get("secrets") or service.get("env_file"):
            issues.append(
                f"{service_name}: raw service không được thêm config/secret/env_file"
            )
    expected_caddy_bind = {
        "type": "bind",
        "source": "./Caddyfile",
        "target": "/etc/caddy/Caddyfile",
        "read_only": True,
    }
    for gateway_name in _GATEWAY_BY_MODE.values():
        gateway_volumes = _as_mapping(services.get(gateway_name)).get("volumes")
        if (
            not isinstance(gateway_volumes, list)
            or len(gateway_volumes) != 3
            or gateway_volumes[0] != expected_caddy_bind
        ):
            issues.append(f"{gateway_name}: raw Caddy mount không đúng contract")
    for service_name, expected_keys in _EXPECTED_ENV_KEYS.items():
        service = _as_mapping(services.get(service_name))
        environment = _as_mapping(service.get("environment"))
        if set(environment) != expected_keys:
            issues.append(f"{service_name}: raw environment key không khớp allowlist")
            continue
        for environment_key, raw_value in environment.items():
            source_key = _CONFIG_ENV_ALIASES.get(
                (service_name, environment_key), environment_key
            )
            if source_key not in config_keys:
                continue
            if source_key in _OPTIONAL_INTERPOLATION_KEYS:
                valid_interpolation = raw_value == f"${{{source_key}:-}}"
            else:
                valid_interpolation = isinstance(raw_value, str) and bool(
                    re.fullmatch(
                        rf"\$\{{{re.escape(source_key)}:\?[^}}\r\n]+\}}",
                        raw_value,
                    )
                )
            if not valid_interpolation:
                issues.append(
                    f"{service_name}.{environment_key}: phải lấy từ env interpolation"
                )
    for service_name, image_key in _IMAGE_KEYS.items():
        raw_image = _as_mapping(services.get(service_name)).get("image")
        if not isinstance(raw_image, str) or not re.fullmatch(
            rf"\$\{{{re.escape(image_key)}:\?[^}}\r\n]+\}}", raw_image
        ):
            issues.append(f"{service_name}: app image phải lấy từ env interpolation")
    for service_name, expected_image in _INFRA_IMAGES.items():
        if _as_mapping(services.get(service_name)).get("image") != expected_image:
            issues.append(f"{service_name}: raw infra image không đúng digest")
    for service_name, expected_ports in _EXPECTED_RAW_GATEWAY_PORTS.items():
        if _as_mapping(services.get(service_name)).get("ports") != expected_ports:
            issues.append(f"{service_name}: raw gateway port không đúng env contract")
    if issues:
        raise StackValidationError(issues)


def validate_compose_source(compose_file: Path, *, config_keys: set[str]) -> None:
    """Đọc an toàn rồi kiểm tra interpolation trong production Compose."""

    try:
        source_text = compose_file.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise StackValidationError("Không đọc được production Compose") from exc
    validate_compose_source_text(source_text, config_keys=config_keys)


def load_and_validate_stack(
    *, compose_file: Path, env_file: Path, template: bool = False
) -> dict[str, Any]:
    """Kiểm tra env, render Compose và kiểm tra stack theo đúng profile."""

    try:
        resolved = validate_config(load_env_file(env_file), template=template)
    except ConfigValidationError as exc:
        raise StackValidationError(exc.issues) from exc
    validate_compose_source(compose_file, config_keys=set(resolved))
    stack = render_compose(compose_file=compose_file, env_file=env_file)
    validate_stack(
        stack,
        deploy_mode=resolved["DEPLOY_MODE"],
        expected_config=resolved,
    )
    return stack


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument(
        "--compose-file",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docker-compose.server.yaml",
    )
    parser.add_argument(
        "--template",
        action="store_true",
        help="Cho phép placeholder trong env example nhưng vẫn khóa stack contract.",
    )
    args = parser.parse_args(argv)
    try:
        stack = load_and_validate_stack(
            compose_file=args.compose_file,
            env_file=args.env_file,
            template=args.template,
        )
    except StackValidationError as exc:
        print("Server stack không hợp lệ:", file=sys.stderr)
        for issue in exc.issues:
            print(f"- {issue}", file=sys.stderr)
        return 1
    mode = "private" if "caddy_private" in stack["services"] else "public"
    print(f"OK: server stack {mode} hợp lệ.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
