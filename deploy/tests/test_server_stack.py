from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

DEPLOY_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = DEPLOY_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Các validator là script độc lập nên chỉ import được sau khi thêm thư mục scripts.
from validate_server_config import load_env_file, validate_config  # noqa: E402
from validate_server_stack import (  # noqa: E402
    StackValidationError,
    load_and_validate_stack,
    validate_compose_source_text,
    validate_profile_pair,
    validate_stack,
)

COMPOSE_FILE = DEPLOY_DIR / "docker-compose.server.yaml"
PRIVATE_ENV = DEPLOY_DIR / "server.private.env.example"
PUBLIC_ENV = DEPLOY_DIR / "server.public.env.example"

APPLICATION_SERVICES = {
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
NAMED_VOLUMES = {
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
PRIVATE_CONFIG = validate_config(load_env_file(PRIVATE_ENV), template=True)


@pytest.fixture(scope="module")
def rendered_stacks() -> tuple[dict, dict]:
    private = load_and_validate_stack(
        compose_file=COMPOSE_FILE,
        env_file=PRIVATE_ENV,
        template=True,
    )
    public = load_and_validate_stack(
        compose_file=COMPOSE_FILE,
        env_file=PUBLIC_ENV,
        template=True,
    )
    return private, public


def test_private_and_public_keep_same_application_graph(
    rendered_stacks: tuple[dict, dict],
) -> None:
    private, public = rendered_stacks

    assert set(private["services"]) == APPLICATION_SERVICES | {"caddy_private"}
    assert set(public["services"]) == APPLICATION_SERVICES | {"caddy_public"}
    assert set(private["volumes"]) == NAMED_VOLUMES
    assert set(public["volumes"]) == NAMED_VOLUMES
    assert set(private["networks"]) == {"app", "data", "edge"}
    assert set(public["networks"]) == {"app", "data", "edge"}

    validate_profile_pair(private, public)


def test_only_active_caddy_gateway_publishes_host_ports(
    rendered_stacks: tuple[dict, dict],
) -> None:
    private, public = rendered_stacks

    assert private["services"]["caddy_private"]["ports"] == [
        {
            "mode": "ingress",
            "protocol": "tcp",
            "target": 80,
            "published": "8080",
            "host_ip": "127.0.0.1",
        }
    ]
    assert public["services"]["caddy_public"]["ports"] == [
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
    for stack in (private, public):
        published = {
            service_name
            for service_name, service in stack["services"].items()
            if service.get("ports")
        }
        assert published == {
            "caddy_private" if "caddy_private" in stack["services"] else "caddy_public"
        }


@pytest.mark.parametrize(
    ("mutate", "expected_issue"),
    [
        (
            lambda stack: stack["services"]["mongo"].update(
                {"ports": [{"target": 27017, "published": "27017"}]}
            ),
            "Chỉ gateway Caddy được publish host port",
        ),
        (
            lambda stack: stack["services"]["toolkit"].update(
                {"image": "hagent-toolkit:latest"}
            ),
            "không được dùng latest",
        ),
        (
            lambda stack: stack["services"]["toolkit"].update(
                {"build": {"context": "../src/backend"}}
            ),
            "không được build image tại server",
        ),
        (
            lambda stack: stack["services"]["worker"]["volumes"].append(
                {"type": "bind", "source": "../src", "target": "/app/src"}
            ),
            "không được bind-mount source",
        ),
        (
            lambda stack: stack["services"]["hagent_bridge"].pop("healthcheck"),
            "phải có healthcheck",
        ),
        (
            lambda stack: stack["services"]["toolkit"].update(
                {"healthcheck": {"test": ["CMD", "true"]}}
            ),
            "healthcheck đúng contract",
        ),
        (
            lambda stack: stack["services"]["toolkit"]["healthcheck"].update(
                {
                    "test": [
                        "CMD-SHELL",
                        "true # python http://127.0.0.1:8585/ready",
                    ]
                }
            ),
            "healthcheck đúng contract",
        ),
        (
            lambda stack: stack["services"]["mongo"].update({"volumes": []}),
            "durable mount contract",
        ),
        (
            lambda stack: stack["services"]["toolkit"]["environment"].update(
                {"SUPER_SECRET_KEY": "hardcoded-in-compose"}
            ),
            "không lấy đúng từ env contract",
        ),
        (
            lambda stack: stack["services"]["toolkit"].update(
                {"stop_grace_period": "10s"}
            ),
            "stop grace period",
        ),
        (
            lambda stack: stack["services"]["caddy_private"].pop("depends_on"),
            "dependency readiness",
        ),
        (
            lambda stack: stack["services"]["toolkit"].update(
                {"command": ["uvicorn", "server.application:app", "--reload"]}
            ),
            "không được chạy dev/reload",
        ),
        (
            lambda stack: stack["services"]["credential_init"].update(
                {"network_mode": "bridge"}
            ),
            "network_mode phải là none",
        ),
        (
            lambda stack: stack["services"]["credential_init"].update(
                {"command": ["true # flock mv sync"]}
            ),
            "command provision không đúng contract",
        ),
        (
            lambda stack: stack["services"]["minio_provision"].update(
                {"entrypoint": ["/bin/sh", "-c"]}
            ),
            "entrypoint provision không đúng contract",
        ),
        (
            lambda stack: stack["services"]["toolkit"].update(
                {"cap_add": ["SYS_ADMIN"]}
            ),
            "capability allowlist không hợp lệ",
        ),
        (
            lambda stack: stack["services"]["toolkit"].update({"user": "0:10001"}),
            "không được override image user",
        ),
        (
            lambda stack: stack["services"]["toolkit"]["depends_on"][
                "mongo_provision"
            ].update({"required": False}),
            "dependency readiness",
        ),
        (
            lambda stack: stack["services"]["toolkit"].update(
                {"volumes_from": ["mongo:ro"]}
            ),
            "không được kế thừa volume",
        ),
        (
            lambda stack: stack["services"]["toolkit"]["security_opt"].append(
                "seccomp=unconfined"
            ),
            "security_opt không đúng allowlist",
        ),
        (
            lambda stack: stack["services"]["toolkit"].update({"ipc": "host"}),
            "không được dùng namespace",
        ),
        (
            lambda stack: stack["services"]["toolkit"].update(
                {"devices": ["/dev/kmsg:/dev/kmsg"]}
            ),
            "không được cấp device host",
        ),
        (
            lambda stack: stack["services"]["mongo"].update({"use_api_socket": True}),
            "không được dùng Docker API socket",
        ),
        (
            lambda stack: stack["volumes"]["mongo_data"].update(
                {
                    "driver": "local",
                    "driver_opts": {"type": "none", "o": "bind", "device": "/"},
                }
            ),
            "Định nghĩa named volume không hợp lệ",
        ),
        (
            lambda stack: (
                stack.update({"configs": {"evil": {"file": "/tmp/evil"}}}),
                stack["services"]["toolkit"].update({"configs": ["evil"]}),
            ),
            "top-level config/secret mount",
        ),
        (
            lambda stack: stack["services"]["kafka"].update({"user": "root"}),
            "không được override image user",
        ),
        (
            lambda stack: stack["services"]["caddy_private"]["volumes"].append(
                {
                    "type": "bind",
                    "source": "/tmp/deploy/Caddyfile",
                    "target": "/etc/caddy/Caddyfile",
                    "read_only": True,
                }
            ),
            "durable mount contract",
        ),
        (
            lambda stack: stack["services"]["toolkit"].update(
                {
                    "post_start": [
                        {
                            "command": ["sh", "-c", "id"],
                            "user": "root",
                            "privileged": True,
                        }
                    ]
                }
            ),
            "field Compose không đúng contract",
        ),
        (
            lambda stack: stack["services"]["toolkit"].update(
                {
                    "command": [
                        "sh",
                        "-c",
                        "printenv; exec uvicorn server.application:app",
                    ]
                }
            ),
            "không được override image command",
        ),
        (
            lambda stack: stack["services"]["caddy_private"].update(
                {"entrypoint": ["/bin/sh", "-c"]}
            ),
            "entrypoint runtime không đúng contract",
        ),
        (
            lambda stack: stack["services"]["frontend"].update({"init": False}),
            "init process không đúng contract",
        ),
        (
            lambda stack: stack["services"]["toolkit"].update(
                {"tmpfs": ["/tmp:rw,size=8g"]}
            ),
            "tmpfs không đúng contract",
        ),
        (
            lambda stack: stack["services"]["toolkit"].update(
                {"deploy": {"replicas": 1}}
            ),
            "field Compose không đúng contract",
        ),
    ],
)
def test_stack_rejects_unsafe_mutations(
    rendered_stacks: tuple[dict, dict],
    mutate,
    expected_issue: str,
) -> None:
    private, _ = rendered_stacks
    mutated = copy.deepcopy(private)
    mutate(mutated)

    with pytest.raises(StackValidationError, match=expected_issue):
        validate_stack(
            mutated,
            deploy_mode="private",
            expected_config=PRIVATE_CONFIG,
        )


def test_data_network_is_internal_and_services_are_unprivileged(
    rendered_stacks: tuple[dict, dict],
) -> None:
    private, public = rendered_stacks
    for stack in (private, public):
        assert stack["networks"]["data"]["internal"] is True
        for service in stack["services"].values():
            assert service.get("privileged") is not True
            assert service.get("network_mode") != "host"
            assert service.get("pid") != "host"
            assert "/var/run/docker.sock" not in str(service.get("volumes", []))


def test_all_services_have_health_restart_and_bounded_logs(
    rendered_stacks: tuple[dict, dict],
) -> None:
    private, public = rendered_stacks
    for stack in (private, public):
        for service_name, service in stack["services"].items():
            if service_name in {
                "credential_init",
                "minio_provision",
                "mongo_provision",
            }:
                assert "healthcheck" not in service
                assert service["restart"] == "on-failure:3"
            else:
                assert service["healthcheck"]["test"]
                assert service["restart"] == "unless-stopped"
            assert service["stop_grace_period"]
            assert service["logging"] == {
                "driver": "json-file",
                "options": {"max-file": "3", "max-size": "10m"},
            }


def test_profile_pair_rejects_app_image_drift(
    rendered_stacks: tuple[dict, dict],
) -> None:
    private, public = rendered_stacks
    mutated_public = copy.deepcopy(public)
    mutated_public["services"]["toolkit"]["image"] = "hagent-toolkit:other-release"

    with pytest.raises(StackValidationError, match="toolkit: thay đổi"):
        validate_profile_pair(private, mutated_public)


def test_raw_compose_rejects_hardcoded_secret() -> None:
    source = COMPOSE_FILE.read_text(encoding="utf-8")
    unsafe = source.replace(
        "SUPER_SECRET_KEY: ${SUPER_SECRET_KEY:?SUPER_SECRET_KEY is required}",
        "SUPER_SECRET_KEY: hardcoded-in-compose",
        1,
    )
    with pytest.raises(StackValidationError, match="phải lấy từ env interpolation"):
        validate_compose_source_text(unsafe, config_keys=set(PRIVATE_CONFIG))


def test_raw_compose_rejects_secret_fallback() -> None:
    source = COMPOSE_FILE.read_text(encoding="utf-8")
    unsafe = source.replace(
        "SUPER_SECRET_KEY: ${SUPER_SECRET_KEY:?SUPER_SECRET_KEY is required}",
        "SUPER_SECRET_KEY: ${SUPER_SECRET_KEY:-hardcoded-fallback}",
        1,
    )
    with pytest.raises(StackValidationError, match="phải lấy từ env interpolation"):
        validate_compose_source_text(unsafe, config_keys=set(PRIVATE_CONFIG))


def test_raw_compose_rejects_https_port_override() -> None:
    source = COMPOSE_FILE.read_text(encoding="utf-8")
    unsafe = source.replace(
        'published: "443"',
        "published: ${GATEWAY_HTTPS_PORT:-443}",
        1,
    )
    with pytest.raises(StackValidationError, match="raw gateway port"):
        validate_compose_source_text(unsafe, config_keys=set(PRIVATE_CONFIG))


def test_raw_compose_rejects_noncanonical_caddy_mount() -> None:
    source = COMPOSE_FILE.read_text(encoding="utf-8")
    unsafe = source.replace("source: ./Caddyfile", "source: /tmp/deploy/Caddyfile", 1)
    with pytest.raises(StackValidationError, match="raw Caddy mount"):
        validate_compose_source_text(unsafe, config_keys=set(PRIVATE_CONFIG))


def test_datastore_root_secrets_are_isolated_and_rotation_is_reconciled(
    rendered_stacks: tuple[dict, dict],
) -> None:
    private, _ = rendered_stacks
    services = private["services"]

    mongo_sources = {mount["source"] for mount in services["mongo"]["volumes"]}
    mongo_provision_sources = {
        mount["source"] for mount in services["mongo_provision"]["volumes"]
    }
    minio_sources = {mount["source"] for mount in services["minio"]["volumes"]}
    minio_provision_sources = {
        mount["source"] for mount in services["minio_provision"]["volumes"]
    }
    assert mongo_sources == {"mongo_data", "mongo_bootstrap_secrets"}
    assert mongo_provision_sources == {"mongo_bootstrap_secrets"}
    assert minio_sources == {"minio_data", "minio_bootstrap_secrets"}
    assert minio_provision_sources == {
        "minio_bootstrap_secrets",
        "minio_provision_state",
    }

    mongo_command = "\n".join(services["mongo_provision"]["command"])
    assert "updateUser" in mongo_command
    assert "dropUser" in mongo_command
    assert 'managedBy = "hagent-server-v1"' in mongo_command
    minio_command = "\n".join(services["minio_provision"]["command"])
    assert "mc admin user remove" in minio_command
    assert "previous_user" in minio_command
    assert "mv -f" in minio_command
    credential_command = "\n".join(services["credential_init"]["command"])
    assert "flock -w 30" in credential_command
    assert "^[0-9a-f]{64}$$" in credential_command
    assert "mv -f --" in credential_command
    assert (
        services["minio"]["healthcheck"]["test"][1]
        == "curl --fail --silent --show-error "
        "http://127.0.0.1:9000/minio/health/ready"
    )
