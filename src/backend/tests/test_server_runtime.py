from __future__ import annotations

import pytest

from config.server_runtime import (
    CookieRuntimePolicy,
    ServerRuntimeConfigError,
    load_cookie_runtime_policy,
    load_server_runtime_config,
)

PRIVATE_SECRET = "private-runtime-8f92c4ab31d7e650-AZURE"
PUBLIC_SECRET = "public-runtime-4d18b0ce62fa7395-AZURE"


def _private_env(**overrides: str) -> dict[str, str]:
    values = {
        "DEPLOY_MODE": "private",
        "APP_ORIGIN": "http://localhost:8080",
        "SUPER_SECRET_KEY": PRIVATE_SECRET,
        "SESSION_HTTPS_ONLY": "false",
        "BACKEND_RELOAD": "false",
        "HAGENT_RUNTIME_MODE": "legacy",
        "HAGENT_CHECKPOINT_BACKEND": "mongodb",
        "MONGODB_CONNECT": "mongodb://runtime-user:runtime-password@mongo:27017/",
        "HAGENT_RUNTIME_DB_NAME": "hagent_journey",
        "HAGENT_CHECKPOINT_TTL_SECONDS": "2592000",
        "HAGENT_EVENT_RETENTION_DAYS": "30",
        "HAGENT_RUNTIME_SERVER_SELECTION_TIMEOUT_MS": "2000",
        "SERVER_READINESS_TIMEOUT_SECONDS": "3",
    }
    values.update(overrides)
    return values


def _public_env(**overrides: str) -> dict[str, str]:
    values = _private_env()
    values.update(
        {
            "DEPLOY_MODE": "public",
            "APP_ORIGIN": "https://hagent.eastus.cloudapp.azure.com",
            "SUPER_SECRET_KEY": PUBLIC_SECRET,
            "SESSION_HTTPS_ONLY": "true",
            "BACKEND_RELOAD": "false",
        }
    )
    values.update(overrides)
    return values


def test_private_mode_uses_exact_origin_and_insecure_cookie() -> None:
    config = load_server_runtime_config(_private_env())

    assert config.deploy_mode == "private"
    assert config.server_mode is True
    assert config.app_origin == "http://localhost:8080"
    assert config.cors_origins == ("http://localhost:8080",)
    assert config.session_https_only is False
    assert config.reload_enabled is False
    assert config.agent_runtime.mode == "legacy"
    assert config.agent_runtime.persistence_mode == "mongodb"
    assert config.agent_runtime.mongodb_uri.startswith("mongodb://")
    assert config.agent_runtime.db_name == "hagent_journey"
    assert config.agent_runtime.checkpoint_ttl_seconds == 2592000
    assert config.agent_runtime.event_retention_days == 30
    assert config.agent_runtime.artifact_retention_days == 180
    assert config.agent_runtime.server_selection_timeout_ms == 2000
    assert config.readiness_timeout_seconds == 3
    assert PRIVATE_SECRET not in repr(config)
    assert "runtime-password" not in repr(config)


def test_public_mode_requires_https_and_secure_cookie() -> None:
    config = load_server_runtime_config(_public_env())

    assert config.deploy_mode == "public"
    assert config.server_mode is True
    assert config.app_origin == "https://hagent.eastus.cloudapp.azure.com"
    assert config.cors_origins == ("https://hagent.eastus.cloudapp.azure.com",)
    assert config.session_https_only is True
    assert config.reload_enabled is False
    assert PUBLIC_SECRET not in repr(config)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"SUPER_SECRET_KEY": "short"}, "SUPER_SECRET_KEY"),
        (
            {"SUPER_SECRET_KEY": "CHANGE_ME_SECRET_KEY_0123456789012345"},
            "SUPER_SECRET_KEY",
        ),
        ({"APP_ORIGIN": "*"}, "APP_ORIGIN"),
        ({"APP_ORIGIN": "https://localhost:8080"}, "private"),
        ({"APP_ORIGIN": "http://localhost:"}, "APP_ORIGIN"),
        ({"APP_ORIGIN": "http://localhost:8080?"}, "APP_ORIGIN"),
        ({"SESSION_HTTPS_ONLY": "true"}, "SESSION_HTTPS_ONLY"),
        ({"BACKEND_RELOAD": "true"}, "BACKEND_RELOAD"),
    ],
)
def test_private_mode_rejects_unsafe_configuration(
    overrides: dict[str, str], message: str
) -> None:
    with pytest.raises(ServerRuntimeConfigError, match=message):
        load_server_runtime_config(_private_env(**overrides))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"APP_ORIGIN": "http://hagent.example.com"}, "public"),
        ({"APP_ORIGIN": "https://127.0.0.1"}, "FQDN"),
        ({"APP_ORIGIN": "https://127.1"}, "FQDN"),
        ({"APP_ORIGIN": "https://sslip.io"}, "FQDN"),
        ({"APP_ORIGIN": "https://nip.io"}, "FQDN"),
        ({"APP_ORIGIN": "https://hagent.localhost"}, "placeholder"),
        ({"APP_ORIGIN": "https://hagent.example.com"}, "placeholder"),
        ({"APP_ORIGIN": "https://hagent.example.invalid"}, "placeholder"),
        (
            {"APP_ORIGIN": "https://hagent.eastus.cloudapp.azure.com:443"},
            "port",
        ),
        ({"APP_ORIGIN": "https://hagent.example.com/path"}, "origin"),
        ({"SESSION_HTTPS_ONLY": "false"}, "SESSION_HTTPS_ONLY"),
        ({"BACKEND_RELOAD": "true"}, "BACKEND_RELOAD"),
    ],
)
def test_public_mode_rejects_unsafe_configuration(
    overrides: dict[str, str], message: str
) -> None:
    with pytest.raises(ServerRuntimeConfigError, match=message):
        load_server_runtime_config(_public_env(**overrides))


def test_development_uses_process_secret_without_literal_fallback() -> None:
    environment = {
        "DEPLOY_MODE": "development",
        "APP_ORIGIN": "http://localhost:3000",
        "BACKEND_RELOAD": "true",
    }

    first = load_server_runtime_config(environment)
    second = load_server_runtime_config(environment)

    assert first.server_mode is False
    assert first.session_secret == second.session_secret
    assert len(first.session_secret) >= 32
    assert first.session_secret not in repr(first)
    assert first.reload_enabled is True
    assert first.agent_runtime.mode == "legacy"
    assert first.agent_runtime.persistence_mode == "memory"
    assert first.agent_runtime.mongodb_uri is None
    assert first.agent_runtime.allow_memory is True
    assert first.agent_runtime.artifact_retention_days == 180


@pytest.mark.parametrize(
    "unsafe_secret",
    [
        "1234567890" * 4,
        "1234-5678-9012-3456-7890-1234-5678-90",
        "Abcd-1234" * 4,
        "CHANGE-ME-runtime-session-0123456789",
        "replace_me-runtime-session-0123456789",
    ],
)
def test_server_mode_rejects_predictable_or_placeholder_secret(
    unsafe_secret: str,
) -> None:
    with pytest.raises(ServerRuntimeConfigError, match="SUPER_SECRET_KEY") as error:
        load_server_runtime_config(_public_env(SUPER_SECRET_KEY=unsafe_secret))

    assert unsafe_secret not in str(error.value)


def test_development_without_origin_disables_cross_origin_access() -> None:
    config = load_server_runtime_config({"DEPLOY_MODE": "test"})

    assert config.app_origin is None
    assert config.cors_origins == ()
    assert config.session_https_only is False
    assert config.reload_enabled is False


def test_invalid_boolean_and_unknown_mode_fail_without_secret_disclosure() -> None:
    secret = "do-not-print-1a2b3c4d5e6f7g8h9i0j"

    with pytest.raises(ServerRuntimeConfigError) as invalid_boolean:
        load_server_runtime_config(
            _public_env(
                SUPER_SECRET_KEY=secret,
                SESSION_HTTPS_ONLY="sometimes",
            )
        )
    with pytest.raises(ServerRuntimeConfigError) as invalid_mode:
        load_server_runtime_config({"DEPLOY_MODE": "staging"})

    assert secret not in str(invalid_boolean.value)
    assert "staging" not in str(invalid_mode.value)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"HAGENT_RUNTIME_MODE": "swarm"}, "HAGENT_RUNTIME_MODE"),
        ({"HAGENT_CHECKPOINT_BACKEND": "memory"}, "HAGENT_CHECKPOINT_BACKEND"),
        ({"MONGODB_CONNECT": ""}, "MONGODB_CONNECT"),
        ({"MONGODB_CONNECT": "mongodb://:@mongo:27017/"}, "MONGODB_CONNECT"),
        (
            {"MONGODB_CONNECT": "mongodb://user:%20@mongo:27017/"},
            "MONGODB_CONNECT",
        ),
        (
            {"MONGODB_CONNECT": "mongodb://%ZZ:password@mongo:27017/"},
            "MONGODB_CONNECT",
        ),
        (
            {"MONGODB_CONNECT": "mongodb://%FF:password@mongo:27017/"},
            "MONGODB_CONNECT",
        ),
        ({"HAGENT_RUNTIME_DB_NAME": "bad/name"}, "HAGENT_RUNTIME_DB_NAME"),
        ({"HAGENT_CHECKPOINT_TTL_SECONDS": "3599"}, "HAGENT_CHECKPOINT_TTL_SECONDS"),
        ({"HAGENT_EVENT_RETENTION_DAYS": "91"}, "HAGENT_EVENT_RETENTION_DAYS"),
        ({"HAGENT_ARTIFACT_RETENTION_DAYS": ""}, "HAGENT_ARTIFACT_RETENTION_DAYS"),
        (
            {"HAGENT_ARTIFACT_RETENTION_DAYS": "invalid"},
            "HAGENT_ARTIFACT_RETENTION_DAYS",
        ),
        ({"HAGENT_ARTIFACT_RETENTION_DAYS": "0"}, "HAGENT_ARTIFACT_RETENTION_DAYS"),
        ({"HAGENT_ARTIFACT_RETENTION_DAYS": "3651"}, "HAGENT_ARTIFACT_RETENTION_DAYS"),
        (
            {"HAGENT_RUNTIME_SERVER_SELECTION_TIMEOUT_MS": "0"},
            "HAGENT_RUNTIME_SERVER_SELECTION_TIMEOUT_MS",
        ),
        ({"SERVER_READINESS_TIMEOUT_SECONDS": "0"}, "SERVER_READINESS_TIMEOUT_SECONDS"),
        (
            {"SERVER_READINESS_TIMEOUT_SECONDS": "nan"},
            "SERVER_READINESS_TIMEOUT_SECONDS",
        ),
    ],
)
def test_server_mode_rejects_invalid_agent_runtime_configuration(
    overrides: dict[str, str], message: str
) -> None:
    secret_uri = "mongodb://runtime-user:sentinel-password@mongo:27017/"
    environment = _private_env(MONGODB_CONNECT=secret_uri)
    environment.update(overrides)

    with pytest.raises(ServerRuntimeConfigError, match=message) as error:
        load_server_runtime_config(environment)

    assert "sentinel-password" not in str(error.value)


def test_journey_development_requires_explicit_memory_backend() -> None:
    config = load_server_runtime_config(
        {
            "DEPLOY_MODE": "test",
            "HAGENT_RUNTIME_MODE": "journey",
            "HAGENT_CHECKPOINT_BACKEND": "memory",
        }
    )

    assert config.agent_runtime.mode == "journey"
    assert config.agent_runtime.persistence_mode == "memory"
    assert config.agent_runtime.allow_memory is True


def test_artifact_retention_accepts_configured_safe_value() -> None:
    config = load_server_runtime_config(
        _public_env(HAGENT_ARTIFACT_RETENTION_DAYS="3650")
    )

    assert config.agent_runtime.artifact_retention_days == 3650


def test_artifact_retention_error_does_not_echo_raw_value() -> None:
    raw_value = "sentinel-retention-secret-value"

    with pytest.raises(
        ServerRuntimeConfigError, match="HAGENT_ARTIFACT_RETENTION_DAYS"
    ) as error:
        load_server_runtime_config(
            _public_env(HAGENT_ARTIFACT_RETENTION_DAYS=raw_value)
        )

    assert raw_value not in str(error.value)


@pytest.mark.parametrize(
    ("environment", "expected_secure"),
    [
        ({"DEPLOY_MODE": "private", "SESSION_HTTPS_ONLY": "false"}, False),
        ({"DEPLOY_MODE": "public", "SESSION_HTTPS_ONLY": "true"}, True),
    ],
)
def test_cookie_policy_consumer_does_not_require_application_dependencies(
    environment: dict[str, str], expected_secure: bool
) -> None:
    policy = load_cookie_runtime_policy(environment)

    assert isinstance(policy, CookieRuntimePolicy)
    assert policy.session_https_only is expected_secure
    assert policy.__slots__ == ("session_https_only",)
    assert PRIVATE_SECRET not in repr(policy)


def test_full_server_config_rejects_missing_mongodb_dependency() -> None:
    environment = {
        "DEPLOY_MODE": "private",
        "APP_ORIGIN": "http://localhost:8080",
        "SUPER_SECRET_KEY": PRIVATE_SECRET,
        "SESSION_HTTPS_ONLY": "false",
        "BACKEND_RELOAD": "false",
    }

    with pytest.raises(ServerRuntimeConfigError, match="MONGODB_CONNECT"):
        load_server_runtime_config(environment)
