from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "deploy" / "scripts" / "validate_server_config.py"
SPEC = importlib.util.spec_from_file_location("validate_server_config", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Không tải được module validate_server_config")
server_config = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server_config)


def valid_private_config() -> dict[str, str]:
    mongo_password = "mongo-test-password-0123456789abcdef"
    return {
        "DEPLOY_MODE": "private",
        "COMPOSE_PROFILES": "private",
        "GATEWAY_BIND_IP": "127.0.0.1",
        "GATEWAY_HTTP_PORT": "8080",
        "GATEWAY_HTTPS_PORT": "",
        "SITE_ADDRESS": "http://",
        "APP_ORIGIN": "http://localhost:8080",
        "NEXTAUTH_URL": "${APP_ORIGIN}",
        "FRONTEND_URL": "${APP_ORIGIN}",
        "REDIRECT_URI": "${APP_ORIGIN}/api/backend",
        "SESSION_HTTPS_ONLY": "false",
        "SKIP_EMAIL_VERIFICATION": "false",
        "AUTH_API_BASE_URL": "http://toolkit:8585",
        "HAGENT_INTERNAL_URL": "http://hagent_bridge:9900",
        "HAUTOML_BASE_URL": "http://toolkit:8585",
        "HAGENT_RUN_API_URL": "http://toolkit:8585/api/v1/runs",
        "RELEASE_TAG": "v1-test-20260809",
        "TOOLKIT_IMAGE": "hagent-toolkit:${RELEASE_TAG}",
        "BRIDGE_IMAGE": "hagent-bridge:${RELEASE_TAG}",
        "WORKER_IMAGE": "hagent-worker:${RELEASE_TAG}",
        "FRONTEND_IMAGE": "hagent-frontend:${RELEASE_TAG}",
        "SECRET_KEY": "secret-test-0123456789abcdef0123456789",
        "SUPER_SECRET_KEY": "session-test-0123456789abcdef01234567",
        "NEXTAUTH_SECRET": "nextauth-test-0123456789abcdef0123456",
        "ALGORITHM": "HS256",
        "ACCESS_EXPIRE": "30",
        "REFRESH_EXPIRE": "7",
        "PASSWORD_RESET_EXPIRE_MINUTES": "5",
        "MONGO_ROOT_USERNAME": "hagent",
        "MONGO_ROOT_PASSWORD": mongo_password,
        "MONGODB_CONNECT": (
            "mongodb://${MONGO_ROOT_USERNAME}:${MONGO_ROOT_PASSWORD}"
            "@mongo:27017/?authSource=admin"
        ),
        "MONGODB_DB_NAME": "hagent",
        "HAGENT_RUNTIME_DB_NAME": "hagent_journey",
        "HAGENT_CHECKPOINT_BACKEND": "mongodb",
        "HAGENT_CHECKPOINT_TTL_SECONDS": "2592000",
        "HAGENT_EVENT_RETENTION_DAYS": "30",
        "HAGENT_ARTIFACT_RETENTION_DAYS": "180",
        "HAGENT_RUNTIME_MODE": "legacy",
        "MINIO_ENDPOINT": "minio:9000",
        "MINIO_ACCESS_KEY": "hagent",
        "MINIO_SECRET_KEY": "minio-test-0123456789abcdef01234567",
        "MINIO_SECURE": "false",
        "KAFKA_SERVER": "kafka:9092",
        "KAFKA_TOPIC": "automl_jobs",
        "MAIL_USERNAME": "noreply@example.test",
        "MAIL_PASSWORD": "mail-test-0123456789abcdef012345678",
        "LOGO": "${APP_ORIGIN}/image.png",
        "LLM_DEFAULT_MODEL": "openai-gpt4o-mini",
        "OPENAI_API_KEY": "sk-test-provider-key",
        "ANTHROPIC_API_KEY": "",
        "OLLAMA_BASE_URL": "",
        "LOCAL_BASE_URL": "",
        "LOCAL_MODEL_NAME": "",
        "LOCAL_API_KEY": "",
    }


def valid_public_config() -> dict[str, str]:
    config = valid_private_config()
    config.update(
        {
            "DEPLOY_MODE": "public",
            "COMPOSE_PROFILES": "public",
            "GATEWAY_BIND_IP": "0.0.0.0",
            "GATEWAY_HTTP_PORT": "80",
            "GATEWAY_HTTPS_PORT": "443",
            "SITE_ADDRESS": "hagent.southeastasia.cloudapp.azure.com",
            "APP_ORIGIN": "https://hagent.southeastasia.cloudapp.azure.com",
            "SESSION_HTTPS_ONLY": "true",
        }
    )
    return config


def set_origin(config: dict[str, str], origin: str) -> None:
    """Đồng bộ origin phụ để từng test chỉ kiểm một policy chính."""
    parsed = urlsplit(origin)
    config["APP_ORIGIN"] = origin
    config["SITE_ADDRESS"] = parsed.hostname or ""
    config["NEXTAUTH_URL"] = origin
    config["FRONTEND_URL"] = origin
    config["REDIRECT_URI"] = f"{origin}/api/backend"
    config["LOGO"] = f"{origin}/image.png"


def set_release_tag(config: dict[str, str], release_tag: str) -> None:
    """Đồng bộ image để test tag không bị che bởi lỗi contract khác."""
    config["RELEASE_TAG"] = release_tag
    config["TOOLKIT_IMAGE"] = f"hagent-toolkit:{release_tag}"
    config["BRIDGE_IMAGE"] = f"hagent-bridge:{release_tag}"
    config["WORKER_IMAGE"] = f"hagent-worker:{release_tag}"
    config["FRONTEND_IMAGE"] = f"hagent-frontend:{release_tag}"


class EnvParserTests(unittest.TestCase):
    def test_parser_rejects_malformed_and_duplicate_keys(self) -> None:
        cases = (
            "VALID=value\nMALFORMED\n",
            "DUPLICATE=one\nDUPLICATE=two\n",
        )
        for content in cases:
            with (
                self.subTest(content=content),
                tempfile.TemporaryDirectory() as temp_dir,
            ):
                env_path = Path(temp_dir) / "server.env"
                env_path.write_text(content, encoding="utf-8")
                with self.assertRaises(server_config.ConfigValidationError):
                    server_config.load_env_file(env_path)

    def test_parser_rejects_ambiguous_compose_syntax(self) -> None:
        cases = (
            "SECRET_KEY='${OTHER_KEY}'\nOTHER_KEY=value\n",
            "SECRET_KEY=x # padding-do-not-count-as-secret\n",
            "SECRET_KEY=$UNDECLARED_SECRET_PADDING_VALUE\n",
            "SECRET_KEY=value\twith-tab\n",
            "SECRET_KEY=value\\with-escape\n",
        )
        for content in cases:
            with (
                self.subTest(content=content),
                tempfile.TemporaryDirectory() as temp_dir,
            ):
                env_path = Path(temp_dir) / "server.env"
                env_path.write_text(content, encoding="utf-8")
                with self.assertRaises(server_config.ConfigValidationError):
                    server_config.load_env_file(env_path)

    def test_parser_wraps_invalid_utf8_without_exposing_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / "sensitive-server-name.env"
            env_path.write_bytes(b"SECRET_KEY=\xff\n")
            with self.assertRaises(server_config.ConfigValidationError) as caught:
                server_config.load_env_file(env_path)
        self.assertNotIn("sensitive-server-name.env", str(caught.exception))

    def test_interpolation_rejects_missing_reference_and_cycle(self) -> None:
        missing = valid_private_config()
        missing["NEXTAUTH_URL"] = "${MISSING_ORIGIN}"
        with self.assertRaisesRegex(
            server_config.ConfigValidationError,
            "NEXTAUTH_URL.*tham chiếu biến chưa khai báo",
        ):
            server_config.validate_config(missing)

        cycle = valid_private_config()
        cycle["APP_ORIGIN"] = "${NEXTAUTH_URL}"
        cycle["NEXTAUTH_URL"] = "${APP_ORIGIN}"
        with self.assertRaisesRegex(
            server_config.ConfigValidationError,
            "vòng lặp",
        ):
            server_config.validate_config(cycle)

    def test_error_does_not_expose_secret_value(self) -> None:
        config = valid_private_config()
        secret_value = "TOP-SECRET-MUST-NOT-LEAK"
        config["SECRET_KEY"] = secret_value
        with self.assertRaises(server_config.ConfigValidationError) as caught:
            server_config.validate_config(config)
        self.assertNotIn(secret_value, str(caught.exception))


class DeploymentModeTests(unittest.TestCase):
    def test_private_config_is_valid(self) -> None:
        resolved = server_config.validate_config(valid_private_config())
        self.assertEqual(resolved["NEXTAUTH_URL"], resolved["APP_ORIGIN"])
        self.assertEqual(
            resolved["MONGO_ROOT_PASSWORD"],
            urlsplit(resolved["MONGODB_CONNECT"]).password,
        )

    def test_private_rejects_public_bind_or_secure_cookie(self) -> None:
        for key, value in (
            ("GATEWAY_BIND_IP", "0.0.0.0"),
            ("SESSION_HTTPS_ONLY", "true"),
            ("APP_ORIGIN", "http://10.0.0.4:8080"),
        ):
            config = valid_private_config()
            if key == "APP_ORIGIN":
                set_origin(config, value)
                config["SITE_ADDRESS"] = "http://"
            else:
                config[key] = value
            with (
                self.subTest(key=key),
                self.assertRaises(server_config.ConfigValidationError),
            ):
                server_config.validate_config(config)

    def test_public_config_is_valid(self) -> None:
        resolved = server_config.validate_config(valid_public_config())
        self.assertEqual(resolved["SESSION_HTTPS_ONLY"], "true")

    def test_public_rejects_http_ip_sslip_and_wrong_profile(self) -> None:
        cases = (
            ("APP_ORIGIN", "http://hagent.example.com"),
            ("APP_ORIGIN", "https://20.1.2.3"),
            ("APP_ORIGIN", "https://20-1-2-3.sslip.io"),
            ("COMPOSE_PROFILES", "private"),
            ("SESSION_HTTPS_ONLY", "false"),
        )
        for key, value in cases:
            config = valid_public_config()
            if key == "APP_ORIGIN":
                set_origin(config, value)
            else:
                config[key] = value
            with (
                self.subTest(key=key, value=value),
                self.assertRaises(server_config.ConfigValidationError),
            ):
                server_config.validate_config(config)

    def test_public_rejects_trailing_slash_and_explicit_port(self) -> None:
        for origin in (
            "https://hagent.example.com/",
            "https://hagent.example.com:443",
            "https://hagent.example.com:",
            "https://hagent.example.com:notaport",
            "https://127.1",
            f"https://{'.'.join(['a' * 63] * 4)}",
            "https://@hagent.example.com",
            "https://:@hagent.example.com",
            "https://hagent.example.com?",
            "https://hagent.example.com#",
        ):
            config = valid_public_config()
            set_origin(config, origin)
            with (
                self.subTest(origin=origin),
                self.assertRaises(server_config.ConfigValidationError),
            ):
                server_config.validate_config(config)

    def test_email_verification_cannot_be_skipped(self) -> None:
        config = valid_public_config()
        config["SKIP_EMAIL_VERIFICATION"] = "true"
        with self.assertRaises(server_config.ConfigValidationError):
            server_config.validate_config(config)

    def test_origins_must_be_consistent(self) -> None:
        for key, value in (
            ("SITE_ADDRESS", "other.example.com"),
            ("NEXTAUTH_URL", "https://other.example.com"),
            ("FRONTEND_URL", "https://other.example.com"),
            ("REDIRECT_URI", "https://other.example.com/api/backend"),
            ("LOGO", "https://other.example.com/image.png"),
        ):
            config = valid_public_config()
            config[key] = value
            with (
                self.subTest(key=key),
                self.assertRaises(server_config.ConfigValidationError),
            ):
                server_config.validate_config(config)


class SecurityConfigTests(unittest.TestCase):
    def test_production_rejects_placeholder_weak_secret_and_latest(self) -> None:
        cases = (
            ("SECRET_KEY", "CHANGE_ME_SECRET_WITH_SUFFICIENT_LENGTH_123456"),
            ("NEXTAUTH_SECRET", "short"),
            ("RELEASE_TAG", "latest"),
        )
        for key, value in cases:
            config = valid_private_config()
            if key == "RELEASE_TAG":
                set_release_tag(config, value)
            else:
                config[key] = value
            with (
                self.subTest(key=key),
                self.assertRaises(server_config.ConfigValidationError),
            ):
                server_config.validate_config(config)

    def test_template_mode_accepts_explicit_placeholders(self) -> None:
        config = valid_private_config()
        for key in (
            "SECRET_KEY",
            "SUPER_SECRET_KEY",
            "NEXTAUTH_SECRET",
            "MONGO_ROOT_PASSWORD",
            "MINIO_SECRET_KEY",
            "MAIL_PASSWORD",
            "OPENAI_API_KEY",
        ):
            config[key] = f"CHANGE_ME_{key}"
        config["MONGODB_CONNECT"] = (
            "mongodb://${MONGO_ROOT_USERNAME}:${MONGO_ROOT_PASSWORD}"
            "@mongo:27017/?authSource=admin"
        )
        server_config.validate_config(config, template=True)

    def test_requires_durable_mongo(self) -> None:
        config = valid_private_config()
        config["HAGENT_CHECKPOINT_BACKEND"] = "memory"
        with self.assertRaises(server_config.ConfigValidationError):
            server_config.validate_config(config)

    def test_rejects_invalid_mongo_endpoint_and_database_names(self) -> None:
        cases = (
            (
                "MONGODB_CONNECT",
                "mongodb://hagent:password@mongo:notaport/?authSource=admin",
            ),
            ("MONGODB_DB_NAME", ""),
            ("HAGENT_RUNTIME_DB_NAME", "bad/name"),
            ("HAGENT_RUNTIME_DB_NAME", "a" * 64),
        )
        for key, value in cases:
            config = valid_private_config()
            config[key] = value
            with (
                self.subTest(key=key),
                self.assertRaises(server_config.ConfigValidationError),
            ):
                server_config.validate_config(config)

    def test_rejects_invalid_kafka_topic(self) -> None:
        for topic in ("", "/", ".", "a" * 250, "CHANGE_ME_KAFKA_TOPIC"):
            config = valid_private_config()
            config["KAFKA_TOPIC"] = topic
            with (
                self.subTest(topic=topic),
                self.assertRaises(server_config.ConfigValidationError),
            ):
                server_config.validate_config(config)

    def test_each_supported_provider_can_satisfy_provider_gate(self) -> None:
        provider_cases = (
            {
                "LLM_DEFAULT_MODEL": "openai-gpt4o-mini",
                "OPENAI_API_KEY": "sk-test-provider-key",
            },
            {
                "LLM_DEFAULT_MODEL": "anthropic-sonnet",
                "ANTHROPIC_API_KEY": "sk-ant-test-provider-key",
            },
            {
                "LLM_DEFAULT_MODEL": "ollama-llama",
                "OLLAMA_BASE_URL": "http://ollama:11434",
            },
            {
                "LLM_DEFAULT_MODEL": "local-compatible",
                "LOCAL_BASE_URL": "http://model-gateway:8000/v1",
                "LOCAL_MODEL_NAME": "internal-model",
            },
        )
        provider_keys = (
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "OLLAMA_BASE_URL",
            "LOCAL_BASE_URL",
            "LOCAL_MODEL_NAME",
            "LOCAL_API_KEY",
        )
        for provider in provider_cases:
            config = valid_private_config()
            config.update(dict.fromkeys(provider_keys, ""))
            config.update(provider)
            with self.subTest(provider=provider):
                server_config.validate_config(config)

    def test_missing_provider_is_rejected(self) -> None:
        config = valid_private_config()
        for key in (
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "OLLAMA_BASE_URL",
            "LOCAL_BASE_URL",
            "LOCAL_MODEL_NAME",
        ):
            config[key] = ""
        with self.assertRaises(server_config.ConfigValidationError):
            server_config.validate_config(config)

    def test_rejects_invalid_provider_url_and_weak_keys(self) -> None:
        cases = (
            {
                "LLM_DEFAULT_MODEL": "openai-gpt4o-mini",
                "OPENAI_API_KEY": "x",
            },
            {
                "LLM_DEFAULT_MODEL": "ollama-llama",
                "OPENAI_API_KEY": "",
                "OLLAMA_BASE_URL": "http://ollama:notaport",
            },
            {
                "LLM_DEFAULT_MODEL": "ollama-llama",
                "OPENAI_API_KEY": "",
                "OLLAMA_BASE_URL": "http://ollama:",
            },
            {
                "LLM_DEFAULT_MODEL": "ollama-llama",
                "OPENAI_API_KEY": "",
                "OLLAMA_BASE_URL": "http://bad_host:11434",
            },
            {
                "LLM_DEFAULT_MODEL": "ollama-llama",
                "OPENAI_API_KEY": "",
                "OLLAMA_BASE_URL": "http://example.invalid:11434",
            },
        )
        for changes in cases:
            config = valid_private_config()
            config.update(changes)
            with (
                self.subTest(changes=changes),
                self.assertRaises(server_config.ConfigValidationError),
            ):
                server_config.validate_config(config)

    def test_rejects_local_provider_placeholders(self) -> None:
        for key, value in (
            ("LOCAL_MODEL_NAME", "CHANGE_ME_LOCAL_MODEL"),
            ("LOCAL_API_KEY", "CHANGE_ME_LOCAL_API_KEY"),
        ):
            config = valid_private_config()
            config.update(
                {
                    "LLM_DEFAULT_MODEL": "local-compatible",
                    "OPENAI_API_KEY": "",
                    "LOCAL_BASE_URL": "http://model-gateway:8000/v1",
                    "LOCAL_MODEL_NAME": "internal-model",
                    key: value,
                }
            )
            with (
                self.subTest(key=key),
                self.assertRaises(server_config.ConfigValidationError),
            ):
                server_config.validate_config(config)


if __name__ == "__main__":
    unittest.main()
