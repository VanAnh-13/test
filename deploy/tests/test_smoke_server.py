"""Regression cho smoke contract và runbook Azure private-first."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SMOKE_SCRIPT = REPOSITORY_ROOT / "deploy" / "scripts" / "smoke_server.sh"
RUNBOOK = REPOSITORY_ROOT / "docs" / "azure-server-deployment.md"

_DEFAULT_RESPONSES = {
    "/hagent": (200, "text/html; charset=utf-8", "<h1>HAgent xử lý gì?</h1>"),
    "/api/backend/home": (
        200,
        "application/json",
        '{"AutoML":"version 1.0","message":"Hi there :P"}',
    ),
    "/api/hagent/api/v1/chat/health": (
        200,
        "application/json",
        (
            '{"hagent_url":"/api/hagent","connected":true,'
            '"hautoml_connected":true,"mode":"hagent","active_provider":"hagent",'
            '"active_model":"hagent-agent","available_providers":["hagent"]}'
        ),
    ),
    "/api/hagent/api/v1/ready": (
        200,
        "application/json",
        '{"status":"ready","dependencies":{"mongodb":"ready","toolkit":"ready"}}',
    ),
    "/api/hagent/api/v1/chat/providers": (
        200,
        "application/json",
        (
            '{"default_provider":"openai","default_model":"openai-gpt4o-mini",'
            '"providers":[{"name":"Openai","provider_id":"openai",'
            '"models":["openai-gpt4o-mini"],"available":true,'
            '"description":"configured"}]}'
        ),
    ),
}


def _bash_executable() -> str:
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    if git_bash.is_file():
        return str(git_bash)
    discovered = shutil.which("bash")
    if discovered:
        return discovered
    pytest.fail("Không tìm thấy Bash để chạy server smoke contract")


class _Handler(BaseHTTPRequestHandler):
    responses: ClassVar[dict[str, tuple[int, str, str]]] = dict(_DEFAULT_RESPONSES)
    delays: ClassVar[dict[str, float]] = {}

    def do_GET(self) -> None:
        delay = self.delays.get(self.path, 0)
        if delay:
            time.sleep(delay)
        status, content_type, body = self.responses.get(
            self.path,
            (404, "application/json", '{"detail":"not found"}'),
        )
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        try:
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format: str, *args: object) -> None:
        del format, args


def _run_smoke(
    origin: str,
    *,
    responses: dict[str, tuple[int, str, str]] | None = None,
    delays: dict[str, float] | None = None,
    timeout_seconds: str = "2",
    curl_config: str | None = None,
) -> subprocess.CompletedProcess[str]:
    handler = type("ConfiguredHandler", (_Handler,), {})
    handler.responses = dict(_DEFAULT_RESPONSES)
    handler.delays = dict(delays or {})
    if responses:
        handler.responses.update(responses)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory() as curl_home:
            if curl_config is not None:
                Path(curl_home, ".curlrc").write_text(curl_config, encoding="utf-8")
            actual_origin = origin.format(port=server.server_port)
            environment = os.environ.copy()
            environment["CURL_HOME"] = curl_home
            environment["HAGENT_SMOKE_TIMEOUT_SECONDS"] = timeout_seconds
            environment["PYTHONUTF8"] = "1"
            return subprocess.run(
                [_bash_executable(), SMOKE_SCRIPT.as_posix(), actual_origin],
                cwd=REPOSITORY_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=20,
                check=False,
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_smoke_server_accepts_complete_same_origin_contract():
    result = _run_smoke("http://127.0.0.1:{port}")

    assert result.returncode == 0, result.stderr
    assert "OK: HAgent server" in result.stdout
    assert "secret" not in result.stdout.lower()


@pytest.mark.parametrize(
    ("path", "response", "expected_message"),
    [
        ("/hagent", (200, "text/html", "wrong page"), "workspace"),
        (
            "/api/hagent/api/v1/ready",
            (503, "application/json", '{"status":"not_ready"}'),
            "readiness",
        ),
        (
            "/api/hagent/api/v1/chat/providers",
            (200, "application/json", '{"default_model":"","providers":[]}'),
            "provider",
        ),
        (
            "/api/hagent/api/v1/chat/providers",
            (
                200,
                "application/json",
                (
                    '{"default_provider":"openai","default_model":"model-a",'
                    '"providers":[{"provider_id":"openai","available":false,'
                    '"models":["old-model"]},{"provider_id":"openai",'
                    '"available":true,"models":["model-a"]}]}'
                ),
            ),
            "provider",
        ),
        (
            "/api/hagent/api/v1/chat/providers",
            (
                200,
                "application/json",
                (
                    '{"default_provider":"openai","default_model":"model-a",'
                    '"providers":[{"provider_id":"openai","available":true,'
                    '"models":{"model-a":true}}]}'
                ),
            ),
            "provider",
        ),
    ],
)
def test_smoke_server_fails_closed_on_incomplete_contract(
    path: str,
    response: tuple[int, str, str],
    expected_message: str,
):
    result = _run_smoke(
        "http://127.0.0.1:{port}",
        responses={path: response},
    )

    assert result.returncode != 0
    assert expected_message in result.stderr.lower()


@pytest.mark.parametrize(
    "origin",
    [
        "http://example.com",
        "http://user:password@localhost:8080",
        "http://localhost:8080/path",
        "http://localhost:0",
        "http://127.1:8080",
        "https://demo.sslip.io",
        " https://hagent.contoso.com",
        "https://hagent.invalid",
        "https://hagent.test",
        "https://hagent.example",
        "https://hagent.localhost",
        "https://hagent.example.com?debug=true",
    ],
)
def test_smoke_server_rejects_unsafe_or_noncanonical_origin(origin: str):
    result = _run_smoke(origin)

    assert result.returncode != 0
    assert "origin" in result.stderr.lower()
    assert "password" not in result.stderr
    assert origin not in result.stderr


def test_smoke_server_enforces_bounded_response_timeout():
    started_at = time.monotonic()
    result = _run_smoke(
        "http://127.0.0.1:{port}",
        delays={"/api/backend/home": 2},
        timeout_seconds="0.2",
    )
    elapsed = time.monotonic() - started_at

    assert result.returncode != 0
    assert "backend" in result.stderr.lower()
    assert elapsed < 1.5


def test_smoke_server_requires_exact_http_200_even_with_valid_body():
    result = _run_smoke(
        "http://127.0.0.1:{port}",
        responses={
            "/hagent": (302, "text/html", "<h1>HAgent xử lý gì?</h1>"),
        },
    )

    assert result.returncode != 0
    assert "workspace" in result.stderr.lower()


def test_smoke_server_ignores_user_curl_configuration():
    result = _run_smoke(
        "http://127.0.0.1:{port}",
        curl_config="url = http://127.0.0.1:1/\nlocation\ninsecure\n",
    )

    assert result.returncode == 0, result.stderr


def test_smoke_server_rejects_oversized_response():
    result = _run_smoke(
        "http://127.0.0.1:{port}",
        responses={
            "/hagent": (
                200,
                "text/html",
                "<h1>HAgent xử lý gì?</h1>" + ("x" * 1_048_576),
            ),
        },
    )

    assert result.returncode != 0
    assert "workspace" in result.stderr.lower()


def test_runbook_covers_private_public_transition_without_false_evidence():
    content = RUNBOOK.read_text(encoding="utf-8")
    normalized = " ".join(content.split())

    required_fragments = [
        "ssh -L 8080:127.0.0.1:8080",
        "chmod 600 deploy/.env.server",
        "validate_server_config.py",
        "up -d --no-build --force-recreate",
        "Standard static Public IP",
        "Azure DNS label",
        "NSG",
        "80/443",
        "logs --since 30m",
        "mongodump",
        ".partial",
        "sha256sum",
        "rollback",
        "deploy/scripts/smoke_server.sh",
        "AZURE-PUBLICATION-VERIFY-001",
    ]
    for fragment in required_fragments:
        assert fragment in normalized

    assert "down -v" not in content
    assert "sslip.io" not in content.lower()
    assert "đã xác minh TLS" not in content
