from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest
import structlog
from starlette.requests import Request
from starlette.responses import Response

from hagent.observability.logging import (
    CORRELATION_ID_HEADER,
    configure_logging,
    correlation_id_middleware,
)

HAGENT_ROOT = Path(__file__).parents[2] / "hagent"


@pytest.fixture(autouse=True)
def _restore_logging_configuration():
    yield
    configure_logging()


def test_production_logging_is_json_correlated_and_redacted():
    output = io.StringIO()
    configure_logging("production", stream=output)
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(correlation_id="corr-123")

    structlog.get_logger("kiem_thu").info(
        "request_completed",
        access_token="khong-duoc-lo",
        nested={"email": "nguoidung@example.com"},
    )

    payload = json.loads(output.getvalue())
    assert payload["event"] == "request_completed"
    assert payload["correlation_id"] == "corr-123"
    assert payload["access_token"] == "[DA_AN]"
    assert payload["nested"]["email"] == "[DA_AN]"
    assert "khong-duoc-lo" not in output.getvalue()
    assert "nguoidung@example.com" not in output.getvalue()


def test_development_logging_is_readable_and_supports_positional_arguments():
    output = io.StringIO()
    configure_logging("development", stream=output)

    structlog.get_logger("kiem_thu").info("Hoàn tất %d bước", 3)

    rendered = output.getvalue()
    assert "Hoàn tất 3 bước" in rendered
    assert not rendered.lstrip().startswith("{")


@pytest.mark.asyncio
async def test_middleware_propagates_valid_correlation_id_to_logs_and_response():
    output = io.StringIO()
    configure_logging("production", stream=output)
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/kiem-tra",
            "headers": [(b"x-correlation-id", b"corr-hop-le")],
        }
    )

    async def call_next(current_request):
        structlog.get_logger("kiem_thu").info("request_seen")
        assert current_request.state.correlation_id == "corr-hop-le"
        return Response()

    response = await correlation_id_middleware(request, call_next)

    assert response.headers[CORRELATION_ID_HEADER] == "corr-hop-le"
    assert json.loads(output.getvalue())["correlation_id"] == "corr-hop-le"


@pytest.mark.asyncio
async def test_middleware_replaces_unsafe_correlation_id():
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/kiem-tra",
            "headers": [(b"x-correlation-id", b"id co ky tu xuong dong")],
        }
    )

    async def call_next(_request):
        return Response()

    response = await correlation_id_middleware(request, call_next)
    generated = response.headers[CORRELATION_ID_HEADER]
    assert generated != "id co ky tu xuong dong"
    assert re.fullmatch(r"[0-9a-f]{32}", generated)


def test_hagent_does_not_use_print_or_standard_get_logger():
    violations: list[str] = []
    for path in HAGENT_ROOT.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if re.search(r"\blogging\.getLogger\s*\(", source):
            violations.append(f"logging.getLogger: {path.relative_to(HAGENT_ROOT)}")
        if re.search(r"\bprint\s*\(", source):
            violations.append(f"print: {path.relative_to(HAGENT_ROOT)}")
    assert not violations, "\n".join(violations)
