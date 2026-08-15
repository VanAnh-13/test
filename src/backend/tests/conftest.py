"""
conftest.py — Pytest fixtures cho HAgent tests.

Cung cấp:
- Mock LLM server (OpenAI-compatible) chạy background
- hagent.yaml override cho test environment
- Reusable fixtures cho agent, tools, config
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

# ── Đường dẫn ────────────────────────────────────────────

BACKEND_DIR = Path(__file__).parent.parent
HAGENT_DIR = BACKEND_DIR / "hagent"
MOCK_SERVER_SCRIPT = Path(__file__).parent / "mock_llm_server.py"

LOOPBACK_HOST = "127.0.0.1"
MOCK_SERVER_START_ATTEMPTS = 3
MOCK_SERVER_START_TIMEOUT_SECONDS = 15.0
MOCK_SERVER_POLL_INTERVAL_SECONDS = 0.1
MOCK_SERVER_REQUEST_TIMEOUT_SECONDS = 0.5
MOCK_SERVER_STOP_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class MockLlmEndpoint:
    """Endpoint bất biến của mock LLM trong một phiên test."""

    root_url: str
    api_base_url: str


# ── Environment setup ────────────────────────────────────


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """
    Cấu hình environment cho toàn bộ test session.
    Chỉ set env vars, KHÔNG hardcode trong code sản phẩm.
    """
    os.environ["HAGENT_CONFIG"] = str(HAGENT_DIR / "config" / "hagent.yaml")
    os.environ["HAUTOML_BASE_URL"] = "http://127.0.0.1:8585"

    # Override LLM config để dùng mock server
    os.environ["LLM_DEFAULT_MODEL"] = "ci-mock"
    os.environ["OPENAI_API_KEY"] = "test-key-not-real"

    # Đảm bảo module path
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))

    yield

    # Cleanup
    for key in ["LLM_DEFAULT_MODEL", "OPENAI_API_KEY"]:
        os.environ.pop(key, None)


# ── Mock LLM Server ──────────────────────────────────────


def _allocate_loopback_port() -> int:
    """Xin hệ điều hành cấp một cổng loopback đang khả dụng."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
        candidate.bind((LOOPBACK_HOST, 0))
        return int(candidate.getsockname()[1])


def _stop_process(proc: subprocess.Popen[bytes]) -> tuple[bytes, bytes]:
    """Dừng tiến trình con có giới hạn và thu hồi hai pipe."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=MOCK_SERVER_STOP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=MOCK_SERVER_STOP_TIMEOUT_SECONDS)
    return proc.communicate()


def _wait_until_ready(proc: subprocess.Popen[bytes], endpoint: MockLlmEndpoint) -> bool:
    """Chờ health endpoint hoặc dừng sớm khi tiến trình bind thất bại."""
    deadline = time.monotonic() + MOCK_SERVER_START_TIMEOUT_SECONDS
    with httpx.Client(timeout=MOCK_SERVER_REQUEST_TIMEOUT_SECONDS) as client:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return False
            try:
                response = client.get(f"{endpoint.root_url}/health")
                if response.status_code == 200:
                    return True
            except httpx.TransportError:
                pass
            time.sleep(MOCK_SERVER_POLL_INTERVAL_SECONDS)
    return False


def _start_mock_server() -> tuple[subprocess.Popen[bytes], MockLlmEndpoint]:
    """Khởi động mock server và thử cổng mới nếu xảy ra race lúc bind."""
    diagnostics: list[str] = []
    for attempt in range(1, MOCK_SERVER_START_ATTEMPTS + 1):
        port = _allocate_loopback_port()
        endpoint = MockLlmEndpoint(
            root_url=f"http://{LOOPBACK_HOST}:{port}",
            api_base_url=f"http://{LOOPBACK_HOST}:{port}/v1",
        )
        try:
            proc = subprocess.Popen(
                [sys.executable, str(MOCK_SERVER_SCRIPT), "--port", str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(BACKEND_DIR),
            )
        except OSError as exc:
            diagnostics.append(
                f"lần {attempt}: không thể tạo tiến trình ({type(exc).__name__})"
            )
            continue

        try:
            ready = _wait_until_ready(proc, endpoint)
        except BaseException:
            # Fixture chưa yield nên pytest không thể chạy teardown thay chúng ta.
            _stop_process(proc)
            raise

        if ready:
            return proc, endpoint

        stdout, stderr = _stop_process(proc)
        detail = (stderr or stdout).decode(errors="replace").strip()
        diagnostics.append(
            f"lần {attempt}: exit={proc.returncode}; {detail[-500:] or 'không có output'}"
        )

    pytest.fail(
        "Mock LLM server không khởi động sau "
        f"{MOCK_SERVER_START_ATTEMPTS} lần thử. " + " | ".join(diagnostics)
    )


@pytest.fixture(scope="session")
def mock_llm_server(setup_test_env):
    """
    Chạy mock LLM server trong background process.
    Server tự tắt khi test session kết thúc.
    """
    proc, endpoint = _start_mock_server()

    yield endpoint

    _stop_process(proc)


# ── Config fixtures ──────────────────────────────────────


@pytest.fixture
def agent_config():
    """Load agent config từ hagent.yaml."""
    from hagent.bridge.config import get_agent_config

    return get_agent_config()


@pytest.fixture
def llm_config():
    """Load LLM config từ hagent.yaml."""
    from hagent.bridge.config import get_llm_config

    return get_llm_config()
