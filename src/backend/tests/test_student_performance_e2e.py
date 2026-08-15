"""
Student Performance — fixture, harness scenarios, mock-api assertions.

Deterministic (no Ollama). Runs in CI unit job.
"""

from __future__ import annotations

import asyncio
import io
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from hagent.agent.harness.loader import load_all_scenarios, load_fixture
from hagent.agent.harness.suite import run_harness_suite


def run(coro):
    return asyncio.run(coro)


def load_student_e2e_module():
    """Nạp runner dưới tên ổn định để dataclass và monkeypatch hoạt động."""
    import importlib.util

    script = BACKEND / "scripts" / "run_student_performance_e2e.py"
    spec = importlib.util.spec_from_file_location("student_e2e", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["student_e2e"] = module
    spec.loader.exec_module(module)
    return module


class _FakeProcess:
    """Process double tối thiểu cho contract lifecycle của mock server."""

    def __init__(self, *, returncode=None):
        self.returncode = returncode
        self.events = []

    def poll(self):
        return self.returncode

    def terminate(self):
        self.events.append("terminate")
        self.returncode = 0

    def wait(self, timeout):
        self.events.append(("wait", timeout))
        return self.returncode

    def kill(self):
        self.events.append("kill")
        self.returncode = -1


class _HungProcess(_FakeProcess):
    """Process double buộc cleanup đi qua nhánh kill có timeout."""

    def terminate(self):
        self.events.append("terminate")

    def wait(self, timeout):
        self.events.append(("wait", timeout))
        if self.returncode is None:
            raise subprocess.TimeoutExpired(cmd="student-mock", timeout=timeout)
        return self.returncode

    def kill(self):
        self.events.append("kill")
        self.returncode = -1


class TestStudentFixture:
    def test_student_wm_shape(self):
        data = load_fixture("student_wm")
        wm = data.get("world_model") or data
        ds = (wm.get("datasets") or {}).get("ds_student_001")
        assert ds is not None
        assert ds["n_rows"] == 395
        assert ds["n_cols"] == 33
        assert ds["target"] == "G3"
        assert "G1" in ds["features"]
        assert "G2" in ds["features"]
        assert "G3" in ds["features"]  # column exists in CSV; also marked as target

    def test_student_scenarios_loaded(self):
        scenarios = load_all_scenarios(tags=["student"])
        ids = {s.id for s in scenarios}
        assert "student_list" in ids
        assert "student_analyze" in ids
        assert "student_train_multi" in ids
        assert "student_evaluate" in ids

    def test_train_scenario_goal(self):
        scenarios = load_all_scenarios(scenario_ids=["student_train_multi"])
        assert len(scenarios) == 1
        s = scenarios[0]
        assert s.goal.get("dataset_id") == "ds_student_001"
        assert s.goal.get("target_column") == "G3"
        assert s.goal.get("problem_type") == "regression"
        assert "RandomForestRegressor" in (s.goal.get("models") or [])
        assert s.expect.has_job is True
        assert "start_training" in (s.expect.tools_include or [])


class TestStudentHarness:
    def test_offline_and_graph_student_pack(self):
        report = run(
            run_harness_suite(
                layers=["offline", "graph"],
                offline_modes=[
                    "single_shot",
                    "plan_executor",
                    "campaign",
                    "hierarchical",
                ],
                tags=["student"],
            )
        )
        assert report["n"] > 0
        failed = [r for r in report["results"] if not r["success"]]
        assert not failed, failed

    def test_graph_train_has_job(self):
        report = run(
            run_harness_suite(
                layers=["graph"],
                scenario_ids=["student_train_multi"],
            )
        )
        assert report["n"] == 1
        row = report["results"][0]
        assert row["success"], row.get("reasons")
        # has_job signal via success + expect
        assert row["tools_called"] >= 1 or "start_training" in (
            row.get("tool_names") or []
        )


class TestStudentMockApiE2E:
    def test_mock_api_layer_assertions(self):
        mod = load_student_e2e_module()

        report = mod.run_mock_api_layer(
            base_url=None,
            models=mod.STUDENT_MODELS,
            start_server=True,
            port=0,
        )
        assert report.ok, [c.to_dict() for c in report.checks if not c.ok]
        endpoint = (report.extra or {}).get("endpoint") or {}
        assert endpoint.get("base_url", "").startswith("http://127.0.0.1:")
        assert endpoint.get("port", 0) > 0
        job = (report.extra or {}).get("job") or {}
        assert job.get("best_model") == mod.EXPECTED_BEST_MODEL
        results = job.get("model_results") or []
        names = {r.get("model") for r in results if isinstance(r, dict)}
        for m in mod.STUDENT_MODELS:
            assert m in names


class TestStudentMockServerLifecycle:
    """Khóa port động, child encoding, retry và cleanup có giới hạn."""

    def test_endpoint_is_immutable(self):
        mod = load_student_e2e_module()
        endpoint = mod.MockServerEndpoint(
            base_url="http://127.0.0.1:54321",
            port=54321,
        )

        with pytest.raises(FrozenInstanceError):
            endpoint.port = 54322

    def test_allocator_asks_os_for_dynamic_loopback_port(self, monkeypatch):
        mod = load_student_e2e_module()
        bind_calls = []

        class FakeSocket:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def bind(self, address):
                bind_calls.append(address)

            def getsockname(self):
                return (mod.LOOPBACK_HOST, 54321)

        monkeypatch.setattr(mod.socket, "socket", lambda: FakeSocket())

        assert mod._allocate_loopback_port() == 54321
        assert bind_calls == [(mod.LOOPBACK_HOST, 0)]

    def test_child_process_has_its_own_utf8_environment(self, monkeypatch):
        mod = load_student_e2e_module()
        process = _FakeProcess()
        captured = {}
        output = io.BytesIO()

        def fake_popen(*args, **kwargs):
            captured.update(kwargs)
            return process

        monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(mod.tempfile, "TemporaryFile", lambda: output)
        endpoint = mod.MockServerEndpoint(
            base_url="http://127.0.0.1:54321",
            port=54321,
        )

        handle = mod._spawn_mock_server(endpoint)

        assert handle.process is process
        assert captured["env"]["PYTHONUTF8"] == "1"
        assert captured["env"]["PYTHONIOENCODING"] == "utf-8"
        assert captured["stderr"] is subprocess.STDOUT

    def test_failed_attempt_is_reaped_before_retry(self, monkeypatch):
        mod = load_student_e2e_module()
        first = mod.MockServerHandle(
            process=_FakeProcess(),
            endpoint=mod.MockServerEndpoint(
                base_url="http://127.0.0.1:54321",
                port=54321,
            ),
            output=io.BytesIO(b"bind failed"),
        )
        second = mod.MockServerHandle(
            process=_FakeProcess(),
            endpoint=mod.MockServerEndpoint(
                base_url="http://127.0.0.1:54322",
                port=54322,
            ),
            output=io.BytesIO(),
        )
        handles = iter([first, second])
        readiness = iter([False, True])
        monkeypatch.setattr(mod, "_allocate_loopback_port", lambda: 54321)
        monkeypatch.setattr(
            mod,
            "_spawn_mock_server",
            lambda endpoint: next(handles),
        )
        monkeypatch.setattr(
            mod,
            "_wait_until_ready",
            lambda handle: next(readiness),
        )

        handle = mod._start_mock_server(port=0)

        assert handle is second
        assert first.process.events[:2] == [
            "terminate",
            ("wait", mod.MOCK_SERVER_STOP_TIMEOUT_SECONDS),
        ]
        assert first.output.closed is True
        assert second.process.events == []

    def test_readiness_exception_reaps_process(self, monkeypatch):
        mod = load_student_e2e_module()
        process = _FakeProcess()
        handle = mod.MockServerHandle(
            process=process,
            endpoint=mod.MockServerEndpoint(
                base_url="http://127.0.0.1:54321",
                port=54321,
            ),
            output=io.BytesIO(),
        )
        monkeypatch.setattr(mod, "_allocate_loopback_port", lambda: 54321)
        monkeypatch.setattr(mod, "_spawn_mock_server", lambda endpoint: handle)

        def fail_readiness(current_handle):
            del current_handle
            raise RuntimeError("readiness lỗi")

        monkeypatch.setattr(mod, "_wait_until_ready", fail_readiness)

        with pytest.raises(RuntimeError, match="readiness lỗi"):
            mod._start_mock_server(port=0)

        assert process.events[:2] == [
            "terminate",
            ("wait", mod.MOCK_SERVER_STOP_TIMEOUT_SECONDS),
        ]
        assert handle.output.closed is True

    def test_process_creation_error_uses_bounded_retry(self, monkeypatch):
        mod = load_student_e2e_module()
        calls = 0

        def fail_to_spawn(endpoint):
            nonlocal calls
            calls += 1
            del endpoint
            raise OSError("PRIVATE_PROCESS_DETAIL")

        monkeypatch.setattr(mod, "_allocate_loopback_port", lambda: 54321)
        monkeypatch.setattr(mod, "_spawn_mock_server", fail_to_spawn)

        with pytest.raises(RuntimeError, match="3 lần thử") as exc_info:
            mod._start_mock_server(port=0)

        assert calls == mod.MOCK_SERVER_START_ATTEMPTS
        assert "PRIVATE_PROCESS_DETAIL" not in str(exc_info.value)

    def test_stop_kills_hung_process_and_limits_diagnostic(self):
        mod = load_student_e2e_module()
        process = _HungProcess()
        output = io.BytesIO(b"x" * (mod.MOCK_SERVER_DIAGNOSTIC_LIMIT + 100))
        handle = mod.MockServerHandle(
            process=process,
            endpoint=mod.MockServerEndpoint(
                base_url="http://127.0.0.1:54321",
                port=54321,
            ),
            output=output,
        )

        diagnostic = mod._stop_mock_server(handle)

        timeout = mod.MOCK_SERVER_STOP_TIMEOUT_SECONDS
        assert process.events == [
            "terminate",
            ("wait", timeout),
            "kill",
            ("wait", timeout),
        ]
        assert len(diagnostic) == mod.MOCK_SERVER_DIAGNOSTIC_LIMIT
        assert output.closed is True


class TestStudentMockEnv:
    def test_mock_invoker_multi_model_scores(self):
        from hagent.agent.harness.loader import scenario_from_mapping
        from hagent.agent.harness.mock_env import (
            STUDENT_TRAINING_RESULTS,
            make_mock_tool_invoker,
        )

        s = scenario_from_mapping(
            {
                "id": "student_train_multi",
                "name": "t",
                "message": "train",
                "tags": ["student"],
                "world_model_fixture": "student_wm",
                "goal": {
                    "goal_type": "train",
                    "dataset_id": "ds_student_001",
                    "target_column": "G3",
                    "problem_type": "regression",
                    "metric": "rmse",
                    "models": list(STUDENT_TRAINING_RESULTS.keys())[:3],
                },
            }
        )
        invoker = make_mock_tool_invoker(s)

        async def _run():
            start = await invoker(
                "start_training",
                {
                    "dataset_id": "ds_student_001",
                    "models": [
                        "RandomForestRegressor",
                        "XGBRegressor",
                        "SVR",
                    ],
                },
            )
            jid = start["job_id"]
            info = await invoker("get_job_info", {"job_id": jid})
            return start, info

        start, info = run(_run())
        assert start.get("job_id")
        assert info.get("best_model") == "XGBRegressor"
        assert abs(float(info.get("best_score")) - 1.65) < 1e-6
        assert len(info.get("model_results") or []) == 3
