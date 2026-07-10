"""
Student Performance — fixture, harness scenarios, mock-api assertions.

Deterministic (no Ollama). Runs in CI unit job.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from hagent.agent.harness.loader import load_all_scenarios, load_fixture
from hagent.agent.harness.suite import run_harness_suite


def run(coro):
    return asyncio.run(coro)


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
                offline_modes=["single_shot", "plan_executor", "campaign", "hierarchical"],
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
        assert row["tools_called"] >= 1 or "start_training" in (row.get("tool_names") or [])


class TestStudentMockApiE2E:
    def test_mock_api_layer_assertions(self):
        import importlib.util

        script = BACKEND / "scripts" / "run_student_performance_e2e.py"
        spec = importlib.util.spec_from_file_location("student_e2e", script)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules["student_e2e"] = mod  # required for dataclasses on 3.12+
        spec.loader.exec_module(mod)

        report = mod.run_mock_api_layer(
            base_url="http://127.0.0.1:18585",
            models=mod.STUDENT_MODELS,
            start_server=True,
            port=18585,
        )
        assert report.ok, [c.to_dict() for c in report.checks if not c.ok]
        job = (report.extra or {}).get("job") or {}
        assert job.get("best_model") == mod.EXPECTED_BEST_MODEL
        results = job.get("model_results") or []
        names = {r.get("model") for r in results if isinstance(r, dict)}
        for m in mod.STUDENT_MODELS:
            assert m in names


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
