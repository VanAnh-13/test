"""Unit tests for Agent Harness."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from hagent.agent.harness.assertions import assert_expectations
from hagent.agent.harness.loader import load_all_scenarios, scenario_from_mapping
from hagent.agent.harness.schema import ExpectSpec
from hagent.agent.harness.suite import run_harness_suite


def run(coro):
    return asyncio.run(coro)


class TestAssertions:
    def test_tools_include_and_order(self):
        ok, reasons = assert_expectations(
            ExpectSpec(
                tools_include=["start_training"],
                tools_order=["get_features", "start_training"],
                has_job=True,
            ),
            tools=["get_features", "start_training", "get_job_info"],
            has_job=True,
        )
        assert ok, reasons

    def test_missing_tool_fails(self):
        ok, reasons = assert_expectations(
            ExpectSpec(tools_include=["start_training"]),
            tools=["list_datasets"],
            has_job=False,
        )
        assert not ok
        assert any("start_training" in r for r in reasons)


class TestLoader:
    def test_load_smoke_pack(self):
        scenarios = load_all_scenarios(tags=["smoke"])
        ids = {s.id for s in scenarios}
        assert "smoke_train_glass" in ids
        assert "smoke_analyze_glass" in ids

    def test_fixture_resolve(self):
        s = scenario_from_mapping(
            {
                "id": "t",
                "name": "t",
                "message": "hi",
                "world_model_fixture": "glass_wm",
                "goal": {"goal_type": "list"},
            }
        )
        assert "ds_glass" in s.world_model.get("datasets", {})


class TestSuite:
    def test_offline_smoke_train(self):
        report = run(
            run_harness_suite(
                layers=["offline"],
                offline_modes=["single_shot", "campaign"],
                scenario_ids=["smoke_train_glass", "tab_clf_glass"],
            )
        )
        assert report["n"] >= 2
        # at least some successes
        assert report["n_failed"] < report["n"]

    def test_graph_smoke(self):
        report = run(
            run_harness_suite(
                layers=["graph"],
                scenario_ids=["smoke_train_glass", "smoke_list"],
            )
        )
        assert report["n"] == 2
        failed = [r for r in report["results"] if not r["success"]]
        assert not failed, failed

    def test_api_soft_skip_without_url(self):
        report = run(
            run_harness_suite(
                layers=["api"],
                scenario_ids=["smoke_list"],
                require_live=False,
            )
        )
        assert report["n"] == 1
        assert report["results"][0]["success"]
        assert "skipped" in " ".join(report["results"][0]["reasons"]).lower() or report[
            "results"
        ][0].get("extra", {}).get("skipped")
