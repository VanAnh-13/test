"""Phase 7 — hierarchy + offline eval harness."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from hagent.agent.eval.metrics import summarize
from hagent.agent.eval.runner import (
    report_markdown,
    run_eval_suite,
    run_scenario,
)
from hagent.agent.eval.scenarios import default_scenarios, scenarios_by_tags
from hagent.agent.planning.hierarchy import (
    decompose_goal,
    subgoal_as_goal,
)


def run(coro):
    return asyncio.run(coro)


class TestHierarchy:
    def test_train_decomposes(self):
        h = decompose_goal(
            {
                "goal_type": "train",
                "dataset_id": "ds1",
                "target_column": "y",
                "problem_type": "classification",
            }
        )
        types = [s.goal_type for s in h.subgoals]
        assert types[0] == "analyze"
        assert "train" in types
        assert types[-1] == "evaluate"
        assert h.current().goal_type == "analyze"
        h.advance()
        assert h.current().goal_type == "select"

    def test_list_is_single(self):
        h = decompose_goal({"goal_type": "list"})
        assert len(h.subgoals) == 1

    def test_subgoal_as_goal(self):
        h = decompose_goal(
            {
                "goal_type": "train",
                "dataset_id": "ds1",
                "target_column": "y",
                "metric": "f1",
            }
        )
        # skip to train leaf
        while h.current() and h.current().goal_type != "train":
            h.advance()
        g = subgoal_as_goal(h)
        assert g["goal_type"] == "train"
        assert g["dataset_id"] == "ds1"
        assert g["target_column"] == "y"


class TestScenarios:
    def test_default_scenarios_nonempty(self):
        s = default_scenarios()
        assert len(s) >= 3
        assert any(x.id == "tab_clf_glass" for x in s)

    def test_filter_tags(self):
        s = scenarios_by_tags(tags=["regression"])
        assert all("regression" in x.tags for x in s)


class TestEvalHarness:
    def test_single_shot_glass(self):
        sc = next(s for s in default_scenarios() if s.id == "tab_clf_glass")
        r = run(run_scenario(sc, "single_shot"))
        assert r.success
        assert r.tools_called >= 1
        assert r.best_job_id

    def test_plan_executor_analyze(self):
        sc = next(s for s in default_scenarios() if s.id == "tab_analyze_glass")
        r = run(run_scenario(sc, "plan_executor"))
        assert r.tools_called >= 1

    def test_campaign_mode(self):
        sc = next(s for s in default_scenarios() if s.id == "tab_clf_glass")
        r = run(run_scenario(sc, "campaign"))
        assert r.success
        assert r.campaign_status == "done"
        assert r.campaign_completed >= 1

    def test_hierarchical_mode(self):
        sc = next(s for s in default_scenarios() if s.id == "tab_clf_glass")
        r = run(run_scenario(sc, "hierarchical"))
        assert r.hierarchy_depth >= 2
        assert r.tools_called >= 1

    def test_suite_summary(self):
        report = run(
            run_eval_suite(
                modes=["single_shot", "campaign"],
                scenario_ids=["tab_clf_glass", "tab_list_datasets"],
            )
        )
        assert report["n_scenarios"] == 2
        assert len(report["summaries"]) >= 1
        md = report_markdown(report)
        assert "Success rate" in md
        assert "single_shot" in md

    def test_summarize_helper(self):
        sc = next(s for s in default_scenarios() if s.id == "tab_clf_glass")
        rows = [
            run(run_scenario(sc, "single_shot")),
            run(run_scenario(sc, "campaign")),
        ]
        sm = summarize(rows)
        assert {s.mode for s in sm} == {"single_shot", "campaign"}
