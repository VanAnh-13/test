"""Adaptive hierarchy: smart-skip + live controller."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from hagent.agent.execution.hierarchy_node import hierarchy_node, hierarchy_route
from hagent.agent.execution.tool_runner import set_tool_invoker
from hagent.agent.planning.hierarchy import (
    apply_smart_skips,
    decompose_goal,
    should_skip_subgoal,
)


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _reset():
    set_tool_invoker(None)
    yield
    set_tool_invoker(None)


RICH_WM = {
    "user_id": "u1",
    "datasets": {
        "ds1": {
            "id": "ds1",
            "name": "glass",
            "features": ["a", "b", "target"],
            "target": "target",
            "n_rows": 100,
            "n_cols": 3,
        }
    },
    "jobs": {},
    "active_dataset_id": "ds1",
}

GOAL = {
    "goal_type": "train",
    "dataset_id": "ds1",
    "target_column": "target",
    "problem_type": "classification",
    "metric": "f1",
}


class TestSmartSkip:
    def test_skip_analyze_when_features_known(self):
        h = decompose_goal(GOAL)
        analyze = h.subgoals[0]
        assert analyze.goal_type == "analyze"
        skip, reason = should_skip_subgoal(
            analyze, root_goal=GOAL, world_model=RICH_WM
        )
        assert skip is True
        assert "features" in reason.lower() or "known" in reason.lower()

    def test_skip_select_when_metric_ready(self):
        h = decompose_goal(GOAL)
        select = next(s for s in h.subgoals if s.goal_type == "select")
        skip, reason = should_skip_subgoal(
            select, root_goal=GOAL, world_model=RICH_WM
        )
        assert skip is True

    def test_do_not_skip_train(self):
        h = decompose_goal(GOAL)
        train = next(s for s in h.subgoals if s.goal_type == "train")
        skip, _ = should_skip_subgoal(train, root_goal=GOAL, world_model=RICH_WM)
        assert skip is False

    def test_apply_smart_skips_lands_on_train(self):
        h = decompose_goal(GOAL)
        events = apply_smart_skips(h, world_model=RICH_WM)
        assert any(e["type"] == "subgoal_skipped" for e in events)
        cur = h.current()
        assert cur is not None
        assert cur.goal_type == "train"

    def test_skip_evaluate_without_jobs(self):
        h = decompose_goal(GOAL)
        # jump index to evaluate
        for i, s in enumerate(h.subgoals):
            if s.goal_type == "evaluate":
                h.current_index = i
                s.status = "active"
                break
        skip, reason = should_skip_subgoal(
            h.current(), root_goal=GOAL, world_model=RICH_WM
        )
        assert skip is True
        assert "no completed jobs" in reason


class TestLiveHierarchy:
    def test_full_run_with_skips_and_campaign(self):
        job_n = {"i": 0}

        async def fake(action_type, params):
            if action_type in ("get_dataset_info", "get_features"):
                return {
                    "id": "ds1",
                    "dataset_id": "ds1",
                    "features": ["a", "b", "target"],
                    "target": "target",
                    "n_rows": 100,
                    "n_cols": 3,
                }
            if action_type in ("get_available_models", "get_metrics"):
                return {"models": ["rf"], "metrics": ["f1"]}
            if action_type == "start_training":
                job_n["i"] += 1
                return {"job_id": f"j{job_n['i']}", "status": "starting"}
            if action_type == "get_job_info":
                return {
                    "id": params.get("job_id"),
                    "status": "completed",
                    "best_score": 0.9,
                    "best_model": "rf",
                    "metrics": {"f1": 0.9},
                }
            if action_type == "list_jobs":
                return {"jobs": []}
            return {}

        set_tool_invoker(fake)

        state = {
            "messages": [],
            "user_id": "u1",
            "goal": GOAL,
            "world_model": dict(RICH_WM),
            "execution_events": [],
            "cost_metrics": {},
            "hierarchy_status": "running",
        }
        # Build hierarchy with smart skip via first tick
        from hagent.agent.planning.hierarchy import decompose_goal, apply_smart_skips

        hier = decompose_goal(GOAL)
        apply_smart_skips(hier, world_model=RICH_WM)
        state["hierarchy"] = hier.to_dict()

        for _ in range(25):
            out = run(hierarchy_node(state))
            state.update(out)
            state["messages"] = []
            if hierarchy_route(state) == "synthesize":
                break

        assert state["hierarchy_status"] == "done"
        assert state.get("evaluation", {}).get("best_job_id")
        # analyze/select should be skipped
        statuses = {
            s["goal_type"]: s["status"]
            for s in state["hierarchy"]["subgoals"]
        }
        assert statuses.get("analyze") == "skipped"
        assert statuses.get("select") == "skipped"
        assert statuses.get("train") == "done"
        # evaluate may be skipped if campaign already produced evaluation
        assert statuses.get("evaluate") in ("done", "skipped")

    def test_hierarchy_route(self):
        assert hierarchy_route({"hierarchy_status": "running"}) == "hierarchy"
        assert hierarchy_route({"hierarchy_status": "done"}) == "synthesize"
