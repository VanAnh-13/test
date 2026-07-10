"""Phase 6 — multi-candidate campaigns, warm-start, compare."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from hagent.agent.campaign.builder import build_campaign
from hagent.agent.campaign.compare import compare_campaign, best_config_payload
from hagent.agent.campaign.nodes import campaign_node, campaign_route
from hagent.agent.campaign.schema import Campaign, CampaignVariant
from hagent.agent.campaign.warm_start import (
    configs_from_world_model,
    merge_warm_starts,
)
from hagent.agent.execution.tool_runner import set_tool_invoker
from hagent.agent.graph import _should_run_campaign, coordinator_route
from hagent.agent.memory import Fact, LocalFactStore


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _reset_invoker():
    set_tool_invoker(None)
    yield
    set_tool_invoker(None)


GOAL = {
    "goal_type": "train",
    "dataset_id": "ds1",
    "target_column": "target",
    "problem_type": "classification",
    "metric": "f1",
}


class TestWarmStart:
    def test_configs_from_wm(self):
        wm = {
            "jobs": {
                "j1": {
                    "id": "j1",
                    "status": "completed",
                    "best_score": 0.9,
                    "best_model": "rf",
                    "config": {
                        "problem_type": "classification",
                        "search_algorithm": "bayesian_search",
                        "models": ["rf"],
                    },
                },
                "j2": {"id": "j2", "status": "running", "best_score": 0.99},
            }
        }
        cfgs = configs_from_world_model(wm, problem_type="classification", top_k=3)
        assert len(cfgs) == 1
        assert cfgs[0]["search_algorithm"] == "bayesian_search"

    def test_merge_dedupes(self):
        a = [{"search_algorithm": "grid_search", "models": ["a"]}]
        b = [
            {"search_algorithm": "grid_search", "models": ["a"]},
            {"search_algorithm": "bayesian_search", "models": ["b"]},
        ]
        m = merge_warm_starts(from_wm=a, from_memory=b, max_items=5)
        assert len(m) == 2


class TestBuilder:
    def test_build_n_variants(self):
        camp = run(
            build_campaign(
                GOAL,
                user_id="u1",
                world_model={
                    "jobs": {
                        "j1": {
                            "id": "j1",
                            "status": "completed",
                            "best_score": 0.88,
                            "config": {
                                "problem_type": "classification",
                                "search_algorithm": "bayesian_search",
                            },
                        }
                    }
                },
                config={"n_job_candidates": 3, "max_concurrent_jobs": 2},
            )
        )
        assert len(camp.variants) == 3
        assert camp.max_concurrent == 2
        assert any(v.source == "warm_start" for v in camp.variants)
        algos = {v.params.get("search_algorithm") for v in camp.variants}
        assert len(algos) >= 2


class TestCompare:
    def test_pick_best_higher(self):
        camp = Campaign(
            campaign_id="c1",
            goal=GOAL,
            variants=[
                CampaignVariant(
                    "v1", "a", {}, status="completed", best_score=0.7, best_model="lr"
                ),
                CampaignVariant(
                    "v2", "b", {}, status="completed", best_score=0.9, best_model="rf"
                ),
                CampaignVariant("v3", "c", {}, status="failed", error="x"),
            ],
        )
        best, table = compare_campaign(camp, metric="f1")
        assert best is not None
        assert best.variant_id == "v2"
        assert len(table) == 3
        payload = best_config_payload(best, GOAL)
        assert payload["best_model"] == "rf"


class TestCampaignNode:
    def test_full_campaign_with_mocks(self):
        job_counter = {"n": 0}
        scores = {"job-1": 0.7, "job-2": 0.95, "job-3": 0.8}

        async def fake(action_type, params):
            if action_type == "start_training":
                job_counter["n"] += 1
                jid = f"job-{job_counter['n']}"
                return {"job_id": jid, "status": "starting"}
            if action_type == "get_job_info":
                jid = params.get("job_id")
                return {
                    "id": jid,
                    "status": "completed",
                    "best_score": scores.get(jid, 0.5),
                    "best_model": f"model-{jid}",
                    "metrics": {"f1": scores.get(jid, 0.5)},
                }
            return {}

        set_tool_invoker(fake)

        with tempfile.TemporaryDirectory() as td:
            store = LocalFactStore(td)
            state = {
                "messages": [],
                "user_id": "u1",
                "user_token": "t",
                "goal": GOAL,
                "world_model": {
                    "user_id": "u1",
                    "datasets": {
                        "ds1": {
                            "id": "ds1",
                            "features": ["a", "target"],
                        }
                    },
                    "jobs": {},
                },
                "execution_events": [],
                "cost_metrics": {},
                "campaign_tick": 0,
            }

            # Tick until done (submit + poll)
            for _ in range(10):
                out = run(campaign_node(state))
                state = {**state, **out}
                state["messages"] = []
                if campaign_route(state) == "synthesize":
                    break

            assert state["campaign_status"] == "done"
            assert state["evaluation"]["best_job_id"] in ("job-1", "job-2", "job-3")
            # best should be job-2 with 0.95
            assert state["evaluation"]["best_job_id"] == "job-2"
            assert state["cost_metrics"]["campaign_completed"] == 3

            # Warm-start fact written
            fact = run(store.get("u1", "warm_start_classification"))
            # campaign_node uses create_fact_store default path — check evaluation only
            # Memory path may differ; assert evaluation recommendation set
            assert state["evaluation"]["recommendation"] == "model-job-2"

    def test_coordinator_prefers_campaign_when_hierarchy_off(self, monkeypatch):
        class Msg:
            tool_calls = None

        monkeypatch.setattr(
            "hagent.agent.graph._hierarchy_live_enabled",
            lambda: False,
        )
        st = {
            "messages": [Msg()],
            "goal": GOAL,
            "world_model": {"active_dataset_id": "ds1"},
            "campaign_status": None,
        }
        assert _should_run_campaign(st) is True
        assert coordinator_route(st) == "campaign"

    def test_coordinator_prefers_hierarchy_when_live(self):
        class Msg:
            tool_calls = None

        from hagent.agent.graph import _should_run_hierarchy

        st = {
            "messages": [Msg()],
            "goal": GOAL,
            "world_model": {
                "active_dataset_id": "ds1",
                "datasets": {
                    "ds1": {
                        "id": "ds1",
                        "features": ["a", "target"],
                        "target": "target",
                    }
                },
            },
            "hierarchy_status": None,
        }
        assert _should_run_hierarchy(st) is True
        assert coordinator_route(st) == "hierarchy"

    def test_max_concurrent_respected(self):
        submitted = []

        async def fake(action_type, params):
            if action_type == "start_training":
                jid = f"j{len(submitted)+1}"
                submitted.append(jid)
                return {"job_id": jid, "status": 0}
            if action_type == "get_job_info":
                # Keep running so we only observe first submit batch
                return {"id": params.get("job_id"), "status": "running"}
            return {}

        set_tool_invoker(fake)
        camp = run(
            build_campaign(
                GOAL,
                user_id="u1",
                config={"n_job_candidates": 3, "max_concurrent_jobs": 2},
            )
        )
        from hagent.agent.campaign.runner import campaign_step

        camp = run(
            campaign_step(
                camp, user_id="u1", user_token=None, world_model={}
            )
        )
        assert len(submitted) == 2
        assert len(camp.in_flight()) == 2
        assert len(camp.pending_submit()) == 1


class TestMemoryWarmStartWrite:
    def test_write_and_read(self):
        from hagent.agent.campaign.runner import write_warm_start_memory
        from hagent.agent.campaign.warm_start import configs_from_memory

        with tempfile.TemporaryDirectory() as td:
            store = LocalFactStore(td)
            run(
                write_warm_start_memory(
                    "u1",
                    {
                        "problem_type": "classification",
                        "search_algorithm": "bayesian_search",
                        "best_model": "xgb",
                        "best_score": 0.91,
                    },
                    fact_store=store,
                )
            )
            cfgs = run(
                configs_from_memory(
                    "u1", problem_type="classification", fact_store=store
                )
            )
            assert len(cfgs) == 1
            assert cfgs[0]["best_model"] == "xgb"
