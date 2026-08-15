"""
Phase 4 — LeWM-style World Model unit tests.

encode / predict / plan / surprise / trajectory / store snapshot
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from hagent.agent.constraints import validate_action, validate_plan_steps
from hagent.agent.planning.goal_parser import is_simple_query, parse_goal
from hagent.world.query import features_of, format_for_prompt, past_best_jobs
from hagent.world.schema import (
    AutoMLAction,
    AutoMLObservation,
    GoalSpec,
    WorldState,
)
from hagent.world.service import WorldModelService
from hagent.world.state_store import WorldStateStore
from hagent.world.surprise import compute_surprise, latent_distance
from hagent.world.updater import apply_plan_event, apply_tool_output


def _service(**overrides) -> WorldModelService:
    cfg = {
        "enabled": True,
        "encoder": {
            "backend": "structured_v1",
            "dim": 32,
            "feature_extractors": [
                "dataset_counts",
                "job_status_histogram",
                "best_score_stats",
                "phase_one_hot",
                "focus_flags",
                "feature_coverage",
                "goal_type_one_hot",
                "active_dataset_hash",
            ],
        },
        "predictor": {"backend": "tabular_transition_v1"},
        "planner": {
            "backend": "cem_lite",
            "horizon": 4,
            "n_candidates": 6,
            "n_return_plans": 2,
        },
        "surprise": {
            "metric": "l2",
            "thresholds": {"medium": 0.05, "high": 0.2},
        },
        "trajectory": {"enabled": True, "max_per_user": 100},
    }
    cfg.update(overrides)
    return WorldModelService.from_config(cfg)


def _obs(**kwargs) -> AutoMLObservation:
    base = {
        "user_id": "u1",
        "datasets": {
            "ds1": {
                "id": "ds1",
                "name": "glass",
                "n_rows": 200,
                "n_cols": 10,
                "features": ["a", "b", "target"],
                "target": "target",
            }
        },
        "jobs": {},
        "focus": {"dataset_id": "ds1"},
        "phase": "idle",
    }
    base.update(kwargs)
    return AutoMLObservation(**base)


class TestEncoder:
    def test_encode_determinism(self):
        wm = _service()
        o = _obs()
        z1 = wm.encode(o)
        z2 = wm.encode(o)
        assert z1.dim == 32
        assert z1.vector == z2.vector

    def test_encode_goal_changes_latent(self):
        wm = _service()
        o = _obs()
        z = wm.encode(o)
        zg = wm.encode_goal({"goal_type": "train", "problem_type": "classification"}, o)
        assert zg.dim == z.dim
        assert zg.vector != z.vector


class TestPredictor:
    def test_predict_shape(self):
        wm = _service()
        o = _obs()
        z = wm.encode(o)
        z2 = wm.predict(
            z, AutoMLAction(type="start_training", params={"dataset_id": "ds1"})
        )
        assert z2.dim == z.dim
        assert len(z2.vector) == z.dim

    def test_start_training_moves_latent(self):
        wm = _service()
        o = _obs()
        z = wm.encode(o)
        z2 = wm.predict(z, AutoMLAction(type="start_training", params={}))
        # Should not be identical after meaningful action
        assert latent_distance(z, z2) > 0


class TestSurprise:
    def test_identical_near_zero(self):
        wm = _service()
        o = _obs()
        z = wm.encode(o)
        s = compute_surprise(
            z, z, {"metric": "l2", "thresholds": {"medium": 0.1, "high": 0.3}}
        )
        assert s.value == pytest.approx(0.0, abs=1e-9)
        assert s.level == "low"

    def test_contradictory_high(self):
        wm = _service()
        o = _obs()
        z = wm.encode(o)
        o2 = _obs(
            jobs={
                "j1": {
                    "id": "j1",
                    "status": "completed",
                    "best_score": 0.99,
                    "best_model": "rf",
                }
            },
            phase="evaluate",
        )
        z_hat = wm.predict(z, AutoMLAction(type="list_datasets", params={}))
        z_act = wm.encode(o2)
        s = wm.measure_surprise(z_hat, z_act)
        assert s.value > 0


class TestPlanner:
    def test_cem_lite_returns_capped_plans(self):
        wm = _service()
        o = _obs()
        goal: GoalSpec = {
            "goal_type": "train",
            "dataset_id": "ds1",
            "problem_type": "classification",
            "target_column": "target",
            "metric": "f1",
        }
        plans = wm.plan(o, goal)
        assert 1 <= len(plans) <= 2
        for p in plans:
            assert p.steps
            for step in p.steps:
                assert step.action.type in wm.action_space

    def test_analyze_goal_prefers_data_tools(self):
        wm = _service()
        o = _obs()
        plans = wm.plan(o, {"goal_type": "analyze", "dataset_id": "ds1"})
        assert plans
        types = [s.action.type for s in plans[0].steps]
        assert any(
            t in types
            for t in (
                "list_datasets",
                "get_dataset_info",
                "get_features",
                "preview_data",
            )
        )


@pytest.mark.asyncio
async def test_update_logs_trajectory():
    wm = _service()
    o = _obs()
    o2 = _obs(phase="analyze")
    action = AutoMLAction(type="get_dataset_info", params={"dataset_id": "ds1"})
    z, z_hat, z_next, surprise = await wm.update(o, action, o2)
    assert z.dim == z_next.dim
    assert surprise.level in ("low", "medium", "high")
    recent = await wm.trajectory_store.list_recent("u1", limit=5)
    assert len(recent) >= 1


class TestStoreSnapshot:
    @pytest.mark.asyncio
    async def test_get_snapshot(self):
        from tests.test_world_state_store import FakeClient, FakeCollection

        collection = FakeCollection()
        store = WorldStateStore(FakeClient(collection), "db", "world_states", 60)
        await store.ensure("user-1")
        snap = await store.get_snapshot("user-1")
        assert snap is not None
        assert snap["user_id"] == "user-1"
        assert "datasets" in snap


class TestUpdater:
    def test_list_datasets_and_plan_event(self):
        state = WorldState(user_id="u1")
        patch = apply_tool_output(
            state, "list_datasets", {"datasets": [{"id": "ds1", "name": "A"}]}
        )
        assert "ds1" in patch["datasets"]
        state.datasets = patch["datasets"]
        pe = apply_plan_event(
            state,
            "plan_selected",
            {"plan_id": "p1", "title": "t", "steps": []},
        )
        assert pe.get("active_plan_id") == "p1"
        assert "p1" in pe.get("plans", {})


class TestQuery:
    def test_features_of(self):
        o = _obs()
        assert "target" in features_of(o, "ds1")

    def test_format_for_prompt(self):
        text = format_for_prompt(_obs())
        assert "Datasets" in text

    def test_past_best_jobs(self):
        o = _obs(
            jobs={
                "j1": {
                    "id": "j1",
                    "status": "completed",
                    "best_score": 0.9,
                    "config": {"problem_type": "classification"},
                },
                "j2": {"id": "j2", "status": "running", "best_score": 0.99},
            }
        )
        best = past_best_jobs(o, problem_type="classification", top_k=3)
        assert len(best) == 1
        assert best[0]["id"] == "j1"


class TestGoalAndConstraints:
    def test_parse_goal_train(self):
        g = parse_goal(
            "Train classification trên dataset ds1 target target, metric f1 trong 5 phút"
        )
        assert g["goal_type"] == "train"
        assert g.get("problem_type") == "classification"
        assert g.get("metric") == "f1"
        assert g.get("target_column") == "target"
        assert g.get("constraints", {}).get("time_limit") == 300

    def test_parse_goal_docker_e2e_prompt(self):
        """Exact CI Docker E2E prompt style (Vietnamese + Mongo ObjectId)."""
        ds = "6a509cdc2bc66674d0756065"
        g = parse_goal(
            f"Hãy train một model classification trên dataset ID {ds} "
            f"với target column là 'Revenue', dùng 3 thuật toán: "
            f"RandomForestClassifier, XGBClassifier, SVC. "
            f"Dùng metric là accuracy."
        )
        assert g["goal_type"] == "train"
        assert g.get("dataset_id") == ds
        assert g.get("target_column") == "Revenue"
        assert g.get("problem_type") == "classification"
        assert g.get("metric") == "accuracy"
        models = (g.get("constraints") or {}).get("models") or []
        assert "RandomForestClassifier" in models
        assert "XGBClassifier" in models
        assert "SVC" in models

    def test_simple_query(self):
        assert is_simple_query("xin chào", ["xin chào", "hello"])
        assert not is_simple_query("train model classification")

    def test_validate_bad_target(self):
        o = _obs()
        action = AutoMLAction(
            type="start_training",
            params={
                "dataset_id": "ds1",
                "target_column": "not_a_feature",
                "problem_type": "classification",
            },
        )
        res = validate_action(action, o)
        assert not res.ok
        assert any("target_column" in r for r in res.reasons)

    def test_validate_plan_ok(self):
        o = _obs()
        steps = [
            {
                "action": {
                    "type": "get_dataset_info",
                    "params": {"dataset_id": "ds1"},
                }
            }
        ]
        res = validate_plan_steps(steps, o)
        assert res.ok


class TestNoCodeExecTools:
    def test_all_tools_safe(self):
        try:
            from hagent.agent.tools.automl_tools import ALL_TOOLS

            names = [getattr(t, "name", str(t)) for t in ALL_TOOLS]
        except ModuleNotFoundError:
            # Env without langchain: assert from source text
            src = (BACKEND_DIR / "hagent/agent/tools/automl_tools.py").read_text(
                encoding="utf-8"
            )
            assert "async def execute" not in src
            assert "run_script" not in src
            assert "run_python" not in src
            return

        forbidden = {"execute", "run_script", "run_python", "exec", "shell"}
        for n in names:
            assert n not in forbidden
            assert "execute" not in n
            assert "run_script" not in n
