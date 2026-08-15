"""
Comprehensive unit tests for World Model latent planners (REFAC-014).
"""

from __future__ import annotations

import pytest

from hagent.world.planner.cem_lite import CEMLitePlanner, _params_for_action
from hagent.world.planner.factory import create_campaign_planner, create_planner
from hagent.world.predictor.tabular_transition_v1 import TabularTransitionV1Predictor
from hagent.world.schema import GoalSpec, LatentState


def test_create_planner_and_campaign_factories() -> None:
    """Planner factories khởi tạo chính xác."""
    predictor = TabularTransitionV1Predictor()
    planner = create_planner(predictor, {"backend": "cem_lite"})
    assert isinstance(planner, CEMLitePlanner)

    with pytest.raises(ValueError, match="Unsupported world_model.planner.backend"):
        create_planner(predictor, {"backend": "unknown_planner"})

    assert create_campaign_planner({"enabled": False}) is None
    c_planner = create_campaign_planner({"backend": "cem_config_v1"})
    assert c_planner is not None

    mpc_planner = create_campaign_planner({"backend": "cem_mpc_v1"})
    assert mpc_planner is not None


def test_params_for_action() -> None:
    """_params_for_action điền tham số tự động từ goal và context."""
    goal: GoalSpec = {
        "goal_type": "train",
        "dataset_id": "ds_123",
        "problem_type": "classification",
        "target_column": "target",
        "metric": "accuracy",
        "constraints": {"time_limit": 300},
    }
    ctx = {"user_id": "usr_abc"}

    params_train = _params_for_action("start_training", goal, ctx)
    assert params_train["dataset_id"] == "ds_123"
    assert params_train["user_id"] == "usr_abc"
    assert params_train["problem_type"] == "classification"
    assert params_train["time_limit"] == 300

    params_data = _params_for_action("get_features", goal, ctx)
    assert params_data["dataset_id"] == "ds_123"


def test_cem_lite_plan_generation() -> None:
    """CEMLitePlanner tạo ra các candidate plans với rank và score."""
    predictor = TabularTransitionV1Predictor()
    planner = CEMLitePlanner(predictor)

    z0 = LatentState(vector=[0.0] * 8, dim=8)
    zg = LatentState(vector=[1.0] * 8, dim=8)

    goal: GoalSpec = {
        "goal_type": "train",
        "dataset_id": "ds_iris",
        "problem_type": "classification",
        "target_column": "species",
    }

    action_space = [
        "list_datasets",
        "get_dataset_info",
        "get_features",
        "get_available_models",
        "start_training",
        "get_job_info",
        "list_jobs",
    ]

    plans = planner.plan(
        z0=z0,
        z_goal=zg,
        goal=goal,
        action_space=action_space,
        observation_context={"user_id": "u1", "dataset_id": "ds_iris"},
    )

    assert len(plans) > 0
    best_plan = plans[0]
    assert isinstance(best_plan.plan_id, str) and len(best_plan.plan_id) > 0
    assert len(best_plan.steps) > 0
    assert best_plan.cost >= 0.0
    # Steps are mapped to appropriate specialist agents
    for step in best_plan.steps:
        assert step.agent is not None


def test_cem_lite_action_space_filtering() -> None:
    """CEMLitePlanner loại bỏ các kế hoạch chứa hành động nằm ngoài action_space."""
    predictor = TabularTransitionV1Predictor()
    planner = CEMLitePlanner(predictor)

    z0 = LatentState(vector=[0.0] * 8, dim=8)
    zg = LatentState(vector=[1.0] * 8, dim=8)

    goal: GoalSpec = {"goal_type": "analyze", "dataset_id": "ds_test"}

    # Action space chỉ cho phép list_datasets
    restricted_space = ["list_datasets"]

    plans = planner.plan(
        z0=z0,
        z_goal=zg,
        goal=goal,
        action_space=restricted_space,
    )

    for p in plans:
        for s in p.steps:
            assert s.action.type in restricted_space
