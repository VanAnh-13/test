"""Test hồi quy cho phép chiếu world state và campaign dùng chung."""

from hagent.agent.campaign.schema import CampaignVariant
from hagent.world.schema import WorldState


def test_execution_snapshot_hydration_preserves_existing_field_contract():
    snapshot = {
        "user_id": "snapshot-owner",
        "datasets": {"dataset-1": {"id": "dataset-1"}},
        "jobs": {"job-1": {"id": "job-1", "status": "running"}},
        "goals": [{"goal_id": "goal-1"}],
        "plans": {"plan-1": {"plan_id": "plan-1"}},
        "active_plan_id": "plan-1",
        "active_dataset_id": "dataset-1",
        "active_job_id": "job-1",
        "active_goal": {"goal_type": "train", "description": "Train"},
        "phase": "execute",
        "last_surprise": {"level": "high"},
    }

    state = WorldState.from_execution_snapshot(snapshot, user_id="request-owner")

    assert state.user_id == "request-owner"
    assert state.datasets == snapshot["datasets"]
    assert state.jobs == snapshot["jobs"]
    assert state.goals == snapshot["goals"]
    assert state.plans == snapshot["plans"]
    assert state.active_plan_id == "plan-1"
    assert state.active_dataset_id == "dataset-1"
    assert state.active_job_id == "job-1"
    assert state.active_goal == snapshot["active_goal"]
    assert state.phase == "execute"
    assert state.last_surprise is None


def test_campaign_variant_owns_submission_and_full_job_entries():
    params = {"dataset_id": "dataset-1", "model": "random_forest"}
    variant = CampaignVariant(
        variant_id="variant-1",
        label="baseline",
        params=params,
        job_id="job-1",
        status="completed",
        metrics={"accuracy": 0.9},
        best_model="RandomForestClassifier",
        best_score=0.9,
    )

    assert variant.to_submission_job_entry() == {
        "id": "job-1",
        "status": "completed",
        "config": params,
        "dataset_id": "dataset-1",
    }
    assert variant.to_job_entry() == {
        "id": "job-1",
        "status": "completed",
        "best_model": "RandomForestClassifier",
        "best_score": 0.9,
        "metrics": {"accuracy": 0.9},
        "config": params,
        "dataset_id": "dataset-1",
    }
