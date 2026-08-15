"""
Unit tests for World Model schema versioning and document migration.
"""

from __future__ import annotations

import pytest

from hagent.world.schema import (
    CURRENT_SCHEMA_VERSION,
    AutoMLAction,
    AutoMLObservation,
    LatentState,
    PlanResult,
    PlanStep,
    SurpriseResult,
    WorldState,
)
from hagent.world.schema_migration import (
    migrate,
    migrate_trajectory_doc,
    migrate_world_state_doc,
)
from hagent.world.state_store import _world_state_from_doc
from hagent.world.trajectory_store import TrajectoryStore


def test_schema_models_have_default_schema_version() -> None:
    """Tất cả Pydantic / dataclass schemas phải có schema_version = '1.0'."""
    assert CURRENT_SCHEMA_VERSION == "1.0"

    obs = AutoMLObservation(user_id="test-user")
    assert obs.schema_version == "1.0"
    assert obs.to_dict()["schema_version"] == "1.0"

    action = AutoMLAction(type="train")
    assert action.schema_version == "1.0"
    assert action.to_dict()["schema_version"] == "1.0"

    latent = LatentState(vector=[0.1, 0.2], dim=2)
    assert latent.schema_version == "1.0"
    assert latent.to_dict()["schema_version"] == "1.0"

    surprise = SurpriseResult(value=0.5, level="medium", predicted_dim=2, actual_dim=2)
    assert surprise.schema_version == "1.0"
    assert surprise.to_dict()["schema_version"] == "1.0"

    step = PlanStep(action=action)
    assert step.schema_version == "1.0"
    assert step.to_dict()["schema_version"] == "1.0"

    plan = PlanResult(plan_id="p1", steps=[step], cost=1.0)
    assert plan.schema_version == "1.0"
    assert plan.to_dict()["schema_version"] == "1.0"

    state = WorldState(user_id="test-user")
    assert state.schema_version == "1.0"
    assert state.to_dict()["schema_version"] == "1.0"


def test_migrate_world_state_doc_unversioned() -> None:
    """Tài liệu legacy không có schema_version phải được nâng cấp lên '1.0'."""
    legacy_doc = {
        "user_id": "u123",
        "datasets": {"d1": {"id": "d1", "name": "iris"}},
        "active_dataset_id": "d1",
    }
    migrated = migrate_world_state_doc(legacy_doc)
    assert migrated["schema_version"] == "1.0"
    assert migrated["datasets"] == legacy_doc["datasets"]
    assert migrated["plans"] == {}
    assert migrated["goals"] == []
    assert migrated["phase"] == "idle"
    assert migrated["cost_metrics"] == {}


def test_migrate_world_state_doc_legacy_version() -> None:
    """Tài liệu schema_version 0.x phải được nâng cấp lên '1.0'."""
    old_doc = {
        "user_id": "u456",
        "schema_version": "0.1",
        "jobs": {"j1": {"id": "j1", "status": "completed"}},
    }
    migrated = migrate(old_doc, doc_type="world_state")
    assert migrated["schema_version"] == "1.0"
    assert migrated["jobs"] == old_doc["jobs"]
    assert migrated["datasets"] == {}


def test_migrate_world_state_doc_already_current() -> None:
    """Tài liệu đã ở version '1.0' phải giữ nguyên."""
    current_doc = {
        "user_id": "u789",
        "schema_version": "1.0",
        "phase": "running",
        "cost_metrics": {"total_cost": 0.05},
    }
    migrated = migrate_world_state_doc(current_doc)
    assert migrated["schema_version"] == "1.0"
    assert migrated["phase"] == "running"
    assert migrated["cost_metrics"] == {"total_cost": 0.05}


def test_migrate_trajectory_doc() -> None:
    """Tài liệu trajectory legacy phải được nâng cấp schema_version cả ở cấp root và sub-docs."""
    legacy_traj = {
        "user_id": "u1",
        "observation": {"user_id": "u1", "phase": "idle"},
        "action": {"type": "fit_model", "params": {}},
        "z": {"vector": [0.0], "dim": 1},
        "surprise": {"value": 0.1, "level": "low", "predicted_dim": 1, "actual_dim": 1},
        "created_at": "2026-08-14T00:00:00Z",
    }
    migrated = migrate_trajectory_doc(legacy_traj)
    assert migrated["schema_version"] == "1.0"
    assert migrated["observation"]["schema_version"] == "1.0"
    assert migrated["action"]["schema_version"] == "1.0"
    assert migrated["z"]["schema_version"] == "1.0"
    assert migrated["surprise"]["schema_version"] == "1.0"


def test_state_store_helper_instantiates_migrated_world_state() -> None:
    """_world_state_from_doc phải trả về đối tượng WorldState hợp lệ với schema_version '1.0'."""
    raw_doc = {"user_id": "owner-1"}
    state = _world_state_from_doc(raw_doc)
    assert isinstance(state, WorldState)
    assert state.schema_version == "1.0"
    assert state.user_id == "owner-1"
    assert state.phase == "idle"


@pytest.mark.asyncio
async def test_trajectory_store_list_returns_migrated_documents() -> None:
    """TrajectoryStore list_recent và list_all phải trả về documents có schema_version '1.0'."""
    store = TrajectoryStore(enabled=True)
    obs = AutoMLObservation(user_id="u1")
    act = AutoMLAction(type="predict")
    z = LatentState(vector=[1.0], dim=1)
    surp = SurpriseResult(value=0.0, level="low", predicted_dim=1, actual_dim=1)

    # Thêm transition vào store
    await store.append(
        user_id="u1",
        observation=obs,
        action=act,
        next_observation=obs,
        z=z,
        z_hat=z,
        z_next=z,
        surprise=surp,
    )

    recent = await store.list_recent("u1", limit=10)
    assert len(recent) == 1
    assert recent[0]["schema_version"] == "1.0"

    all_docs = await store.list_all(user_id="u1")
    assert len(all_docs) == 1
    assert all_docs[0]["schema_version"] == "1.0"
