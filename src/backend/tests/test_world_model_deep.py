"""
Deep World Model integration tests — neural predictor, trajectory factory, hooks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from hagent.world.predictor.factory import create_predictor
from hagent.world.predictor.neural_jepa_v1 import (
    NeuralJepaV1Predictor,
    train_neural_jepa,
)
from hagent.world.schema import AutoMLAction, AutoMLObservation, LatentState
from hagent.world.service import WorldModelService
from hagent.world.trajectory_store import TrajectoryStore, create_trajectory_store


def _obs(user_id: str = "u1") -> AutoMLObservation:
    return AutoMLObservation(
        user_id=user_id,
        datasets={
            "ds1": {
                "id": "ds1",
                "name": "glass",
                "n_rows": 200,
                "n_cols": 10,
                "features": ["a", "b", "target"],
                "target": "target",
            }
        },
        jobs={},
        phase="analyze",
    )


class TestNeuralPredictor:
    def test_factory_neural_without_checkpoint_falls_back(self):
        pred = create_predictor(
            {
                "backend": "neural_jepa_v1",
                "hidden_dim": 32,
                "fallback": "tabular_transition_v1",
            }
        )
        z = LatentState(vector=[0.1] * 16, dim=16)
        a = AutoMLAction(type="list_datasets", params={})
        z2 = pred.predict(z, a)
        assert z2.dim == 16
        assert len(z2.vector) == 16

    def test_train_and_save_load(self, tmp_path):
        # Build synthetic traj via tabular service
        wm = WorldModelService.from_config(
            {
                "encoder": {"backend": "structured_v1", "dim": 32},
                "predictor": {"backend": "tabular_transition_v1"},
                "planner": {
                    "backend": "cem_lite",
                    "horizon": 2,
                    "n_candidates": 2,
                    "n_return_plans": 1,
                },
                "trajectory": {"enabled": True, "max_per_user": 100},
            }
        )
        docs = []
        for at in (
            "list_datasets",
            "get_dataset_info",
            "start_training",
            "get_job_info",
        ):
            obs = _obs()
            next_obs = _obs()
            next_obs.jobs = {
                "j1": {"id": "j1", "status": "completed", "best_score": 0.9}
            }
            action = AutoMLAction(type=at, params={})
            z = wm.encode(obs)
            z_hat = wm.predict(z, action)
            z_next = wm.encode(next_obs)
            docs.append(
                {
                    "action": action.to_dict(),
                    "z": z.to_dict(),
                    "z_hat": z_hat.to_dict(),
                    "z_next": z_next.to_dict(),
                }
            )
        # Repeat for more samples
        docs = docs * 8
        pred = train_neural_jepa(
            docs, latent_dim=32, hidden_dim=32, epochs=5, lr=0.05, seed=1
        )
        path = tmp_path / "jepa.npz"
        pred.save(str(path), latent_dim=32)
        loaded = NeuralJepaV1Predictor(
            {
                "checkpoint_path": str(path),
                "hidden_dim": 32,
                "fallback": "tabular_transition_v1",
            }
        )
        assert loaded.loaded
        z = LatentState(vector=[0.05] * 32, dim=32)
        out = loaded.predict(z, AutoMLAction(type="list_datasets", params={}))
        assert out.meta.get("predictor") == "neural_jepa_v1"
        assert out.meta.get("mode") == "neural"


class TestTrajectoryFactory:
    def test_memory_store_append_and_list(self):
        store = create_trajectory_store(None)
        assert store.collection is None

    @pytest.mark.asyncio
    async def test_append_memory(self):
        store = TrajectoryStore(collection=None, max_per_user=10)
        from hagent.world.surprise import compute_surprise

        obs = _obs()
        action = AutoMLAction(type="list_datasets", params={})
        z = LatentState(vector=[0.1, 0.2], dim=2)
        z2 = LatentState(vector=[0.2, 0.1], dim=2)
        s = compute_surprise(
            z, z2, {"metric": "l2", "thresholds": {"medium": 0.1, "high": 0.5}}
        )
        await store.append(
            user_id="u1",
            observation=obs,
            action=action,
            next_observation=obs,
            z=z,
            z_hat=z,
            z_next=z2,
            surprise=s,
        )
        recent = await store.list_recent("u1", limit=5)
        assert len(recent) == 1
        all_docs = await store.list_all(user_id="u1")
        assert len(all_docs) == 1


class TestWmHooks:
    @pytest.mark.asyncio
    async def test_campaign_wm_step(self):
        from hagent.agent.campaign.wm_hooks import (
            blend_score_with_surprise,
            campaign_wm_step,
        )

        wm = WorldModelService.from_config(
            {
                "encoder": {"backend": "structured_v1", "dim": 32},
                "predictor": {"backend": "tabular_transition_v1"},
                "planner": {"backend": "cem_lite", "horizon": 2, "n_candidates": 2},
                "trajectory": {"enabled": True},
            }
        )
        before = {
            "user_id": "u1",
            "datasets": {"ds1": {"id": "ds1", "name": "g"}},
            "jobs": {},
            "phase": "train",
        }
        after = {
            **before,
            "jobs": {"j1": {"id": "j1", "status": "starting"}},
        }
        surprise, snap = await campaign_wm_step(
            wm_service=wm,
            world_model=before,
            user_id="u1",
            action_type="start_training",
            params={"dataset_id": "ds1"},
            next_world_model=after,
        )
        assert surprise is not None
        assert "value" in surprise or "level" in surprise
        assert snap is not None
        assert blend_score_with_surprise(0.9, surprise) <= 0.9


class TestServiceMongoOptional:
    def test_from_config_with_none_client(self):
        wm = WorldModelService.from_config(
            {
                "encoder": {"backend": "structured_v1", "dim": 16},
                "predictor": {"backend": "tabular_transition_v1"},
                "planner": {"backend": "cem_lite", "horizon": 2, "n_candidates": 2},
                "trajectory": {"enabled": True, "max_per_user": 50},
            },
            mongo_client=None,
        )
        z = wm.encode(_obs())
        assert z.dim == 16
