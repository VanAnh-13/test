"""
Tests cho DynamicsEnsemble và latent surprise chuẩn hóa theo σ per-dim.
"""

from __future__ import annotations

import numpy as np
import pytest

from hagent.world.predictor import DynamicsEnsemble, train_dynamics_ensemble
from hagent.world.predictor.factory import create_predictor
from hagent.world.schema import AutoMLAction, LatentState
from hagent.world.surprise import (
    compute_normalized_latent_surprise,
    compute_surprise,
)

DIM = 16
ACTIONS = ["start_training", "get_job_info", "list_datasets"]


def _unit(rng, dim=DIM):
    v = rng.normal(0, 1, dim)
    return v / np.linalg.norm(v)


def _traj_docs(n=80, seed=0):
    """z_next = xoay z một góc cố định phụ thuộc action — học được."""
    rng = np.random.default_rng(seed)
    shift = {a: rng.normal(0, 0.3, DIM) for a in ACTIONS}
    docs = []
    for _ in range(n):
        z = _unit(rng)
        a = str(rng.choice(ACTIONS))
        z_next = z + shift[a]
        z_next = z_next / np.linalg.norm(z_next)
        docs.append(
            {
                "z": {"vector": z.tolist(), "dim": DIM},
                "z_next": {"vector": z_next.tolist(), "dim": DIM},
                "action": {"type": a},
            }
        )
    return docs


def _z(vec):
    return LatentState(vector=list(vec), dim=len(vec))


class TestDynamicsEnsemble:
    def test_untrained_identity_no_fake_std(self):
        ens = DynamicsEnsemble({})
        z = _z([1.0] + [0.0] * (DIM - 1))
        out = ens.predict(z, AutoMLAction(type="start_training"))
        assert out.meta["mode"] == "identity"
        assert "std" not in out.meta
        assert not ens.is_ready

    def test_train_predict_with_std(self):
        ens = train_dynamics_ensemble(
            _traj_docs(), latent_dim=DIM, k=3, hidden_dim=32, epochs=20, seed=0
        )
        assert ens.is_ready
        rng = np.random.default_rng(9)
        out = ens.predict(_z(_unit(rng)), AutoMLAction(type="start_training"))
        assert out.meta["mode"] == "ensemble"
        assert out.meta["n_members"] == 3
        assert len(out.meta["std"]) == DIM
        assert out.meta["std_mean"] > 0
        assert np.isclose(np.linalg.norm(out.vector), 1.0)

    def test_members_disagree(self):
        ens = train_dynamics_ensemble(
            _traj_docs(), latent_dim=DIM, k=3, hidden_dim=32, epochs=10, seed=0
        )
        rng = np.random.default_rng(3)
        out = ens.predict(_z(_unit(rng)), AutoMLAction(type="list_datasets"))
        assert max(out.meta["std"]) > 0

    def test_save_load_roundtrip(self, tmp_path):
        ens = train_dynamics_ensemble(
            _traj_docs(), latent_dim=DIM, k=2, hidden_dim=32, epochs=10, seed=0
        )
        rng = np.random.default_rng(5)
        z = _z(_unit(rng))
        a = AutoMLAction(type="get_job_info")
        before = ens.predict(z, a)

        d = str(tmp_path / "dyn")
        ens.save(d)
        ens2 = DynamicsEnsemble({"checkpoint_dir": d, "hidden_dim": 32})
        assert ens2.is_ready
        after = ens2.predict(z, a)
        assert np.allclose(before.vector, after.vector)
        assert np.allclose(before.meta["std"], after.meta["std"])

    def test_save_empty_raises(self, tmp_path):
        with pytest.raises(RuntimeError):
            DynamicsEnsemble({}).save(str(tmp_path / "x"))

    def test_factory_backend(self):
        ens = create_predictor({"backend": "dynamics_ensemble", "k": 2})
        assert isinstance(ens, DynamicsEnsemble)
        assert ens.k == 2


class TestNormalizedSurprise:
    def _pred_with_std(self, vec, std):
        return LatentState(
            vector=list(vec), dim=len(vec), meta={"std": list(std)}
        )

    def test_zscore_math(self):
        pred = self._pred_with_std([0.0, 0.0], [0.1, 0.1])
        actual = _z([0.1, 0.1])
        r = compute_normalized_latent_surprise(pred, actual)
        assert r.value == pytest.approx(1.0)  # mỗi chiều lệch đúng 1σ
        assert r.level == "low"

    def test_far_actual_is_high(self):
        pred = self._pred_with_std([0.0, 0.0], [0.1, 0.1])
        r = compute_normalized_latent_surprise(pred, _z([0.5, 0.5]))
        assert r.value == pytest.approx(5.0)
        assert r.level == "high"

    def test_sigma_floor_guards_zero_std(self):
        pred = self._pred_with_std([0.0], [0.0])
        r = compute_normalized_latent_surprise(pred, _z([0.5]))
        assert np.isfinite(r.value)

    def test_thresholds_from_config(self):
        pred = self._pred_with_std([0.0], [0.1])
        cfg = {"normalized_thresholds": {"medium": 0.5, "high": 1.0}}
        assert compute_normalized_latent_surprise(pred, _z([0.06]), cfg).level == "medium"
        assert compute_normalized_latent_surprise(pred, _z([0.2]), cfg).level == "high"

    def test_compute_surprise_autodetects_std(self):
        """meta.std → z-units; không std → khoảng cách thô như cũ."""
        pred_std = self._pred_with_std([0.0, 0.0], [0.1, 0.1])
        actual = _z([0.3, 0.3])
        r_norm = compute_surprise(pred_std, actual)
        assert r_norm.value == pytest.approx(3.0)  # z-units

        pred_raw = _z([0.0, 0.0])
        r_raw = compute_surprise(pred_raw, actual)
        assert r_raw.value == pytest.approx(np.sqrt(0.18))  # L2 thô

    def test_ensemble_end_to_end_surprise(self):
        """Transition đúng phân phối → surprise thấp hơn transition lạ."""
        docs = _traj_docs(120, seed=0)
        ens = train_dynamics_ensemble(
            docs, latent_dim=DIM, k=3, hidden_dim=32, epochs=30, seed=0
        )
        doc = docs[0]
        z = _z(doc["z"]["vector"])
        a = AutoMLAction(type=doc["action"]["type"])
        pred = ens.predict(z, a)

        in_dist = compute_surprise(pred, _z(doc["z_next"]["vector"]))
        rng = np.random.default_rng(42)
        out_dist = compute_surprise(pred, _z(_unit(rng)))
        assert in_dist.value < out_dist.value
