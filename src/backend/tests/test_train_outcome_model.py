"""
Tests cho T3 — pipeline train checkpoint outcome model + memoize.
Không cần Mongo: nguồn JSONL, path checkpoint trỏ tmp.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).parent.parent / "scripts" / "train_outcome_model.py"
_spec = importlib.util.spec_from_file_location("train_outcome_model", _SCRIPT)
tom = importlib.util.module_from_spec(_spec)
sys.modules["train_outcome_model"] = tom
_spec.loader.exec_module(tom)


def _traj_doc(job_id, algo="grid_search", t=180, score=0.85):
    return {
        "next_observation": {
            "jobs": {
                job_id: {
                    "id": job_id,
                    "dataset_id": "ds1",
                    "status": "completed",
                    "best_score": score,
                    "config": {
                        "search_algorithm": algo,
                        "problem_type": "classification",
                        "metric": "accuracy",
                        "time_limit": t,
                    },
                }
            },
            "datasets": {"ds1": {"n_rows": 500, "n_cols": 8}},
        },
        "z_next": {"vector": [0.1] * 64, "dim": 64},
    }


def _make_docs(n=30, seed=0):
    rng = np.random.default_rng(seed)
    algos = ["grid_search", "bayesian_search", "genetic_algorithm"]
    return [
        _traj_doc(
            f"j{i}",
            algo=str(rng.choice(algos)),
            t=int(rng.choice([60, 180, 600])),
            score=float(0.7 + rng.normal(0, 0.05)),
        )
        for i in range(n)
    ]


class TestJsonlLoading:
    def test_roundtrip(self, tmp_path):
        docs = _make_docs(5)
        p = tmp_path / "traj.jsonl"
        p.write_text(
            "\n".join(json.dumps(d) for d in docs), encoding="utf-8"
        )
        loaded = tom.load_docs_jsonl(str(p))
        assert len(loaded) == 5
        assert loaded[0]["z_next"]["dim"] == 64


class TestTrainAndSave:
    def test_dry_run_counts_only(self, tmp_path):
        report = tom.train_and_save(_make_docs(10), dry_run=True)
        assert report["n_samples"] == 10
        assert "head_path" not in report

    def test_too_few_samples_fails_loud(self, tmp_path):
        with pytest.raises(SystemExit):
            tom.train_and_save(
                _make_docs(3),
                head_path=str(tmp_path / "h.npz"),
                ensemble_dir=str(tmp_path / "e"),
                min_samples=8,
            )

    def test_trains_and_writes_checkpoints(self, tmp_path):
        head_path = tmp_path / "outcome_head.npz"
        ens_dir = tmp_path / "ensemble"
        report = tom.train_and_save(
            _make_docs(30),
            head_path=str(head_path),
            ensemble_dir=str(ens_dir),
            epochs=30,
            seed=0,
            k=2,
        )
        assert head_path.is_file()
        assert len(list(ens_dir.glob("member_*.npz"))) == 2
        assert len(report["head_sha256"]) == 16
        assert len(report["ensemble_members"]) == 2
        assert report["final_nll"] is not None

    def test_checkpoint_loads_via_auto_path(self, tmp_path, monkeypatch):
        """Checkpoint script sinh ra phải nạp được qua _default_outcome_model
        — đúng đường 'auto' mà builder/runner production dùng."""
        head_path = tmp_path / "outcome_head.npz"
        ens_dir = tmp_path / "ensemble"
        tom.train_and_save(
            _make_docs(30),
            head_path=str(head_path),
            ensemble_dir=str(ens_dir),
            epochs=20,
            k=2,
        )

        import hagent.agent.campaign.wm_hooks as wm_hooks

        monkeypatch.setattr(
            wm_hooks,
            "get_world_model_config",
            lambda: {
                "outcome_head": {"checkpoint_path": str(head_path), "use_latent": False},
                "outcome_ensemble": {"checkpoint_dir": str(ens_dir), "use_latent": False},
            },
            raising=False,
        )
        # wm_hooks import get_world_model_config bên trong hàm → patch module đích
        import hagent.bridge.config as bridge_config

        monkeypatch.setattr(
            bridge_config,
            "get_world_model_config",
            lambda: {
                "outcome_head": {"checkpoint_path": str(head_path), "use_latent": False},
                "outcome_ensemble": {"checkpoint_dir": str(ens_dir), "use_latent": False},
            },
        )
        wm_hooks._outcome_model_cache["fingerprint"] = None  # xóa memo cũ

        model = wm_hooks._default_outcome_model()
        assert model is not None and model.is_ready
        pred = model.predict(
            {"search_algorithm": "grid_search", "problem_type": "classification",
             "metric": "accuracy", "time_limit": 180},
            {"n_rows": 500, "n_cols": 8},
        )
        assert pred is not None and np.isfinite(pred.mean)


class TestMemoization:
    def _patch_cfg(self, monkeypatch, head_path, ens_dir):
        import hagent.bridge.config as bridge_config

        cfg = {
            "outcome_head": {"checkpoint_path": str(head_path), "use_latent": False},
            "outcome_ensemble": {"checkpoint_dir": str(ens_dir), "use_latent": False},
        }
        monkeypatch.setattr(bridge_config, "get_world_model_config", lambda: cfg)

    def test_same_object_until_file_changes(self, tmp_path, monkeypatch):
        head_path = tmp_path / "h.npz"
        ens_dir = tmp_path / "e"
        tom.train_and_save(
            _make_docs(30), head_path=str(head_path), ensemble_dir=str(ens_dir),
            epochs=10, k=2,
        )
        import hagent.agent.campaign.wm_hooks as wm_hooks

        self._patch_cfg(monkeypatch, head_path, ens_dir)
        wm_hooks._outcome_model_cache["fingerprint"] = None

        m1 = wm_hooks._default_outcome_model()
        m2 = wm_hooks._default_outcome_model()
        assert m1 is m2  # memo hit — không dựng lại từ đĩa

        # Retrain (mtime đổi) → nạp model mới
        time.sleep(0.05)
        tom.train_and_save(
            _make_docs(30, seed=1), head_path=str(head_path),
            ensemble_dir=str(ens_dir), epochs=10, k=2,
        )
        import os
        os.utime(head_path)
        m3 = wm_hooks._default_outcome_model()
        assert m3 is not m1

    def test_missing_checkpoints_memoized_none(self, tmp_path, monkeypatch):
        import hagent.agent.campaign.wm_hooks as wm_hooks

        self._patch_cfg(monkeypatch, tmp_path / "nope.npz", tmp_path / "noe")
        wm_hooks._outcome_model_cache["fingerprint"] = None
        assert wm_hooks._default_outcome_model() is None
        assert wm_hooks._default_outcome_model() is None  # không nổ khi gọi lại
