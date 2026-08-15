"""
Tests cho T3 — pipeline train checkpoint outcome model + memoize.
Không cần Mongo: nguồn JSONL, path checkpoint trỏ tmp.
"""

from __future__ import annotations

import hashlib
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _rewrite_npz(path: Path, **updates) -> None:
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    values.update(updates)
    np.savez(path, **values)


class TestJsonlLoading:
    def test_roundtrip(self, tmp_path):
        docs = _make_docs(5)
        p = tmp_path / "traj.jsonl"
        p.write_text("\n".join(json.dumps(d) for d in docs), encoding="utf-8")
        loaded = tom.load_docs_jsonl(str(p))
        assert len(loaded) == 5
        assert loaded[0]["z_next"]["dim"] == 64


class TestTrainAndSave:
    def test_manifest_records_and_validates_complete_artifact_set(self, tmp_path):
        head_path = tmp_path / "outcome_head.npz"
        ens_dir = tmp_path / "ensemble"
        source_sha256 = "ab" * 32

        report = tom.train_from_docs(
            _make_docs(30),
            head_path=str(head_path),
            ensemble_dir=str(ens_dir),
            epochs=5,
            seed=7,
            k=2,
            source_sha256=source_sha256,
        )

        manifest_path = ens_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["schema_version"] == 1
        assert manifest["vocabulary_version"] == "outcome-v2"
        assert len(manifest["vocabulary_sha256"]) == 64
        assert manifest["source_sha256"] == source_sha256
        assert len(manifest["config_sha256"]) == 64
        assert manifest["config"]["outcome_ensemble"]["k"] == 2
        assert "checkpoint_path" not in manifest["config"]["outcome_head"]
        assert "checkpoint_dir" not in manifest["config"]["outcome_ensemble"]
        with np.load(head_path, allow_pickle=False) as checkpoint:
            assert (
                checkpoint["search_algorithms"].tolist()
                == manifest["vocabulary"]["outcome_head"]["search_algorithms"]
            )
        assert manifest["training"] == {
            "epochs": 5,
            "seed": 7,
            "ensemble_size": 2,
            "min_samples": 8,
            "n_trajectory_docs": 30,
            "n_samples": 30,
        }
        assert manifest["head"] == {
            "filename": "outcome_head.npz",
            "sha256": _sha256(head_path),
        }
        assert manifest["ensemble_members"] == [
            {"filename": "member_0.npz", "sha256": _sha256(ens_dir / "member_0.npz")},
            {"filename": "member_1.npz", "sha256": _sha256(ens_dir / "member_1.npz")},
        ]
        assert report["manifest_path"] == str(manifest_path)
        assert tom.validate_checkpoint_manifest(head_path, ens_dir) == manifest
        assert not list(ens_dir.glob("manifest.json.*"))

    @pytest.mark.parametrize(
        "failure_mode",
        ["missing_member", "extra_member", "tampered_member", "tampered_head"],
    )
    def test_validator_rejects_artifact_set_mismatch(self, tmp_path, failure_mode):
        head_path = tmp_path / "outcome_head.npz"
        ens_dir = tmp_path / "ensemble"
        tom.train_from_docs(
            _make_docs(30),
            head_path=str(head_path),
            ensemble_dir=str(ens_dir),
            epochs=5,
            k=2,
        )

        if failure_mode == "missing_member":
            (ens_dir / "member_1.npz").unlink()
        elif failure_mode == "extra_member":
            (ens_dir / "member_2.npz").write_bytes(b"unexpected")
        elif failure_mode == "tampered_member":
            with (ens_dir / "member_0.npz").open("ab") as fh:
                fh.write(b"tampered")
        else:
            with head_path.open("ab") as fh:
                fh.write(b"tampered")

        with pytest.raises(ValueError, match="Invalid checkpoint manifest"):
            tom.validate_checkpoint_manifest(head_path, ens_dir, expected_k=2)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("schema_version", 999),
            ("vocabulary_version", "unknown"),
            ("source_sha256", "not-a-sha"),
            ("config_sha256", "0" * 64),
            ("vocabulary_sha256", "f" * 64),
        ],
    )
    def test_validator_rejects_invalid_manifest_metadata(self, tmp_path, field, value):
        head_path = tmp_path / "outcome_head.npz"
        ens_dir = tmp_path / "ensemble"
        tom.train_from_docs(
            _make_docs(30),
            head_path=str(head_path),
            ensemble_dir=str(ens_dir),
            epochs=5,
            k=2,
        )
        manifest_path = ens_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest[field] = value
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid checkpoint manifest"):
            tom.validate_checkpoint_manifest(head_path, ens_dir, expected_k=2)

    def test_validator_rejects_rehashed_vocabulary_checkpoint_mismatch(self, tmp_path):
        head_path = tmp_path / "outcome_head.npz"
        ens_dir = tmp_path / "ensemble"
        tom.train_from_docs(
            _make_docs(30),
            head_path=str(head_path),
            ensemble_dir=str(ens_dir),
            epochs=5,
            k=2,
        )
        manifest_path = ens_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["vocabulary"]["outcome_head"]["search_algorithms"] = [
            "forged_algorithm"
        ]
        manifest["vocabulary_sha256"] = _canonical_sha256(manifest["vocabulary"])
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="checkpoint vocabulary mismatch"):
            tom.validate_checkpoint_manifest(head_path, ens_dir, expected_k=2)

    @pytest.mark.parametrize(
        ("field", "value", "error"),
        [
            (
                "search_algorithms",
                np.asarray(["forged_algorithm"], dtype=np.str_),
                "checkpoint vocabulary mismatch",
            ),
            ("hidden_dim", np.asarray(999), "checkpoint config mismatch"),
        ],
    )
    def test_validator_rejects_rehashed_checkpoint_metadata_mismatch(
        self, tmp_path, field, value, error
    ):
        head_path = tmp_path / "outcome_head.npz"
        ens_dir = tmp_path / "ensemble"
        tom.train_from_docs(
            _make_docs(30),
            head_path=str(head_path),
            ensemble_dir=str(ens_dir),
            epochs=5,
            k=2,
        )
        manifest_path = ens_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _rewrite_npz(head_path, **{field: value})
        manifest["head"]["sha256"] = _sha256(head_path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match=error):
            tom.validate_checkpoint_manifest(head_path, ens_dir, expected_k=2)

    def test_validator_rejects_incomplete_training_metadata(self, tmp_path):
        head_path = tmp_path / "outcome_head.npz"
        ens_dir = tmp_path / "ensemble"

        tom.train_from_docs(
            _make_docs(30),
            head_path=str(head_path),
            ensemble_dir=str(ens_dir),
            epochs=5,
            k=2,
        )
        manifest_path = ens_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        del manifest["training"]["epochs"]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="training fields"):
            tom.validate_checkpoint_manifest(head_path, ens_dir, expected_k=2)

    def test_validator_rejects_wrong_expected_member_count(self, tmp_path):

        head_path = tmp_path / "outcome_head.npz"
        ens_dir = tmp_path / "ensemble"
        tom.train_from_docs(
            _make_docs(30),
            head_path=str(head_path),
            ensemble_dir=str(ens_dir),
            epochs=5,
            k=2,
        )

        with pytest.raises(ValueError, match="unexpected ensemble_size"):
            tom.validate_checkpoint_manifest(head_path, ens_dir, expected_k=5)

    def test_atomic_replace_failure_leaves_no_manifest(self, tmp_path, monkeypatch):
        head_path = tmp_path / "outcome_head.npz"
        ens_dir = tmp_path / "ensemble"

        def fail_replace(_source, _destination):
            raise OSError("replace failed")

        monkeypatch.setattr(tom.os, "replace", fail_replace)
        with pytest.raises(OSError, match="replace failed"):
            tom.train_from_docs(
                _make_docs(30),
                head_path=str(head_path),
                ensemble_dir=str(ens_dir),
                epochs=5,
                k=2,
            )

        assert not (ens_dir / "manifest.json").exists()
        assert not (ens_dir / "manifest.json.tmp").exists()

    def test_self_validation_io_failure_unpublishes_manifest(
        self, tmp_path, monkeypatch
    ):
        head_path = tmp_path / "outcome_head.npz"
        ens_dir = tmp_path / "ensemble"
        original_read_bytes = Path.read_bytes
        hash_reads = 0

        def fail_during_self_validation(path):
            nonlocal hash_reads
            hash_reads += 1
            if hash_reads == 4:
                raise OSError("checkpoint became unreadable")
            return original_read_bytes(path)

        monkeypatch.setattr(Path, "read_bytes", fail_during_self_validation)
        with pytest.raises(OSError, match="checkpoint became unreadable"):
            tom.train_from_docs(
                _make_docs(30),
                head_path=str(head_path),
                ensemble_dir=str(ens_dir),
                epochs=5,
                k=2,
            )

        assert hash_reads == 4
        assert not (ens_dir / "manifest.json").exists()
        assert not (ens_dir / "manifest.json.tmp").exists()

    def test_partial_member_save_invalidates_stale_manifest(self, tmp_path):
        head_path = tmp_path / "outcome_head.npz"
        ens_dir = tmp_path / "ensemble"
        ens_dir.mkdir()
        (ens_dir / "manifest.json").write_text("{}", encoding="utf-8")
        (ens_dir / "manifest.json.tmp").write_text("stale", encoding="utf-8")
        (ens_dir / "member_1.npz").mkdir()

        with pytest.raises(OSError):
            tom.train_from_docs(
                _make_docs(30),
                head_path=str(head_path),
                ensemble_dir=str(ens_dir),
                epochs=5,
                k=2,
            )

        assert not (ens_dir / "manifest.json").exists()
        assert not (ens_dir / "manifest.json.tmp").exists()

    def test_retraining_removes_stale_extra_member(self, tmp_path):
        head_path = tmp_path / "outcome_head.npz"
        ens_dir = tmp_path / "ensemble"
        ens_dir.mkdir()
        (ens_dir / "member_9.npz").write_bytes(b"stale")

        tom.train_from_docs(
            _make_docs(30),
            head_path=str(head_path),
            ensemble_dir=str(ens_dir),
            epochs=5,
            k=2,
        )

        assert sorted(path.name for path in ens_dir.glob("member_*.npz")) == [
            "member_0.npz",
            "member_1.npz",
        ]
        tom.validate_checkpoint_manifest(head_path, ens_dir, expected_k=2)

    def test_validator_rejects_manifest_member_hash_mismatch(self, tmp_path):
        head_path = tmp_path / "outcome_head.npz"
        ens_dir = tmp_path / "ensemble"
        tom.train_from_docs(
            _make_docs(30),
            head_path=str(head_path),
            ensemble_dir=str(ens_dir),
            epochs=5,
            k=2,
        )
        manifest_path = ens_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["ensemble_members"][0]["sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="hash mismatch for member_0.npz"):
            tom.validate_checkpoint_manifest(head_path, ens_dir, expected_k=2)

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
        assert report["head_sha256"] == _sha256(head_path)
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

        from hagent.agent.campaign import wm_hooks

        monkeypatch.setattr(
            wm_hooks,
            "get_world_model_config",
            lambda: {
                "outcome_head": {
                    "checkpoint_path": str(head_path),
                    "use_latent": False,
                },
                "outcome_ensemble": {
                    "checkpoint_dir": str(ens_dir),
                    "use_latent": False,
                },
            },
            raising=False,
        )
        # wm_hooks import get_world_model_config bên trong hàm → patch module đích
        import hagent.bridge.config as bridge_config

        monkeypatch.setattr(
            bridge_config,
            "get_world_model_config",
            lambda: {
                "outcome_head": {
                    "checkpoint_path": str(head_path),
                    "use_latent": False,
                },
                "outcome_ensemble": {
                    "checkpoint_dir": str(ens_dir),
                    "use_latent": False,
                },
            },
        )
        wm_hooks._outcome_model_cache["fingerprint"] = None  # xóa memo cũ

        model = wm_hooks._default_outcome_model()
        assert model is not None and model.is_ready
        pred = model.predict(
            {
                "search_algorithm": "grid_search",
                "problem_type": "classification",
                "metric": "accuracy",
                "time_limit": 180,
            },
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
            _make_docs(30),
            head_path=str(head_path),
            ensemble_dir=str(ens_dir),
            epochs=10,
            k=2,
        )
        from hagent.agent.campaign import wm_hooks

        self._patch_cfg(monkeypatch, head_path, ens_dir)
        wm_hooks._outcome_model_cache["fingerprint"] = None

        m1 = wm_hooks._default_outcome_model()
        m2 = wm_hooks._default_outcome_model()
        assert m1 is m2  # memo hit — không dựng lại từ đĩa

        # Retrain (mtime đổi) → nạp model mới
        time.sleep(0.05)
        tom.train_and_save(
            _make_docs(30, seed=1),
            head_path=str(head_path),
            ensemble_dir=str(ens_dir),
            epochs=10,
            k=2,
        )
        import os

        os.utime(head_path)
        m3 = wm_hooks._default_outcome_model()
        assert m3 is not m1

    def test_missing_checkpoints_memoized_none(self, tmp_path, monkeypatch):
        from hagent.agent.campaign import wm_hooks

        self._patch_cfg(monkeypatch, tmp_path / "nope.npz", tmp_path / "noe")
        wm_hooks._outcome_model_cache["fingerprint"] = None
        assert wm_hooks._default_outcome_model() is None
        assert wm_hooks._default_outcome_model() is None  # không nổ khi gọi lại
