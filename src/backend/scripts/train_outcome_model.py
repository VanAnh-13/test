#!/usr/bin/env python3
"""
Train outcome model (head + ensemble) và ghi checkpoint vào path trong
hagent.yaml — đây chính là bước "nối world model vào agent thật": đường
"auto" của builder/runner đã trỏ sẵn vào các checkpoint này.

Nguồn dữ liệu:
  --source mongo   : Mongo world_trajectories (production)
  --source jsonl   : file JSONL trajectory docs (artifact versionable —
                     dùng cho thí nghiệm để data train có thể commit/tái lập)

Config head/ensemble (vocab, meta_profile, hidden_dim...) đọc từ
world_model.outcome_head / world_model.outcome_ensemble trong hagent.yaml —
MỘT nguồn sự thật, tránh train checkpoint lệch vocab với lúc serve.

Usage:
  cd src/backend
  python scripts/train_outcome_model.py --source jsonl --jsonl data/traj.jsonl
  python scripts/train_outcome_model.py --source mongo --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parent.parent
MANIFEST_SCHEMA_VERSION = 1
VOCABULARY_FIELDS = (
    "search_algorithms",
    "problem_types",
    "metrics",
    "model_vocab",
    "meta_profile",
)
CHECKPOINT_CONFIG_FIELDS = (
    "hidden_dim",
    "time_limit_norm",
    "use_latent",
    "latent_dim",
)
VOCABULARY_VERSION = "outcome-v2"
MANIFEST_FILENAME = "manifest.json"

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def load_docs_jsonl(path: str) -> list[dict[str, Any]]:
    docs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def load_docs_mongo(limit: int) -> list[dict[str, Any]]:
    import asyncio

    from pymongo import AsyncMongoClient

    from hagent.bridge.config import get_mongodb_config
    from hagent.world.trajectory_store import create_trajectory_store

    mongo = get_mongodb_config()
    uri = mongo.get("uri") or "mongodb://localhost:27017"

    async def _load():
        client = AsyncMongoClient(uri)
        try:
            store = create_trajectory_store(client)
            return await store.list_all(limit=limit)
        finally:
            await client.close()

    return asyncio.new_event_loop().run_until_complete(_load())


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _full_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _normalize_checkpoint_archive(path: Path) -> None:
    """Rewrite a freshly-created checkpoint without pickled object arrays."""
    import numpy as np

    # allow_pickle is confined to artifacts created moments earlier by this process.
    with np.load(path, allow_pickle=True) as archive:
        values = {name: archive[name] for name in archive.files}
    for field in VOCABULARY_FIELDS[:-1]:
        if field in values:
            values[field] = np.asarray(
                [str(value) for value in values[field].tolist()],
                dtype=np.str_,
            )
    if "meta_profile" in values:
        values["meta_profile"] = np.asarray(
            str(np.asarray(values["meta_profile"]).item()),
            dtype=np.str_,
        )
    np.savez(path, **values)


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    """Read only non-pickled metadata needed to bind manifest to checkpoint."""
    import numpy as np

    required = set(VOCABULARY_FIELDS) | set(CHECKPOINT_CONFIG_FIELDS)
    try:
        with np.load(path, allow_pickle=False) as archive:
            if not required.issubset(archive.files):
                raise ValueError("missing checkpoint metadata")
            metadata = {
                field: [str(value) for value in archive[field].tolist()]
                for field in VOCABULARY_FIELDS[:-1]
            }
            metadata.update(
                {
                    "meta_profile": str(archive["meta_profile"].item()),
                    "hidden_dim": int(archive["hidden_dim"].item()),
                    "time_limit_norm": float(archive["time_limit_norm"].item()),
                    "use_latent": bool(archive["use_latent"].item()),
                    "latent_dim": int(archive["latent_dim"].item()),
                }
            )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid checkpoint manifest: unreadable checkpoint metadata: {path.name}"
        ) from exc
    return metadata


def _expected_checkpoint_metadata(config: dict[str, Any]) -> dict[str, Any]:
    from hagent.world.predictor.outcome_head_v1 import outcome_feature_config

    feature_config = outcome_feature_config(config)
    return {
        **{field: feature_config[field] for field in VOCABULARY_FIELDS},
        "hidden_dim": int(config.get("hidden_dim", 64)),
        "time_limit_norm": feature_config["time_limit_norm"],
        "use_latent": feature_config["use_latent"],
        "latent_dim": feature_config["latent_dim"],
    }


def validate_checkpoint_manifest(
    head_path: str | Path,
    ensemble_dir: str | Path,
    *,
    expected_k: int | None = None,
) -> dict[str, Any]:
    """Load and verify a published head/ensemble artifact manifest."""
    head = Path(head_path)
    ensemble = Path(ensemble_dir)
    manifest_path = ensemble / MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid checkpoint manifest: {exc}") from exc

    if not isinstance(manifest, dict):
        raise TypeError("Invalid checkpoint manifest: root must be an object")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Invalid checkpoint manifest: unsupported schema_version")
    if manifest.get("vocabulary_version") != VOCABULARY_VERSION:
        raise ValueError("Invalid checkpoint manifest: unsupported vocabulary_version")
    for field in ("source_sha256", "config_sha256", "vocabulary_sha256"):
        if not _is_sha256(manifest.get(field)):
            raise ValueError(f"Invalid checkpoint manifest: {field}")

    config = manifest.get("config")
    if not isinstance(config, dict) or set(config) != {
        "outcome_head",
        "outcome_ensemble",
    }:
        raise ValueError("Invalid checkpoint manifest: config")
    if manifest["config_sha256"] != _canonical_sha256(config):
        raise ValueError("Invalid checkpoint manifest: config hash mismatch")

    vocabulary = manifest.get("vocabulary")
    if not isinstance(vocabulary, dict) or set(vocabulary) != {
        "outcome_head",
        "outcome_ensemble",
    }:
        raise ValueError("Invalid checkpoint manifest: vocabulary")
    for component in ("outcome_head", "outcome_ensemble"):
        record = vocabulary.get(component)
        if not isinstance(record, dict) or set(record) != set(VOCABULARY_FIELDS):
            raise ValueError("Invalid checkpoint manifest: vocabulary fields")
        for field in VOCABULARY_FIELDS[:-1]:
            values = record.get(field)
            if not isinstance(values, list) or not all(
                isinstance(value, str) for value in values
            ):
                raise ValueError("Invalid checkpoint manifest: vocabulary values")
        if not isinstance(record.get("meta_profile"), str):
            raise TypeError("Invalid checkpoint manifest: meta_profile")
    if manifest["vocabulary_sha256"] != _canonical_sha256(vocabulary):
        raise ValueError("Invalid checkpoint manifest: vocabulary hash mismatch")

    training = manifest.get("training")
    if not isinstance(training, dict):
        raise TypeError("Invalid checkpoint manifest: training must be an object")
    required_training_fields = {
        "epochs",
        "seed",
        "ensemble_size",
        "min_samples",
        "n_trajectory_docs",
        "n_samples",
    }
    if set(training) != required_training_fields:
        raise ValueError("Invalid checkpoint manifest: training fields")
    if any(type(training[field]) is not int for field in required_training_fields):
        raise ValueError("Invalid checkpoint manifest: training field types")
    if any(training[field] < 1 for field in ("epochs", "ensemble_size", "min_samples")):
        raise ValueError("Invalid checkpoint manifest: training field values")
    if training["n_trajectory_docs"] < 0 or training["n_samples"] < 0:
        raise ValueError("Invalid checkpoint manifest: training sample counts")
    if training["n_samples"] < training["min_samples"]:
        raise ValueError("Invalid checkpoint manifest: insufficient training samples")
    member_count = training["ensemble_size"]
    if expected_k is not None and member_count != expected_k:
        raise ValueError("Invalid checkpoint manifest: unexpected ensemble_size")
    if not all(
        isinstance(config[component], dict)
        for component in ("outcome_head", "outcome_ensemble")
    ):
        raise ValueError("Invalid checkpoint manifest: model config")
    if config["outcome_ensemble"].get("k") != member_count:
        raise ValueError("Invalid checkpoint manifest: config ensemble_size mismatch")

    head_record = manifest.get("head")
    if not isinstance(head_record, dict):
        raise TypeError("Invalid checkpoint manifest: head must be an object")
    if head_record.get("filename") != head.name or not head.is_file():
        raise ValueError("Invalid checkpoint manifest: head checkpoint missing")
    if not _is_sha256(head_record.get("sha256")):
        raise ValueError("Invalid checkpoint manifest: head sha256")
    if head_record["sha256"] != _full_sha256(head):
        raise ValueError("Invalid checkpoint manifest: head hash mismatch")

    expected_names = [f"member_{index}.npz" for index in range(member_count)]
    actual_names = {path.name for path in ensemble.glob("member_*.npz")}
    if actual_names != set(expected_names):
        raise ValueError("Invalid checkpoint manifest: ensemble member set mismatch")
    member_records = manifest.get("ensemble_members")
    if not isinstance(member_records, list):
        raise TypeError("Invalid checkpoint manifest: ensemble_members must be a list")
    declared_names = [
        record.get("filename") for record in member_records if isinstance(record, dict)
    ]
    if declared_names != expected_names:
        raise ValueError("Invalid checkpoint manifest: declared member set mismatch")
    for record in member_records:
        digest = record.get("sha256")
        member_path = ensemble / record["filename"]
        if not _is_sha256(digest) or digest != _full_sha256(member_path):
            raise ValueError(
                f"Invalid checkpoint manifest: hash mismatch for {member_path.name}"
            )

    expected_metadata = {
        "outcome_head": _expected_checkpoint_metadata(config["outcome_head"]),
        "outcome_ensemble": _expected_checkpoint_metadata(config["outcome_ensemble"]),
    }

    def validate_metadata(path: Path, component: str) -> None:
        expected = expected_metadata[component]
        declared_vocabulary = vocabulary[component]
        expected_vocabulary = {field: expected[field] for field in VOCABULARY_FIELDS}
        if declared_vocabulary != expected_vocabulary:
            raise ValueError(
                "Invalid checkpoint manifest: checkpoint vocabulary mismatch"
            )
        actual = _checkpoint_metadata(path)
        actual_vocabulary = {field: actual[field] for field in VOCABULARY_FIELDS}
        if actual_vocabulary != declared_vocabulary:
            raise ValueError(
                "Invalid checkpoint manifest: checkpoint vocabulary mismatch"
            )
        if any(actual[field] != expected[field] for field in CHECKPOINT_CONFIG_FIELDS):
            raise ValueError("Invalid checkpoint manifest: checkpoint config mismatch")

    validate_metadata(head, "outcome_head")
    for member_name in expected_names:
        validate_metadata(ensemble / member_name, "outcome_ensemble")
    return manifest


def train_from_docs(
    docs: list[dict[str, Any]],
    *,
    head_path: str | None = None,
    ensemble_dir: str | None = None,
    epochs: int = 200,
    seed: int = 0,
    k: int | None = None,
    min_samples: int = 8,
    source_sha256: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Trích sample → train head + ensemble → ghi checkpoint. Trả manifest."""
    from hagent.bridge.config import get_world_model_config
    from hagent.world.predictor.ensemble import train_outcome_ensemble
    from hagent.world.predictor.outcome_head_v1 import (
        extract_outcome_samples,
        train_outcome_head,
    )

    samples = extract_outcome_samples(docs)
    report: dict[str, Any] = {
        "n_trajectory_docs": len(docs),
        "n_samples": len(samples),
    }
    if dry_run:
        return report
    if len(samples) < min_samples:
        raise SystemExit(
            f"Chỉ trích được {len(samples)} sample (< {min_samples}). "
            f"Cần thêm trajectories — chạy warmup batch hoặc giảm --min-samples."
        )

    wm_cfg = get_world_model_config() or {}
    head_cfg = dict(wm_cfg.get("outcome_head") or {})
    ens_cfg = dict(wm_cfg.get("outcome_ensemble") or {})

    head_out = Path(head_path or head_cfg.get("checkpoint_path") or "")
    ens_out = Path(ensemble_dir or ens_cfg.get("checkpoint_dir") or "")
    if not str(head_out) or not str(ens_out):
        raise SystemExit(
            "Thiếu checkpoint path — cấu hình world_model.outcome_head."
            "checkpoint_path / outcome_ensemble.checkpoint_dir trong hagent.yaml"
        )
    if not head_out.is_absolute():
        head_out = BACKEND / head_out
    if not ens_out.is_absolute():
        ens_out = BACKEND / ens_out

    head = train_outcome_head(samples, config=head_cfg, epochs=epochs, seed=seed)
    ens = train_outcome_ensemble(samples, config=ens_cfg, k=k, epochs=epochs, seed=seed)
    member_count = len(ens.members)
    manifest_path = ens_out / MANIFEST_FILENAME
    source_digest = source_sha256 or _canonical_sha256(docs)
    if not _is_sha256(source_digest):
        raise ValueError("source_sha256 must be a lowercase, 64-character SHA-256")
    if member_count < 1:
        raise RuntimeError("Outcome ensemble training produced no members")
    effective_head_cfg = {
        key: value for key, value in head_cfg.items() if key != "checkpoint_path"
    }
    effective_head_cfg.update(head.feature_cfg)
    effective_head_cfg["hidden_dim"] = head.hidden_dim
    first_member = ens.members[0]
    effective_ens_cfg = {
        key: value for key, value in ens_cfg.items() if key != "checkpoint_dir"
    }
    effective_ens_cfg.update(first_member.feature_cfg)
    effective_ens_cfg["hidden_dim"] = first_member.hidden_dim
    effective_ens_cfg["k"] = member_count
    config = {
        "outcome_head": effective_head_cfg,
        "outcome_ensemble": effective_ens_cfg,
    }
    config_digest = _canonical_sha256(config)
    vocabulary = {
        component: {field: feature_config[field] for field in VOCABULARY_FIELDS}
        for component, feature_config in (
            ("outcome_head", head.feature_cfg),
            ("outcome_ensemble", first_member.feature_cfg),
        )
    }
    vocabulary_digest = _canonical_sha256(vocabulary)

    ens_out.mkdir(parents=True, exist_ok=True)
    manifest_path.unlink(missing_ok=True)
    manifest_path.with_name(f"{manifest_path.name}.tmp").unlink(missing_ok=True)
    head.save(str(head_out))
    ens.save(str(ens_out))

    expected_paths = [ens_out / f"member_{index}.npz" for index in range(member_count)]
    expected_names = {path.name for path in expected_paths}
    for stale_path in ens_out.glob("member_*.npz"):
        if stale_path.name not in expected_names:
            stale_path.unlink()
    if not all(path.is_file() for path in expected_paths):
        raise RuntimeError("Ensemble save did not create every expected member")
    if not head_out.is_file():
        raise RuntimeError("Outcome head save did not create its checkpoint")
    _normalize_checkpoint_archive(head_out)
    for path in expected_paths:
        _normalize_checkpoint_archive(path)

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "vocabulary_version": VOCABULARY_VERSION,
        "vocabulary": vocabulary,
        "vocabulary_sha256": vocabulary_digest,
        "config": config,
        "source_sha256": source_digest,
        "config_sha256": config_digest,
        "training": {
            "epochs": epochs,
            "seed": seed,
            "ensemble_size": member_count,
            "min_samples": min_samples,
            "n_trajectory_docs": len(docs),
            "n_samples": len(samples),
        },
        "head": {"filename": head_out.name, "sha256": _full_sha256(head_out)},
        "ensemble_members": [
            {"filename": path.name, "sha256": _full_sha256(path)}
            for path in expected_paths
        ],
    }
    _atomic_write_json(manifest_path, manifest)
    try:
        validate_checkpoint_manifest(head_out, ens_out, expected_k=member_count)
    except BaseException:
        manifest_path.unlink(missing_ok=True)
        raise

    history = head.config.get("train_history") or []
    report.update(
        {
            "head_path": str(head_out),
            "head_sha256": sha256_of(head_out),
            "ensemble_dir": str(ens_out),
            "ensemble_members": [
                f"{path.name}:{sha256_of(path)}" for path in expected_paths
            ],
            "manifest_path": str(manifest_path),
            "final_nll": history[-1] if history else None,
            "epochs": epochs,
            "seed": seed,
        }
    )
    return report


# Backward-compatible name used by existing callers.
train_and_save = train_from_docs


def main() -> int:
    parser = argparse.ArgumentParser(description="Train outcome model checkpoints")
    parser.add_argument("--source", choices=["mongo", "jsonl"], default="mongo")
    parser.add_argument("--jsonl", help="File JSONL trajectory docs (source=jsonl)")
    parser.add_argument(
        "--limit", type=int, default=10000, help="Giới hạn docs từ Mongo"
    )
    parser.add_argument("--out-head", dest="out_head", help="Override checkpoint_path")
    parser.add_argument(
        "--out-ensemble", dest="out_ensemble", help="Override checkpoint_dir"
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--k", type=int, default=None, help="Số member ensemble (mặc định theo yaml)"
    )
    parser.add_argument("--min-samples", type=int, default=8, dest="min_samples")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ đếm sample")
    args = parser.parse_args()

    source_digest = None
    if args.source == "jsonl":
        if not args.jsonl:
            parser.error("--source jsonl cần --jsonl PATH")
        docs = load_docs_jsonl(args.jsonl)
        source_digest = _full_sha256(Path(args.jsonl))
    else:
        docs = load_docs_mongo(args.limit)

    report = train_from_docs(
        docs,
        head_path=args.out_head,
        ensemble_dir=args.out_ensemble,
        epochs=args.epochs,
        seed=args.seed,
        k=args.k,
        min_samples=args.min_samples,
        dry_run=args.dry_run,
        source_sha256=source_digest,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
