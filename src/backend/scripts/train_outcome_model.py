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
import sys
from pathlib import Path
from typing import Any, Dict, List

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def load_docs_jsonl(path: str) -> List[Dict[str, Any]]:
    docs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def load_docs_mongo(limit: int) -> List[Dict[str, Any]]:
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
    return h.hexdigest()[:16]


def train_and_save(
    docs: List[Dict[str, Any]],
    *,
    head_path: str | None = None,
    ensemble_dir: str | None = None,
    epochs: int = 200,
    seed: int = 0,
    k: int | None = None,
    min_samples: int = 8,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Trích sample → train head + ensemble → ghi checkpoint. Trả manifest."""
    from hagent.bridge.config import get_world_model_config
    from hagent.world.predictor.ensemble import train_outcome_ensemble
    from hagent.world.predictor.outcome_head_v1 import (
        extract_outcome_samples,
        train_outcome_head,
    )

    samples = extract_outcome_samples(docs)
    report: Dict[str, Any] = {
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
    head.save(str(head_out))

    ens = train_outcome_ensemble(
        samples, config=ens_cfg, k=k, epochs=epochs, seed=seed
    )
    ens.save(str(ens_out))

    history = head.config.get("train_history") or []
    report.update(
        {
            "head_path": str(head_out),
            "head_sha256": sha256_of(head_out),
            "ensemble_dir": str(ens_out),
            "ensemble_members": sorted(
                f"{p.name}:{sha256_of(p)}" for p in ens_out.glob("member_*.npz")
            ),
            "final_nll": history[-1] if history else None,
            "epochs": epochs,
            "seed": seed,
        }
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Train outcome model checkpoints")
    parser.add_argument("--source", choices=["mongo", "jsonl"], default="mongo")
    parser.add_argument("--jsonl", help="File JSONL trajectory docs (source=jsonl)")
    parser.add_argument("--limit", type=int, default=10000, help="Giới hạn docs từ Mongo")
    parser.add_argument("--out-head", dest="out_head", help="Override checkpoint_path")
    parser.add_argument("--out-ensemble", dest="out_ensemble", help="Override checkpoint_dir")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=None, help="Số member ensemble (mặc định theo yaml)")
    parser.add_argument("--min-samples", type=int, default=8, dest="min_samples")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ đếm sample")
    args = parser.parse_args()

    if args.source == "jsonl":
        if not args.jsonl:
            parser.error("--source jsonl cần --jsonl PATH")
        docs = load_docs_jsonl(args.jsonl)
    else:
        docs = load_docs_mongo(args.limit)

    report = train_and_save(
        docs,
        head_path=args.out_head,
        ensemble_dir=args.out_ensemble,
        epochs=args.epochs,
        seed=args.seed,
        k=args.k,
        min_samples=args.min_samples,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
