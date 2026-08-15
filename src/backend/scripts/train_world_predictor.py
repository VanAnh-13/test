#!/usr/bin/env python3
"""
Offline train neural JEPA-lite world predictor from trajectories.

Usage:
  cd src/backend
  python scripts/train_world_predictor.py --from-memory
  python scripts/train_world_predictor.py --mongo --out ./data/world_model/jepa_v1.npz
  python scripts/train_world_predictor.py --jsonl /tmp/traj.jsonl --epochs 30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("train_world_predictor")


def _load_jsonl(path: str) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


async def _load_mongo(limit: int) -> list:
    try:
        from pymongo import MongoClient

        from hagent.bridge.config import get_mongodb_config, get_world_model_config
        from hagent.world.trajectory_store import create_trajectory_store
    except Exception as exc:
        logger.error("Mongo load deps failed: %s", exc)
        return []

    mongo = get_mongodb_config()
    wm = get_world_model_config()
    traj = wm.get("trajectory") or {}
    connect = mongo.get("connect") or "localhost:27017"
    if not str(connect).startswith("mongodb"):
        uri = f"mongodb://{connect}"
    else:
        uri = connect
    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    store = create_trajectory_store(
        client,
        db_name=mongo.get("db_name"),
        collection_name=traj.get("collection"),
    )
    return await store.list_all(limit=limit)


def _synthetic_trajectories(n: int = 64, dim: int = 32) -> list:
    """Minimal demo trajectories when no data available."""
    import numpy as np

    from hagent.world.schema import AutoMLAction, AutoMLObservation
    from hagent.world.service import WorldModelService

    wm = WorldModelService.from_config(
        {
            "encoder": {"backend": "structured_v1", "dim": dim},
            "predictor": {"backend": "tabular_transition_v1"},
            "planner": {"backend": "cem_lite", "horizon": 2, "n_candidates": 2},
            "trajectory": {"enabled": True, "max_per_user": 1000},
        }
    )
    actions = [
        "list_datasets",
        "get_dataset_info",
        "get_features",
        "start_training",
        "get_job_info",
        "list_jobs",
    ]
    docs = []
    rng = np.random.default_rng(0)
    for i in range(n):
        a_type = actions[i % len(actions)]
        obs = AutoMLObservation(
            user_id="synth",
            datasets={
                "ds1": {
                    "id": "ds1",
                    "name": "demo",
                    "n_rows": 100 + i,
                    "n_cols": 8,
                    "features": ["a", "b", "t"],
                    "target": "t",
                }
            },
            jobs={},
            phase="analyze" if "dataset" in a_type or "feature" in a_type else "train",
        )
        next_obs = AutoMLObservation(
            user_id="synth",
            datasets=obs.datasets,
            jobs=(
                {
                    "j1": {
                        "id": "j1",
                        "status": "completed"
                        if a_type == "get_job_info"
                        else "running",
                        "best_score": float(rng.random()),
                    }
                }
                if "job" in a_type or a_type == "start_training"
                else {}
            ),
            phase=obs.phase,
        )
        action = AutoMLAction(type=a_type, params={})
        z = wm.encode(obs)
        z_hat = wm.predict(z, action)
        z_next = wm.encode(next_obs)
        docs.append(
            {
                "user_id": "synth",
                "action": action.to_dict(),
                "z": z.to_dict(),
                "z_hat": z_hat.to_dict(),
                "z_next": z_next.to_dict(),
            }
        )
    return docs


def main() -> int:
    ap = argparse.ArgumentParser(description="Train neural JEPA world predictor")
    ap.add_argument("--jsonl", type=str, help="Trajectory JSONL path")
    ap.add_argument(
        "--mongo", action="store_true", help="Load from Mongo world_trajectories"
    )
    ap.add_argument(
        "--from-memory", action="store_true", help="Use synthetic demo trajectories"
    )
    ap.add_argument("--out", type=str, default="./data/world_model/jepa_v1.npz")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--latent-dim", type=int, default=64)
    ap.add_argument("--limit", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    trajectories = []
    if args.jsonl:
        trajectories = _load_jsonl(args.jsonl)
        logger.info("Loaded %d from jsonl", len(trajectories))
    elif args.mongo:
        trajectories = asyncio.run(_load_mongo(args.limit))
        logger.info("Loaded %d from mongo", len(trajectories))
    elif args.from_memory:
        trajectories = _synthetic_trajectories(n=96, dim=args.latent_dim)
        logger.info("Synthetic trajectories: %d", len(trajectories))
    else:
        logger.error("Specify --jsonl, --mongo, or --from-memory")
        return 2

    if not trajectories:
        logger.error("No trajectories; abort")
        return 1

    # Infer latent dim from first sample
    try:
        dim = len(trajectories[0]["z"]["vector"])
    except Exception:
        dim = args.latent_dim

    from hagent.world.predictor.neural_jepa_v1 import train_neural_jepa

    pred = train_neural_jepa(
        trajectories,
        latent_dim=dim,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pred.save(str(out), latent_dim=dim)
    logger.info("Saved checkpoint → %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
