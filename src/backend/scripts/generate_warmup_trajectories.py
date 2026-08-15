#!/usr/bin/env python3
"""
Sinh warmup trajectories cho outcome model — search sklearn THẬT trên
dataset SYNTHETIC tách biệt hoàn toàn 6 bộ eval của ma trận.

Nguyên tắc liêm chính (bài học audit hpo benchmark):
  - Chống leakage: checkpoint train từ đây CHƯA TỪNG thấy iris/wine/
    breast_cancer/digits/glass/online_shoppers — điều kiện B/C của ma trận
    đo TRANSFER qua meta-features, không phải tra bảng đáp án.
  - Train/serve consistency: meta tính bằng đúng meta_features_from_frame
    (một nguồn sự thật với datasets_real + serve); params doc khớp shape
    variant của builder (không nhét key serve không có).
  - Cùng không gian search giữa các thuật toán: infer_dimensions=False
    (BO mặc định True sẽ tự nới grid thành không gian liên tục — khác
    hẳn 4 thuật toán còn lại).

Usage:
  cd src/backend
  python scripts/generate_warmup_trajectories.py --dry-run
  python scripts/generate_warmup_trajectories.py            # resumable
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import numpy as np

# 6 bộ eval của agent_matrix_config.yaml — generator này KHÔNG ĐƯỢC đụng tới
EVAL_DATASETS = frozenset(
    {"iris", "wine", "breast_cancer", "digits", "glass", "online_shoppers"}
)

DEFAULT_OUT = BACKEND / "data" / "world_model" / "warmup_trajectories.jsonl"

# Cùng job config với ma trận (agent_matrix_config.yaml) — outcome phân bố
# giống môi trường serve
JOB_CFG: dict[str, Any] = {
    "cv": 3,
    "time_limit": 60,
    "param_grid": {
        "n_estimators": [50, 100, 200],
        "max_depth": [4, 8, 16],
        "min_samples_split": [2, 10],
    },
}


def build_profiles() -> list[dict[str, Any]]:
    """Lưới 24 profile phủ meta-space: rows × classes × frac categorical."""
    profiles = []
    for n_rows in (300, 1000, 3000, 10000):
        for n_classes in (2, 3, 5):
            for frac_cat in (0.0, 0.4):
                profiles.append(
                    {
                        "name": f"synth_r{n_rows}_c{n_classes}_"
                        f"cat{int(frac_cat * 100)}",
                        "n_rows": n_rows,
                        "n_classes": n_classes,
                        "frac_cat": frac_cat,
                        # biến thiên phụ để meta-space không suy biến
                        "n_features": 10 + (n_classes * 4),
                        "imbalanced": (n_rows + n_classes) % 2 == 0,
                    }
                )
    return profiles


def make_dataset(profile: dict[str, Any], seed: int) -> dict[str, Any]:
    """Synthetic dataset + meta tính bằng ĐÚNG meta_features_from_frame."""
    import pandas as pd
    from sklearn.datasets import make_classification

    from hagent.world.meta_features import meta_features_from_frame

    name = str(profile["name"])
    assert name not in EVAL_DATASETS, f"leakage guard: {name} là dataset eval"

    n_classes = int(profile["n_classes"])
    n_features = int(profile["n_features"])
    n_informative = max(4, n_features // 2)
    weights = None
    if profile.get("imbalanced"):
        rest = (1.0 - 0.6) / max(1, n_classes - 1)
        weights = [0.6] + [rest] * (n_classes - 1)

    X, y = make_classification(
        n_samples=int(profile["n_rows"]),
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=max(0, n_features // 5),
        n_classes=n_classes,
        weights=weights,
        random_state=seed,
    )

    # 1/3 cột đầu biến đổi lognormal — phủ dải mean_abs_skew của meta v2
    n_skew = n_features // 3
    if n_skew:
        X[:, :n_skew] = np.expm1(np.clip(X[:, :n_skew], -5, 5) / 2.0)

    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    n_cat = int(round(float(profile["frac_cat"]) * n_features))
    for col in frame.columns[n_features - n_cat :] if n_cat else []:
        # binning phân vị → dtype category: meta thấy categorical thật,
        # X cho model dùng mã số (RF nhận numeric)
        binned = pd.qcut(frame[col], q=5, duplicates="drop")
        frame[col] = binned.astype("category")
        X[:, frame.columns.get_loc(col)] = binned.cat.codes.to_numpy(dtype=float)
    frame["target"] = y

    return {
        "name": name,
        "X": X,
        "y": y,
        "meta": meta_features_from_frame(frame, target="target"),
    }


def run_search(
    algo: str, X: np.ndarray, y: np.ndarray, job_cfg: dict, seed: int
) -> dict[str, Any]:
    """Search THẬT — cùng cấu hình với RealJobEnv của ma trận."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold

    from automl.search.factory.search_strategy_factory import SearchStrategyFactory

    cfg = dict(
        cv=StratifiedKFold(
            n_splits=int(job_cfg.get("cv", 3)), shuffle=True, random_state=seed
        ),
        scoring={"accuracy": "accuracy"},
        metric_sort="accuracy",
        n_jobs=-1,
        random_state=seed,
        save_log=False,
        verbose=0,
        max_time=float(job_cfg.get("time_limit") or 60),
        # Cùng không gian 18 điểm cho MỌI thuật toán
        infer_dimensions=False,
    )
    strategy = SearchStrategyFactory.create_strategy(algo, cfg)
    t0 = time.perf_counter()
    best_params, best_score, _, _, time_limited = strategy.search(
        RandomForestClassifier(random_state=seed),
        dict(job_cfg.get("param_grid") or {}),
        X,
        y,
    )
    return {
        "best_params": best_params,
        "best_score": float(best_score),
        "seconds": round(time.perf_counter() - t0, 2),
        "time_limited": bool(time_limited),
    }


def make_doc(
    profile_name: str,
    algo: str,
    meta: dict[str, float],
    outcome: dict[str, Any],
    job_cfg: dict,
) -> dict[str, Any]:
    """Trajectory doc đúng schema extract_outcome_samples; params khớp shape
    variant serve của builder (search_algorithm/problem_type/metric/time_limit)."""
    job_id = f"warmup:{profile_name}:{algo}"
    return {
        "kind": "warmup_synthetic",
        "profile": profile_name,
        "next_observation": {
            "jobs": {
                job_id: {
                    "status": "completed",
                    "best_score": outcome["best_score"],
                    "dataset_id": profile_name,
                    "config": {
                        "search_algorithm": algo,
                        "problem_type": "classification",
                        "metric": "accuracy",
                        "time_limit": job_cfg.get("time_limit", 60),
                    },
                }
            },
            "datasets": {profile_name: dict(meta)},
        },
        "search_seconds": outcome["seconds"],
        "time_limited": outcome["time_limited"],
        "best_params": outcome["best_params"],
    }


def load_algos() -> list[str]:
    """Vocab thuật toán từ hagent.yaml (một nguồn sự thật với checkpoint)."""
    from hagent.bridge.config import get_world_model_config

    cfg = (get_world_model_config() or {}).get("outcome_head") or {}
    algos = list(cfg.get("search_algorithms") or [])
    if not algos:
        raise SystemExit("hagent.yaml thiếu world_model.outcome_head.search_algorithms")
    return algos


def generate(
    out_path: Path,
    *,
    seed: int = 0,
    limit: int | None = None,
    dry_run: bool = False,
    search_fn: Callable[..., dict[str, Any]] = run_search,
) -> dict[str, Any]:
    profiles = build_profiles()
    algos = load_algos()

    done: set = set()
    if out_path.is_file():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                doc = json.loads(line)
                for jid in (doc.get("next_observation") or {}).get("jobs", {}):
                    done.add(jid)
            except Exception:
                continue

    todo = [
        (p, a) for p in profiles for a in algos if f"warmup:{p['name']}:{a}" not in done
    ]
    if limit:
        todo = todo[:limit]
    report = {
        "profiles": len(profiles),
        "algos": algos,
        "total": len(profiles) * len(algos),
        "done": len(done),
        "todo": len(todo),
    }
    if dry_run:
        return report

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_cache: dict[str, dict[str, Any]] = {}
    with open(out_path, "a", encoding="utf-8") as fh:
        for i, (profile, algo) in enumerate(todo, 1):
            name = str(profile["name"])
            if name not in ds_cache:
                ds_cache[name] = make_dataset(profile, seed)
            ds = ds_cache[name]
            outcome = search_fn(algo, ds["X"], ds["y"], JOB_CFG, seed)
            doc = make_doc(name, algo, ds["meta"], outcome, JOB_CFG)
            fh.write(json.dumps(doc, ensure_ascii=False, default=str) + "\n")
            fh.flush()
            print(
                f"[{i}/{len(todo)}] {name} {algo}: "
                f"score={outcome['best_score']:.4f} {outcome['seconds']}s",
                flush=True,
            )
    report["written"] = len(todo)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Sinh warmup trajectories")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Chỉ chạy N cặp đầu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = generate(
        Path(args.out), seed=args.seed, limit=args.limit, dry_run=args.dry_run
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
