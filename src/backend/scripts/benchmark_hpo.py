#!/usr/bin/env python3
"""
Benchmark 5 thuật toán HPO trên dữ liệu THẬT.

grid_search chạy vét cạn (mốc tham chiếu); random/bayesian/genetic/
successive_halving chạy cùng budget số lần đánh giá. Đo: best CV score,
wall-clock, số lần đánh giá, và accuracy trên holdout test (generalization).

Chạy TUẦN TỰ, độc chiếm CPU — song song hóa nhiều run cùng lúc sẽ làm hỏng
số liệu thời gian.

Usage:
  cd src/backend
  python scripts/benchmark_hpo.py
  python scripts/benchmark_hpo.py --datasets iris,wine --budget 8
  python scripts/benchmark_hpo.py --out benchmarks/hpo_real.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import numpy as np  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402

# Không gian tham số dùng chung cho mọi strategy (18 tổ hợp)
PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [4, 8, 16],
    "min_samples_split": [2, 10],
}
N_GRID_COMBOS = 18


def strategy_configs(budget: int) -> dict:
    """Cấu hình mỗi strategy sao cho ngân sách đánh giá tương đương `budget`."""
    return {
        "grid_search": {},  # vét cạn — mốc tham chiếu
        "random_search": {"n_iter": budget},
        "bayesian_search": {
            "n_calls": budget,
            "n_initial_points": max(3, budget // 3),
            "early_stopping_enabled": False,  # giữ đúng budget để so công bằng
        },
        "genetic_algorithm": {
            "population_size": max(4, budget // 2),
            "generation": 2,
            "elite_size": 1,
        },
        "successive_halving": {
            "n_candidates": budget + 1,
            "eta": 3,
            "min_resource_frac": 1 / 9,
            "min_subsample_rows": 60,
        },
    }


def _warm_up_pool(n_jobs: int) -> None:
    """Khởi động sẵn joblib pool để strategy đầu tiên không gánh phí spawn."""
    from joblib import Parallel, delayed

    Parallel(n_jobs=n_jobs)(delayed(int)(i) for i in range(4))


def run_one(dataset: dict, strategy_name: str, cfg: dict, args) -> dict:
    from automl.search.factory.search_strategy_factory import SearchStrategyFactory

    X, y = dataset["X"], dataset["y"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed
    )

    base = dict(
        cv=StratifiedKFold(
            n_splits=args.cv, shuffle=True, random_state=args.seed
        ),
        scoring={"accuracy": "accuracy"},
        metric_sort="accuracy",
        n_jobs=args.n_jobs,
        random_state=args.seed,
        save_log=False,
        verbose=0,
    )
    base.update(cfg)

    strategy = SearchStrategyFactory.create_strategy(strategy_name, base)
    model = RandomForestClassifier(random_state=args.seed)

    t0 = time.perf_counter()
    best_params, best_score, _, cv_results, time_limited = strategy.search(
        model, PARAM_GRID, X_tr, y_tr
    )
    elapsed = time.perf_counter() - t0

    # Refit tham số tốt nhất trên toàn train, đo trên holdout test
    final = RandomForestClassifier(random_state=args.seed)
    if best_params:
        final.set_params(**best_params)
    final.fit(X_tr, y_tr)
    test_score = float(final.score(X_te, y_te))

    n_evals = len(cv_results.get("params", [])) if cv_results else 0
    return {
        "dataset": dataset["name"],
        "strategy": strategy_name,
        "best_params": best_params,
        "cv_score": float(best_score),
        "test_score": test_score,
        "seconds": elapsed,
        "n_evaluations": n_evals,
        "time_limited": bool(time_limited),
    }


def main() -> int:
    from automl.search.datasets_real import available_datasets, load_real_datasets

    parser = argparse.ArgumentParser(description="HPO benchmark trên dữ liệu thật")
    parser.add_argument(
        "--datasets", default=",".join(available_datasets()),
        help=f"Có sẵn: {', '.join(available_datasets())}",
    )
    parser.add_argument(
        "--strategies",
        default="grid_search,random_search,bayesian_search,genetic_algorithm,successive_halving",
    )
    parser.add_argument("--budget", type=int, default=8, help="Số lần đánh giá cho non-grid")
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=-1, dest="n_jobs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="benchmarks/hpo_real.json")
    args = parser.parse_args()

    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    cfgs = strategy_configs(args.budget)
    unknown = [s for s in strategies if s not in cfgs]
    if unknown:
        parser.error(f"Strategy chưa cấu hình budget: {unknown}")

    datasets = load_real_datasets(names)
    if not datasets:
        parser.error("Không nạp được dataset nào")

    _warm_up_pool(args.n_jobs)
    print(
        f"Benchmark HPO | {len(datasets)} dataset × {len(strategies)} strategy | "
        f"budget={args.budget} (grid vét cạn {N_GRID_COMBOS}) | cv={args.cv} | n_jobs={args.n_jobs}\n"
    )

    results = []
    for ds in datasets:
        print(f"=== {ds['name']} ({ds['n_rows']}×{ds['n_cols']}, {ds['n_classes']} lớp) ===")
        for s in strategies:
            r = run_one(ds, s, cfgs[s], args)
            results.append(r)
            print(
                f"  {s:20s} cv={r['cv_score']:.4f} test={r['test_score']:.4f} "
                f"{r['seconds']:7.1f}s  evals={r['n_evaluations']}"
            )
        print()

    # Tổng hợp: so với grid vét cạn trên cùng dataset
    by_ds = {}
    for r in results:
        by_ds.setdefault(r["dataset"], {})[r["strategy"]] = r
    for ds_name, runs in by_ds.items():
        ref = runs.get("grid_search")
        for s, r in runs.items():
            if ref and ref["seconds"] > 0:
                r["speedup_vs_grid"] = ref["seconds"] / r["seconds"]
                r["cv_pct_of_grid"] = (
                    r["cv_score"] / ref["cv_score"] if ref["cv_score"] else None
                )

    summary = {}
    for s in strategies:
        rs = [r for r in results if r["strategy"] == s]
        if not rs:
            continue
        summary[s] = {
            "mean_cv": float(np.mean([r["cv_score"] for r in rs])),
            "mean_test": float(np.mean([r["test_score"] for r in rs])),
            "total_seconds": float(np.sum([r["seconds"] for r in rs])),
            "mean_evals": float(np.mean([r["n_evaluations"] for r in rs])),
            "mean_speedup_vs_grid": float(
                np.mean([r.get("speedup_vs_grid", 1.0) for r in rs])
            ),
            "mean_cv_pct_of_grid": float(
                np.mean([r.get("cv_pct_of_grid", 1.0) or 1.0 for r in rs])
            ),
        }

    out_path = (BACKEND / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "budget": args.budget,
                "cv": args.cv,
                "seed": args.seed,
                "n_jobs": args.n_jobs,
                "param_grid": PARAM_GRID,
                "datasets": [
                    {k: v for k, v in d.items() if k not in ("X", "y")}
                    for d in datasets
                ],
                "results": results,
                "summary": summary,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    print("=" * 78)
    print(
        f"{'strategy':22s} {'cv':>7s} {'test':>7s} {'evals':>6s} {'tổng s':>9s} {'×grid':>7s}"
    )
    print("-" * 78)
    for s, agg in summary.items():
        print(
            f"{s:22s} {agg['mean_cv']:7.4f} {agg['mean_test']:7.4f} "
            f"{agg['mean_evals']:6.1f} {agg['total_seconds']:9.1f} "
            f"{agg['mean_speedup_vs_grid']:6.1f}x"
        )
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
