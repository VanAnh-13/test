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

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

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
            # BẮT BUỘC cho tính công bằng: nếu bật, BO suy ra Integer/Real và
            # tìm trên hộp liên tục 17.667 điểm trong khi 4 strategy kia bị
            # giới hạn ở đúng 18 tổ hợp — gấp 981 lần, không còn cùng bài toán.
            "infer_dimensions": False,
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


def _shutdown_pool() -> None:
    """
    Đóng loky executor sau mỗi strategy.

    Không dọn thì worker tồn đọng giữa các lần đo (đã quan sát 146 tiến trình
    còn sống), ăn CPU/RAM và làm sai lệch thời gian của strategy chạy sau.
    """
    try:
        from joblib.externals.loky import get_reusable_executor

        get_reusable_executor().shutdown(wait=True)
    except Exception:
        pass


def _warm_up_dataset(dataset: dict, args) -> None:
    """Một fit rẻ để phân tán dữ liệu tới worker trước khi bấm giờ strategy đầu."""
    from sklearn.model_selection import cross_validate

    try:
        cross_validate(
            RandomForestClassifier(n_estimators=5, max_depth=3, random_state=args.seed),
            dataset["X"],
            dataset["y"],
            cv=2,
            n_jobs=args.n_jobs,
            error_score="raise",
        )
    except Exception:
        pass


def _machine_state() -> dict:
    """Ghi lại tải máy để số liệu thời gian có ngữ cảnh (đo phải độc chiếm CPU)."""
    state: dict = {}
    try:
        import psutil

        state["cpu_count"] = psutil.cpu_count()
        state["cpu_percent"] = psutil.cpu_percent(interval=1.0)
        state["ram_available_gb"] = round(psutil.virtual_memory().available / 1e9, 2)
        state["python_processes"] = sum(
            1
            for p in psutil.process_iter(["name"])
            if (p.info.get("name") or "").lower().startswith("python")
        )
    except Exception as exc:
        state["error"] = str(exc)
    return state


def run_one(dataset: dict, strategy_name: str, cfg: dict, args, seed: int) -> dict:
    from automl.search.factory.search_strategy_factory import SearchStrategyFactory

    X, y = dataset["X"], dataset["y"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    base = dict(
        cv=StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=seed),
        scoring={"accuracy": "accuracy"},
        metric_sort="accuracy",
        n_jobs=args.n_jobs,
        random_state=seed,
        save_log=False,
        verbose=0,
    )
    if args.max_time is not None:
        base["max_time"] = args.max_time
    base.update(cfg)

    strategy = SearchStrategyFactory.create_strategy(strategy_name, base)
    model = RandomForestClassifier(random_state=seed)

    t0 = time.perf_counter()
    best_params, best_score, _, cv_results, time_limited = strategy.search(
        model, PARAM_GRID, X_tr, y_tr
    )
    elapsed = time.perf_counter() - t0

    # Refit tham số tốt nhất trên toàn train, đo trên holdout test
    final = RandomForestClassifier(random_state=seed)
    if best_params:
        final.set_params(**best_params)
    final.fit(X_tr, y_tr)
    test_score = float(final.score(X_te, y_te))

    params_log = cv_results.get("params", []) if cv_results else []
    n_evals = len(params_log)
    # Số cấu hình PHÂN BIỆT: GA log lại cả cá thể trùng (cache hit), nên
    # "8 đánh giá" của nó có thể chỉ là 5–6 cấu hình thật sự được khám phá.
    distinct = len(
        {tuple(sorted(p.items())) for p in params_log if isinstance(p, dict)}
    )
    # Ngân sách quy đổi full-fidelity: successive_halving đánh giá phần lớn ở
    # fidelity thấp nên đếm đầu lượt sẽ phóng đại chi phí thật của nó.
    fracs = cv_results.get("resource_frac") if cv_results else None
    budget_equiv = float(sum(fracs)) if fracs else float(n_evals)
    off_grid = [
        p
        for p in params_log
        if isinstance(p, dict)
        and any(k in PARAM_GRID and v not in PARAM_GRID[k] for k, v in p.items())
    ]
    return {
        "dataset": dataset["name"],
        "strategy": strategy_name,
        "best_params": best_params,
        "cv_score": float(best_score),
        "test_score": test_score,
        "seconds": elapsed,
        "n_evaluations": n_evals,
        "n_distinct_configs": distinct,
        "full_fidelity_budget": budget_equiv,
        "n_off_grid_configs": len(off_grid),
        "time_limited": bool(time_limited),
        "seed": seed,
    }


def main() -> int:
    from automl.search.datasets_real import available_datasets, load_real_datasets

    parser = argparse.ArgumentParser(description="HPO benchmark trên dữ liệu thật")
    parser.add_argument(
        # Mặc định BỎ dataset lớn — phải chỉ định tường minh (vd. --datasets covtype)
        "--datasets",
        default=",".join(available_datasets(include_large=False)),
        help=f"Có sẵn: {', '.join(available_datasets())}",
    )
    parser.add_argument(
        "--strategies",
        default="grid_search,random_search,bayesian_search,genetic_algorithm,successive_halving",
    )
    parser.add_argument(
        "--budget", type=int, default=8, help="Số lần đánh giá cho non-grid"
    )
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=-1, dest="n_jobs")
    parser.add_argument("--seed", type=int, default=42, help="Seed đầu tiên")
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=3,
        dest="n_seeds",
        help="Số seed lặp lại mỗi ô (n=1 thì KHÔNG có sai số để báo cáo)",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=None,
        dest="max_time",
        help="Giới hạn giây cho MỖI lần search (đẩy vào config max_time)",
    )
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

    seeds = [args.seed + i for i in range(max(1, args.n_seeds))]
    _warm_up_pool(args.n_jobs)
    print(
        f"Benchmark HPO | {len(datasets)} dataset × {len(strategies)} strategy × "
        f"{len(seeds)} seed | budget={args.budget} (grid vét cạn {N_GRID_COMBOS}) | "
        f"cv={args.cv} | n_jobs={args.n_jobs}\n"
    )

    machine_before = _machine_state()
    if machine_before.get("cpu_percent", 0) > 40:
        print(
            f"CẢNH BÁO: CPU đang tải {machine_before['cpu_percent']:.0f}% "
            f"({machine_before.get('python_processes')} tiến trình python) — "
            f"số liệu thời gian sẽ KHÔNG đáng tin.\n"
        )

    results = []
    for ds in datasets:
        mb = ds["X"].nbytes / 1e6
        print(
            f"=== {ds['name']} ({ds['n_rows']}×{ds['n_cols']}, "
            f"{ds['n_classes']} lớp, {mb:.0f}MB) ==="
        )
        # Làm nóng theo TỪNG dataset: strategy chạy đầu tiên (grid) từng phải
        # gánh chi phí nạp dữ liệu vào worker + JIT, làm mọi tỉ số speedup
        # so với nó bị thổi phồng.
        _warm_up_dataset(ds, args)
        for s in strategies:
            runs = []
            for seed in seeds:
                r = run_one(ds, s, cfgs[s], args, seed)
                results.append(r)
                runs.append(r)
                # Dọn pool giữa các lần đo: worker tồn đọng làm sai lệch lần sau
                _shutdown_pool()
            secs = [r["seconds"] for r in runs]
            tests = [r["test_score"] for r in runs]
            cvs = [r["cv_score"] for r in runs]
            print(
                f"  {s:20s} cv={np.mean(cvs):.4f}±{np.std(cvs):.4f} "
                f"test={np.mean(tests):.4f}±{np.std(tests):.4f} "
                f"{np.mean(secs):7.1f}±{np.std(secs):.1f}s  "
                f"evals={runs[0]['n_evaluations']} "
                f"(distinct={runs[0]['n_distinct_configs']}, "
                f"full-fid={runs[0]['full_fidelity_budget']:.1f})"
                + ("  [hết giờ]" if any(r["time_limited"] for r in runs) else "")
            )
        print()

    # Tổng hợp. Speedup = TỈ SỐ CỦA TỔNG thời gian, không phải trung bình
    # cộng của các tỉ số — trung bình cộng tỉ số luôn thiên vị lên trên và
    # để một dataset bé xíu chi phối con số headline.
    grid_total = (
        float(np.sum([r["seconds"] for r in results if r["strategy"] == "grid_search"]))
        or None
    )

    summary = {}
    for s in strategies:
        rs = [r for r in results if r["strategy"] == s]
        if not rs:
            continue
        total = float(np.sum([r["seconds"] for r in rs]))
        # Sai số giữa các seed, gộp theo dataset
        per_cell_std = []
        for ds_name in {r["dataset"] for r in rs}:
            cell = [r["test_score"] for r in rs if r["dataset"] == ds_name]
            if len(cell) > 1:
                per_cell_std.append(float(np.std(cell)))
        summary[s] = {
            "mean_cv": float(np.mean([r["cv_score"] for r in rs])),
            "std_cv": float(np.std([r["cv_score"] for r in rs])),
            "mean_test": float(np.mean([r["test_score"] for r in rs])),
            "std_test": float(np.std([r["test_score"] for r in rs])),
            "mean_within_cell_std_test": (
                float(np.mean(per_cell_std)) if per_cell_std else None
            ),
            "total_seconds": total,
            "mean_evals": float(np.mean([r["n_evaluations"] for r in rs])),
            "mean_distinct_configs": float(
                np.mean([r["n_distinct_configs"] for r in rs])
            ),
            "mean_full_fidelity_budget": float(
                np.mean([r["full_fidelity_budget"] for r in rs])
            ),
            "n_off_grid_configs": int(np.sum([r["n_off_grid_configs"] for r in rs])),
            "speedup_vs_grid_total": (grid_total / total)
            if grid_total and total
            else None,
            "n_runs": len(rs),
        }

    out_path = (BACKEND / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "budget": args.budget,
                "cv": args.cv,
                "seeds": seeds,
                "n_jobs": args.n_jobs,
                "max_time": args.max_time,
                "machine_before": machine_before,
                "machine_after": _machine_state(),
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

    print("=" * 96)
    print(
        f"{'strategy':21s} {'cv':>16s} {'test':>16s} {'evals':>6s} {'fid':>6s} "
        f"{'tổng s':>9s} {'×grid':>7s}"
    )
    print("-" * 96)
    for s, agg in summary.items():
        sp = agg["speedup_vs_grid_total"]
        print(
            f"{s:21s} {agg['mean_cv']:.4f}±{agg['std_cv']:.4f}  "
            f"{agg['mean_test']:.4f}±{agg['std_test']:.4f}  "
            f"{agg['mean_evals']:6.1f} {agg['mean_full_fidelity_budget']:6.1f} "
            f"{agg['total_seconds']:9.1f} " + (f"{sp:6.2f}x" if sp else "     —")
        )
    print(
        f"\nn={len(seeds)} seed/ô; speedup = tỉ số TỔNG thời gian; "
        f"'fid' = ngân sách quy đổi full-fidelity; "
        f"off-grid configs: "
        + ", ".join(f"{s}={a['n_off_grid_configs']}" for s, a in summary.items())
    )
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
