#!/usr/bin/env python3
"""
Benchmark CLI — sample-efficiency của campaign có/không world model.

Usage:
  cd src/backend
  python scripts/run_benchmark.py
  python scripts/run_benchmark.py --conditions wm,no_wm,random,fixed_grid_search \
      --profiles synth_strong,synth_noisy --budget 20 --seeds 5 \
      --out benchmarks/bench_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def _main() -> int:
    from hagent.agent.eval.benchmark import PROFILES, run_benchmark_matrix
    from hagent.agent.eval.metrics import aggregate_curves

    parser = argparse.ArgumentParser(
        description="HAgent world-model campaign benchmark (simulated env)"
    )
    parser.add_argument(
        "--conditions",
        default="wm,no_wm,random,fixed_grid_search",
        help="Danh sách condition, phân tách bằng dấu phẩy",
    )
    parser.add_argument(
        "--profiles",
        default="synth_strong,synth_noisy",
        help=f"Profiles: {', '.join(PROFILES)}",
    )
    parser.add_argument("--budget", type=int, default=20, help="Số job mỗi run")
    parser.add_argument("--seeds", type=int, default=3, help="Số seed (0..n-1)")
    parser.add_argument("--campaign-size", type=int, default=3, dest="campaign_size")
    parser.add_argument(
        "--out",
        default="benchmarks/bench_results.json",
        help="File JSON kết quả (tương đối với src/backend)",
    )
    args = parser.parse_args()

    from hagent.agent.eval.benchmark import validate_condition

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    for c in conditions:
        try:
            validate_condition(c)
        except ValueError as exc:
            parser.error(str(exc))
    for p in profiles:
        if p not in PROFILES:
            parser.error(f"Unknown profile {p!r}. Available: {', '.join(PROFILES)}")
    seeds = list(range(max(1, args.seeds)))

    results = run_benchmark_matrix(
        conditions=conditions,
        profiles=profiles,
        budget_jobs=args.budget,
        seeds=seeds,
        campaign_size=args.campaign_size,
    )

    # Aggregate theo (profile, condition)
    aggregates = {}
    for prof in profiles:
        for cond in conditions:
            rs = [r for r in results if r["profile"] == prof and r["condition"] == cond]
            if not rs:
                continue
            finals = [r["final_best"] for r in rs if r["final_best"] is not None]
            regrets = [
                r["normalized_regret"] for r in rs if r["normalized_regret"] is not None
            ]
            jt = [r["jobs_to_95pct"] for r in rs if r["jobs_to_95pct"] is not None]
            aggregates[f"{prof}::{cond}"] = {
                "n_seeds": len(rs),
                "final_best_mean": sum(finals) / len(finals) if finals else None,
                "normalized_regret_mean": (
                    sum(regrets) / len(regrets) if regrets else None
                ),
                "jobs_to_95pct_mean": sum(jt) / len(jt) if jt else None,
                "jobs_to_95pct_hit_rate": len(jt) / len(rs),
                "curve": aggregate_curves([r["curve"] for r in rs]),
            }

    out_path = (BACKEND / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_version": 1,
        "budget_jobs": args.budget,
        "seeds": seeds,
        "conditions": conditions,
        "profiles": profiles,
        "results": results,
        "aggregates": aggregates,
    }
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Bảng tóm tắt
    def _fmt(value, width, spec):
        return f"{value:>{width}{spec}}" if value is not None else f"{'-':>{width}}"

    print(
        f"\n{'profile':<14} {'condition':<20} {'final_best':>10} {'n.regret':>9} {'jobs95':>7}"
    )
    print("-" * 64)
    for key, agg in aggregates.items():
        prof, cond = key.split("::")
        print(
            f"{prof:<14} {cond:<20} "
            + _fmt(agg["final_best_mean"], 10, ".4f")
            + " "
            + _fmt(agg["normalized_regret_mean"], 9, ".4f")
            + " "
            + _fmt(agg["jobs_to_95pct_mean"], 7, ".1f")
        )
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
