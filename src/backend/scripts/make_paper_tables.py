#!/usr/bin/env python3
"""
Sinh bảng LaTeX cho bài báo từ dữ liệu thí nghiệm THẬT — không gõ số tay.

Input:
  - benchmarks/agent_matrix_results.jsonl  (T10)
  - benchmarks/hpo_real_fair.json + hpo_large_clean.json  (benchmark HPO)
Output:
  - paper/tables/agent_matrix.tex
  - paper/tables/hpo_two_scales.tex

Usage:
  cd src/backend
  python scripts/make_paper_tables.py
  python scripts/make_paper_tables.py --out-dir ../../paper/tables
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import numpy as np  # noqa: E402


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not row.get("error"):
            rows.append(row)
    return rows


def _mean_std(values: List[float]) -> str:
    if not values:
        return "--"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{np.mean(values):.4f} $\\pm$ {np.std(values):.4f}"


def agent_matrix_table(rows: List[dict]) -> str:
    """Điều kiện × dataset: best score thật (mean±std theo seed) + cột cơ chế."""
    by_cell: Dict[tuple, List[dict]] = defaultdict(list)
    for r in rows:
        by_cell[(r["condition"], r["dataset"])].append(r)

    conditions = sorted({r["condition"] for r in rows})
    datasets = sorted({r["dataset"] for r in rows})

    lines = [
        "% Sinh tự động bởi scripts/make_paper_tables.py — KHÔNG sửa tay",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Agent-level results: best real training-job score per"
        " condition (mean $\\pm$ std over seeds), with replanning activity"
        " and LLM cost.}",
        "\\label{tab:agent-matrix}",
        "\\small",
        "\\begin{tabular}{ll" + "c" * 4 + "}",
        "\\toprule",
        "Dataset & Condition & Best score & Jobs & Extensions & Tokens \\\\",
        "\\midrule",
    ]
    for ds in datasets:
        first = True
        for cond in conditions:
            cell = by_cell.get((cond, ds), [])
            if not cell:
                continue
            scores = [
                r["best_real_score"] for r in cell if r.get("best_real_score") is not None
            ]
            jobs = [r.get("n_real_jobs") or 0 for r in cell]
            exts = sum(r.get("n_extended") or 0 for r in cell)
            tokens = [
                (r.get("cost_metrics") or {}).get("total_input_tokens", 0)
                + (r.get("cost_metrics") or {}).get("total_output_tokens", 0)
                for r in cell
            ]
            ds_col = ds.replace("_", "\\_") if first else ""
            first = False
            lines.append(
                f"{ds_col} & {cond} & {_mean_std(scores)} & "
                f"{np.mean(jobs):.1f} & {exts} & {np.mean(tokens):.0f} \\\\"
            )
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines.pop()
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def hpo_two_scales_table(small: dict, large: dict) -> str:
    """Bảng HPO 2 quy mô — thứ hạng đảo chiều theo scale."""
    strategies = list(small["summary"].keys())

    def row(strategy: str) -> str:
        s = small["summary"].get(strategy) or {}
        l = large["summary"].get(strategy) or {}
        s_sp = s.get("speedup_vs_grid_total")
        l_sp = l.get("speedup_vs_grid_total")
        return (
            f"{strategy.replace('_', '\\_')} & "
            f"{s.get('mean_test', 0):.4f} $\\pm$ {s.get('std_test', 0):.4f} & "
            + (f"{s_sp:.2f}$\\times$" if s_sp else "--")
            + f" & {l.get('mean_test', 0):.4f} & "
            + (f"{l_sp:.2f}$\\times$" if l_sp else "--")
            + " \\\\"
        )

    lines = [
        "% Sinh tự động bởi scripts/make_paper_tables.py — KHÔNG sửa tay",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{HPO strategies at two data scales (equal search space,"
        " equal budget). Small: 6 datasets $\\times$ 3 seeds. Large:"
        " Covertype 581k$\\times$54 (251\\,MB), single seed --- rankings"
        " invert with scale.}",
        "\\label{tab:hpo-two-scales}",
        "\\small",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        " & \\multicolumn{2}{c}{Small (150--12k rows)} &"
        " \\multicolumn{2}{c}{Covertype (581k rows)} \\\\",
        "Strategy & Test acc. & Speedup & Test acc. & Speedup \\\\",
        "\\midrule",
    ]
    lines += [row(s) for s in strategies]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sinh bảng LaTeX từ kết quả thật")
    parser.add_argument(
        "--matrix", default="benchmarks/agent_matrix_results.jsonl"
    )
    parser.add_argument("--hpo-small", default="benchmarks/hpo_real_fair.json")
    parser.add_argument("--hpo-large", default="benchmarks/hpo_large_clean.json")
    parser.add_argument(
        "--out-dir", default=str(BACKEND.parent.parent / "paper" / "tables")
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    matrix_path = BACKEND / args.matrix
    if matrix_path.is_file():
        rows = load_jsonl(matrix_path)
        if rows:
            (out_dir / "agent_matrix.tex").write_text(
                agent_matrix_table(rows), encoding="utf-8"
            )
            written.append(f"agent_matrix.tex ({len(rows)} rows)")
    else:
        print(f"(bỏ qua) chưa có {matrix_path} — chạy run_experiment_matrix.py trước")

    small_p = BACKEND / args.hpo_small
    large_p = BACKEND / args.hpo_large
    if small_p.is_file() and large_p.is_file():
        small = json.loads(small_p.read_text(encoding="utf-8"))
        large = json.loads(large_p.read_text(encoding="utf-8"))
        (out_dir / "hpo_two_scales.tex").write_text(
            hpo_two_scales_table(small, large), encoding="utf-8"
        )
        written.append("hpo_two_scales.tex")

    for name in written:
        print(f"  {out_dir / name.split(' ')[0]}")
    if not written:
        print("Không có input nào — chưa sinh bảng.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
