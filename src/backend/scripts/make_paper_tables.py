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
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import numpy as np

from scripts import run_experiment_matrix as matrix_protocol

FROZEN_MATRIX_DESIGN_SHA = (
    "0860d3662ed8a2420aa46887cce84fdb70787c34f5b2271690976905bb4893bb"
)


class TableGateError(ValueError):
    """Input evidence is unsafe for publication."""


def _strict_json(text: str, *, artifact: str, line_number: int | None = None):
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate field {key}")
            result[key] = value
        return result

    location = f" line {line_number}" if line_number is not None else ""
    try:
        return json.loads(
            text,
            object_pairs_hook=no_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise TableGateError(f"{artifact}{location} is malformed") from exc


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise TableGateError(f"matrix input is missing: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise TableGateError(f"matrix input is unreadable: {path}") from exc
    rows = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        row = _strict_json(line, artifact="matrix", line_number=line_number)
        if not isinstance(row, dict):
            raise TableGateError(f"matrix line {line_number} is not an object")
        rows.append(row)
    if not rows:
        raise TableGateError("matrix input is empty")
    return rows


def _is_hex_digest(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(char in "0123456789abcdef" for char in value)
    )


def validate_matrix_rows(
    rows: list[dict],
    *,
    cfg: dict,
    accepted_advices: dict[str, dict],
    current_dataset_hashes: dict[str, str],
    current_advice_keys: dict[tuple[str, str], str],
) -> list[dict]:
    try:
        design = matrix_protocol.build_experiment_design(cfg)
        design_sha = matrix_protocol.design_sha256(cfg)
    except (KeyError, TypeError, ValueError, matrix_protocol.ProtocolError) as exc:
        raise TableGateError("matrix design is invalid") from exc
    if design_sha != FROZEN_MATRIX_DESIGN_SHA:
        raise TableGateError(
            "matrix design SHA differs from the frozen publication design"
        )

    expected_cells = [
        (condition, dataset, model, seed)
        for condition in design["conditions"]
        for dataset in design["datasets"]
        for model in design["models"]
        for seed in design["seeds"]
    ]
    if len(expected_cells) != 54 or len(rows) != 54:
        raise TableGateError(
            f"matrix must contain exactly 54 rows; received {len(rows)}"
        )
    for line_number, advice in enumerate(accepted_advices.values(), 1):
        try:
            matrix_protocol._validate_advice_record(advice, line_number=line_number)
        except matrix_protocol.ProtocolError as exc:
            raise TableGateError("accepted advice evidence is invalid") from exc
        if advice.get("status") != "accepted":
            raise TableGateError("advice evidence is not accepted")

    try:
        partition = matrix_protocol.partition_resume_rows(
            rows,
            design_sha,
            accepted_advices,
            current_dataset_hashes,
            current_advice_keys,
        )
    except matrix_protocol.ProtocolError as exc:
        raise TableGateError(str(exc)) from exc
    if partition["rejected"]:
        reasons = sorted(
            {
                reason
                for rejected in partition["rejected"]
                for reason in rejected["reason_codes"]
            }
        )
        raise TableGateError("matrix evidence invalid: " + ",".join(reasons))

    expected_keys = {matrix_protocol.cell_key(*cell) for cell in expected_cells}
    if partition["done"] != expected_keys:
        missing = expected_keys - partition["done"]
        extra = partition["done"] - expected_keys
        raise TableGateError(
            f"matrix Cartesian grid mismatch: missing={len(missing)} extra={len(extra)}"
        )

    expected_experiment_id = f"matrix-{design_sha[:16]}"
    pair_evidence = {}
    git_shas = set()
    checkpoint_shas = set()
    for row in rows:
        if row.get("experiment_id") != expected_experiment_id:
            raise TableGateError("matrix experiment_id is not frozen")
        if not _is_hex_digest(row.get("git_sha"), 40):
            raise TableGateError("matrix git_sha is invalid")
        if not _is_hex_digest(row.get("checkpoint_sha"), 64):
            raise TableGateError("matrix checkpoint_sha is invalid")
        git_shas.add(row["git_sha"])
        checkpoint_shas.add(row["checkpoint_sha"])

        pair = (row["dataset"], row["model"])
        signature = (
            row["dataset_sha256"],
            matrix_protocol._canonical_json(row["advice_provenance"]),
        )
        previous = pair_evidence.setdefault(pair, signature)
        if previous != signature:
            raise TableGateError("paired advice was not reused across all cells")

        score = row.get("best_real_score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
        ):
            raise TableGateError("best_real_score is invalid")
        trace_scores = [float(item["score"]) for item in row["budget_score_trace"]]
        if not math.isclose(float(score), max(trace_scores), abs_tol=1e-12):
            raise TableGateError("best_real_score does not match the execution trace")
        if row.get("n_extended") != row["event_types"].count("campaign_extended"):
            raise TableGateError("extension count does not match event evidence")

    expected_pairs = {
        (dataset, model) for dataset in design["datasets"] for model in design["models"]
    }
    if set(pair_evidence) != expected_pairs:
        raise TableGateError("paired advice coverage is incomplete")
    if len(git_shas) != 1 or len(checkpoint_shas) != 1:
        raise TableGateError("matrix code or checkpoint provenance is not frozen")

    condition_rank = {value: index for index, value in enumerate(design["conditions"])}
    dataset_rank = {value: index for index, value in enumerate(design["datasets"])}
    model_rank = {value: index for index, value in enumerate(design["models"])}
    return sorted(
        rows,
        key=lambda row: (
            dataset_rank[row["dataset"]],
            condition_rank[row["condition"]],
            model_rank[row["model"]],
            row["seed"],
        ),
    )


def _mean_std(values: list[float]) -> str:
    if not values:
        return "--"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{np.mean(values):.4f} $\\pm$ {np.std(values):.4f}"


def agent_matrix_table(rows: list[dict]) -> str:
    """Điều kiện × dataset: best score thật (mean±std theo seed) + cột cơ chế."""
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
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
                r["best_real_score"]
                for r in cell
                if r.get("best_real_score") is not None
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
    parser.add_argument("--matrix", default="benchmarks/agent_matrix_results.jsonl")
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
