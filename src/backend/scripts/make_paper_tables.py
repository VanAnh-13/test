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
import hashlib
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
import yaml

from automl.search.datasets_real import load_dataset
from scripts import run_experiment_matrix as matrix_protocol

FROZEN_MATRIX_DESIGN_SHA = (
    "0860d3662ed8a2420aa46887cce84fdb70787c34f5b2271690976905bb4893bb"
)
DEFAULT_WM_MANIFEST = (
    BACKEND / "data" / "world_model" / "outcome_ensemble_v2" / "manifest.json"
)
MATRIX_CONDITIONS = ("A", "B", "C")
MATRIX_DATASETS = (
    "iris",
    "wine",
    "breast_cancer",
    "digits",
    "glass",
    "online_shoppers",
)
MATRIX_SEEDS = (0, 1, 2)
MATRIX_MODEL = "meta-ai"
HPO_STRATEGIES = (
    "grid_search",
    "random_search",
    "bayesian_search",
    "genetic_algorithm",
    "successive_halving",
)
FROZEN_HPO_CONFIG = {
    "budget": 8,
    "cv": 3,
    "param_grid": {
        "n_estimators": [50, 100, 200],
        "max_depth": [4, 8, 16],
        "min_samples_split": [2, 10],
    },
}


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
    expected_checkpoint_sha: str,
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
    if not _is_hex_digest(expected_checkpoint_sha, 64):
        raise TableGateError("manifest checkpoint SHA is invalid")

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
        git_shas.add(row["git_sha"])

        condition = row["condition"]
        checkpoint_sha = row.get("checkpoint_sha")
        variant_sources = row.get("variant_sources")
        if not isinstance(variant_sources, list) or any(
            not isinstance(source, str) for source in variant_sources
        ):
            raise TableGateError(f"condition {condition} variant evidence is invalid")
        if condition == "A":
            if checkpoint_sha is not None or "wm_planner" in variant_sources:
                raise TableGateError(
                    "condition A must not use a checkpoint or wm_planner"
                )
        else:
            if checkpoint_sha != expected_checkpoint_sha:
                raise TableGateError(
                    f"condition {condition} checkpoint does not match manifest"
                )
            if "wm_planner" not in variant_sources:
                raise TableGateError(f"condition {condition} lacks wm_planner evidence")
            checkpoint_shas.add(checkpoint_sha)

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
    if len(git_shas) != 1:
        raise TableGateError("matrix code provenance is not frozen")
    if checkpoint_shas != {expected_checkpoint_sha}:
        raise TableGateError("matrix WM checkpoint provenance is not frozen")

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


def load_json_document(path: Path, artifact: str) -> dict:
    if not path.is_file():
        raise TableGateError(f"{artifact} input is missing: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise TableGateError(f"{artifact} input is unreadable: {path}") from exc
    value = _strict_json(text, artifact=artifact)
    if not isinstance(value, dict):
        raise TableGateError(f"{artifact} input is not an object")
    return value


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise TableGateError(f"checkpoint file is unreadable: {path}") from exc


def _load_expected_checkpoint_sha(manifest_path: Path) -> str:
    manifest = load_json_document(manifest_path, "WM manifest")
    head = manifest.get("head")
    if (
        not isinstance(head, dict)
        or not isinstance(head.get("filename"), str)
        or not _is_hex_digest(head.get("sha256"), 64)
    ):
        raise TableGateError("WM manifest head entry is invalid")
    filename = head["filename"]
    if Path(filename).name != filename:
        raise TableGateError("WM manifest head filename is not local")
    checkpoint_path = manifest_path.parent.parent / filename
    if not checkpoint_path.is_file():
        raise TableGateError(f"checkpoint file is missing: {checkpoint_path}")
    if _sha256_file(checkpoint_path) != head["sha256"]:
        raise TableGateError("WM manifest head SHA does not match checkpoint file")
    return head["sha256"]


def _finite_number(
    value: Any,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise TableGateError(f"{label} is not finite")
    number = float(value)
    if minimum is not None and number < minimum:
        raise TableGateError(f"{label} is below its minimum")
    if maximum is not None and number > maximum:
        raise TableGateError(f"{label} is above its maximum")
    return number


def _validate_hpo_scale(
    document: dict,
    *,
    label: str,
    expected_datasets: tuple[str, ...],
    expected_seeds: tuple[int, ...],
) -> None:
    datasets = document.get("datasets")
    if not isinstance(datasets, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("name"), str)
        for item in datasets
    ):
        raise TableGateError(f"{label} dataset metadata is invalid")
    dataset_names = tuple(item["name"] for item in datasets)
    if dataset_names != expected_datasets:
        raise TableGateError(f"{label} dataset grid is not frozen")
    seeds = document.get("seeds")
    if (
        not isinstance(seeds, list)
        or any(type(seed) is not int for seed in seeds)
        or tuple(seeds) != expected_seeds
    ):
        raise TableGateError(f"{label} seed grid is not frozen")

    results = document.get("results")
    expected_count = len(expected_datasets) * len(expected_seeds) * len(HPO_STRATEGIES)
    if not isinstance(results, list) or len(results) != expected_count:
        raise TableGateError(
            f"{label} must contain exactly {expected_count} raw results"
        )
    expected_coordinates = {
        (dataset, strategy, seed)
        for dataset in expected_datasets
        for strategy in HPO_STRATEGIES
        for seed in expected_seeds
    }
    coordinates = set()
    by_strategy = defaultdict(list)
    for index, result in enumerate(results, 1):
        if not isinstance(result, dict):
            raise TableGateError(f"{label} result {index} is not an object")
        coordinate = (
            result.get("dataset"),
            result.get("strategy"),
            result.get("seed"),
        )
        if coordinate in coordinates:
            raise TableGateError(f"{label} contains a duplicate raw result")
        coordinates.add(coordinate)
        test_score = _finite_number(
            result.get("test_score"),
            label=f"{label} test_score",
            minimum=0.0,
            maximum=1.0,
        )
        _finite_number(
            result.get("cv_score"),
            label=f"{label} cv_score",
            minimum=0.0,
            maximum=1.0,
        )
        seconds = _finite_number(
            result.get("seconds"), label=f"{label} seconds", minimum=0.0
        )
        if seconds <= 0:
            raise TableGateError(f"{label} seconds must be positive")
        by_strategy[result.get("strategy")].append((test_score, seconds))
    if coordinates != expected_coordinates:
        raise TableGateError(f"{label} raw Cartesian grid is incomplete")

    summary = document.get("summary")
    if not isinstance(summary, dict) or set(summary) != set(HPO_STRATEGIES):
        raise TableGateError(f"{label} summary strategy set is invalid")
    grid_total = sum(seconds for _, seconds in by_strategy["grid_search"])
    for strategy in HPO_STRATEGIES:
        entry = summary[strategy]
        if not isinstance(entry, dict):
            raise TableGateError(f"{label} summary {strategy} is not an object")
        values = by_strategy[strategy]
        scores = [score for score, _ in values]
        total_seconds = sum(seconds for _, seconds in values)
        expected = {
            "mean_test": float(np.mean(scores)),
            "std_test": float(np.std(scores)),
            "speedup_vs_grid_total": grid_total / total_seconds,
        }
        if type(entry.get("n_runs")) is not int or entry["n_runs"] != len(values):
            raise TableGateError(f"{label} summary n_runs is invalid")
        for field, expected_value in expected.items():
            actual = _finite_number(
                entry.get(field), label=f"{label} summary {strategy}.{field}"
            )
            if not math.isclose(actual, expected_value, rel_tol=1e-12, abs_tol=1e-12):
                raise TableGateError(
                    f"{label} summary {strategy}.{field} disagrees with raw results"
                )


def validate_hpo_inputs(small: dict, large: dict) -> None:
    _validate_hpo_scale(
        small,
        label="HPO small",
        expected_datasets=MATRIX_DATASETS,
        expected_seeds=(42, 43, 44),
    )
    _validate_hpo_scale(
        large,
        label="HPO large",
        expected_datasets=("covtype",),
        expected_seeds=(42,),
    )
    for field, expected_value in FROZEN_HPO_CONFIG.items():
        expected_json = matrix_protocol._canonical_json(expected_value)
        for label, document in (("HPO small", small), ("HPO large", large)):
            if matrix_protocol._canonical_json(document.get(field)) != expected_json:
                raise TableGateError(
                    f"{label} {field} differs from the frozen HPO protocol"
                )


def _mean_std(values: list[float]) -> str:
    if not values:
        return "--"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{np.mean(values):.4f} $\\pm$ {np.std(values):.4f}"


def agent_matrix_table(rows: list[dict]) -> str:
    """Render the complete, already validated 54-cell matrix."""
    if len(rows) != 54:
        raise TableGateError(
            f"matrix renderer requires exactly 54 rows; received {len(rows)}"
        )
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        by_cell[(row["condition"], row["dataset"])].append(row)
    expected_cells = {
        (condition, dataset)
        for dataset in MATRIX_DATASETS
        for condition in MATRIX_CONDITIONS
    }
    if set(by_cell) != expected_cells:
        raise TableGateError("matrix renderer received a non-frozen Cartesian grid")
    for cell, cell_rows in by_cell.items():
        if len(cell_rows) != len(MATRIX_SEEDS) or sorted(
            row["seed"] for row in cell_rows
        ) != list(MATRIX_SEEDS):
            raise TableGateError(
                f"matrix renderer received incomplete seeds for {cell}"
            )

    git_sha = rows[0]["git_sha"]
    checkpoint_sha = next(
        row["checkpoint_sha"] for row in rows if row["condition"] != "A"
    )

    lines = [
        "% Sinh tự động bởi scripts/make_paper_tables.py — KHÔNG sửa tay",
        f"% design_sha256={FROZEN_MATRIX_DESIGN_SHA}",
        f"% git_sha={git_sha}",
        f"% wm_checkpoint_sha256={checkpoint_sha}",
        "\\begin{table}[t]",
        "\\centering",
        (
            "\\caption{Agent-level results using one shared paired-advice"
            " decision per dataset/model. Best observed search score and jobs"
            " per cell are descriptive mean $\\pm$ population standard"
            " deviation over three seeds; extensions are summed and cell"
            " tokens are averaged.}"
        ),
        "\\label{tab:agent-matrix}",
        "\\small",
        "\\begin{tabular}{ll" + "c" * 4 + "}",
        "\\toprule",
        "Dataset & Condition & Best score & Jobs/cell & Extensions & Cell tokens \\\\",
        "\\midrule",
    ]
    for dataset in MATRIX_DATASETS:
        first = True
        for condition in MATRIX_CONDITIONS:
            cell_rows = by_cell[(condition, dataset)]
            scores = [float(row["best_real_score"]) for row in cell_rows]
            jobs = [int(row["n_real_jobs"]) for row in cell_rows]
            extensions = sum(int(row["n_extended"]) for row in cell_rows)
            tokens = [
                int(row["cost_metrics"]["total_input_tokens"])
                + int(row["cost_metrics"]["total_output_tokens"])
                for row in cell_rows
            ]
            dataset_column = dataset.replace("_", "\\_") if first else ""
            first = False
            lines.append(
                f"{dataset_column} & {condition} & {_mean_std(scores)} & "
                f"{np.mean(jobs):.1f} & {extensions} & {np.mean(tokens):.0f} \\\\"
            )
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines.pop()
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def hpo_two_scales_table(small: dict, large: dict) -> str:
    """Render validated HPO evidence at the two committed scales."""
    validate_hpo_inputs(small, large)

    def row(strategy: str) -> str:
        small_summary = small["summary"][strategy]
        large_summary = large["summary"][strategy]
        strategy_label = strategy.replace("_", "\\_")
        return (
            f"{strategy_label} & "
            f"{float(small_summary['mean_test']):.4f} $\\pm$ "
            f"{float(small_summary['std_test']):.4f} & "
            f"{float(small_summary['speedup_vs_grid_total']):.2f}$\\times$ & "
            f"{float(large_summary['mean_test']):.4f} & "
            f"{float(large_summary['speedup_vs_grid_total']):.2f}$\\times$ \\\\"
        )

    lines = [
        "% Sinh tự động bởi scripts/make_paper_tables.py — KHÔNG sửa tay",
        "\\begin{table}[t]",
        "\\centering",
        (
            "\\caption{Descriptive HPO results at two data scales with equal"
            " search space and budget. Small-scale values aggregate six"
            " datasets $\\times$ three seeds; the Covertype result uses a"
            " single seed and does not support an inferential ranking claim.}"
        ),
        "\\label{tab:hpo-two-scales}",
        "\\small",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        (
            " & \\multicolumn{2}{c}{Small (150--12k rows)} &"
            " \\multicolumn{2}{c}{Covertype (581k rows)} \\\\"
        ),
        "Strategy & Test acc. & Speedup & Test acc. & Speedup \\\\",
        "\\midrule",
    ]
    lines += [row(strategy) for strategy in HPO_STRATEGIES]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def _resolve_input(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else BACKEND / path


def _load_matrix_config(path: Path) -> dict:
    if not path.is_file():
        raise TableGateError(f"matrix config is missing: {path}")
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise TableGateError(f"matrix config is unreadable: {path}") from exc
    if not isinstance(cfg, dict):
        raise TableGateError("matrix config is not an object")
    return cfg


def _current_matrix_bindings(
    cfg: dict,
) -> tuple[dict[str, str], dict[tuple[str, str], str]]:
    design = matrix_protocol.build_experiment_design(cfg)
    design_sha = matrix_protocol.design_sha256(cfg)
    dataset_hashes = {}
    prompt_hashes = {}
    for dataset_name in design["datasets"]:
        dataset = load_dataset(dataset_name)
        dataset_hashes[dataset_name] = matrix_protocol.dataset_sha256(dataset)
        prompt = matrix_protocol._advice_prompt(
            matrix_protocol.anonymized_advice_payload(dataset)
        )
        prompt_hashes[dataset_name] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    advice_keys = {
        (dataset_name, model): matrix_protocol._advice_key(
            design_sha,
            dataset_hashes[dataset_name],
            model,
            prompt_hashes[dataset_name],
        )
        for dataset_name in design["datasets"]
        for model in design["models"]
    }
    return dataset_hashes, advice_keys


def _snapshot_output(path: Path) -> tuple[bool, bytes]:
    if not path.exists():
        return False, b""
    try:
        return True, path.read_bytes()
    except OSError as exc:
        raise TableGateError(f"cannot snapshot existing output: {path}") from exc


def _restore_output(path: Path, existed: bool, data: bytes) -> None:
    if existed:
        path.write_bytes(data)
    elif path.exists():
        path.unlink()


def _write_tables_all_or_none(
    agent_path: Path,
    agent_table: str,
    hpo_path: Path,
    hpo_table: str,
) -> None:
    snapshots = {
        agent_path: _snapshot_output(agent_path),
        hpo_path: _snapshot_output(hpo_path),
    }
    try:
        agent_path.write_text(agent_table, encoding="utf-8", newline="\n")
        hpo_path.write_text(hpo_table, encoding="utf-8", newline="\n")
    except OSError as exc:
        rollback_errors = []
        for path, (existed, data) in snapshots.items():
            try:
                _restore_output(path, existed, data)
            except OSError as rollback_exc:
                rollback_errors.append(f"{path}: {rollback_exc}")
        if rollback_errors:
            detail = "; ".join(rollback_errors)
            raise TableGateError(
                f"table output failed and rollback failed: {detail}"
            ) from exc
        raise TableGateError(f"table output failed: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sinh bảng LaTeX từ kết quả thật")
    parser.add_argument("--matrix", default="benchmarks/agent_matrix_results.jsonl")
    parser.add_argument("--advice", default="benchmarks/agent_matrix_advice.jsonl")
    parser.add_argument("--config", default="benchmarks/agent_matrix_config.yaml")
    parser.add_argument("--wm-manifest", default=str(DEFAULT_WM_MANIFEST))
    parser.add_argument("--hpo-small", default="benchmarks/hpo_real_fair.json")
    parser.add_argument("--hpo-large", default="benchmarks/hpo_large_clean.json")
    parser.add_argument(
        "--out-dir", default=str(BACKEND.parent.parent / "paper" / "tables")
    )
    args = parser.parse_args(argv)

    try:
        cfg = _load_matrix_config(_resolve_input(args.config))
        matrix_rows = load_jsonl(_resolve_input(args.matrix))
        advice_path = _resolve_input(args.advice)
        if not advice_path.is_file():
            raise TableGateError(f"advice input is missing: {advice_path}")
        advice_state = matrix_protocol.load_advice_index(advice_path)
        dataset_hashes, advice_keys = _current_matrix_bindings(cfg)
        expected_checkpoint_sha = _load_expected_checkpoint_sha(
            _resolve_input(args.wm_manifest)
        )
        matrix_rows = validate_matrix_rows(
            matrix_rows,
            cfg=cfg,
            accepted_advices=advice_state["accepted"],
            current_dataset_hashes=dataset_hashes,
            current_advice_keys=advice_keys,
            expected_checkpoint_sha=expected_checkpoint_sha,
        )
        hpo_small = load_json_document(_resolve_input(args.hpo_small), "HPO small")
        hpo_large = load_json_document(_resolve_input(args.hpo_large), "HPO large")
        validate_hpo_inputs(hpo_small, hpo_large)
        agent_table = agent_matrix_table(matrix_rows)
        hpo_table = hpo_two_scales_table(hpo_small, hpo_large)
    except (
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        matrix_protocol.ProtocolError,
    ) as exc:
        print(f"Table evidence rejected: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    agent_path = out_dir / "agent_matrix.tex"
    hpo_path = out_dir / "hpo_two_scales.tex"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_tables_all_or_none(agent_path, agent_table, hpo_path, hpo_table)
    except (OSError, TableGateError) as exc:
        print(f"Table output failed: {exc}", file=sys.stderr)
        return 2

    print(f"  {agent_path}")
    print(f"  {hpo_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
