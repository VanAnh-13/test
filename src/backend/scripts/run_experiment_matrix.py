#!/usr/bin/env python3
"""
Runner ma trận thí nghiệm agent — conditions × datasets × models × seeds.

Thiết kế:
  - LLM THẬT (per-request model, T12) + job sklearn THẬT chạy in-process:
    RealJobEnv chặn invoke_tool, start_training chạy đúng
    SearchStrategyFactory trên dataset thật từ automl/search/datasets_real
    → không docker, không mock score, vô hiệu phản biện "toàn mock".
  - Điều kiện A/B/C/C_mpc render thành yaml tạm từ hagent.yaml gốc,
    HAGENT_CONFIG trỏ vào + load_config.cache_clear() mỗi ô.
  - JSONL mỗi ô một dòng, RESUMABLE (ô đã có bị bỏ qua).

Usage:
  cd src/backend
  python scripts/run_experiment_matrix.py --dry-run
  python scripts/run_experiment_matrix.py                 # chạy theo config
  python scripts/run_experiment_matrix.py --only A:iris:openai-gpt4o-mini:0
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import math
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

# ── Điều kiện → patch yaml ───────────────────────────────

_MISSING = "./data/world_model/__disabled__"
PROTOCOL_VERSION = "paired-meta-advice-v1"
PROTOCOL_METRIC = "accuracy"
PROTOCOL_CONDITIONS = ("A", "B", "C")
PROTOCOL_SEEDS = (0, 1, 2)
ADVICE_ALGORITHMS = (
    "grid_search",
    "bayesian_search",
    "genetic_algorithm",
    "random_search",
    "successive_halving",
)
META_FEATURE_KEYS = (
    "n_rows",
    "n_cols",
    "n_classes",
    "class_imbalance",
    "frac_categorical",
    "missing_frac",
    "mean_abs_skew",
)


class ProtocolError(RuntimeError):
    """Fail-closed protocol violation."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_experiment_design(cfg: dict) -> Dict[str, Any]:
    dimensions: Dict[str, list] = {}
    for name in ("conditions", "datasets", "models", "seeds"):
        values = cfg.get(name)
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise ProtocolError(f"{name} must be a non-empty unique list")
        dimensions[name] = list(values)
    if tuple(dimensions["conditions"]) != PROTOCOL_CONDITIONS:
        raise ProtocolError("conditions must be exactly A, B, C")
    if tuple(dimensions["seeds"]) != PROTOCOL_SEEDS:
        raise ProtocolError("seeds must be exactly 0, 1, 2")
    cell_count = (
        len(dimensions["conditions"])
        * len(dimensions["datasets"])
        * len(dimensions["models"])
        * len(dimensions["seeds"])
    )
    if cell_count != 54:
        raise ProtocolError(f"main matrix must contain 54 cells, got {cell_count}")
    prompt = cfg.get("prompt")
    job = cfg.get("job")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ProtocolError("prompt must be a non-empty string")
    if not isinstance(job, dict):
        raise ProtocolError("job must be an object")
    return {
        "protocol_version": PROTOCOL_VERSION,
        **dimensions,
        "metric": PROTOCOL_METRIC,
        "search_algorithms": list(ADVICE_ALGORITHMS),
        "job": job,
        "prompt": prompt,
    }


def design_sha256(cfg: dict) -> str:
    return hashlib.sha256(
        _canonical_json(build_experiment_design(cfg)).encode("utf-8")
    ).hexdigest()


def dataset_sha256(dataset: Dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in ("X", "y"):
        if name not in dataset:
            raise ProtocolError(f"dataset missing {name}")
        array = np.ascontiguousarray(np.asarray(dataset[name], dtype="<f8"))
        digest.update(name.encode("ascii"))
        digest.update(_canonical_json(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def anonymized_advice_payload(
    dataset: Dict[str, Any],
    *,
    metric: str = PROTOCOL_METRIC,
    algorithms: tuple[str, ...] = ADVICE_ALGORITHMS,
) -> Dict[str, Any]:
    if metric != PROTOCOL_METRIC:
        raise ProtocolError(f"metric must be {PROTOCOL_METRIC}")
    if tuple(algorithms) != ADVICE_ALGORITHMS:
        raise ProtocolError("search algorithm pool differs from frozen protocol")
    meta = dataset.get("meta")
    if not isinstance(meta, dict):
        raise ProtocolError("dataset meta-features are missing")
    safe_meta: Dict[str, float] = {}
    for key in META_FEATURE_KEYS:
        value = meta.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise ProtocolError(f"meta-feature {key} is not numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ProtocolError(f"meta-feature {key} is not finite")
        safe_meta[key] = number
    return {
        "meta_features": safe_meta,
        "metric": metric,
        "search_algorithms": list(algorithms),
    }


def _advice_prompt(payload: Dict[str, Any]) -> str:
    return (
        "Select exactly one HPO search algorithm from the supplied ordered pool. "
        'Return exactly one JSON object: {"search_algorithm":"<enum>"}. '
        "Do not add prose, markdown, or fields.\n"
        + _canonical_json(payload)
    )


def _invoke_advice_model(model: str, prompt: str) -> tuple[str, dict]:
    from hagent.agent.llm_config import create_chat_model, require_model_config
    from hagent.agent.middlewares.usage_tracker import (
        UsageTracker,
        UsageTrackingCallback,
    )

    require_model_config(model)
    tracker = UsageTracker()
    chat_model = create_chat_model(
        model,
        temperature=0,
        max_tokens=64,
        callbacks=[UsageTrackingCallback(tracker)],
    )
    message = chat_model.invoke(prompt)
    usage = tracker.summary()
    if usage.get("total_calls") == 0:
        direct = getattr(message, "usage_metadata", None) or {}
        metadata = getattr(message, "response_metadata", None) or {}
        token_usage = metadata.get("token_usage") or metadata.get("usage") or {}
        usage = {
            "total_input_tokens": direct.get("input_tokens")
            or token_usage.get("prompt_tokens")
            or token_usage.get("input_tokens")
            or 0,
            "total_output_tokens": direct.get("output_tokens")
            or token_usage.get("completion_tokens")
            or token_usage.get("output_tokens")
            or 0,
            "total_calls": 1,
        }
    return getattr(message, "content", None), usage


def _strict_advice_algorithm(response: str) -> str:
    if not isinstance(response, str):
        raise ProtocolError("advice response must be text")

    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ProtocolError("advice response contains duplicate fields")
            result[key] = value
        return result

    try:
        value = json.loads(response, object_pairs_hook=no_duplicates)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ProtocolError("advice response is not strict JSON") from exc
    if not isinstance(value, dict) or set(value) != {"search_algorithm"}:
        raise ProtocolError("advice response must contain only search_algorithm")
    algorithm = value["search_algorithm"]
    if not isinstance(algorithm, str) or algorithm not in ADVICE_ALGORITHMS:
        raise ProtocolError("advice search_algorithm is outside the frozen enum")
    return algorithm


def _strict_usage(usage: dict) -> Dict[str, int]:
    if not isinstance(usage, dict):
        raise ProtocolError("advice usage is missing")
    values = [
        usage.get("total_input_tokens"),
        usage.get("total_output_tokens"),
        usage.get("total_calls"),
    ]
    if any(type(value) is not int or value < 0 for value in values):
        raise ProtocolError("advice usage values must be non-negative integers")
    input_tokens, output_tokens, calls = values
    if calls != 1 or input_tokens + output_tokens <= 0:
        raise ProtocolError("advice usage must prove exactly one non-zero-token call")
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "total_calls": calls,
    }


def _advice_key(design_sha: str, data_sha: str, model: str) -> str:
    material = {
        "design_sha256": design_sha,
        "dataset_sha256": data_sha,
        "model": model,
        "metric": PROTOCOL_METRIC,
        "search_algorithms": list(ADVICE_ALGORITHMS),
    }
    return hashlib.sha256(_canonical_json(material).encode("utf-8")).hexdigest()


def request_paired_advice(
    dataset: Dict[str, Any],
    *,
    model: str,
    design_sha: str,
    experiment_id: str,
    invoke: Optional[Callable[[str, str], tuple[str, dict]]] = None,
) -> Dict[str, Any]:
    if (
        not isinstance(design_sha, str)
        or len(design_sha) != 64
        or any(char not in "0123456789abcdef" for char in design_sha)
    ):
        raise ProtocolError("design_sha must be a lowercase SHA-256")
    if not isinstance(model, str) or not model:
        raise ProtocolError("model is required")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ProtocolError("experiment_id is required")

    payload = anonymized_advice_payload(dataset)
    prompt = _advice_prompt(payload)
    boundary = invoke or _invoke_advice_model
    try:
        response, raw_usage = boundary(model, prompt)
    except Exception as exc:
        raise ProtocolError("Meta advice invocation failed") from exc
    algorithm = _strict_advice_algorithm(response)
    usage = _strict_usage(raw_usage)
    data_sha = dataset_sha256(dataset)
    return {
        "record_type": "paired_meta_advice",
        "status": "accepted",
        "protocol_version": PROTOCOL_VERSION,
        "experiment_id": experiment_id,
        "design_sha256": design_sha,
        "advice_key": _advice_key(design_sha, data_sha, model),
        "dataset_sha256": data_sha,
        "model": model,
        "metric": PROTOCOL_METRIC,
        "search_algorithms": list(ADVICE_ALGORITHMS),
        "algorithm": algorithm,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
        "token_usage": usage,
        "cost_usd": None,
        "accepted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


_ADVICE_COMMON_FIELDS = {
    "record_type",
    "status",
    "protocol_version",
    "experiment_id",
    "design_sha256",
    "advice_key",
    "dataset_sha256",
    "model",
    "metric",
    "search_algorithms",
    "prompt_sha256",
}
_PENDING_FIELDS = _ADVICE_COMMON_FIELDS | {"created_at"}
_ACCEPTED_FIELDS = _ADVICE_COMMON_FIELDS | {
    "algorithm",
    "response_sha256",
    "token_usage",
    "cost_usd",
    "accepted_at",
}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _validate_advice_record(record: dict, *, line_number: int) -> None:
    if not isinstance(record, dict):
        raise ProtocolError(f"advice line {line_number} is not an object")
    status = record.get("status")
    fields = _PENDING_FIELDS if status == "pending" else _ACCEPTED_FIELDS
    if status not in ("pending", "accepted") or set(record) != fields:
        raise ProtocolError(f"advice line {line_number} has an invalid schema")
    if (
        record["record_type"] != "paired_meta_advice"
        or record["protocol_version"] != PROTOCOL_VERSION
        or record["metric"] != PROTOCOL_METRIC
        or record["search_algorithms"] != list(ADVICE_ALGORITHMS)
    ):
        raise ProtocolError(f"advice line {line_number} conflicts with protocol")
    for field in (
        "design_sha256",
        "advice_key",
        "dataset_sha256",
        "prompt_sha256",
    ):
        if not _is_sha256(record.get(field)):
            raise ProtocolError(f"advice line {line_number} has invalid {field}")
    if not isinstance(record.get("model"), str) or not record["model"]:
        raise ProtocolError(f"advice line {line_number} has invalid model")
    if not isinstance(record.get("experiment_id"), str) or not record["experiment_id"]:
        raise ProtocolError(f"advice line {line_number} has invalid experiment_id")
    expected_key = _advice_key(
        record["design_sha256"], record["dataset_sha256"], record["model"]
    )
    if record["advice_key"] != expected_key:
        raise ProtocolError(f"advice line {line_number} has invalid advice_key")
    if status == "accepted":
        if record.get("algorithm") not in ADVICE_ALGORITHMS:
            raise ProtocolError(f"advice line {line_number} has invalid algorithm")
        if not _is_sha256(record.get("response_sha256")):
            raise ProtocolError(f"advice line {line_number} has invalid response_sha256")
        usage = record.get("token_usage")
        if not isinstance(usage, dict) or set(usage) != {
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "total_calls",
        }:
            raise ProtocolError(f"advice line {line_number} has invalid token usage")
        values = list(usage.values())
        if any(type(value) is not int or value < 0 for value in values):
            raise ProtocolError(f"advice line {line_number} has invalid token usage")
        if (
            usage["total_calls"] != 1
            or usage["total_tokens"]
            != usage["input_tokens"] + usage["output_tokens"]
            or usage["total_tokens"] <= 0
        ):
            raise ProtocolError(f"advice line {line_number} has invalid token usage")
        if record.get("cost_usd") is not None:
            raise ProtocolError(f"advice line {line_number} must use null unknown cost")


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(_canonical_json(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_advice_index(path: Path) -> Dict[str, Dict[str, dict]]:
    state: Dict[str, Dict[str, dict]] = {"pending": {}, "accepted": {}}
    if not path.is_file():
        return state
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"advice line {line_number} is malformed") from exc
        _validate_advice_record(record, line_number=line_number)
        key = record["advice_key"]
        status = record["status"]
        if status == "pending":
            if key in state["pending"] or key in state["accepted"]:
                raise ProtocolError(f"duplicate pending advice key {key}")
            state["pending"][key] = record
            continue
        if key in state["accepted"]:
            raise ProtocolError(f"duplicate accepted advice key {key}")
        pending = state["pending"].get(key)
        if pending is None:
            raise ProtocolError(f"accepted advice key {key} has no pending claim")
        for field in _ADVICE_COMMON_FIELDS - {"status"}:
            if record[field] != pending[field]:
                raise ProtocolError(f"accepted advice key {key} conflicts with pending")
        state["accepted"][key] = record
    return state


def ensure_paired_advices(
    *,
    cells: List[tuple[str, str, str, int]],
    datasets: Dict[str, Dict[str, Any]],
    sidecar_path: Path,
    design_sha: str,
    experiment_id: str,
    invoke: Optional[Callable[[str, str], tuple[str, dict]]] = None,
) -> Dict[tuple[str, str], dict]:
    state = load_advice_index(sidecar_path)
    pairs: List[tuple[str, str]] = []
    for cell in cells:
        if not isinstance(cell, tuple) or len(cell) != 4:
            raise ProtocolError("cell must be condition, dataset, model, seed")
        pair = (cell[1], cell[2])
        if pair not in pairs:
            pairs.append(pair)
    result: Dict[tuple[str, str], dict] = {}
    for dataset_name, model in pairs:
        dataset = datasets.get(dataset_name)
        if dataset is None:
            raise ProtocolError(f"dataset {dataset_name} is not loaded")
        data_sha = dataset_sha256(dataset)
        key = _advice_key(design_sha, data_sha, model)
        accepted = state["accepted"].get(key)
        if accepted is not None:
            result[(dataset_name, model)] = accepted
            continue
        if key in state["pending"]:
            raise ProtocolError(f"advice key {key} is pending; refusing duplicate call")
        prompt_sha = hashlib.sha256(
            _advice_prompt(anonymized_advice_payload(dataset)).encode("utf-8")
        ).hexdigest()
        pending = {
            "record_type": "paired_meta_advice",
            "status": "pending",
            "protocol_version": PROTOCOL_VERSION,
            "experiment_id": experiment_id,
            "design_sha256": design_sha,
            "advice_key": key,
            "dataset_sha256": data_sha,
            "model": model,
            "metric": PROTOCOL_METRIC,
            "search_algorithms": list(ADVICE_ALGORITHMS),
            "prompt_sha256": prompt_sha,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _append_jsonl(sidecar_path, pending)
        state["pending"][key] = pending
        accepted = request_paired_advice(
            dataset,
            model=model,
            design_sha=design_sha,
            experiment_id=experiment_id,
            invoke=invoke,
        )
        for field in _ADVICE_COMMON_FIELDS - {"status"}:
            if accepted[field] != pending[field]:
                raise ProtocolError(f"accepted advice key {key} conflicts with claim")
        _append_jsonl(sidecar_path, accepted)
        state["accepted"][key] = accepted
        result[(dataset_name, model)] = accepted
    return result



def validate_result_evidence(
    row: dict,
    design_sha: str,
    accepted_advices: Dict[str, dict],
) -> List[str]:
    reasons: List[str] = []

    def reject(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    if not isinstance(row, dict):
        return ["result_not_object"]
    if "error" not in row or row["error"] is not None:
        reject("cell_error")
    if row.get("design_sha256") != design_sha:
        reject("design_sha_mismatch")
    try:
        expected_key = cell_key(
            row["condition"], row["dataset"], row["model"], row["seed"]
        )
    except (KeyError, TypeError):
        expected_key = None
    if expected_key is None or row.get("key") != expected_key:
        reject("cell_key_invalid")
    if (
        row.get("condition") not in PROTOCOL_CONDITIONS
        or row.get("seed") not in PROTOCOL_SEEDS
    ):
        reject("cell_dimension_invalid")
    if not _is_sha256(row.get("dataset_sha256")):
        reject("dataset_sha_invalid")

    provenance = row.get("advice_provenance")
    provenance_fields = {
        "experiment_id",
        "design_sha256",
        "advice_key",
        "algorithm",
        "prompt_sha256",
        "response_sha256",
        "token_usage",
    }
    advice = None
    if not isinstance(provenance, dict) or set(provenance) != provenance_fields:
        reject("advice_provenance_invalid")
    else:
        advice = accepted_advices.get(provenance.get("advice_key"))
        if advice is None:
            reject("advice_not_accepted")
        else:
            for field in provenance_fields:
                if provenance[field] != advice[field]:
                    reject("advice_provenance_mismatch")
                    break
            if row.get("design_sha256") != advice["design_sha256"]:
                reject("advice_provenance_mismatch")
            if row.get("dataset_sha256") != advice["dataset_sha256"]:
                reject("advice_provenance_mismatch")
            if row.get("model") != advice["model"]:
                reject("advice_provenance_mismatch")
            if row.get("experiment_id") != advice["experiment_id"]:
                reject("advice_provenance_mismatch")

    cost = row.get("cost_metrics")
    if not isinstance(cost, dict):
        reject("cell_usage_invalid")
    else:
        usage_values = [
            cost.get("total_input_tokens"),
            cost.get("total_output_tokens"),
            cost.get("total_calls"),
        ]
        if (
            any(type(value) is not int or value < 0 for value in usage_values)
            or usage_values[2] <= 0
            or usage_values[0] + usage_values[1] <= 0
        ):
            reject("cell_usage_invalid")

    trace = row.get("budget_score_trace")
    trace_valid = isinstance(trace, list) and bool(trace)
    trace_algorithms: List[str] = []
    trace_jobs: Dict[str, str] = {}
    if trace_valid:
        for item in trace:
            required = {
                "sequence",
                "job_id",
                "search_algorithm",
                "budget_seconds",
                "score",
                "elapsed_seconds",
                "time_limited",
            }
            if not isinstance(item, dict) or not required.issubset(item):
                trace_valid = False
                break
            numeric_values = (
                item["budget_seconds"],
                item["score"],
                item["elapsed_seconds"],
            )
            if (
                type(item["sequence"]) is not int
                or item["sequence"] <= 0
                or not isinstance(item["job_id"], str)
                or not item["job_id"]
                or item["search_algorithm"] not in ADVICE_ALGORITHMS
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in numeric_values
                )
                or float(item["budget_seconds"]) <= 0
                or float(item["elapsed_seconds"]) < 0
                or type(item["time_limited"]) is not bool
            ):
                trace_valid = False
                break
            trace_algorithms.append(item["search_algorithm"])
            trace_jobs[item["job_id"]] = item["search_algorithm"]
    if not trace_valid or row.get("n_real_jobs") != len(trace or []):
        reject("budget_score_trace_invalid")

    executed = row.get("executed_algorithms")
    expected_executed = list(dict.fromkeys(trace_algorithms))
    if not isinstance(executed, list) or not executed or executed != expected_executed:
        reject("executed_algorithms_invalid")
    if not isinstance(row.get("event_types"), list) or any(
        not isinstance(value, str) for value in row.get("event_types") or []
    ):
        reject("event_types_invalid")
    if row.get("campaign_status") != "done":
        reject("campaign_not_done")
    if "requested" not in (row.get("variant_sources") or []):
        reject("requested_variant_missing")

    requested = row.get("requested_variant")
    if (
        not isinstance(requested, dict)
        or requested.get("source") != "requested"
        or requested.get("status") != "completed"
        or requested.get("job_id") not in trace_jobs
    ):
        reject("requested_variant_invalid")
    elif advice is not None and (
        requested.get("algorithm") != advice["algorithm"]
        or trace_jobs[requested["job_id"]] != advice["algorithm"]
    ):
        reject("requested_variant_invalid")
    if advice is not None and advice["algorithm"] not in (executed or []):
        reject("advised_algorithm_not_executed")
    return reasons


def partition_resume_rows(
    rows: List[dict],
    design_sha: str,
    accepted_advices: Dict[str, dict],
) -> Dict[str, Any]:
    seen: set[str] = set()
    done: set[str] = set()
    kept: List[dict] = []
    rejected: List[dict] = []
    for row in rows:
        key = row.get("key") if isinstance(row, dict) else None
        if isinstance(key, str):
            if key in seen:
                raise ProtocolError(f"duplicate result key {key}")
            seen.add(key)
        reasons = validate_result_evidence(row, design_sha, accepted_advices)
        if not reasons:
            done.add(key)
            kept.append(row)
        else:
            rejected.append({"row": row, "reason_codes": reasons})
    return {"done": done, "kept": kept, "rejected": rejected}


def _read_result_rows(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    rows: List[dict] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(
                line,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(value)
                ),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProtocolError(f"result line {line_number} is malformed") from exc
        if not isinstance(row, dict):
            raise ProtocolError(f"result line {line_number} is not an object")
        rows.append(row)
    return rows


_SAFE_REJECTED_ROW_FIELDS = {
    "key",
    "condition",
    "dataset",
    "model",
    "seed",
    "route",
    "campaign_status",
    "n_variants",
    "variant_sources",
    "extension_rounds",
    "best_real_score",
    "n_real_jobs",
    "job_seconds_total",
    "n_outcome_surprise",
    "n_extended",
    "cost_metrics",
    "checkpoint_sha",
    "git_sha",
    "wall_seconds",
    "ts",
    "design_sha256",
    "experiment_id",
    "dataset_sha256",
    "advice_provenance",
    "requested_variant",
    "budget_score_trace",
    "executed_algorithms",
    "event_types",
}


def _rejection_record(row: dict, reason_codes: List[str]) -> dict:
    row_sha = hashlib.sha256(_canonical_json(row).encode("utf-8")).hexdigest()
    rejection_id = hashlib.sha256(
        (row_sha + ":" + _canonical_json(reason_codes)).encode("utf-8")
    ).hexdigest()
    return {
        "record_type": "matrix_preflight_rejection",
        "rejection_id": rejection_id,
        "row_sha256": row_sha,
        "key": row.get("key") if isinstance(row.get("key"), str) else None,
        "reason_codes": list(reason_codes),
        "row": {
            key: row[key]
            for key in _SAFE_REJECTED_ROW_FIELDS
            if key in row
        },
        "had_error": row.get("error") is not None,
        "rejected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _load_rejection_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    ids: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"rejection line {line_number} is malformed") from exc
        rejection_id = record.get("rejection_id") if isinstance(record, dict) else None
        if (
            not isinstance(record, dict)
            or record.get("record_type") != "matrix_preflight_rejection"
            or not _is_sha256(rejection_id)
            or rejection_id in ids
        ):
            raise ProtocolError(f"rejection line {line_number} is invalid")
        ids.add(rejection_id)
    return ids


def migrate_rejected_rows(
    results_path: Path,
    rejected_path: Path,
    *,
    design_sha: str,
    accepted_advices: Dict[str, dict],
) -> set[str]:
    temp_path = results_path.with_name(results_path.name + ".tmp")
    if temp_path.exists():
        raise ProtocolError(f"stale migration temp exists: {temp_path.name}")
    rows = _read_result_rows(results_path)
    partition = partition_resume_rows(rows, design_sha, accepted_advices)
    if not partition["rejected"]:
        return partition["done"]
    rejection_ids = _load_rejection_ids(rejected_path)
    for rejected in partition["rejected"]:
        record = _rejection_record(rejected["row"], rejected["reason_codes"])
        if record["rejection_id"] not in rejection_ids:
            _append_jsonl(rejected_path, record)
            rejection_ids.add(record["rejection_id"])

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with temp_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in partition["kept"]:
            handle.write(_canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, results_path)
    if not partition["kept"]:
        results_path.unlink()
    return partition["done"]


CONDITION_PATCHES: Dict[str, Dict[str, Any]] = {
    "A": {
        "campaign": {
            "wm_variant_proposal": False,
            "wm_rank_variants": False,
            "surprise_extension": {"enabled": False},
        },
        # Trỏ checkpoint vào path không tồn tại — model không thể bị nạp lén
        "world_model": {
            "outcome_head": {"checkpoint_path": _MISSING + ".npz"},
            "outcome_ensemble": {"checkpoint_dir": _MISSING},
        },
    },
    "B": {
        "campaign": {
            "wm_variant_proposal": True,
            "wm_rank_variants": True,
            "surprise_extension": {"enabled": False},
        },
    },
    "C": {
        "campaign": {
            "wm_variant_proposal": True,
            "wm_rank_variants": True,
            "surprise_extension": {
                "enabled": True,
                "max_rounds": 1,
                "n_extra": 2,
                "exploration_weight": 0.5,
            },
        },
    },
    "C_mpc": {
        "campaign": {
            "wm_variant_proposal": True,
            "wm_rank_variants": True,
            "surprise_extension": {
                "enabled": True,
                "max_rounds": 1,
                "n_extra": 2,
                "exploration_weight": 0.5,
            },
        },
        "world_model": {"campaign_planner": {"backend": "cem_mpc_v1"}},
    },
}


def _deep_merge(base: dict, patch: dict) -> dict:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def render_condition_yaml(condition: str, scratch_dir: Path) -> Path:
    """hagent.yaml gốc + patch điều kiện → file tạm; trả path."""
    if condition not in CONDITION_PATCHES:
        raise ValueError(
            f"Condition {condition!r} không hợp lệ. Có: {list(CONDITION_PATCHES)}"
        )
    base_path = BACKEND / "hagent" / "hagent.yaml"
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    patch = CONDITION_PATCHES[condition]

    merged = dict(base)
    if "campaign" in patch:
        agent = dict(merged.get("agent") or {})
        agent["campaign"] = _deep_merge(dict(agent.get("campaign") or {}), patch["campaign"])
        merged["agent"] = agent
    if "world_model" in patch:
        merged["world_model"] = _deep_merge(
            dict(merged.get("world_model") or {}), patch["world_model"]
        )

    out = scratch_dir / f"hagent_{condition}.yaml"
    out.write_text(yaml.safe_dump(merged, allow_unicode=True), encoding="utf-8")
    return out


def apply_condition(condition: str, scratch_dir: Path) -> None:
    """Đặt HAGENT_CONFIG + xóa cache config — PHẢI gọi trước mỗi ô."""
    path = render_condition_yaml(condition, scratch_dir)
    os.environ["HAGENT_CONFIG"] = str(path)
    from hagent.bridge import config as bridge_config

    bridge_config.load_config.cache_clear()
    # Memo outcome model phụ thuộc config path → xóa để nạp đúng điều kiện
    from hagent.agent.campaign import wm_hooks

    wm_hooks._outcome_model_cache["fingerprint"] = None


# ── RealJobEnv: job sklearn thật in-process ──────────────


class RealJobEnv:
    """Tool invoker: start_training chạy THẬT search strategy trên dataset thật."""

    def __init__(self, dataset: Dict[str, Any], *, job_cfg: dict, seed: int):
        self.dataset = dataset
        self.job_cfg = dict(job_cfg or {})
        self.seed = seed
        self.jobs: Dict[str, Dict[str, Any]] = {}

    async def invoke(self, action_type: str, params: dict) -> dict:
        d = self.dataset
        if action_type in ("list_datasets",):
            return {
                "datasets": [
                    {"id": d["name"], "name": d["name"], "n_rows": d["n_rows"],
                     "n_cols": d["n_cols"]}
                ]
            }
        if action_type in ("get_dataset_info", "get_features", "preview_data"):
            return {
                "id": d["name"],
                "name": d["name"],
                "n_rows": d["n_rows"],
                "n_cols": d["n_cols"],
                "features": [f"f{i}" for i in range(d["n_cols"])] + ["target"],
                "target": "target",
            }
        if action_type == "get_available_models":
            return {"models": ["RandomForestClassifier"]}
        if action_type == "start_training":
            job_id = f"real_{len(self.jobs) + 1}_{uuid.uuid4().hex[:6]}"
            record = await asyncio.to_thread(self._run_real_job, dict(params))
            record["job_id"] = job_id
            self.jobs[job_id] = record
            return {"job_id": job_id, "status": 0}
        if action_type == "get_job_info":
            job = self.jobs.get(str(params.get("job_id")))
            if not job:
                return {"error": "job not found"}
            return {
                "id": job["job_id"],
                "status": "completed",
                "best_model": job["best_model"],
                "best_score": job["best_score"],
                "metrics": {"accuracy": job["best_score"]},
            }
        if action_type == "list_jobs":
            return {"jobs": list(self.jobs.values())}
        return {}

    def _run_real_job(self, params: dict) -> dict:
        """Job sklearn thật: strategy.search trên dataset thật (blocking)."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold

        from automl.search.factory.search_strategy_factory import (
            SearchStrategyFactory,
        )

        algo = str(params.get("search_algorithm") or "grid_search")
        cfg = dict(
            cv=StratifiedKFold(
                n_splits=int(self.job_cfg.get("cv", 3)), shuffle=True,
                random_state=self.seed,
            ),
            scoring={"accuracy": "accuracy"},
            metric_sort="accuracy",
            n_jobs=-1,
            random_state=self.seed,
            save_log=False,
            verbose=0,
            max_time=float(params.get("time_limit") or self.job_cfg.get("time_limit") or 60),
            # BO mặc định infer_dimensions=True sẽ tự nới grid thành không
            # gian liên tục — mọi thuật toán PHẢI cùng không gian 18 điểm
            infer_dimensions=False,
        )
        strategy = SearchStrategyFactory.create_strategy(algo, cfg)
        grid = dict(self.job_cfg.get("param_grid") or {})
        t0 = time.perf_counter()
        best_params, best_score, _, _, time_limited = strategy.search(
            RandomForestClassifier(random_state=self.seed),
            grid,
            self.dataset["X"],
            self.dataset["y"],
        )
        return {
            "search_algorithm": algo,
            "best_params": best_params,
            "best_score": float(best_score),
            "best_model": "RandomForestClassifier",
            "seconds": round(time.perf_counter() - t0, 2),
            "time_limited": bool(time_limited),
        }


# ── Cell execution ───────────────────────────────────────


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(BACKEND), text=True
        ).strip()
    except Exception:
        return "unknown"


def _checkpoint_sha() -> Optional[str]:
    try:
        from hagent.bridge.config import get_world_model_config

        path = (get_world_model_config().get("outcome_head") or {}).get("checkpoint_path")
        p = Path(path) if path else None
        if p and not p.is_absolute():
            p = BACKEND / p
        if p and p.is_file():
            return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    except Exception:
        pass
    return None


def cell_key(condition: str, dataset: str, model: str, seed: int) -> str:
    return f"{condition}:{dataset}:{model}:{seed}"


def run_cell(
    condition: str,
    dataset_name: str,
    model: str,
    seed: int,
    *,
    cfg: dict,
    scratch_dir: Path,
) -> Dict[str, Any]:
    from automl.search.datasets_real import load_dataset
    from hagent.agent.execution.tool_runner import set_tool_invoker
    from hagent.agent.graph import run_agent

    apply_condition(condition, scratch_dir)
    dataset = load_dataset(dataset_name)
    env = RealJobEnv(dataset, job_cfg=cfg.get("job") or {}, seed=seed)
    set_tool_invoker(env.invoke)

    prompt = str(cfg.get("prompt") or "Train a model on {dataset}, target {target}.")
    message = prompt.format(dataset=dataset_name, target="target")

    t0 = time.perf_counter()
    error = None
    result: Dict[str, Any] = {}
    try:
        result = asyncio.new_event_loop().run_until_complete(
            run_agent(
                message,
                user_id=f"mx_{condition}_{dataset_name}_{seed}",
                world_model={
                    "user_id": f"mx_{condition}_{dataset_name}_{seed}",
                    "datasets": {
                        dataset_name: {
                            "id": dataset_name,
                            "name": dataset_name,
                            **{k: v for k, v in dataset["meta"].items()},
                            "features": [f"f{i}" for i in range(dataset["n_cols"])]
                            + ["target"],
                            "target": "target",
                        }
                    },
                },
                model_name=model,
            )
        )
    except Exception as exc:  # ghi lỗi vào row, không sập cả ma trận
        error = f"{type(exc).__name__}: {exc}"
    finally:
        set_tool_invoker(None)
    elapsed = time.perf_counter() - t0

    campaign = result.get("campaign") or {}
    variants = campaign.get("variants") or []
    events = result.get("execution_events") or []
    ev_types = [e.get("type") for e in events if isinstance(e, dict)]
    real_scores = [j["best_score"] for j in env.jobs.values()]

    return {
        "key": cell_key(condition, dataset_name, model, seed),
        "condition": condition,
        "dataset": dataset_name,
        "model": model,
        "seed": seed,
        "error": error,
        "route": result.get("route"),
        "response_chars": len(result.get("response") or ""),
        "campaign_status": result.get("campaign_status"),
        "n_variants": len(variants),
        "variant_sources": sorted({v.get("source") for v in variants if isinstance(v, dict)}),
        "extension_rounds": campaign.get("extension_rounds"),
        "best_real_score": max(real_scores) if real_scores else None,
        "n_real_jobs": len(env.jobs),
        "job_seconds_total": round(
            sum(j.get("seconds") or 0 for j in env.jobs.values()), 2
        ),
        "n_outcome_surprise": ev_types.count("campaign_outcome_surprise"),
        "n_extended": ev_types.count("campaign_extended"),
        "cost_metrics": result.get("cost_metrics"),
        "checkpoint_sha": _checkpoint_sha(),
        "git_sha": _git_sha(),
        "wall_seconds": round(elapsed, 2),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Ma trận thí nghiệm agent")
    parser.add_argument(
        "--config", default="benchmarks/agent_matrix_config.yaml"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only", help="Chạy đúng một ô: COND:DATASET:MODEL:SEED"
    )
    args = parser.parse_args()

    cfg_path = BACKEND / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    cells = [
        (c, d, m, s)
        for c in cfg["conditions"]
        for d in cfg["datasets"]
        for m in cfg["models"]
        for s in cfg["seeds"]
    ]
    if args.only:
        c, d, m, s = args.only.split(":")
        cells = [(c, d, m, int(s))]

    out_path = BACKEND / str(cfg.get("output") or "benchmarks/agent_matrix_results.jsonl")
    done: set = set()
    if out_path.is_file():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if not row.get("error"):
                    done.add(row["key"])
            except Exception:
                continue

    todo = [c for c in cells if cell_key(*c) not in done]
    print(
        f"Ma trận: {len(cells)} ô | đã xong {len(cells) - len(todo)} | còn {len(todo)}"
    )
    if args.dry_run:
        for c in todo:
            print("  ", cell_key(*c))
        return 0

    # Cảnh báo checkpoint cho điều kiện B/C
    needs_ckpt = [c for c in todo if c[0] != "A"]
    if needs_ckpt:
        head = BACKEND / "data" / "world_model" / "outcome_head_v2.npz"
        if not head.is_file():
            print(
                f"CẢNH BÁO: điều kiện B/C cần checkpoint {head} — chạy "
                f"scripts/train_outcome_model.py trước, nếu không WM sẽ bất hoạt."
            )

    scratch = Path(tempfile.mkdtemp(prefix="hagent_matrix_"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as fh:
        for i, (c, d, m, s) in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {cell_key(c, d, m, s)} ...", flush=True)
            row = run_cell(c, d, m, s, cfg=cfg, scratch_dir=scratch)
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            fh.flush()
            status = row["error"] or (
                f"route={row['route']} jobs={row['n_real_jobs']} "
                f"best={row['best_real_score']} ext={row['n_extended']} "
                f"{row['wall_seconds']}s"
            )
            print(f"    -> {status}", flush=True)

    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
