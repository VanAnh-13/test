"""Dataset profiler xác định, chỉ đọc dữ liệu qua capability broker."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from hagent.agent.capabilities.broker import InvocationBroker
from hagent.agent.capabilities.models import (
    CapabilityInvocationError,
    CapabilitySnapshot,
)
from hagent.agent.journey.artifacts import DatasetAudit, EvidenceRef
from hagent.agent.journey.request_scope import request_scope_from_context
from hagent.agent.runtime.context import GraphRequestContext

_DATASET_CAPABILITY_ID = "automl.dataset.inspect@1"
_DATASET_PATTERN = re.compile(
    r"(?:dataset(?:\s+id)?|dữ\s+liệu)\s*[:=]?\s*([A-Za-z0-9][A-Za-z0-9._:-]{0,127})",
    re.IGNORECASE,
)
_TARGET_PATTERN = re.compile(
    r"(?:target(?:\s+column)?|cột\s+mục\s+tiêu|nhãn)\s*[:=]?\s*"
    r"([A-Za-z_][A-Za-z0-9_.-]{0,127})",
    re.IGNORECASE,
)


def interpret_audit_goal(message: str) -> dict[str, Any]:
    """Trích dataset/target cho audit request Việt hoặc Anh mà không dùng LLM."""
    dataset_match = _DATASET_PATTERN.search(message)
    target_match = _TARGET_PATTERN.search(message)
    return {
        "operation": "dataset_audit",
        "dataset_id": dataset_match.group(1) if dataset_match else None,
        "target_column": target_match.group(1) if target_match else None,
    }


def _columns_from_output(output: Mapping[str, Any]) -> tuple[str, ...]:
    raw_columns = (
        output.get("columns") or output.get("list_feature") or output.get("features")
    )
    if isinstance(raw_columns, str) or not isinstance(raw_columns, Sequence):
        return ()
    columns: list[str] = []
    for item in raw_columns:
        if isinstance(item, str):
            columns.append(item)
        elif isinstance(item, Mapping):
            name = item.get("name") or item.get("column")
            if isinstance(name, str):
                columns.append(name)
    return tuple(dict.fromkeys(columns))


def _numeric_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, item in value.items():
        if isinstance(item, int | float) and not isinstance(item, bool):
            result[str(key)] = float(item)
    return result


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if not isinstance(value, Sequence):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


async def profile_dataset(
    state: Mapping[str, Any],
    *,
    context: GraphRequestContext,
) -> dict[str, Any]:
    """Gọi đúng một read capability và chuyển output thành DatasetAudit."""
    goal = state.get("goal")
    if not isinstance(goal, Mapping) or not isinstance(goal.get("dataset_id"), str):
        return {
            "error_code": "DATASET_ID_REQUIRED",
            "error_message": "Dataset ID is required for audit",
        }
    if not isinstance(context.capability_snapshot, CapabilitySnapshot):
        return {
            "error_code": "CAPABILITY_SNAPSHOT_REQUIRED",
            "error_message": "Capability snapshot is required",
        }

    broker = InvocationBroker(context.capability_snapshot)
    request_scope = request_scope_from_context(context)
    try:
        capability_result = await broker.invoke(
            _DATASET_CAPABILITY_ID,
            {"dataset_id": goal["dataset_id"]},
            scope=request_scope,
        )
    except CapabilityInvocationError as exc:
        return {
            "error_code": exc.code,
            "error_message": "Dataset audit capability failed",
        }

    output = capability_result.output
    if not isinstance(output, Mapping):
        return {
            "error_code": "INVALID_OUTPUT",
            "error_message": "Dataset audit capability returned invalid output",
        }
    canonical_output = json.dumps(
        output,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    evidence_hash = hashlib.sha256(canonical_output.encode("utf-8")).hexdigest()
    dataset_fingerprint = output.get("dataset_fingerprint")
    if not isinstance(dataset_fingerprint, str) or not re.fullmatch(
        r"[a-fA-F0-9]{64}",
        dataset_fingerprint,
    ):
        dataset_fingerprint = evidence_hash
    target = (
        goal.get("target_column") or output.get("target") or output.get("target_column")
    )
    target_hypothesis = target if isinstance(target, str) and target else None
    columns = _columns_from_output(output)
    artifact_digest = hashlib.sha256(
        (
            f"{context.principal_id}\0{state['run_id']}\0{goal['dataset_id']}\0"
            f"{evidence_hash}"
        ).encode()
    ).hexdigest()
    evidence = EvidenceRef(
        evidence_id=f"evidence-{evidence_hash[:24]}",
        source=f"capability:{_DATASET_CAPABILITY_ID}",
        content_hash=evidence_hash,
        summary="Schema và thống kê dataset từ capability snapshot của run.",
    )
    quality_blockers = list(_string_tuple(output.get("quality_blockers")))
    if not columns:
        quality_blockers.append("schema_columns_missing")
    artifact = DatasetAudit(
        artifact_id=f"audit-{artifact_digest[:24]}",
        owner_id=context.principal_id,
        run_id=str(state["run_id"]),
        status="draft",
        evidence=(evidence,),
        dataset_id=str(goal["dataset_id"]),
        dataset_fingerprint=dataset_fingerprint,
        target_hypothesis=target_hypothesis,
        columns=columns,
        missingness=_numeric_mapping(output.get("missingness")),
        class_balance=_numeric_mapping(output.get("class_balance")),
        leakage_risks=_string_tuple(output.get("leakage_risks")),
        quality_blockers=tuple(dict.fromkeys(quality_blockers)),
    )
    return {"artifact": artifact}
