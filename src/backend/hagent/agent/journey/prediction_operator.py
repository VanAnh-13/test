"""Prediction operator có schema gate, action digest và provenance xác định."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from hagent.agent.capabilities.broker import InvocationBroker
from hagent.agent.capabilities.models import (
    CapabilityInvocationError,
    freeze_json,
    thaw_json,
)
from hagent.agent.journey.artifacts import EvidenceRef, PredictionArtifact
from hagent.agent.journey.request_scope import request_scope_from_context
from hagent.agent.runtime.context import GraphRequestContext

PREDICTION_INPUT_INSPECT_CAPABILITY_ID = "automl.prediction.input.inspect@1"
PREDICTION_WRITE_CAPABILITY_ID = "automl.prediction.batch@1"

_SAFE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_HASH_PATTERN = re.compile(r"[a-fA-F0-9]{64}")
_RESULT_URI_PATTERN = re.compile(
    r"(?:artifact|minio)://[A-Za-z0-9][A-Za-z0-9._:/-]{0,511}"
)
_PREDICTION_PATTERN = re.compile(
    r"(?:\bpredict(?:ion)?\b|\bdự\s+đoán\b|prediction_input\s*=)",
    re.IGNORECASE,
)
_DEPLOY_PATTERN = re.compile(
    r"(?:\bdeploy\b|\btriển\s+khai\s+(?:model|mô\s+hình)\b)",
    re.IGNORECASE,
)
_INPUT_REFERENCE_PATTERNS = (
    re.compile(r"prediction_input\s*=\s*(\S+)", re.IGNORECASE),
    re.compile(r"input\s+artifact\s+(\S+)", re.IGNORECASE),
)
_DEFINITE_WRITE_FAILURES = frozenset(
    {
        "AUTH_SCOPE_REQUIRED",
        "CAPABILITY_NOT_FOUND",
        "INVALID_INPUT",
        "SCOPE_DENIED",
    }
)
_MAX_ROW_ERRORS = 1_000
_MAX_ROW_ERROR_LENGTH = 512


class _InvalidPredictionData(ValueError):
    pass


def requests_prediction(message: str) -> bool:
    return isinstance(message, str) and _PREDICTION_PATTERN.search(message) is not None


def requests_deploy(message: str) -> bool:
    return isinstance(message, str) and _DEPLOY_PATTERN.search(message) is not None


def prediction_input_reference(message: str) -> str | None:
    """Chỉ nhận artifact ID; path, URI và token tự do đều bị loại."""
    if not requests_prediction(message):
        return None
    for pattern in _INPUT_REFERENCE_PATTERNS:
        match = pattern.search(message)
        if match is None:
            continue
        candidate = match.group(1).rstrip(",.;")
        if _SAFE_ID_PATTERN.fullmatch(candidate):
            return candidate
        return None
    return None


def _canonical_json(value: Any, *, field_name: str) -> tuple[Any, str]:
    try:
        safe_value = thaw_json(freeze_json(value))
        encoded = json.dumps(
            safe_value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError):
        raise _InvalidPredictionData(f"{field_name} không phải JSON hợp lệ") from None
    return safe_value, encoded


def _parse_input_metadata(
    output: Any,
    *,
    expected_input_artifact_id: str,
) -> dict[str, Any]:
    if not isinstance(output, Mapping):
        raise _InvalidPredictionData("input metadata không phải object")
    if output.get("input_artifact_id") != expected_input_artifact_id:
        raise _InvalidPredictionData("input artifact ID không khớp")
    content_hash = output.get("content_hash")
    if not isinstance(content_hash, str) or not _HASH_PATTERN.fullmatch(content_hash):
        raise _InvalidPredictionData("input content hash không hợp lệ")
    schema = output.get("schema")
    if not isinstance(schema, Mapping) or not schema:
        raise _InvalidPredictionData("input schema không hợp lệ")
    safe_schema, canonical_schema = _canonical_json(schema, field_name="input schema")
    row_count = output.get("row_count")
    if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count < 0:
        raise _InvalidPredictionData("row_count không hợp lệ")
    return {
        "content_hash": content_hash.lower(),
        "row_count": row_count,
        "schema": safe_schema,
        "canonical_schema": canonical_schema,
    }


def _action_digest(
    *,
    owner_id: str,
    run_id: str,
    release_candidate_id: str,
    model_version: str,
    input_artifact_id: str,
    input_content_hash: str,
) -> str:
    payload = {
        "input_artifact_id": input_artifact_id,
        "input_content_hash": input_content_hash,
        "model_version": model_version,
        "owner_id": owner_id,
        "release_candidate_id": release_candidate_id,
        "run_id": run_id,
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _row_errors(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or len(value) > _MAX_ROW_ERRORS:
        raise _InvalidPredictionData("row_errors không hợp lệ")
    errors: dict[str, str] = {}
    for row_id, message in value.items():
        key = str(row_id)
        if (
            not key
            or not isinstance(message, str)
            or not message
            or len(message) > _MAX_ROW_ERROR_LENGTH
        ):
            raise _InvalidPredictionData("row_errors không hợp lệ")
        errors[key] = message
    return errors


def _parse_prediction_output(output: Any) -> dict[str, Any]:
    if not isinstance(output, Mapping):
        raise _InvalidPredictionData("prediction output không phải object")
    status = output.get("status")
    if status not in {"success", "replayed"}:
        raise _InvalidPredictionData("prediction status không hợp lệ")
    prediction_id = output.get("prediction_id")
    if not isinstance(prediction_id, str) or not _SAFE_ID_PATTERN.fullmatch(
        prediction_id
    ):
        raise _InvalidPredictionData("prediction_id không hợp lệ")
    result_uri = output.get("result_uri")
    if (
        not isinstance(result_uri, str)
        or not _RESULT_URI_PATTERN.fullmatch(result_uri)
        or any(part in {".", ".."} for part in result_uri.split("/"))
    ):
        raise _InvalidPredictionData("result_uri không hợp lệ")
    return {
        "prediction_id": prediction_id,
        "result_uri": result_uri,
        "row_errors": _row_errors(output.get("row_errors", {})),
        "status": status,
    }


def _prediction_evidence(
    *,
    prediction_id: str,
    result_uri: str,
    input_content_hash: str,
    model_input_hash: str,
) -> EvidenceRef:
    payload = {
        "input_content_hash": input_content_hash,
        "model_input_hash": model_input_hash,
        "prediction_id": prediction_id,
        "result_uri": result_uri,
    }
    content_hash = hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    return EvidenceRef(
        evidence_id=f"prediction-{content_hash[:24]}",
        source=f"capability:{PREDICTION_WRITE_CAPABILITY_ID}",
        content_hash=content_hash,
        summary=f"Dự đoán {prediction_id} được lưu tại artifact URI ổn định.",
    )


async def run_prediction(
    state: Mapping[str, Any],
    *,
    context: GraphRequestContext,
) -> dict[str, Any]:
    """Inspect schema trước write; không retry khi outcome write chưa xác định."""
    input_artifact_id = prediction_input_reference(state["message"])
    if input_artifact_id is None:
        return {
            "prediction_error_code": "PREDICTION_INPUT_REQUIRED",
            "result": {
                "status": "prediction_failed",
                "error_code": "PREDICTION_INPUT_REQUIRED",
            },
        }
    release = state["release_candidate"]
    if release.readiness_verdict != "ready":
        return {
            "prediction_error_code": "RELEASE_CANDIDATE_NOT_READY",
            "result": {
                "status": "prediction_failed",
                "error_code": "RELEASE_CANDIDATE_NOT_READY",
            },
        }
    broker = InvocationBroker(context.capability_snapshot, max_cache_entries=0)
    scope = request_scope_from_context(context)
    try:
        inspected = await broker.invoke(
            PREDICTION_INPUT_INSPECT_CAPABILITY_ID,
            {"input_artifact_id": input_artifact_id},
            scope=scope,
        )
        metadata = _parse_input_metadata(
            inspected.output,
            expected_input_artifact_id=input_artifact_id,
        )
        _, release_schema = _canonical_json(
            release.input_schema,
            field_name="release input schema",
        )
    except CapabilityInvocationError as exc:
        return {
            "prediction_error_code": exc.code,
            "result": {"status": "prediction_failed", "error_code": exc.code},
        }
    except _InvalidPredictionData:
        return {
            "prediction_error_code": "INVALID_PREDICTION_INPUT_METADATA",
            "result": {
                "status": "prediction_failed",
                "error_code": "INVALID_PREDICTION_INPUT_METADATA",
            },
        }
    if metadata["canonical_schema"] != release_schema:
        return {
            "prediction_error_code": "PREDICTION_SCHEMA_MISMATCH",
            "result": {
                "status": "prediction_failed",
                "error_code": "PREDICTION_SCHEMA_MISMATCH",
            },
        }

    digest = _action_digest(
        owner_id=context.principal_id,
        run_id=state["run_id"],
        release_candidate_id=release.artifact_id,
        model_version=release.model_version,
        input_artifact_id=input_artifact_id,
        input_content_hash=metadata["content_hash"],
    )
    action = {
        "input_artifact_id": input_artifact_id,
        "release_candidate_id": release.artifact_id,
    }
    try:
        invoked = await broker.invoke(
            PREDICTION_WRITE_CAPABILITY_ID,
            {
                "action_digest": digest,
                "input_artifact_id": input_artifact_id,
                "input_content_hash": metadata["content_hash"],
                "model_version": release.model_version,
                "release_candidate_id": release.artifact_id,
            },
            scope=scope,
        )
    except CapabilityInvocationError as exc:
        uncertain = exc.code not in _DEFINITE_WRITE_FAILURES
        return {
            "prediction_action": action,
            "prediction_error_code": exc.code,
            "prediction_outcome": ("needs_reconciliation" if uncertain else "rejected"),
            "result": {
                "status": (
                    "prediction_needs_reconciliation"
                    if uncertain
                    else "prediction_failed"
                ),
                "error_code": exc.code,
            },
        }
    try:
        parsed = _parse_prediction_output(invoked.output)
    except _InvalidPredictionData:
        return {
            "prediction_action": action,
            "prediction_error_code": "INVALID_PREDICTION_OUTPUT",
            "prediction_outcome": "needs_reconciliation",
            "result": {
                "status": "prediction_needs_reconciliation",
                "error_code": "INVALID_PREDICTION_OUTPUT",
            },
        }

    model_input_hash = hashlib.sha256(
        f"{release.model_version}\0{metadata['content_hash']}".encode()
    ).hexdigest()
    evidence = _prediction_evidence(
        prediction_id=parsed["prediction_id"],
        result_uri=parsed["result_uri"],
        input_content_hash=metadata["content_hash"],
        model_input_hash=model_input_hash,
    )
    artifact = PredictionArtifact(
        owner_id=context.principal_id,
        run_id=state["run_id"],
        status="draft",
        evidence=(evidence,),
        lineage=(release.artifact_id,),
        release_candidate_id=release.artifact_id,
        model_input_hash=model_input_hash,
        result_uri=parsed["result_uri"],
        row_errors=parsed["row_errors"],
        provenance={
            "capability_id": PREDICTION_WRITE_CAPABILITY_ID,
            "input_artifact_id": input_artifact_id,
            "input_content_hash": metadata["content_hash"],
            "model_version": release.model_version,
            "prediction_id": parsed["prediction_id"],
            "provider_id": invoked.provider_id,
            "row_count": metadata["row_count"],
        },
    )
    return {
        "prediction_action": action,
        "prediction_artifact": artifact,
        "prediction_outcome": (
            "replayed" if parsed["status"] == "replayed" else "completed"
        ),
        "prediction_verdicts": (),
    }
