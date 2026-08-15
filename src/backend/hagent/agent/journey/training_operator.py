"""Training operator có idempotency và reconciliation cho Journey v1."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hagent.agent.capabilities.broker import InvocationBroker
from hagent.agent.capabilities.models import CapabilityInvocationError
from hagent.agent.journey.artifacts import EvidenceRef, TrainingRunSet
from hagent.agent.journey.canonical import canonical_mapping_hash
from hagent.agent.journey.request_scope import request_scope_from_context
from hagent.agent.runtime import RequestScope
from hagent.agent.runtime.context import GraphRequestContext

TRAINING_START_CAPABILITY_ID = "automl.training.start@1"
TRAINING_LOOKUP_CAPABILITY_ID = "automl.training.lookup@1"

_DEFAULT_TRAINING_TIME_LIMIT_SECONDS = 300
_RECONCILABLE_CODES = frozenset({"INVALID_OUTPUT", "PROVIDER_FAILURE", "TIMEOUT"})


def _training_arguments(state: Mapping[str, Any], context: GraphRequestContext):
    audit = state["artifact"]
    spec = state["experiment_spec"]
    raw_time_limit = context.services.get(
        "training_time_limit_seconds",
        _DEFAULT_TRAINING_TIME_LIMIT_SECONDS,
    )
    time_limit = (
        raw_time_limit
        if isinstance(raw_time_limit, int)
        and not isinstance(raw_time_limit, bool)
        and raw_time_limit > 0
        else _DEFAULT_TRAINING_TIME_LIMIT_SECONDS
    )
    config = {
        "dataset_id": audit.dataset_id,
        "problem_type": spec.problem_type,
        "target_column": spec.target_column,
        "metric": spec.metric,
        "models": list(spec.model_families),
        "time_limit": time_limit,
        "list_feature": [
            column for column in audit.columns if column != spec.target_column
        ],
    }
    config_hash = canonical_mapping_hash(config)
    action_digest = canonical_mapping_hash(
        {
            "config_hash": config_hash,
            "experiment_spec_id": spec.artifact_id,
            "owner_id": context.principal_id,
            "run_id": state["run_id"],
        }
    )
    return {**config, "idempotency_key": action_digest}, config_hash, action_digest


def _safe_cost(value: Any) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool) and value >= 0:
        return float(value)
    return 0.0


def _training_evidence(
    *,
    source: str,
    action_digest: str,
    job_id: str | None,
    dispatch_status: str,
) -> EvidenceRef:
    evidence_payload = {
        "action_digest": action_digest,
        "dispatch_status": dispatch_status,
        "job_id": job_id,
    }
    content_hash = canonical_mapping_hash(evidence_payload)
    return EvidenceRef(
        evidence_id=f"training-{content_hash[:24]}",
        source=source,
        content_hash=content_hash,
        summary=(
            f"Training dispatch status={dispatch_status}; job_id={job_id or 'unknown'}."
        ),
    )


def _artifact(
    state: Mapping[str, Any],
    context: GraphRequestContext,
    *,
    config_hash: str,
    action_digest: str,
    job_id: str | None,
    dispatch_status: str,
    reconciliation_status: str,
    cost: float,
    source: str,
) -> TrainingRunSet:
    spec = state["experiment_spec"]
    return TrainingRunSet(
        owner_id=context.principal_id,
        run_id=state["run_id"],
        status=(
            "accepted"
            if reconciliation_status in {"not_required", "reconciled"}
            else "draft"
        ),
        evidence=(
            _training_evidence(
                source=source,
                action_digest=action_digest,
                job_id=job_id,
                dispatch_status=dispatch_status,
            ),
        ),
        lineage=(spec.artifact_id,),
        experiment_spec_id=spec.artifact_id,
        config_hash=config_hash,
        idempotency_key=action_digest,
        job_ids=(job_id,) if job_id else (),
        job_statuses={job_id: dispatch_status} if job_id else {},
        cost=cost,
        reconciliation_status=reconciliation_status,
    )


async def _lookup_after_uncertain_submit(
    broker: InvocationBroker,
    *,
    action_digest: str,
    scope: RequestScope,
) -> Mapping[str, Any] | None:
    try:
        result = await broker.invoke(
            TRAINING_LOOKUP_CAPABILITY_ID,
            {"idempotency_key": action_digest},
            scope=scope,
        )
    except CapabilityInvocationError:
        return None
    return result.output if isinstance(result.output, Mapping) else None


def _resolved_job(output: Mapping[str, Any]) -> tuple[str, str, float] | None:
    if output.get("found") is False:
        return None
    job_id = output.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        return None
    dispatch_status = str(output.get("dispatch_status") or "sent")
    if dispatch_status == "needs_reconciliation":
        return None
    return job_id, dispatch_status, _safe_cost(output.get("cost"))


async def dispatch_training(
    state: Mapping[str, Any],
    *,
    context: GraphRequestContext,
) -> dict[str, Any]:
    """Submit đúng một lần; khi outcome mơ hồ chỉ lookup bằng cùng digest."""
    if context.capability_snapshot is None:
        return {
            "training_error_code": "CAPABILITY_UNAVAILABLE",
            "result": {"status": "capability_unavailable"},
        }
    arguments, config_hash, action_digest = _training_arguments(state, context)
    broker = InvocationBroker(context.capability_snapshot)
    scope = request_scope_from_context(context)

    try:
        start_result = await broker.invoke(
            TRAINING_START_CAPABILITY_ID,
            arguments,
            scope=scope,
        )
        output = start_result.output
        if not isinstance(output, Mapping):
            raise CapabilityInvocationError(
                "INVALID_OUTPUT",
                "Training capability returned an invalid output",
                capability_id=TRAINING_START_CAPABILITY_ID,
            )
        job_id = output.get("job_id")
        status = str(output.get("status") or "")
        dispatch_status = str(output.get("dispatch_status") or "sent")
        if (
            status == "success"
            and isinstance(job_id, str)
            and job_id
            and dispatch_status != "needs_reconciliation"
        ):
            artifact = _artifact(
                state,
                context,
                config_hash=config_hash,
                action_digest=action_digest,
                job_id=job_id,
                dispatch_status=dispatch_status,
                reconciliation_status="not_required",
                cost=_safe_cost(output.get("cost")),
                source=f"capability:{TRAINING_START_CAPABILITY_ID}",
            )
            return {
                "training_run_set": artifact,
                "training_outcome": "replayed"
                if output.get("replayed")
                else "submitted",
                "result": {
                    "status": "training_dispatched",
                    "artifact_id": artifact.artifact_id,
                    "job_ids": list(artifact.job_ids),
                },
            }
    except CapabilityInvocationError as exc:
        if exc.code not in _RECONCILABLE_CODES:
            return {
                "training_error_code": exc.code,
                "result": {"status": "training_failed", "error_code": exc.code},
            }

    lookup = await _lookup_after_uncertain_submit(
        broker,
        action_digest=action_digest,
        scope=scope,
    )
    resolved = _resolved_job(lookup) if lookup is not None else None
    if resolved is not None:
        job_id, dispatch_status, cost = resolved
        artifact = _artifact(
            state,
            context,
            config_hash=config_hash,
            action_digest=action_digest,
            job_id=job_id,
            dispatch_status=dispatch_status,
            reconciliation_status="reconciled",
            cost=cost,
            source=f"capability:{TRAINING_LOOKUP_CAPABILITY_ID}",
        )
        return {
            "training_run_set": artifact,
            "training_outcome": "reconciled",
            "result": {
                "status": "training_dispatched",
                "artifact_id": artifact.artifact_id,
                "job_ids": list(artifact.job_ids),
            },
        }

    artifact = _artifact(
        state,
        context,
        config_hash=config_hash,
        action_digest=action_digest,
        job_id=None,
        dispatch_status="unknown",
        reconciliation_status="needs_reconciliation",
        cost=0.0,
        source=f"capability:{TRAINING_LOOKUP_CAPABILITY_ID}",
    )
    return {
        "training_run_set": artifact,
        "training_outcome": "needs_reconciliation",
        "result": {
            "status": "needs_reconciliation",
            "artifact_id": artifact.artifact_id,
            "job_ids": [],
        },
    }
