"""AgentRuntime adapter cho DatasetAudit và ExperimentSpec approval journey."""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import AsyncIterator, Mapping
from datetime import datetime
from typing import Any

from langgraph.types import Command

from hagent.agent.capabilities.models import CapabilitySnapshot
from hagent.agent.journey.artifact_store import (
    ArtifactMetadataStore,
    InMemoryArtifactMetadataStore,
)
from hagent.agent.journey.experiment_designer import valid_edit_changes
from hagent.agent.journey.graph import (
    compile_audit_graph,
    compile_evaluation_graph,
    compile_experiment_graph,
    compile_prediction_graph,
    compile_training_graph,
    initial_audit_state,
)
from hagent.agent.journey.persistence import (
    JOURNEY_DURABILITY,
    journey_graph_config,
    prepare_journey_checkpointer,
)
from hagent.agent.journey.prediction_operator import (
    PREDICTION_INPUT_INSPECT_CAPABILITY_ID,
    PREDICTION_WRITE_CAPABILITY_ID,
    prediction_input_reference,
    requests_deploy,
    requests_prediction,
)
from hagent.agent.journey.result_critic import TRAINING_RESULTS_CAPABILITY_ID
from hagent.agent.journey.training_operator import TRAINING_START_CAPABILITY_ID
from hagent.agent.runtime import (
    ActionCompleted,
    ApprovalRequired,
    ArtifactProduced,
    CancelRun,
    CheckCompleted,
    EvidenceAdded,
    InMemoryRuntimeEventStore,
    PlanProposed,
    RequestScope,
    ResolveApproval,
    RunCancelled,
    RunCompleted,
    RunFailed,
    RunStarted,
    RuntimeCommand,
    RuntimeCommandConflict,
    RuntimeEvent,
    RuntimeEventStore,
    StartTurn,
    UnsupportedRuntimeCommand,
)
from hagent.agent.runtime.context import GraphRequestContext


def _created_at() -> str:
    return datetime.now().astimezone().isoformat()


def _safe_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _safe_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _safe_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_safe_value(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _resume_payload(command: ResolveApproval) -> dict[str, Any]:
    response = dict(command.response)
    if set(response) - {"decision", "changes"}:
        return {"approval_id": command.approval_id, "decision": "invalid"}
    requested = response.get("decision")
    expected = "approve" if command.approved else "reject"
    if requested is None:
        decision = expected
    elif requested == "edit":
        decision = "edit"
    elif requested in {"approve", "reject"} and requested == expected:
        decision = requested
    else:
        decision = "invalid"
    payload: dict[str, Any] = {
        "approval_id": command.approval_id,
        "decision": decision,
    }
    changes = response.get("changes")
    if decision == "edit":
        if not valid_edit_changes(changes):
            payload["decision"] = "invalid"
        else:
            payload["changes"] = _safe_value(changes)
    return payload


def _preflight_terminal_result(
    message: str,
    *,
    prediction_enabled: bool,
) -> dict[str, Any] | None:
    if requests_deploy(message):
        return {
            "status": "capability_unavailable",
            "error_code": "CAPABILITY_UNAVAILABLE",
            "capability": "automl.deploy@1",
        }
    if not requests_prediction(message):
        return None
    if prediction_input_reference(message) is None:
        return {
            "status": "prediction_failed",
            "error_code": "PREDICTION_INPUT_REQUIRED",
        }
    if not prediction_enabled:
        return {
            "status": "capability_unavailable",
            "error_code": "CAPABILITY_UNAVAILABLE",
            "capability": PREDICTION_WRITE_CAPABILITY_ID,
        }
    return None


class JourneyRuntime:
    """Runtime cho audit, experiment approval và training có idempotency."""

    def __init__(
        self,
        *,
        capability_snapshot: CapabilitySnapshot,
        event_store: RuntimeEventStore | None = None,
        artifact_store: ArtifactMetadataStore | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        self._capability_snapshot = capability_snapshot
        self._event_store = event_store or InMemoryRuntimeEventStore()
        self._artifact_store = artifact_store or InMemoryArtifactMetadataStore()
        self._checkpointer = prepare_journey_checkpointer(checkpointer)
        self._training_enabled = (
            TRAINING_START_CAPABILITY_ID in capability_snapshot.descriptors
        )
        self._evaluation_enabled = (
            self._training_enabled
            and TRAINING_RESULTS_CAPABILITY_ID in capability_snapshot.descriptors
        )
        self._prediction_enabled = (
            self._evaluation_enabled
            and PREDICTION_INPUT_INSPECT_CAPABILITY_ID
            in capability_snapshot.descriptors
            and PREDICTION_WRITE_CAPABILITY_ID in capability_snapshot.descriptors
        )
        if self._checkpointer is None:
            self._graph = compile_audit_graph()
        elif self._prediction_enabled:
            self._graph = compile_prediction_graph(checkpointer=self._checkpointer)
        elif self._evaluation_enabled:
            self._graph = compile_evaluation_graph(checkpointer=self._checkpointer)
        elif self._training_enabled:
            self._graph = compile_training_graph(checkpointer=self._checkpointer)
        else:
            self._graph = compile_experiment_graph(checkpointer=self._checkpointer)

    async def dispatch(
        self,
        command: RuntimeCommand,
        *,
        scope: RequestScope,
    ) -> AsyncIterator[RuntimeEvent]:
        if isinstance(command, StartTurn):
            stream = self._dispatch_start(command, scope=scope)
            try:
                async for event in stream:
                    yield event
            finally:
                await stream.aclose()
            return
        if isinstance(command, ResolveApproval) and self._checkpointer is not None:
            stream = self._dispatch_approval(command, scope=scope)
            try:
                async for event in stream:
                    yield event
            finally:
                await stream.aclose()
            return
        if isinstance(command, CancelRun):
            stream = self._dispatch_cancel(command, scope=scope)
            try:
                async for event in stream:
                    yield event
            finally:
                await stream.aclose()
            return
        raise UnsupportedRuntimeCommand(type(command).__name__)

    def _context(self, scope: RequestScope) -> GraphRequestContext:
        return GraphRequestContext(
            principal_id=scope.principal_id,
            credential=scope.credential,
            trace_id=scope.trace_id,
            deadline=scope.deadline,
            services=scope.services,
            capability_snapshot=self._capability_snapshot,
        )

    def _append(self, record, event: RuntimeEvent) -> RuntimeEvent:
        if isinstance(event, ArtifactProduced):
            self._artifact_store.put(owner_id=record.owner_id, event=event)
        elif isinstance(event, RunCompleted | RunFailed | RunCancelled):
            self._artifact_store.seal_run(
                owner_id=record.owner_id,
                run_id=record.run_id,
                terminal_at=datetime.fromisoformat(event.created_at),
            )
        self._event_store.append(record, event)
        return event

    async def _dispatch_start(
        self,
        command: StartTurn,
        *,
        scope: RequestScope,
    ) -> AsyncIterator[RuntimeEvent]:
        record, is_new = self._event_store.begin(command, owner_id=scope.principal_id)
        if not is_new:
            await record.completed.wait()
            if record.needs_reconciliation:
                raise RuntimeCommandConflict()
            for event in self._event_store.snapshot(record):
                yield event
            return

        command_finished = False
        try:
            yield self._append(
                record,
                RunStarted(
                    run_id=command.run_id,
                    command_id=command.command_id,
                    sequence=1,
                    created_at=_created_at(),
                    metadata={
                        "trace_id": scope.trace_id,
                        "journey": "automl_journey_v1",
                    },
                ),
            )
            yield self._append(
                record,
                PlanProposed(
                    run_id=command.run_id,
                    command_id=command.command_id,
                    sequence=2,
                    created_at=_created_at(),
                    plan={
                        "stages": [
                            "interpret",
                            "dataset_profiler",
                            "contract_checker",
                            "statistical_checker",
                            "policy_checker",
                            "experiment_designer",
                            "approval",
                        ]
                        + (["training_operator"] if self._training_enabled else [])
                        + (
                            [
                                "evaluation_operator",
                                "evaluation_checkers",
                                "release_candidate",
                            ]
                            if self._evaluation_enabled
                            else []
                        )
                        + (
                            [
                                "prediction_input_inspect",
                                "prediction_operator",
                                "prediction_checkers",
                            ]
                            if self._prediction_enabled
                            else []
                        )
                    },
                ),
            )
            preflight_result = _preflight_terminal_result(
                command.message,
                prediction_enabled=self._prediction_enabled,
            )
            if preflight_result is not None:
                completed = self._append(
                    record,
                    RunCompleted(
                        run_id=command.run_id,
                        command_id=command.command_id,
                        sequence=len(record.events) + 1,
                        created_at=_created_at(),
                        result=preflight_result,
                    ),
                )
                command_finished = True
                yield completed
                return
            state = await self._graph.ainvoke(
                initial_audit_state(
                    message=command.message,
                    run_id=command.run_id,
                    capability_snapshot_digest=self._capability_snapshot.digest,
                    training_enabled=self._training_enabled,
                    evaluation_enabled=self._evaluation_enabled,
                    prediction_enabled=self._prediction_enabled,
                ),
                config=journey_graph_config(
                    principal_id=scope.principal_id,
                    run_id=command.run_id,
                ),
                context=self._context(scope),
                durability=(
                    JOURNEY_DURABILITY if self._checkpointer is not None else None
                ),
            )
            if state.get("error_code"):
                failure = self._append(
                    record,
                    RunFailed(
                        run_id=command.run_id,
                        command_id=command.command_id,
                        sequence=len(record.events) + 1,
                        created_at=_created_at(),
                        error_code=str(state["error_code"]),
                    ),
                )
                command_finished = True
                yield failure
                return
            async for event in self._state_events(
                record,
                command_id=command.command_id,
                state=state,
                include_audit=True,
                include_experiment=True,
                include_training=False,
                include_evaluation=False,
                include_prediction=False,
            ):
                yield event
            if state.get("__interrupt__") and state.get("approval"):
                approval_event = self._append(
                    record,
                    self._approval_event(record, command.command_id, state["approval"]),
                )
                command_finished = True
                yield approval_event
                return
            completed = self._append(
                record,
                RunCompleted(
                    run_id=command.run_id,
                    command_id=command.command_id,
                    sequence=len(record.events) + 1,
                    created_at=_created_at(),
                    result=_safe_value(state["result"]),
                ),
            )
            command_finished = True
            yield completed
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except (KeyError, RuntimeError, TypeError, ValueError):
            if not record.is_terminal:
                failure = self._append(
                    record,
                    RunFailed(
                        run_id=command.run_id,
                        command_id=command.command_id,
                        sequence=len(record.events) + 1,
                        created_at=_created_at(),
                        error_code="JOURNEY_RUNTIME_ERROR",
                    ),
                )
                command_finished = True
                yield failure
        finally:
            if command_finished:
                self._event_store.finish(record)
            else:
                self._event_store.abandon(record)

    async def _dispatch_approval(
        self,
        command: ResolveApproval,
        *,
        scope: RequestScope,
    ) -> AsyncIterator[RuntimeEvent]:
        record, claimed, is_new = self._event_store.claim_command(
            command,
            owner_id=scope.principal_id,
        )
        if not is_new:
            await claimed.completed.wait()
            if claimed.needs_reconciliation:
                raise RuntimeCommandConflict()
            for event in self._event_store.command_snapshot(record, claimed):
                yield event
            return

        command_finished = False
        try:
            if not await self._resume_snapshot_matches(
                run_id=command.run_id,
                scope=scope,
            ):
                failure = self._append(
                    record,
                    RunFailed(
                        run_id=command.run_id,
                        command_id=command.command_id,
                        sequence=len(record.events) + 1,
                        created_at=_created_at(),
                        error_code="CAPABILITY_SNAPSHOT_MISMATCH",
                    ),
                )
                command_finished = True
                yield failure
                return
            state = await self._graph.ainvoke(
                Command(resume=_resume_payload(command)),
                config=journey_graph_config(
                    principal_id=scope.principal_id,
                    run_id=command.run_id,
                ),
                context=self._context(scope),
                durability=JOURNEY_DURABILITY,
            )
            if state.get("__interrupt__") and state.get("approval"):
                async for event in self._state_events(
                    record,
                    command_id=command.command_id,
                    state=state,
                    include_audit=False,
                    include_experiment=True,
                    include_training=True,
                    include_evaluation=True,
                    include_prediction=True,
                ):
                    yield event
                approval_event = self._append(
                    record,
                    self._approval_event(record, command.command_id, state["approval"]),
                )
                command_finished = True
                yield approval_event
                return
            if self._training_enabled:
                async for event in self._state_events(
                    record,
                    command_id=command.command_id,
                    state=state,
                    include_audit=False,
                    include_experiment=False,
                    include_training=True,
                    include_evaluation=True,
                    include_prediction=True,
                ):
                    yield event
            completed = self._append(
                record,
                RunCompleted(
                    run_id=command.run_id,
                    command_id=command.command_id,
                    sequence=len(record.events) + 1,
                    created_at=_created_at(),
                    result=_safe_value(state["result"]),
                ),
            )
            command_finished = True
            yield completed
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except (KeyError, RuntimeError, TypeError, ValueError):
            if not record.is_terminal:
                failure = self._append(
                    record,
                    RunFailed(
                        run_id=command.run_id,
                        command_id=command.command_id,
                        sequence=len(record.events) + 1,
                        created_at=_created_at(),
                        error_code="JOURNEY_APPROVAL_ERROR",
                    ),
                )
                command_finished = True
                yield failure
        finally:
            if command_finished:
                self._event_store.finish_command(claimed)
            else:
                self._event_store.abandon_command(claimed)

    async def _dispatch_cancel(
        self,
        command: CancelRun,
        *,
        scope: RequestScope,
    ) -> AsyncIterator[RuntimeEvent]:
        """Hủy run đang chờ duyệt mà không resume graph hoặc phản chiếu reason."""
        record, claimed, is_new = self._event_store.claim_command(
            command,
            owner_id=scope.principal_id,
        )
        if not is_new:
            await claimed.completed.wait()
            if claimed.needs_reconciliation:
                raise RuntimeCommandConflict()
            for event in self._event_store.command_snapshot(record, claimed):
                yield event
            return

        command_finished = False
        try:
            cancelled = self._append(
                record,
                RunCancelled(
                    run_id=command.run_id,
                    command_id=command.command_id,
                    sequence=len(record.events) + 1,
                    created_at=_created_at(),
                    reason="user_requested",
                ),
            )
            command_finished = True
            yield cancelled
        finally:
            if command_finished:
                self._event_store.finish_command(claimed)
            else:
                self._event_store.abandon_command(claimed)

    async def _resume_snapshot_matches(
        self,
        *,
        run_id: str,
        scope: RequestScope,
    ) -> bool:
        config = journey_graph_config(
            principal_id=scope.principal_id,
            run_id=run_id,
        )
        snapshot = await self._graph.aget_state(config)
        values = snapshot.values
        if not isinstance(values, Mapping):
            return False
        persisted_digest = values.get("capability_snapshot_digest")
        persisted_training = values.get("training_enabled")
        persisted_evaluation = values.get("evaluation_enabled", False)
        persisted_prediction = values.get("prediction_enabled", False)
        if persisted_digest is None:
            # Checkpoint cũ chỉ được resume bằng topology experiment cũ.
            return not self._training_enabled
        return (
            persisted_digest == self._capability_snapshot.digest
            and persisted_training is self._training_enabled
            and persisted_evaluation is self._evaluation_enabled
            and persisted_prediction is self._prediction_enabled
        )

    async def _state_events(
        self,
        record,
        *,
        command_id: str,
        state: Mapping[str, Any],
        include_audit: bool,
        include_experiment: bool,
        include_training: bool,
        include_evaluation: bool,
        include_prediction: bool,
    ) -> AsyncIterator[RuntimeEvent]:
        if include_audit and state.get("artifact") is not None:
            artifact = state["artifact"]
            yield self._append(
                record,
                ArtifactProduced(
                    run_id=record.run_id,
                    command_id=command_id,
                    sequence=len(record.events) + 1,
                    created_at=_created_at(),
                    artifact_type=type(artifact).__name__,
                    artifact=_safe_value(artifact),
                ),
            )
            for evidence in artifact.evidence:
                yield self._append(
                    record,
                    EvidenceAdded(
                        run_id=record.run_id,
                        command_id=command_id,
                        sequence=len(record.events) + 1,
                        created_at=_created_at(),
                        evidence_type="dataset_profile",
                        evidence=_safe_value(evidence),
                    ),
                )
            async for event in self._checker_events(
                record,
                command_id=command_id,
                verdicts=state.get("verdicts", ()),
            ):
                yield event
        if include_experiment and state.get("experiment_spec") is not None:
            artifact = state["experiment_spec"]
            yield self._append(
                record,
                ArtifactProduced(
                    run_id=record.run_id,
                    command_id=command_id,
                    sequence=len(record.events) + 1,
                    created_at=_created_at(),
                    artifact_type=type(artifact).__name__,
                    artifact=_safe_value(artifact),
                ),
            )
            async for event in self._checker_events(
                record,
                command_id=command_id,
                verdicts=state.get("experiment_verdicts", ()),
            ):
                yield event
        if include_training and state.get("training_run_set") is not None:
            artifact = state["training_run_set"]
            yield self._append(
                record,
                ArtifactProduced(
                    run_id=record.run_id,
                    command_id=command_id,
                    sequence=len(record.events) + 1,
                    created_at=_created_at(),
                    artifact_type=type(artifact).__name__,
                    artifact=_safe_value(artifact),
                ),
            )
            for evidence in artifact.evidence:
                yield self._append(
                    record,
                    EvidenceAdded(
                        run_id=record.run_id,
                        command_id=command_id,
                        sequence=len(record.events) + 1,
                        created_at=_created_at(),
                        evidence_type="training_dispatch",
                        evidence=_safe_value(evidence),
                    ),
                )
            yield self._append(
                record,
                ActionCompleted(
                    run_id=record.run_id,
                    command_id=command_id,
                    sequence=len(record.events) + 1,
                    created_at=_created_at(),
                    action="automl.training.start",
                    outcome=str(state.get("training_outcome", "unknown")),
                    details={
                        "artifact_id": artifact.artifact_id,
                        "job_ids": list(artifact.job_ids),
                        "reconciliation_status": artifact.reconciliation_status,
                    },
                ),
            )
        if include_evaluation and state.get("evaluation_report") is not None:
            report = state["evaluation_report"]
            yield self._append(
                record,
                ArtifactProduced(
                    run_id=record.run_id,
                    command_id=command_id,
                    sequence=len(record.events) + 1,
                    created_at=_created_at(),
                    artifact_type=type(report).__name__,
                    artifact=_safe_value(report),
                ),
            )
            for evidence in report.evidence:
                yield self._append(
                    record,
                    EvidenceAdded(
                        run_id=record.run_id,
                        command_id=command_id,
                        sequence=len(record.events) + 1,
                        created_at=_created_at(),
                        evidence_type="model_evaluation",
                        evidence=_safe_value(evidence),
                    ),
                )
            async for event in self._checker_events(
                record,
                command_id=command_id,
                verdicts=state.get("evaluation_verdicts", ()),
            ):
                yield event
        if include_evaluation and state.get("release_candidate") is not None:
            release = state["release_candidate"]
            yield self._append(
                record,
                ArtifactProduced(
                    run_id=record.run_id,
                    command_id=command_id,
                    sequence=len(record.events) + 1,
                    created_at=_created_at(),
                    artifact_type=type(release).__name__,
                    artifact=_safe_value(release),
                ),
            )
        if include_prediction and state.get("prediction_action") is not None:
            action = state["prediction_action"]
            yield self._append(
                record,
                ActionCompleted(
                    run_id=record.run_id,
                    command_id=command_id,
                    sequence=len(record.events) + 1,
                    created_at=_created_at(),
                    action="automl.prediction.batch",
                    outcome=str(state.get("prediction_outcome", "unknown")),
                    details={
                        "input_artifact_id": action.get("input_artifact_id"),
                        "release_candidate_id": action.get("release_candidate_id"),
                    },
                ),
            )
        if include_prediction and state.get("prediction_artifact") is not None:
            prediction = state["prediction_artifact"]
            yield self._append(
                record,
                ArtifactProduced(
                    run_id=record.run_id,
                    command_id=command_id,
                    sequence=len(record.events) + 1,
                    created_at=_created_at(),
                    artifact_type=type(prediction).__name__,
                    artifact=_safe_value(prediction),
                ),
            )
            for evidence in prediction.evidence:
                yield self._append(
                    record,
                    EvidenceAdded(
                        run_id=record.run_id,
                        command_id=command_id,
                        sequence=len(record.events) + 1,
                        created_at=_created_at(),
                        evidence_type="batch_prediction",
                        evidence=_safe_value(evidence),
                    ),
                )
            async for event in self._checker_events(
                record,
                command_id=command_id,
                verdicts=state.get("prediction_verdicts", ()),
            ):
                yield event

    async def _checker_events(
        self,
        record,
        *,
        command_id: str,
        verdicts,
    ) -> AsyncIterator[RuntimeEvent]:
        for verdict in verdicts:
            yield self._append(
                record,
                CheckCompleted(
                    run_id=record.run_id,
                    command_id=command_id,
                    sequence=len(record.events) + 1,
                    created_at=_created_at(),
                    checker=verdict.checker,
                    verdict="blocked" if verdict.blocked else "passed",
                    details={
                        "findings": [_safe_value(item) for item in verdict.findings],
                        "computed": _safe_value(verdict.computed),
                    },
                ),
            )

    @staticmethod
    def _approval_event(record, command_id: str, proposal) -> ApprovalRequired:
        safe_proposal = _safe_value(proposal)
        return ApprovalRequired(
            run_id=record.run_id,
            command_id=command_id,
            sequence=len(record.events) + 1,
            created_at=_created_at(),
            approval_id=str(safe_proposal["approval_id"]),
            proposal=safe_proposal,
        )

    async def get_checkpoint_state(
        self,
        *,
        run_id: str,
        scope: RequestScope,
    ) -> dict[str, Any]:
        """Đọc checkpoint theo owner-derived thread; không nhận checkpoint ID ngoài."""
        if self._checkpointer is None:
            return {}
        config = journey_graph_config(
            principal_id=scope.principal_id,
            run_id=run_id,
        )
        snapshot = await self._graph.aget_state(config)
        values = snapshot.values
        return _safe_value(values) if isinstance(values, Mapping) else {}

    async def replay(
        self,
        run_id: str,
        *,
        after_sequence: int,
        scope: RequestScope,
    ) -> AsyncIterator[RuntimeEvent]:
        if after_sequence < 0:
            raise ValueError("after_sequence must not be negative")
        record = self._event_store.find(run_id, owner_id=scope.principal_id)
        for event in self._event_store.snapshot(record, after_sequence=after_sequence):
            yield event


# Alias giữ import cũ ổn định trong drain window.
JourneyAuditRuntime = JourneyRuntime
