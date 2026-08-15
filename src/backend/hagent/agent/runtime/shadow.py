"""Shadow runtime quan sát Journey mà không thay đổi response của primary."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import math
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from time import perf_counter

import structlog

from hagent.agent.runtime.contracts import (
    AgentRuntime,
    ApprovalRequired,
    ArtifactProduced,
    CancelRun,
    CheckCompleted,
    EvidenceAdded,
    RequestScope,
    RunCancelled,
    RunCompleted,
    RunFailed,
    RuntimeCommand,
    RuntimeEvent,
    StartTurn,
)

logger = structlog.get_logger(__name__)
_ARTIFACT_TYPES = frozenset(
    {
        "DatasetAudit",
        "EvaluationReport",
        "ExperimentSpec",
        "PredictionArtifact",
        "ReleaseCandidate",
        "TrainingRunSet",
        "response_delta",
    }
)
_EVIDENCE_TYPES = frozenset(
    {
        "batch_prediction",
        "dataset_profile",
        "model_evaluation",
        "surprise",
        "tool_result",
        "training_dispatch",
    }
)
_CHECKER_TYPES = frozenset(
    {"aggregate", "contract", "legacy_plan_executor", "policy", "statistical"}
)
_CHECKER_VERDICTS = frozenset({"blocked", "observed", "passed"})
_COMPLETED_STATUSES = frozenset(
    {
        "approved",
        "blocked",
        "capability_unavailable",
        "completed",
        "evaluation_completed",
        "evaluation_failed",
        "failed",
        "needs_reconciliation",
        "prediction_completed",
        "prediction_failed",
        "prediction_partial",
        "rejected",
        "success",
        "training_completed",
        "training_failed",
    }
)
_CANCEL_REASONS = frozenset(
    {"cancelled", "consumer_cancelled", "shadow_observation_complete", "user_requested"}
)


@dataclass(frozen=True, slots=True)
class RuntimeObservation:
    """Số liệu đã loại raw payload của một nhánh runtime."""

    outcome: str
    artifact_types: tuple[str, ...]
    evidence_types: tuple[str, ...]
    checker_verdicts: tuple[str, ...]
    latency_ms: float
    total_tokens: int | None
    total_cost: float | None
    event_count: int


@dataclass(frozen=True, slots=True)
class ShadowComparisonReport:
    """Kết quả so sánh chỉ chứa label và số liệu an toàn."""

    run_id: str
    primary: RuntimeObservation
    observer: RuntimeObservation
    outcome_match: bool
    artifact_match: bool
    evidence_match: bool
    checker_match: bool
    latency_ratio: float | None
    token_ratio: float | None
    cost_ratio: float | None


ReportSink = Callable[[ShadowComparisonReport], object]


def _allowed_label(
    value: object,
    *,
    allowed: frozenset[str],
    fallback: str,
) -> str:
    return value if isinstance(value, str) and value in allowed else fallback


def _safe_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _ratio(observer: float | None, primary: float | None) -> float | None:
    if observer is None or primary is None or primary <= 0:
        return None
    return round(float(observer) / float(primary), 6)


def _cost_metrics(events: list[RuntimeEvent]) -> tuple[int | None, float | None]:
    completed = next(
        (event for event in reversed(events) if isinstance(event, RunCompleted)),
        None,
    )
    if completed is None:
        return None, None
    raw_metrics = completed.result.get("cost_metrics")
    if not isinstance(raw_metrics, Mapping):
        return None, None
    token_value = raw_metrics.get("total_tokens", raw_metrics.get("tokens"))
    cost_value = raw_metrics.get("total_cost", raw_metrics.get("cost"))
    tokens = _safe_number(token_value)
    return (int(tokens) if tokens is not None else None, _safe_number(cost_value))


def _outcome(events: list[RuntimeEvent], *, observer_failed: bool = False) -> str:
    if observer_failed:
        return "failed:observer_error"
    terminal = next(
        (
            event
            for event in reversed(events)
            if isinstance(event, RunCompleted | RunFailed | RunCancelled)
        ),
        None,
    )
    if isinstance(terminal, RunCompleted):
        status = _allowed_label(
            terminal.result.get("status"),
            allowed=_COMPLETED_STATUSES,
            fallback="completed",
        )
        return f"completed:{status}"
    if isinstance(terminal, RunFailed):
        return "failed:runtime_error"
    if isinstance(terminal, RunCancelled):
        reason = _allowed_label(
            terminal.reason,
            allowed=_CANCEL_REASONS,
            fallback="cancelled",
        )
        return f"cancelled:{reason}"
    return "incomplete:no_terminal"


def _observation(
    events: list[RuntimeEvent],
    *,
    elapsed_seconds: float,
    observer_failed: bool = False,
) -> RuntimeObservation:
    artifact_types = tuple(
        _allowed_label(
            event.artifact_type,
            allowed=_ARTIFACT_TYPES,
            fallback="unknown_artifact",
        )
        for event in events
        if isinstance(event, ArtifactProduced)
    )
    evidence_types = tuple(
        _allowed_label(
            event.evidence_type,
            allowed=_EVIDENCE_TYPES,
            fallback="unknown_evidence",
        )
        for event in events
        if isinstance(event, EvidenceAdded)
    )
    checker_verdicts = tuple(
        f"{_allowed_label(event.checker, allowed=_CHECKER_TYPES, fallback='unknown_checker')}:"
        f"{_allowed_label(event.verdict, allowed=_CHECKER_VERDICTS, fallback='unknown_verdict')}"
        for event in events
        if isinstance(event, CheckCompleted)
    )
    total_tokens, total_cost = _cost_metrics(events)
    return RuntimeObservation(
        outcome=_outcome(events, observer_failed=observer_failed),
        artifact_types=artifact_types,
        evidence_types=evidence_types,
        checker_verdicts=checker_verdicts,
        latency_ms=round(max(elapsed_seconds, 0) * 1000, 3),
        total_tokens=total_tokens,
        total_cost=total_cost,
        event_count=len(events),
    )


def _cancel_command(command: StartTurn) -> CancelRun:
    digest = hashlib.sha256(
        f"{command.run_id}\0{command.command_id}\0shadow-cancel".encode()
    ).hexdigest()[:32]
    return CancelRun(
        run_id=command.run_id,
        command_id=f"shadow-cancel-{digest}",
        reason="shadow_observation_complete",
    )


class ShadowAgentRuntime:
    """Trả primary stream và dùng observer read-only để đo cutover."""

    def __init__(
        self,
        *,
        primary: AgentRuntime,
        observer: AgentRuntime,
        report_sink: ReportSink | None = None,
    ) -> None:
        self._primary = primary
        self._observer = observer
        self._report_sink = report_sink
        self._active_observers: set[asyncio.Task] = set()
        self._closed = False

    async def dispatch(
        self,
        command: RuntimeCommand,
        *,
        scope: RequestScope,
    ) -> AsyncIterator[RuntimeEvent]:
        if self._closed:
            raise RuntimeError("Shadow runtime is closed")
        if not isinstance(command, StartTurn):
            async for event in self._primary.dispatch(command, scope=scope):
                yield event
            return

        observer_task = asyncio.create_task(
            self._observe(command, scope=scope),
            name=f"hagent-shadow-{command.run_id}",
        )
        self._active_observers.add(observer_task)
        observer_task.add_done_callback(self._active_observers.discard)
        primary_events: list[RuntimeEvent] = []
        primary_started_at = perf_counter()
        primary_finished_at = primary_started_at
        try:
            async for event in self._primary.dispatch(command, scope=scope):
                primary_events.append(event)
                primary_finished_at = perf_counter()
                yield event
            observer, observer_failed = await observer_task
        except (asyncio.CancelledError, GeneratorExit):
            observer_task.cancel()
            await asyncio.gather(observer_task, return_exceptions=True)
            raise
        except Exception:
            observer_task.cancel()
            await asyncio.gather(observer_task, return_exceptions=True)
            raise

        primary = _observation(
            primary_events,
            elapsed_seconds=primary_finished_at - primary_started_at,
        )
        report = ShadowComparisonReport(
            run_id=command.run_id,
            primary=primary,
            observer=observer,
            outcome_match=primary.outcome == observer.outcome,
            artifact_match=primary.artifact_types == observer.artifact_types,
            evidence_match=primary.evidence_types == observer.evidence_types,
            checker_match=primary.checker_verdicts == observer.checker_verdicts,
            latency_ratio=_ratio(observer.latency_ms, primary.latency_ms),
            token_ratio=_ratio(observer.total_tokens, primary.total_tokens),
            cost_ratio=_ratio(observer.total_cost, primary.total_cost),
        )
        await self._emit_report(report)

    async def replay(
        self,
        run_id: str,
        *,
        after_sequence: int,
        scope: RequestScope,
    ) -> AsyncIterator[RuntimeEvent]:
        async for event in self._primary.replay(
            run_id,
            after_sequence=after_sequence,
            scope=scope,
        ):
            yield event

    async def _observe(
        self,
        command: StartTurn,
        *,
        scope: RequestScope,
    ) -> tuple[RuntimeObservation, bool]:
        events: list[RuntimeEvent] = []
        started_at = perf_counter()
        failed = False
        try:
            async for event in self._observer.dispatch(command, scope=scope):
                events.append(event)
            has_approval = any(isinstance(event, ApprovalRequired) for event in events)
            has_terminal = any(
                isinstance(event, RunCompleted | RunFailed | RunCancelled)
                for event in events
            )
            if has_approval and not has_terminal:
                async for event in self._observer.dispatch(
                    _cancel_command(command),
                    scope=scope,
                ):
                    events.append(event)
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except Exception as exc:  # noqa: BLE001 - observer không được phá primary
            failed = True
            logger.warning(
                "Shadow observer thất bại",
                extra={"error_type": type(exc).__name__},
            )
        return (
            _observation(
                events,
                elapsed_seconds=perf_counter() - started_at,
                observer_failed=failed,
            ),
            failed,
        )

    async def _emit_report(self, report: ShadowComparisonReport) -> None:
        if self._report_sink is None:
            return
        try:
            result = self._report_sink(report)
            if inspect.isawaitable(result):
                await result
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except Exception as exc:  # noqa: BLE001 - metrics sink không được phá primary
            logger.warning(
                "Không ghi được shadow comparison report",
                extra={"error_type": type(exc).__name__},
            )

    def close(self) -> None:
        """Ngăn dispatch mới và hủy observer còn chạy khi shutdown."""
        self._closed = True
        for task in tuple(self._active_observers):
            task.cancel()

    async def aclose(self) -> None:
        """Hủy và đợi toàn bộ observer trước khi storage bị đóng."""
        self.close()
        if self._active_observers:
            await asyncio.gather(*tuple(self._active_observers), return_exceptions=True)
