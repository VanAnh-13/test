"""Ranh giới runtime công khai và adapter tương thích graph cũ của HAgent.

Module này không sở hữu kiểu LangGraph. Graph hiện tại vẫn là chi tiết triển khai
phía sau :class:`LegacyGraphRuntime` cho tới khi các journey slice sau thay thế nó.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import re
import threading
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Literal, Mapping, Protocol, Union

logger = logging.getLogger(__name__)

_DEFAULT_MAX_REPLAY_RUNS = 256
_DEFAULT_MAX_EVENTS_PER_RUN = 2048
_DEFAULT_MAX_EVENT_BYTES_PER_RUN = 2 * 1024 * 1024
_DEFAULT_TOMBSTONE_BYTES = 64 * 1024
_TERMINAL_BYTE_RESERVE = 1024
_MAX_RUNTIME_ID_LENGTH = 128
_MAX_PRINCIPAL_ID_LENGTH = 256
_RUNTIME_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_SENSITIVE_KEY_ALIASES = frozenset(
    {
        "accesstoken",
        "apikey",
        "authorization",
        "bearer",
        "clientsecret",
        "cookie",
        "credential",
        "jwt",
        "otp",
        "password",
        "privatekey",
        "refreshtoken",
        "secret",
        "token",
    }
)


def _new_id() -> str:
    return uuid.uuid4().hex


def _validate_runtime_id(name: str, value: str) -> None:
    if not isinstance(value, str) or not _RUNTIME_ID_PATTERN.fullmatch(value):
        raise ValueError(
            f"{name} must be 1-{_MAX_RUNTIME_ID_LENGTH} safe identifier characters"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RequestScope:
    """Danh tính và tài nguyên tạm thời được truyền bên ngoài lệnh của model."""

    principal_id: str
    credential: str | None = field(default=None, repr=False, compare=False)
    trace_id: str | None = None
    deadline: datetime | None = None
    services: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not self.principal_id.strip():
            raise ValueError("RequestScope.principal_id must not be empty")
        if len(self.principal_id) > _MAX_PRINCIPAL_ID_LENGTH:
            raise ValueError("RequestScope.principal_id is too long")
        if self.trace_id is not None:
            _validate_runtime_id("RequestScope.trace_id", self.trace_id)
        if self.deadline is not None and self.deadline.tzinfo is None:
            raise ValueError("RequestScope.deadline must be timezone-aware")


@dataclass(frozen=True, slots=True, kw_only=True)
class StartTurn:
    message: str
    command_id: str = field(default_factory=_new_id)
    run_id: str = field(default_factory=_new_id)
    history: tuple[Mapping[str, str], ...] = ()
    world_model: Mapping[str, Any] | None = None
    memory_context: str | None = None
    model_name: str | None = None

    def __post_init__(self) -> None:
        _validate_runtime_id("StartTurn.command_id", self.command_id)
        _validate_runtime_id("StartTurn.run_id", self.run_id)
        if self.model_name is not None and len(self.model_name) > _MAX_RUNTIME_ID_LENGTH:
            raise ValueError("StartTurn.model_name is too long")


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolveApproval:
    run_id: str
    approval_id: str
    approved: bool
    command_id: str = field(default_factory=_new_id)
    response: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_runtime_id("ResolveApproval.command_id", self.command_id)
        _validate_runtime_id("ResolveApproval.run_id", self.run_id)
        _validate_runtime_id("ResolveApproval.approval_id", self.approval_id)


@dataclass(frozen=True, slots=True, kw_only=True)
class CancelRun:
    run_id: str
    command_id: str = field(default_factory=_new_id)
    reason: str = "user_requested"

    def __post_init__(self) -> None:
        _validate_runtime_id("CancelRun.command_id", self.command_id)
        _validate_runtime_id("CancelRun.run_id", self.run_id)


RuntimeCommand = Union[StartTurn, ResolveApproval, CancelRun]


@dataclass(frozen=True, slots=True, kw_only=True)
class _RuntimeEventBase:
    run_id: str
    command_id: str
    sequence: int
    created_at: str
    compatibility_event: Mapping[str, Any] | None = field(
        default=None,
        repr=False,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class RunStarted(_RuntimeEventBase):
    type: Literal["run_started"] = field(default="run_started", init=False)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class PlanProposed(_RuntimeEventBase):
    type: Literal["plan_proposed"] = field(default="plan_proposed", init=False)
    plan: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactProduced(_RuntimeEventBase):
    type: Literal["artifact_produced"] = field(
        default="artifact_produced",
        init=False,
    )
    artifact_type: str
    artifact: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckCompleted(_RuntimeEventBase):
    type: Literal["check_completed"] = field(default="check_completed", init=False)
    checker: str
    verdict: str
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class ApprovalRequired(_RuntimeEventBase):
    type: Literal["approval_required"] = field(
        default="approval_required",
        init=False,
    )
    approval_id: str
    proposal: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class ActionCompleted(_RuntimeEventBase):
    type: Literal["action_completed"] = field(default="action_completed", init=False)
    action: str
    outcome: str
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class EvidenceAdded(_RuntimeEventBase):
    type: Literal["evidence_added"] = field(default="evidence_added", init=False)
    evidence_type: str
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class RunCompleted(_RuntimeEventBase):
    type: Literal["run_completed"] = field(default="run_completed", init=False)
    result: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class RunFailed(_RuntimeEventBase):
    type: Literal["run_failed"] = field(default="run_failed", init=False)
    error_code: str
    message: str = "Agent runtime failed"


@dataclass(frozen=True, slots=True, kw_only=True)
class RunCancelled(_RuntimeEventBase):
    type: Literal["run_cancelled"] = field(default="run_cancelled", init=False)
    reason: str = "cancelled"


RuntimeEvent = Union[
    RunStarted,
    PlanProposed,
    ArtifactProduced,
    CheckCompleted,
    ApprovalRequired,
    ActionCompleted,
    EvidenceAdded,
    RunCompleted,
    RunFailed,
    RunCancelled,
]
TerminalRuntimeEvent = Union[RunCompleted, RunFailed, RunCancelled]


class AgentRuntime(Protocol):
    def dispatch(
        self,
        command: RuntimeCommand,
        *,
        scope: RequestScope,
    ) -> AsyncIterator[RuntimeEvent]: ...

    def replay(
        self,
        run_id: str,
        *,
        after_sequence: int,
        scope: RequestScope,
    ) -> AsyncIterator[RuntimeEvent]: ...


class AgentRuntimeError(RuntimeError):
    """Lỗi runtime công khai an toàn với mã ổn định để máy đọc."""

    def __init__(self, code: str, message: str = "Agent runtime failed"):
        self.code = code
        super().__init__(message)


class RuntimeCommandConflict(AgentRuntimeError):
    def __init__(self) -> None:
        super().__init__("COMMAND_ID_CONFLICT", "Runtime command conflicts with prior use")


class RuntimeCommandExpired(AgentRuntimeError):
    def __init__(self) -> None:
        super().__init__(
            "COMMAND_REPLAY_EXPIRED",
            "Runtime command replay window expired",
        )


class RuntimeRunNotFound(AgentRuntimeError):
    def __init__(self) -> None:
        super().__init__("RUN_NOT_FOUND", "Runtime run was not found")


class RuntimeAccessDenied(AgentRuntimeError):
    def __init__(self) -> None:
        super().__init__("RUN_ACCESS_DENIED", "Runtime run access denied")


class UnsupportedRuntimeCommand(AgentRuntimeError):
    def __init__(self, command_type: str) -> None:
        super().__init__(
            "COMMAND_UNSUPPORTED",
            f"{command_type} is not available in the compatibility runtime",
        )


class RuntimeCapacityExceeded(AgentRuntimeError):
    def __init__(self) -> None:
        super().__init__("RUNTIME_CAPACITY_EXCEEDED", "Runtime capacity exceeded")


class RuntimeEventLimitExceeded(AgentRuntimeError):
    def __init__(self) -> None:
        super().__init__("EVENT_LIMIT_EXCEEDED", "Runtime event limit exceeded")


class _CommandTombstones:
    """Bloom filter hữu hạn chỉ từ chối, không replay lại lệnh cũ."""

    _HASH_COUNT = 4

    def __init__(self, size_bytes: int):
        if size_bytes < 1:
            raise ValueError("tombstone size must be positive")
        self._bits = bytearray(size_bytes)
        self._bit_count = size_bytes * 8

    def add(self, value: str) -> None:
        for index in self._indexes(value):
            byte_index, bit_index = divmod(index, 8)
            self._bits[byte_index] |= 1 << bit_index

    def might_contain(self, value: str) -> bool:
        return all(
            self._bits[byte_index] & (1 << bit_index)
            for byte_index, bit_index in (
                divmod(index, 8) for index in self._indexes(value)
            )
        )

    def _indexes(self, value: str) -> tuple[int, ...]:
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        return tuple(
            int.from_bytes(digest[offset : offset + 4], "big") % self._bit_count
            for offset in range(0, self._HASH_COUNT * 4, 4)
        )


@dataclass(slots=True)
class _RunRecord:
    owner_id: str
    run_id: str
    command_id: str
    fingerprint: str
    events: list[RuntimeEvent] = field(default_factory=list)
    stored_bytes: int = 0
    completed: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def is_terminal(self) -> bool:
        return bool(self.events) and isinstance(
            self.events[-1],
            (RunCompleted, RunFailed, RunCancelled),
        )


class InMemoryRuntimeEventStore:
    """Ledger tương thích hữu hạn; lưu bền vững thuộc journey slice sau."""

    def __init__(
        self,
        *,
        max_runs: int = _DEFAULT_MAX_REPLAY_RUNS,
        max_events_per_run: int = _DEFAULT_MAX_EVENTS_PER_RUN,
        max_event_bytes_per_run: int = _DEFAULT_MAX_EVENT_BYTES_PER_RUN,
        tombstone_bytes: int = _DEFAULT_TOMBSTONE_BYTES,
    ):
        if max_runs < 1:
            raise ValueError("max_runs must be positive")
        if max_events_per_run < 2:
            raise ValueError("max_events_per_run must allow start and terminal events")
        if max_event_bytes_per_run < _TERMINAL_BYTE_RESERVE * 2:
            raise ValueError("max_event_bytes_per_run is too small")
        self._max_runs = max_runs
        self._max_events_per_run = max_events_per_run
        self._max_event_bytes_per_run = max_event_bytes_per_run
        self._runs: OrderedDict[str, _RunRecord] = OrderedDict()
        self._command_runs: dict[tuple[str, str], str] = {}
        self._tombstones = _CommandTombstones(tombstone_bytes)
        self._run_tombstones = _CommandTombstones(tombstone_bytes)
        self._lock = threading.RLock()

    def begin(
        self,
        command: StartTurn,
        *,
        owner_id: str,
    ) -> tuple[_RunRecord, bool]:
        fingerprint = _command_fingerprint(command)
        command_key = (owner_id, command.command_id)
        with self._lock:
            prior_run_id = self._command_runs.get(command_key)
            if prior_run_id is not None:
                record = self._runs.get(prior_run_id)
                if record is None or record.fingerprint != fingerprint:
                    raise RuntimeCommandConflict()
                self._runs.move_to_end(prior_run_id)
                return record, False

            existing = self._runs.get(command.run_id)
            if existing is not None:
                if existing.owner_id != owner_id:
                    raise RuntimeAccessDenied()
                raise RuntimeCommandConflict()

            if self._tombstones.might_contain(
                _command_tombstone_key(owner_id, command.command_id)
            ):
                raise RuntimeCommandExpired()
            if self._run_tombstones.might_contain(
                _run_tombstone_key(command.run_id)
            ):
                raise RuntimeCommandExpired()

            self._evict_completed_runs()
            if len(self._runs) >= self._max_runs:
                raise RuntimeCapacityExceeded()
            record = _RunRecord(
                owner_id=owner_id,
                run_id=command.run_id,
                command_id=command.command_id,
                fingerprint=fingerprint,
            )
            self._runs[command.run_id] = record
            self._command_runs[command_key] = command.run_id
            return record, True

    def append(self, record: _RunRecord, event: RuntimeEvent) -> None:
        with self._lock:
            if record.is_terminal:
                raise RuntimeError("Runtime run already has a terminal event")
            expected_sequence = len(record.events) + 1
            if event.sequence != expected_sequence:
                raise RuntimeError("Runtime event sequence is not monotonic")
            is_terminal = isinstance(event, (RunCompleted, RunFailed, RunCancelled))
            stored_event = copy.deepcopy(event)
            event_bytes = _event_storage_size(stored_event)
            if not is_terminal and len(record.events) >= self._max_events_per_run - 1:
                raise RuntimeEventLimitExceeded()
            if len(record.events) >= self._max_events_per_run:
                raise RuntimeEventLimitExceeded()
            byte_limit = self._max_event_bytes_per_run
            if not is_terminal:
                byte_limit -= _TERMINAL_BYTE_RESERVE
            if record.stored_bytes + event_bytes > byte_limit:
                raise RuntimeEventLimitExceeded()
            record.events.append(stored_event)
            record.stored_bytes += event_bytes

    def finish(self, record: _RunRecord) -> None:
        record.completed.set()

    def snapshot(self, record: _RunRecord, *, after_sequence: int = 0) -> list[RuntimeEvent]:
        with self._lock:
            return [
                copy.deepcopy(event)
                for event in record.events
                if event.sequence > after_sequence
            ]

    def find(self, run_id: str, *, owner_id: str) -> _RunRecord:
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                raise RuntimeRunNotFound()
            if record.owner_id != owner_id:
                raise RuntimeAccessDenied()
            self._runs.move_to_end(run_id)
            return record

    def _evict_completed_runs(self) -> None:
        while len(self._runs) >= self._max_runs:
            completed_id = next(
                (run_id for run_id, record in self._runs.items() if record.is_terminal),
                None,
            )
            if completed_id is None:
                return
            record = self._runs.pop(completed_id)
            self._command_runs.pop((record.owner_id, record.command_id), None)
            self._tombstones.add(
                _command_tombstone_key(record.owner_id, record.command_id)
            )
            self._run_tombstones.add(_run_tombstone_key(record.run_id))


LegacyEventSource = Callable[
    [StartTurn, RequestScope],
    AsyncIterator[Mapping[str, Any]],
]


class _DeadlineExceeded(Exception):
    pass


async def _default_legacy_event_source(
    command: StartTurn,
    scope: RequestScope,
) -> AsyncIterator[Mapping[str, Any]]:
    from hagent.agent.graph import _stream_legacy_graph_events

    graph_events = _stream_legacy_graph_events(
        command.message,
        user_id=scope.principal_id,
        user_token=scope.credential,
        history=[dict(item) for item in command.history],
        world_model=dict(command.world_model) if command.world_model is not None else None,
        memory_context=command.memory_context,
        mongo_client=scope.services.get("mongo_client"),
        db_name=scope.services.get("db_name"),
        world_store=scope.services.get("world_store"),
        wm_service=scope.services.get("wm_service"),
        model_name=command.model_name,
    )
    try:
        async for event in graph_events:
            yield event
    finally:
        await graph_events.aclose()


class LegacyGraphRuntime:
    """Chuyển event stream của graph cũ sang contract runtime công khai."""

    def __init__(
        self,
        *,
        event_source: LegacyEventSource = _default_legacy_event_source,
        event_store: InMemoryRuntimeEventStore | None = None,
    ):
        self._event_source = event_source
        self._event_store = event_store or InMemoryRuntimeEventStore()

    async def dispatch(
        self,
        command: RuntimeCommand,
        *,
        scope: RequestScope,
    ) -> AsyncIterator[RuntimeEvent]:
        if not isinstance(command, StartTurn):
            raise UnsupportedRuntimeCommand(type(command).__name__)

        record, is_new = self._event_store.begin(command, owner_id=scope.principal_id)
        if not is_new:
            await record.completed.wait()
            for event in self._event_store.snapshot(record):
                yield event
            return

        terminal_emitted = False
        legacy_events: AsyncIterator[Mapping[str, Any]] | None = None
        try:
            started = RunStarted(
                **_event_identity(record),
                sequence=1,
                metadata={
                    "trace_id": scope.trace_id,
                    "model_name": command.model_name,
                },
            )
            self._event_store.append(record, started)
            yield started

            if scope.deadline is not None and datetime.now(timezone.utc) >= scope.deadline:
                failure = self._failure_event(record, "DEADLINE_EXCEEDED")
                self._event_store.append(record, failure)
                terminal_emitted = True
                yield failure
                return

            legacy_events = self._event_source(command, scope)
            while True:
                try:
                    if scope.deadline is None:
                        legacy_event = await anext(legacy_events)
                    else:
                        remaining = (
                            scope.deadline - datetime.now(timezone.utc)
                        ).total_seconds()
                        if remaining <= 0:
                            raise _DeadlineExceeded()
                        deadline_timeout = asyncio.timeout(remaining)
                        try:
                            async with deadline_timeout:
                                legacy_event = await anext(legacy_events)
                        except TimeoutError:
                            if deadline_timeout.expired():
                                raise _DeadlineExceeded() from None
                            raise
                except StopAsyncIteration:
                    break

                runtime_event = self._adapt_legacy_event(
                    record,
                    legacy_event,
                    credential=scope.credential,
                )
                if runtime_event is None:
                    continue
                self._event_store.append(record, runtime_event)
                if isinstance(runtime_event, (RunCompleted, RunFailed, RunCancelled)):
                    terminal_emitted = True
                yield runtime_event
                if terminal_emitted:
                    break

            if not terminal_emitted:
                failure = self._failure_event(record, "LEGACY_STREAM_INCOMPLETE")
                self._event_store.append(record, failure)
                terminal_emitted = True
                yield failure
        except (asyncio.CancelledError, GeneratorExit):
            if not terminal_emitted and not record.is_terminal:
                cancelled = RunCancelled(
                    **_event_identity(record),
                    sequence=len(record.events) + 1,
                    reason="consumer_cancelled",
                )
                self._event_store.append(record, cancelled)
            raise
        except _DeadlineExceeded:
            if not terminal_emitted and not record.is_terminal:
                failure = self._failure_event(record, "DEADLINE_EXCEEDED")
                self._event_store.append(record, failure)
                yield failure
        except RuntimeEventLimitExceeded:
            if not terminal_emitted and not record.is_terminal:
                failure = self._failure_event(record, "EVENT_LIMIT_EXCEEDED")
                self._event_store.append(record, failure)
                yield failure
        except TimeoutError:
            if not terminal_emitted and not record.is_terminal:
                failure = self._failure_event(record, "LEGACY_RUNTIME_TIMEOUT")
                self._event_store.append(record, failure)
                yield failure
        except Exception:
            if not terminal_emitted and not record.is_terminal:
                failure = self._failure_event(record, "LEGACY_RUNTIME_ERROR")
                self._event_store.append(record, failure)
                yield failure
        finally:
            try:
                if legacy_events is not None:
                    close = getattr(legacy_events, "aclose", None)
                    if close is not None:
                        try:
                            await close()
                        except (asyncio.CancelledError, GeneratorExit):
                            raise
                        except Exception as exc:
                            logger.error(
                                "Đóng legacy event stream thất bại "
                                "code=LEGACY_STREAM_CLOSE_FAILED "
                                "run_id=%s trace_id=%s type=%s",
                                record.run_id,
                                scope.trace_id or "none",
                                type(exc).__name__,
                            )
            finally:
                self._event_store.finish(record)

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
        for event in self._event_store.snapshot(
            record,
            after_sequence=after_sequence,
        ):
            yield event

    def _adapt_legacy_event(
        self,
        record: _RunRecord,
        legacy_event: Mapping[str, Any],
        *,
        credential: str | None,
    ) -> RuntimeEvent | None:
        safe_event = _redact_value(dict(legacy_event), credential=credential)
        if not isinstance(safe_event, dict):
            raise TypeError("Legacy runtime event must be an object")
        event_type = safe_event.get("type")
        if not isinstance(event_type, str):
            raise TypeError("Legacy runtime event requires a type")

        common = {
            **_event_identity(record),
            "sequence": len(record.events) + 1,
            "compatibility_event": safe_event,
        }
        if event_type == "done":
            response = safe_event.get("response")
            if not isinstance(response, dict):
                raise TypeError("Legacy done response must be an object")
            return RunCompleted(**common, result=response)
        if event_type == "error":
            error = safe_event.get("error")
            raw_code = error.get("code") if isinstance(error, dict) else None
            code = (
                str(raw_code).lower()
                if isinstance(raw_code, str)
                and re.fullmatch(r"[a-zA-Z0-9_.-]{1,64}", raw_code)
                else "legacy_runtime_error"
            )
            compatibility_event = {
                "type": "error",
                "error": {"code": code, "message": "Agent runtime failed"},
            }
            return RunFailed(
                **{**common, "compatibility_event": compatibility_event},
                error_code=code.upper(),
            )
        if event_type == "plan":
            return PlanProposed(**common, plan=safe_event)
        if event_type == "token":
            return ArtifactProduced(
                **common,
                artifact_type="response_delta",
                artifact={"content": safe_event.get("content", "")},
            )
        if event_type == "tool_result":
            return EvidenceAdded(
                **common,
                evidence_type="tool_result",
                evidence={
                    "tool": safe_event.get("tool", ""),
                    "output": safe_event.get("output"),
                },
            )
        if event_type == "plan_event":
            return CheckCompleted(
                **common,
                checker="legacy_plan_executor",
                verdict="observed",
                details={"event": safe_event.get("event")},
            )
        if event_type == "surprise":
            return EvidenceAdded(
                **common,
                evidence_type="surprise",
                evidence=safe_event,
            )
        return ActionCompleted(
            **common,
            action=event_type,
            outcome="requested" if event_type == "tool_call" else "observed",
            details=safe_event,
        )

    @staticmethod
    def _failure_event(record: _RunRecord, code: str) -> RunFailed:
        compatibility_event = {
            "type": "error",
            "error": {"code": code.lower(), "message": "Agent runtime failed"},
        }
        return RunFailed(
            **_event_identity(record),
            sequence=len(record.events) + 1,
            compatibility_event=compatibility_event,
            error_code=code,
        )


def build_start_turn(
    message: str,
    *,
    user_id: str | None = None,
    user_token: str | None = None,
    history: list[dict[str, str]] | None = None,
    world_model: dict[str, Any] | None = None,
    memory_context: str | None = None,
    mongo_client: Any | None = None,
    db_name: str | None = None,
    world_store: Any | None = None,
    wm_service: Any | None = None,
    model_name: str | None = None,
    trace_id: str | None = None,
) -> tuple[StartTurn, RequestScope]:
    command = StartTurn(
        message=message,
        history=tuple(dict(item) for item in (history or [])),
        world_model=dict(world_model) if world_model is not None else None,
        memory_context=memory_context,
        model_name=model_name,
    )
    safe_trace_id = trace_id
    if trace_id is not None and not _RUNTIME_ID_PATTERN.fullmatch(trace_id):
        safe_trace_id = hashlib.sha256(trace_id.encode("utf-8")).hexdigest()
    scope = RequestScope(
        principal_id=str(user_id) if user_id else f"anonymous:{command.run_id}",
        credential=user_token or None,
        trace_id=safe_trace_id,
        services={
            "mongo_client": mongo_client,
            "db_name": db_name,
            "world_store": world_store,
            "wm_service": wm_service,
        },
    )
    return command, scope


async def collect_runtime_result(
    runtime: AgentRuntime,
    command: StartTurn,
    *,
    scope: RequestScope,
) -> dict[str, Any]:
    result: dict[str, Any] | None = None
    failure: RunFailed | None = None
    runtime_events = runtime.dispatch(command, scope=scope)
    try:
        async for event in runtime_events:
            if isinstance(event, RunCompleted):
                result = dict(event.result)
            elif isinstance(event, RunFailed):
                failure = event
            elif isinstance(event, RunCancelled):
                raise asyncio.CancelledError(event.reason)
    finally:
        close = getattr(runtime_events, "aclose", None)
        if close is not None:
            await close()
    if result is not None:
        return result
    if failure is not None:
        raise AgentRuntimeError(failure.error_code, failure.message)
    raise AgentRuntimeError("RUNTIME_TERMINAL_MISSING")


async def stream_legacy_events(
    runtime: AgentRuntime,
    command: StartTurn,
    *,
    scope: RequestScope,
) -> AsyncIterator[dict[str, Any]]:
    runtime_events = runtime.dispatch(command, scope=scope)
    try:
        async for event in runtime_events:
            legacy_event = runtime_event_to_legacy(event)
            if legacy_event is not None:
                yield legacy_event
    finally:
        close = getattr(runtime_events, "aclose", None)
        if close is not None:
            await close()


def runtime_event_to_legacy(event: RuntimeEvent) -> dict[str, Any] | None:
    if event.compatibility_event is not None:
        return dict(event.compatibility_event)
    if isinstance(event, RunStarted):
        return None
    if isinstance(event, RunCompleted):
        return {"type": "done", "response": dict(event.result)}
    if isinstance(event, RunFailed):
        return {
            "type": "error",
            "error": {
                "code": event.error_code.lower(),
                "message": event.message,
            },
        }
    if isinstance(event, RunCancelled):
        return {
            "type": "error",
            "error": {"code": "run_cancelled", "message": "Run cancelled"},
        }
    if isinstance(event, PlanProposed):
        return {**dict(event.plan), "type": "plan"}
    if isinstance(event, ArtifactProduced) and event.artifact_type == "response_delta":
        return {
            "type": "token",
            "content": event.artifact.get("content", ""),
        }
    if isinstance(event, CheckCompleted):
        return {
            "type": "plan_event",
            "event": {
                **dict(event.details),
                "checker": event.checker,
                "verdict": event.verdict,
            },
        }
    if isinstance(event, ActionCompleted) and event.action in {
        "route",
        "phase",
        "tool_call",
    }:
        return {**dict(event.details), "type": event.action}
    if isinstance(event, EvidenceAdded) and event.evidence_type in {
        "tool_result",
        "surprise",
    }:
        return {**dict(event.evidence), "type": event.evidence_type}
    return {"type": "meta", "runtime_event": runtime_event_to_dict(event)}


def runtime_event_to_dict(event: RuntimeEvent) -> dict[str, Any]:
    """Trả event tương thích JSON mà không lộ dữ liệu scope tạm thời."""
    return {
        item.name: _json_value(getattr(event, item.name))
        for item in fields(event)
        if item.name != "compatibility_event"
    }


_runtime_lock = threading.RLock()
_runtime: AgentRuntime | None = None


def get_agent_runtime() -> AgentRuntime:
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = LegacyGraphRuntime()
        return _runtime


def set_agent_runtime(runtime: AgentRuntime | None) -> AgentRuntime | None:
    """Đổi process runtime để test deterministic hoặc wiring ứng dụng."""
    global _runtime
    with _runtime_lock:
        previous = _runtime
        _runtime = runtime
        return previous


def _event_identity(record: _RunRecord) -> dict[str, Any]:
    return {
        "run_id": record.run_id,
        "command_id": record.command_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _command_fingerprint(command: StartTurn) -> str:
    payload = json.dumps(
        asdict(command),
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _command_tombstone_key(owner_id: str, command_id: str) -> str:
    return json.dumps(
        [owner_id, command_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _run_tombstone_key(run_id: str) -> str:
    return json.dumps(["run", run_id], ensure_ascii=False, separators=(",", ":"))


def _event_storage_size(event: RuntimeEvent) -> int:
    payload = {
        item.name: _json_value(getattr(event, item.name))
        for item in fields(event)
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return len(serialized.encode("utf-8"))


def _is_sensitive_key(key: Any) -> bool:
    compact = re.sub(r"[^a-z0-9]", "", str(key).casefold())
    if compact == "tokencount":
        return False
    return compact in _SENSITIVE_KEY_ALIASES or compact.endswith(
        ("apikey", "password", "privatekey", "secret", "token")
    )


def _redact_value(value: Any, *, credential: str | None) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): (
                "[REDACTED]"
                if _is_sensitive_key(key)
                else _redact_value(item, credential=credential)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(item, credential=credential) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item, credential=credential) for item in value)
    if isinstance(value, str) and credential and credential in value:
        return value.replace(credential, "[REDACTED]")
    return value


def _json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value
