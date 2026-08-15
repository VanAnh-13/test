"""Interface ổn định cho AgentRuntime command, event và event ledger."""

from typing import TYPE_CHECKING, Any

from hagent.agent.runtime.context import GraphRequestContext, bind_request_context
from hagent.agent.runtime.contracts import (
    ActionCompleted,
    AgentRuntime,
    AgentRuntimeError,
    ApprovalRequired,
    ArtifactProduced,
    CancelRun,
    CheckCompleted,
    EvidenceAdded,
    InMemoryRuntimeEventStore,
    LegacyEventSource,
    LegacyGraphRuntime,
    PlanProposed,
    RequestScope,
    ResolveApproval,
    RunCancelled,
    RunCompleted,
    RunFailed,
    RunStarted,
    RuntimeAccessDenied,
    RuntimeCapacityExceeded,
    RuntimeCommand,
    RuntimeCommandConflict,
    RuntimeCommandExpired,
    RuntimeEvent,
    RuntimeEventLimitExceeded,
    RuntimeEventStore,
    RuntimeRunNotFound,
    StartTurn,
    TerminalRuntimeEvent,
    UnsupportedRuntimeCommand,
    _command_fingerprint,
    _CommandRecord,
    _event_storage_size,
    _is_sensitive_key,
    _RunRecord,
    build_start_turn,
    collect_runtime_result,
    get_agent_runtime,
    runtime_event_to_dict,
    runtime_event_to_legacy,
    set_agent_runtime,
    stream_legacy_events,
)
from hagent.agent.runtime.shadow import (
    ReportSink,
    RuntimeObservation,
    ShadowAgentRuntime,
    ShadowComparisonReport,
)
from hagent.agent.runtime.store import (
    MongoRuntimeEventStore,
    RuntimeLedgerSensitiveData,
    RuntimeLedgerUnavailable,
)

if TYPE_CHECKING:
    from hagent.agent.runtime.factory import (
        AgentRuntimeFactoryError,
        AgentRuntimeHandle,
        RuntimeMode,
        create_agent_runtime,
    )

_LAZY_FACTORY_EXPORTS = frozenset(
    {
        "AgentRuntimeFactoryError",
        "AgentRuntimeHandle",
        "RuntimeMode",
        "create_agent_runtime",
    }
)


def __getattr__(name: str) -> Any:
    """Nạp composition factory muộn để contract import không tạo vòng lặp."""
    if name in _LAZY_FACTORY_EXPORTS:
        from hagent.agent.runtime import factory

        return getattr(factory, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = (
    "ActionCompleted",
    "AgentRuntime",
    "AgentRuntimeError",
    "AgentRuntimeFactoryError",
    "AgentRuntimeHandle",
    "ApprovalRequired",
    "ArtifactProduced",
    "CancelRun",
    "CheckCompleted",
    "EvidenceAdded",
    "GraphRequestContext",
    "InMemoryRuntimeEventStore",
    "LegacyEventSource",
    "LegacyGraphRuntime",
    "MongoRuntimeEventStore",
    "PlanProposed",
    "ReportSink",
    "RequestScope",
    "ResolveApproval",
    "RunCancelled",
    "RunCompleted",
    "RunFailed",
    "RunStarted",
    "RuntimeAccessDenied",
    "RuntimeCapacityExceeded",
    "RuntimeCommand",
    "RuntimeCommandConflict",
    "RuntimeCommandExpired",
    "RuntimeEvent",
    "RuntimeEventLimitExceeded",
    "RuntimeEventStore",
    "RuntimeLedgerSensitiveData",
    "RuntimeLedgerUnavailable",
    "RuntimeMode",
    "RuntimeRunNotFound",
    "RuntimeObservation",
    "ShadowAgentRuntime",
    "ShadowComparisonReport",
    "StartTurn",
    "TerminalRuntimeEvent",
    "UnsupportedRuntimeCommand",
    # Giữ tương thích regression cũ; store mới không còn phụ thuộc seam này.
    "_CommandRecord",
    "_RunRecord",
    "_command_fingerprint",
    "_event_storage_size",
    "_is_sensitive_key",
    "build_start_turn",
    "bind_request_context",
    "collect_runtime_result",
    "create_agent_runtime",
    "get_agent_runtime",
    "runtime_event_to_dict",
    "runtime_event_to_legacy",
    "set_agent_runtime",
    "stream_legacy_events",
)
