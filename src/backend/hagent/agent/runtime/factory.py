"""Composition root explicit cho legacy và durable AutoML Journey runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import structlog

from hagent.agent.capabilities.catalog import CapabilityCatalog
from hagent.agent.capabilities.native import (
    NATIVE_PROVIDER_ID,
    HAutoMLNativeAdapter,
    native_journey_descriptors,
)
from hagent.agent.journey.artifact_store import (
    ArtifactMetadataUnavailable,
    InMemoryArtifactMetadataStore,
    MongoArtifactMetadataStore,
)
from hagent.agent.journey.persistence import (
    JourneyPersistenceError,
    JourneyPersistenceHandle,
    PersistenceMode,
    create_journey_persistence,
)
from hagent.agent.journey.runtime_adapter import JourneyRuntime
from hagent.agent.runtime.contracts import (
    AgentRuntime,
    InMemoryRuntimeEventStore,
    LegacyGraphRuntime,
)
from hagent.agent.runtime.shadow import ReportSink, ShadowAgentRuntime
from hagent.agent.runtime.store import (
    MongoRuntimeEventStore,
    RuntimeLedgerUnavailable,
)

RuntimeMode = Literal["legacy", "shadow", "journey"]
_DEFAULT_RETENTION_DAYS = 30
_DEFAULT_ARTIFACT_RETENTION_DAYS = 180
_SECONDS_PER_DAY = 24 * 60 * 60
logger = structlog.get_logger(__name__)


class AgentRuntimeFactoryError(RuntimeError):
    """Lỗi composition an toàn, không chứa URI hoặc credential."""


@dataclass(slots=True)
class AgentRuntimeHandle:
    """Sở hữu runtime và toàn bộ storage client do factory tạo."""

    mode: RuntimeMode
    runtime: AgentRuntime = field(repr=False)
    capability_snapshot_digest: str | None = None
    _persistence: JourneyPersistenceHandle | None = field(
        default=None,
        repr=False,
    )
    _event_store: object | None = field(default=None, repr=False)
    _artifact_store: object | None = field(default=None, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def _close_resources(self) -> None:
        close_error = False
        artifact_close = getattr(self._artifact_store, "close", None)
        if callable(artifact_close):
            try:
                artifact_close()
            except Exception:  # noqa: BLE001 - Cleanup phải tiếp tục với resource sau.
                close_error = True
        event_close = getattr(self._event_store, "close", None)
        if callable(event_close):
            try:
                event_close()
            except Exception:  # noqa: BLE001 - Cleanup phải tiếp tục với resource sau.
                close_error = True
        if self._persistence is not None:
            try:
                self._persistence.close()
            except Exception:  # noqa: BLE001 - Cleanup phải trả lỗi tổng quát an toàn.
                close_error = True
        if close_error:
            raise AgentRuntimeFactoryError("Agent runtime resources could not close")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close_error = False
        runtime_close = getattr(self.runtime, "close", None)
        if callable(runtime_close):
            try:
                runtime_close()
            except Exception:  # noqa: BLE001 - Runtime plugin có close contract mở.
                close_error = True
        try:
            self._close_resources()
        except AgentRuntimeFactoryError:
            close_error = True
        if close_error:
            raise AgentRuntimeFactoryError("Agent runtime resources could not close")

    async def aclose(self) -> None:
        """Đợi shadow observer dừng trước khi đóng durable storage."""
        if self._closed:
            return
        self._closed = True
        close_error = False
        runtime_aclose = getattr(self.runtime, "aclose", None)
        if callable(runtime_aclose):
            try:
                await runtime_aclose()
            except Exception:  # noqa: BLE001 - Runtime plugin có async close contract mở.
                close_error = True
        else:
            runtime_close = getattr(self.runtime, "close", None)
            if callable(runtime_close):
                try:
                    runtime_close()
                except Exception:  # noqa: BLE001 - Runtime plugin có close contract mở.
                    close_error = True
        try:
            self._close_resources()
        except AgentRuntimeFactoryError:
            close_error = True
        if close_error:
            raise AgentRuntimeFactoryError("Agent runtime resources could not close")


def _close_partial(
    persistence: JourneyPersistenceHandle | None,
    event_store: object | None,
    artifact_store: object | None,
) -> None:
    artifact_close = getattr(artifact_store, "close", None)
    if callable(artifact_close):
        try:
            artifact_close()
        except Exception as exc:  # noqa: BLE001 - Partial cleanup không được chặn owner sau.
            logger.error(
                "Không đóng được partial artifact store",
                extra={"error_type": type(exc).__name__},
            )
    event_close = getattr(event_store, "close", None)
    if callable(event_close):
        try:
            event_close()
        except Exception as exc:  # noqa: BLE001 - Partial cleanup không được chặn owner sau.
            logger.error(
                "Không đóng được partial runtime event store",
                extra={"error_type": type(exc).__name__},
            )
    if persistence is not None:
        try:
            persistence.close()
        except Exception as exc:  # noqa: BLE001 - Partial cleanup không được chặn owner sau.
            logger.error(
                "Không đóng được partial journey persistence",
                extra={"error_type": type(exc).__name__},
            )


def create_agent_runtime(
    *,
    mode: RuntimeMode,
    persistence_mode: PersistenceMode = "mongodb",
    mongodb_uri: str | None = None,
    db_name: str = "hagent_journey",
    checkpoint_collection_name: str = "checkpoints",
    checkpoint_writes_collection_name: str = "checkpoint_writes",
    ledger_collection_name: str = "runtime_runs",
    artifact_collection_name: str = "runtime_artifacts",
    checkpoint_ttl_seconds: int | None = (_DEFAULT_RETENTION_DAYS * _SECONDS_PER_DAY),
    event_retention_days: int = _DEFAULT_RETENTION_DAYS,
    artifact_retention_days: int = _DEFAULT_ARTIFACT_RETENTION_DAYS,
    server_selection_timeout_ms: int = 2000,
    allow_memory: bool = False,
    native_adapter: HAutoMLNativeAdapter | None = None,
    legacy_runtime: AgentRuntime | None = None,
    shadow_report_sink: ReportSink | None = None,
) -> AgentRuntimeHandle:
    """Tạo runtime theo mode; journey không bao giờ tự fallback sang memory."""
    if mode == "legacy":
        return AgentRuntimeHandle(
            mode="legacy",
            runtime=legacy_runtime or LegacyGraphRuntime(),
        )
    if mode not in {"shadow", "journey"}:
        raise AgentRuntimeFactoryError("Unsupported agent runtime mode")

    catalog = CapabilityCatalog()
    descriptors = native_journey_descriptors()
    if mode == "shadow":
        descriptors = tuple(
            descriptor for descriptor in descriptors if descriptor.effect == "read"
        )
    catalog.register_provider(
        NATIVE_PROVIDER_ID,
        descriptors,
        native_adapter or HAutoMLNativeAdapter(),
    )
    snapshot = catalog.snapshot()
    persistence: JourneyPersistenceHandle | None = None
    event_store: object | None = None
    artifact_store: object | None = None
    try:
        persistence = create_journey_persistence(
            mode=persistence_mode,
            allow_memory=allow_memory,
            mongodb_uri=mongodb_uri,
            db_name=db_name,
            checkpoint_collection_name=checkpoint_collection_name,
            writes_collection_name=checkpoint_writes_collection_name,
            ttl_seconds=checkpoint_ttl_seconds,
            server_selection_timeout_ms=server_selection_timeout_ms,
        )
        if persistence_mode == "memory":
            event_store = InMemoryRuntimeEventStore()
            artifact_store = InMemoryArtifactMetadataStore(
                retention_days=artifact_retention_days,
            )
        else:
            event_store = MongoRuntimeEventStore.connect(
                mongodb_uri,
                db_name=db_name,
                collection_name=ledger_collection_name,
                retention_days=event_retention_days,
                server_selection_timeout_ms=server_selection_timeout_ms,
            )
            artifact_store = MongoArtifactMetadataStore.connect(
                mongodb_uri,
                db_name=db_name,
                collection_name=artifact_collection_name,
                retention_days=artifact_retention_days,
                server_selection_timeout_ms=server_selection_timeout_ms,
            )
        journey_runtime = JourneyRuntime(
            capability_snapshot=snapshot,
            event_store=event_store,
            artifact_store=artifact_store,
            checkpointer=persistence.checkpointer,
        )
        runtime: AgentRuntime = journey_runtime
        if mode == "shadow":
            runtime = ShadowAgentRuntime(
                primary=legacy_runtime or LegacyGraphRuntime(),
                observer=journey_runtime,
                report_sink=shadow_report_sink,
            )
    except (
        ArtifactMetadataUnavailable,
        JourneyPersistenceError,
        RuntimeLedgerUnavailable,
    ):
        _close_partial(persistence, event_store, artifact_store)
        raise AgentRuntimeFactoryError("Durable agent runtime is unavailable") from None
    except Exception:
        _close_partial(persistence, event_store, artifact_store)
        raise

    return AgentRuntimeHandle(
        mode=mode,
        runtime=runtime,
        capability_snapshot_digest=snapshot.digest,
        _persistence=persistence,
        _event_store=event_store,
        _artifact_store=artifact_store,
    )
