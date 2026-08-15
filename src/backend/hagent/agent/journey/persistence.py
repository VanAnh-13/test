"""Persistence boundary fail-closed cho durable LangGraph journey."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from hagent.agent.journey.artifacts import (
    DatasetAudit,
    EvaluationReport,
    EvidenceRef,
    ExperimentSpec,
    PredictionArtifact,
    ReleaseCandidate,
    TrainingRunSet,
)
from hagent.agent.journey.checkers import CheckerVerdict, CheckFinding

JOURNEY_CHECKPOINT_NAMESPACE = "journey-v1"
JOURNEY_DURABILITY = "sync"

PersistenceMode = Literal["memory", "mongodb"]

_TYPE_KEY = "__hagent_journey_checkpoint_type_v1__"
_VALUE_KEY = "value"
_CHECKPOINT_TYPES = {
    checkpoint_type.__name__: checkpoint_type
    for checkpoint_type in (
        EvidenceRef,
        DatasetAudit,
        ExperimentSpec,
        TrainingRunSet,
        EvaluationReport,
        ReleaseCandidate,
        PredictionArtifact,
        CheckFinding,
        CheckerVerdict,
    )
}


class _Serializer(Protocol):
    def dumps_typed(self, obj: Any) -> tuple[str, bytes]: ...

    def loads_typed(self, data: tuple[str, bytes]) -> Any: ...


def _encode_checkpoint_value(value: Any) -> Any:
    value_type = type(value)
    if (
        value_type.__name__ in _CHECKPOINT_TYPES
        and _CHECKPOINT_TYPES[value_type.__name__] is value_type
    ):
        return {
            _TYPE_KEY: value_type.__name__,
            _VALUE_KEY: {
                item.name: _encode_checkpoint_value(getattr(value, item.name))
                for item in value_type.__dataclass_fields__.values()
            },
        }
    if isinstance(value, Mapping):
        return {
            str(key): _encode_checkpoint_value(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return {
            _TYPE_KEY: "tuple",
            _VALUE_KEY: [_encode_checkpoint_value(item) for item in value],
        }
    if isinstance(value, frozenset):
        return {
            _TYPE_KEY: "frozenset",
            _VALUE_KEY: [_encode_checkpoint_value(item) for item in value],
        }
    if isinstance(value, set):
        return {
            _TYPE_KEY: "set",
            _VALUE_KEY: [_encode_checkpoint_value(item) for item in value],
        }
    if isinstance(value, list):
        return [_encode_checkpoint_value(item) for item in value]
    return value


def _decode_checkpoint_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_checkpoint_value(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    if set(value) == {_TYPE_KEY, _VALUE_KEY}:
        type_name = value[_TYPE_KEY]
        payload = value[_VALUE_KEY]
        if type_name == "tuple" and isinstance(payload, list):
            return tuple(_decode_checkpoint_value(item) for item in payload)
        if type_name == "frozenset" and isinstance(payload, list):
            return frozenset(_decode_checkpoint_value(item) for item in payload)
        if type_name == "set" and isinstance(payload, list):
            return {_decode_checkpoint_value(item) for item in payload}
        checkpoint_type = _CHECKPOINT_TYPES.get(type_name)
        if checkpoint_type is not None and isinstance(payload, Mapping):
            return checkpoint_type(
                **{
                    str(key): _decode_checkpoint_value(item)
                    for key, item in payload.items()
                }
            )
    return {
        str(key): _decode_checkpoint_value(item)
        for key, item in value.items()
    }


class JourneyCheckpointSerializer:
    """Serializer allowlist cho artifact; không dùng pickle hoặc dynamic import."""

    def __init__(self, delegate: _Serializer | None = None) -> None:
        self._delegate = delegate or JsonPlusSerializer()

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        return self._delegate.dumps_typed(_encode_checkpoint_value(obj))

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        return _decode_checkpoint_value(self._delegate.loads_typed(data))


def _replace_namespace(config: dict[str, Any] | None, namespace: str):
    if config is None:
        return None
    replaced = dict(config)
    replaced["configurable"] = dict(config.get("configurable", {}))
    replaced["configurable"]["checkpoint_ns"] = namespace
    return replaced


def _to_storage_config(config: dict[str, Any] | None):
    return _replace_namespace(config, JOURNEY_CHECKPOINT_NAMESPACE)


def _to_graph_config(config: dict[str, Any] | None):
    return _replace_namespace(config, "")


def _to_graph_tuple(value: CheckpointTuple | None) -> CheckpointTuple | None:
    if value is None:
        return None
    return CheckpointTuple(
        config=_to_graph_config(value.config),
        checkpoint=value.checkpoint,
        metadata=value.metadata,
        parent_config=_to_graph_config(value.parent_config),
        pending_writes=value.pending_writes,
    )


class JourneyNamespacedSaver(BaseCheckpointSaver):
    """Map namespace root nội bộ sang namespace versioned trong storage."""

    def __init__(self, delegate: BaseCheckpointSaver) -> None:
        super().__init__(serde=delegate.serde)
        self._delegate = delegate

    @property
    def config_specs(self) -> list:
        return self._delegate.config_specs

    def get_next_version(self, current, channel):
        return self._delegate.get_next_version(current, channel)

    def get_tuple(self, config):
        return _to_graph_tuple(self._delegate.get_tuple(_to_storage_config(config)))

    def list(
        self,
        config,
        *,
        filter=None,
        before=None,
        limit=None,
    ) -> Iterator[CheckpointTuple]:
        for item in self._delegate.list(
            _to_storage_config(config),
            filter=filter,
            before=_to_storage_config(before),
            limit=limit,
        ):
            graph_item = _to_graph_tuple(item)
            if graph_item is not None:
                yield graph_item

    def put(self, config, checkpoint, metadata, new_versions):
        stored = self._delegate.put(
            _to_storage_config(config),
            checkpoint,
            metadata,
            new_versions,
        )
        return _to_graph_config(stored)

    def put_writes(
        self,
        config,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self._delegate.put_writes(
            _to_storage_config(config),
            writes,
            task_id,
            task_path,
        )

    def delete_thread(self, thread_id: str) -> None:
        self._delegate.delete_thread(thread_id)

    async def aget_tuple(self, config):
        value = await self._delegate.aget_tuple(_to_storage_config(config))
        return _to_graph_tuple(value)

    async def alist(
        self,
        config,
        *,
        filter=None,
        before=None,
        limit=None,
    ) -> AsyncIterator[CheckpointTuple]:
        async for item in self._delegate.alist(
            _to_storage_config(config),
            filter=filter,
            before=_to_storage_config(before),
            limit=limit,
        ):
            graph_item = _to_graph_tuple(item)
            if graph_item is not None:
                yield graph_item

    async def aput(self, config, checkpoint, metadata, new_versions):
        stored = await self._delegate.aput(
            _to_storage_config(config),
            checkpoint,
            metadata,
            new_versions,
        )
        return _to_graph_config(stored)

    async def aput_writes(
        self,
        config,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await self._delegate.aput_writes(
            _to_storage_config(config),
            writes,
            task_id,
            task_path,
        )

    async def adelete_thread(self, thread_id: str) -> None:
        await self._delegate.adelete_thread(thread_id)


def prepare_journey_checkpointer(checkpointer: Any | None) -> Any | None:
    """Gắn serializer typed đúng một lần tại boundary compile graph."""
    if checkpointer is None:
        return None
    if isinstance(checkpointer, JourneyNamespacedSaver):
        return checkpointer
    if not isinstance(checkpointer, BaseCheckpointSaver):
        raise TypeError("checkpointer must implement BaseCheckpointSaver")
    if not isinstance(checkpointer.serde, JourneyCheckpointSerializer):
        checkpointer.serde = JourneyCheckpointSerializer(checkpointer.serde)
    return JourneyNamespacedSaver(checkpointer)


class JourneyPersistenceError(RuntimeError):
    """Lỗi cấu hình/storage an toàn, không chứa URI hoặc credential."""


def _validate_thread_component(name: str, value: str) -> None:
    if not isinstance(value, str) or not value or len(value) > 256 or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty safe identifier")


def journey_thread_id(principal_id: str, run_id: str) -> str:
    """Tách tenant bằng hash, không để raw owner/run thành Mongo key."""
    _validate_thread_component("principal_id", principal_id)
    _validate_thread_component("run_id", run_id)
    return hashlib.sha256(f"{principal_id}\0{run_id}".encode()).hexdigest()


def journey_checkpoint_config(*, principal_id: str, run_id: str) -> dict[str, Any]:
    return {
        "configurable": {
            "thread_id": journey_thread_id(principal_id, run_id),
            "checkpoint_ns": JOURNEY_CHECKPOINT_NAMESPACE,
        }
    }


def journey_graph_config(*, principal_id: str, run_id: str) -> dict[str, Any]:
    """Config root graph; saver boundary ánh xạ sang namespace storage versioned."""
    config = journey_checkpoint_config(principal_id=principal_id, run_id=run_id)
    config["configurable"]["checkpoint_ns"] = ""
    return config


@dataclass(slots=True)
class JourneyPersistenceHandle:
    mode: PersistenceMode
    checkpointer: Any
    _client: MongoClient | None = field(default=None, repr=False)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


def create_journey_persistence(
    *,
    mode: PersistenceMode,
    allow_memory: bool = False,
    mongodb_uri: str | None = None,
    db_name: str = "hagent_journey",
    checkpoint_collection_name: str = "checkpoints",
    writes_collection_name: str = "checkpoint_writes",
    ttl_seconds: int | None = None,
    server_selection_timeout_ms: int = 2000,
) -> JourneyPersistenceHandle:
    """Tạo saver theo mode rõ ràng; lỗi Mongo không bao giờ fallback sang RAM."""
    if mode == "memory":
        if not allow_memory:
            raise JourneyPersistenceError(
                "Memory persistence must be explicitly allowed for dev/test"
            )
        return JourneyPersistenceHandle(
            mode="memory",
            checkpointer=InMemorySaver(serde=JourneyCheckpointSerializer()),
        )
    if mode != "mongodb":
        raise JourneyPersistenceError("Unsupported journey persistence mode")
    if not isinstance(mongodb_uri, str) or not mongodb_uri.strip():
        raise JourneyPersistenceError("MongoDB URI is required")
    if ttl_seconds is not None and ttl_seconds <= 0:
        raise JourneyPersistenceError("MongoDB checkpoint TTL must be positive")
    if server_selection_timeout_ms < 1:
        raise JourneyPersistenceError("MongoDB timeout must be positive")

    client: MongoClient | None = None
    try:
        client = MongoClient(
            mongodb_uri,
            serverSelectionTimeoutMS=server_selection_timeout_ms,
        )
        client.admin.command("ping")
        saver = MongoDBSaver(
            client,
            db_name=db_name,
            checkpoint_collection_name=checkpoint_collection_name,
            writes_collection_name=writes_collection_name,
            ttl=ttl_seconds,
            serde=JourneyCheckpointSerializer(),
        )
    except (PyMongoError, ValueError, TypeError):
        if client is not None:
            client.close()
        raise JourneyPersistenceError("MongoDB checkpoint backend is unavailable") from None
    return JourneyPersistenceHandle(
        mode="mongodb",
        checkpointer=saver,
        _client=client,
    )
