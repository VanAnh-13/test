"""Artifact metadata store độc lập với checkpoint và runtime event retention."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any, Protocol

from pymongo import ASCENDING, MongoClient
from pymongo.errors import DuplicateKeyError, PyMongoError

from hagent.core.mongo import connect_mongo_client
from hagent.agent.runtime.contracts import (
    ArtifactProduced,
    RuntimeAccessDenied,
    _is_sensitive_key,
)

_DEFAULT_RETENTION_DAYS = 180
_MAX_ID_LENGTH = 256


class ArtifactMetadataUnavailable(RuntimeError):
    """Lỗi storage đã khử URI, credential và document nội bộ."""


class ArtifactMetadataConflict(RuntimeError):
    """Cùng identity nhưng payload artifact không còn bất biến."""


class ArtifactMetadataSensitiveData(ValueError):
    """Artifact metadata còn chứa trường nhạy cảm chưa redact."""


@dataclass(frozen=True, slots=True)
class ArtifactMetadataRecord:
    """Bản ghi metadata immutable trả về qua owner-scoped store seam."""

    owner_id: str = field(repr=False)
    run_id: str = field(repr=False)
    artifact_id: str = field(repr=False)
    artifact_type: str
    payload: Mapping[str, Any] = field(repr=False)
    digest: str = field(repr=False)
    created_at: datetime
    expires_at: datetime
    terminal_at: datetime | None = None


class ArtifactMetadataStore(Protocol):
    """Seam tối thiểu để persist, đọc và chốt retention của artifact."""

    def put(self, *, owner_id: str, event: ArtifactProduced) -> None: ...

    def list_for_run(
        self,
        *,
        owner_id: str,
        run_id: str,
    ) -> tuple[ArtifactMetadataRecord, ...]: ...

    def seal_run(
        self,
        *,
        owner_id: str,
        run_id: str,
        terminal_at: datetime,
    ) -> None: ...


def _validate_id(name: str, value: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > _MAX_ID_LENGTH
    ):
        raise ValueError(f"{name} must be a non-empty bounded identifier")
    return value


def _contains_sensitive_data(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _is_sensitive_key(str(key)) and item not in (None, "[REDACTED]"):
                return True
            if _contains_sensitive_data(item):
                return True
        return False
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return any(_contains_sensitive_data(item) for item in value)
    return False


def _parse_created_at(value: str) -> datetime:
    try:
        created_at = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        raise ValueError("Artifact event created_at must be ISO-8601") from None
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("Artifact event created_at must include timezone")
    return created_at.astimezone(UTC)


def _canonical_payload(event: ArtifactProduced, owner_id: str) -> tuple[dict, str]:
    owner_id = _validate_id("owner_id", owner_id)
    run_id = _validate_id("run_id", event.run_id)
    artifact_type = _validate_id("artifact_type", event.artifact_type)
    if not isinstance(event.artifact, Mapping):
        raise TypeError("Artifact payload must be a mapping")
    payload = copy.deepcopy(dict(event.artifact))
    _validate_id("artifact_id", payload.get("artifact_id"))
    if payload.get("owner_id") != owner_id or payload.get("run_id") != run_id:
        raise ValueError("Artifact identity does not match runtime authority")
    if _contains_sensitive_data(payload):
        raise ArtifactMetadataSensitiveData(
            "Artifact metadata contains unredacted sensitive data"
        )
    try:
        canonical = json.dumps(
            {"artifact_type": artifact_type, "payload": payload},
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise ValueError("Artifact payload must be JSON serializable") from None
    return payload, hashlib.sha256(canonical).hexdigest()


def _record_from_document(document: Mapping[str, Any]) -> ArtifactMetadataRecord:
    try:
        payload = copy.deepcopy(dict(document["payload"]))
        if _contains_sensitive_data(payload):
            raise ArtifactMetadataUnavailable("Artifact metadata is unavailable")
        return ArtifactMetadataRecord(
            owner_id=str(document["owner_id"]),
            run_id=str(document["run_id"]),
            artifact_id=str(document["artifact_id"]),
            artifact_type=str(document["artifact_type"]),
            payload=MappingProxyType(payload),
            digest=str(document["digest"]),
            created_at=document["created_at"],
            expires_at=document["expires_at"],
            terminal_at=document.get("terminal_at"),
        )
    except (KeyError, TypeError, ValueError):
        raise ArtifactMetadataUnavailable("Artifact metadata is unavailable") from None


class InMemoryArtifactMetadataStore:
    """Artifact store cho dev/test với cùng invariant như Mongo adapter."""

    def __init__(self, *, retention_days: int = _DEFAULT_RETENTION_DAYS) -> None:
        if retention_days < 1:
            raise ValueError("retention_days must be positive")
        self._retention = timedelta(days=retention_days)
        self._documents: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._lock = threading.RLock()

    def put(self, *, owner_id: str, event: ArtifactProduced) -> None:
        payload, digest = _canonical_payload(event, owner_id)
        artifact_id = str(payload["artifact_id"])
        key = (owner_id, event.run_id, artifact_id)
        created_at = _parse_created_at(event.created_at)
        document = {
            "owner_id": owner_id,
            "run_id": event.run_id,
            "artifact_id": artifact_id,
            "artifact_type": event.artifact_type,
            "payload": payload,
            "digest": digest,
            "created_at": created_at,
            "expires_at": created_at + self._retention,
        }
        with self._lock:
            prior = self._documents.get(key)
            if prior is not None:
                if prior["digest"] != digest:
                    raise ArtifactMetadataConflict(
                        "Artifact metadata conflicts with immutable identity"
                    )
                return
            self._documents[key] = document

    def list_for_run(
        self,
        *,
        owner_id: str,
        run_id: str,
    ) -> tuple[ArtifactMetadataRecord, ...]:
        _validate_id("owner_id", owner_id)
        _validate_id("run_id", run_id)
        with self._lock:
            documents = [
                copy.deepcopy(document)
                for (stored_owner, stored_run, _), document in self._documents.items()
                if stored_owner == owner_id and stored_run == run_id
            ]
            if not documents and any(
                stored_run == run_id and stored_owner != owner_id
                for stored_owner, stored_run, _ in self._documents
            ):
                raise RuntimeAccessDenied()
        documents.sort(key=lambda item: (item["created_at"], item["artifact_id"]))
        return tuple(_record_from_document(document) for document in documents)

    def seal_run(
        self,
        *,
        owner_id: str,
        run_id: str,
        terminal_at: datetime,
    ) -> None:
        terminal_at = _validate_terminal_at(terminal_at)
        with self._lock:
            matched = False
            wrong_owner = False
            for (stored_owner, stored_run, _), document in self._documents.items():
                if stored_run != run_id:
                    continue
                if stored_owner != owner_id:
                    wrong_owner = True
                    continue
                matched = True
                prior_terminal = document.get("terminal_at")
                if prior_terminal is not None and prior_terminal != terminal_at:
                    raise ArtifactMetadataConflict(
                        "Artifact retention is already sealed"
                    )
                document["terminal_at"] = terminal_at
                document["expires_at"] = terminal_at + self._retention
            if not matched and wrong_owner:
                raise RuntimeAccessDenied()

    def close(self) -> None:
        """Giữ interface ownership đồng nhất với Mongo adapter."""


def _validate_terminal_at(value: datetime) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError("terminal_at must include timezone")
    return value.astimezone(UTC)


class MongoArtifactMetadataStore:
    """Mongo adapter owner-scoped với immutable identity và TTL 180 ngày."""

    def __init__(
        self,
        client: MongoClient,
        *,
        db_name: str,
        collection_name: str = "runtime_artifacts",
        retention_days: int = _DEFAULT_RETENTION_DAYS,
    ) -> None:
        if not isinstance(db_name, str) or not db_name.strip():
            raise ValueError("db_name must not be empty")
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("collection_name must not be empty")
        if retention_days < 1:
            raise ValueError("retention_days must be positive")
        self._client = client
        self._collection = client[db_name][collection_name]
        self._retention = timedelta(days=retention_days)
        try:
            self._collection.create_index(
                [
                    ("owner_id", ASCENDING),
                    ("run_id", ASCENDING),
                    ("artifact_id", ASCENDING),
                ],
                unique=True,
                name="uq_journey_owner_run_artifact",
            )
            self._collection.create_index(
                [("expires_at", ASCENDING)],
                expireAfterSeconds=0,
                name="ttl_journey_artifact",
            )
        except PyMongoError:
            raise ArtifactMetadataUnavailable(
                "Artifact metadata store is unavailable"
            ) from None

    @classmethod
    def connect(
        cls,
        mongodb_uri: str,
        *,
        db_name: str = "hagent_journey",
        collection_name: str = "runtime_artifacts",
        retention_days: int = _DEFAULT_RETENTION_DAYS,
        server_selection_timeout_ms: int = 2000,
    ) -> MongoArtifactMetadataStore:
        client = connect_mongo_client(
            mongodb_uri,
            error_type=ArtifactMetadataUnavailable,
            unavailable_message="Artifact metadata store is unavailable",
            server_selection_timeout_ms=server_selection_timeout_ms,
        )
        return cls(
            client,
            db_name=db_name,
            collection_name=collection_name,
            retention_days=retention_days,
        )

    def close(self) -> None:
        self._client.close()

    def put(self, *, owner_id: str, event: ArtifactProduced) -> None:
        payload, digest = _canonical_payload(event, owner_id)
        artifact_id = str(payload["artifact_id"])
        created_at = _parse_created_at(event.created_at)
        document = {
            "owner_id": owner_id,
            "run_id": event.run_id,
            "artifact_id": artifact_id,
            "artifact_type": event.artifact_type,
            "payload": payload,
            "digest": digest,
            "created_at": created_at,
            "expires_at": created_at + self._retention,
        }
        try:
            self._collection.insert_one(document)
            return
        except DuplicateKeyError:
            pass
        except PyMongoError:
            raise ArtifactMetadataUnavailable(
                "Artifact metadata store is unavailable"
            ) from None
        try:
            prior = self._collection.find_one(
                {
                    "owner_id": owner_id,
                    "run_id": event.run_id,
                    "artifact_id": artifact_id,
                },
                {"digest": 1},
            )
        except PyMongoError:
            raise ArtifactMetadataUnavailable(
                "Artifact metadata store is unavailable"
            ) from None
        if prior is None or prior.get("digest") != digest:
            raise ArtifactMetadataConflict(
                "Artifact metadata conflicts with immutable identity"
            )

    def list_for_run(
        self,
        *,
        owner_id: str,
        run_id: str,
    ) -> tuple[ArtifactMetadataRecord, ...]:
        _validate_id("owner_id", owner_id)
        _validate_id("run_id", run_id)
        try:
            documents = list(
                self._collection.find(
                    {"owner_id": owner_id, "run_id": run_id},
                ).sort([("created_at", ASCENDING), ("artifact_id", ASCENDING)])
            )
            if not documents:
                wrong_owner = self._collection.find_one(
                    {"run_id": run_id, "owner_id": {"$ne": owner_id}},
                    {"_id": 1},
                )
                if wrong_owner is not None:
                    raise RuntimeAccessDenied()
        except RuntimeAccessDenied:
            raise
        except PyMongoError:
            raise ArtifactMetadataUnavailable(
                "Artifact metadata store is unavailable"
            ) from None
        return tuple(_record_from_document(document) for document in documents)

    def seal_run(
        self,
        *,
        owner_id: str,
        run_id: str,
        terminal_at: datetime,
    ) -> None:
        _validate_id("owner_id", owner_id)
        _validate_id("run_id", run_id)
        terminal_at = _validate_terminal_at(terminal_at)
        try:
            total = self._collection.count_documents(
                {"owner_id": owner_id, "run_id": run_id}
            )
            result = self._collection.update_many(
                {
                    "owner_id": owner_id,
                    "run_id": run_id,
                    "$or": [
                        {"terminal_at": {"$exists": False}},
                        {"terminal_at": terminal_at},
                    ],
                },
                {
                    "$set": {
                        "terminal_at": terminal_at,
                        "expires_at": terminal_at + self._retention,
                    }
                },
            )
            if total > result.matched_count:
                raise ArtifactMetadataConflict("Artifact retention is already sealed")
            if total == 0:
                wrong_owner = self._collection.find_one(
                    {"run_id": run_id, "owner_id": {"$ne": owner_id}},
                    {"_id": 1},
                )
                if wrong_owner is not None:
                    raise RuntimeAccessDenied()
        except RuntimeAccessDenied:
            raise
        except ArtifactMetadataConflict:
            raise
        except PyMongoError:
            raise ArtifactMetadataUnavailable(
                "Artifact metadata store is unavailable"
            ) from None
