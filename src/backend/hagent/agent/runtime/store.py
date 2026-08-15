"""MongoDB implementation cho runtime event ledger owner-scoped."""

from __future__ import annotations

import copy
import threading
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from pymongo import ASCENDING, MongoClient
from pymongo.errors import DuplicateKeyError, PyMongoError

from hagent.core.mongo import connect_mongo_client
from hagent.agent.runtime.contracts import (
    ActionCompleted,
    ApprovalRequired,
    ArtifactProduced,
    CheckCompleted,
    EvidenceAdded,
    PlanProposed,
    ResolveApproval,
    RunCancelled,
    RunCompleted,
    RunFailed,
    RunStarted,
    RuntimeAccessDenied,
    RuntimeCommand,
    RuntimeCommandConflict,
    RuntimeEvent,
    RuntimeEventLimitExceeded,
    RuntimeRunNotFound,
    StartTurn,
    _command_fingerprint,
    _CommandRecord,
    _event_storage_size,
    _is_sensitive_key,
    _RunRecord,
    runtime_event_to_dict,
)

_DEFAULT_MAX_EVENTS_PER_RUN = 2048
_DEFAULT_MAX_EVENT_BYTES_PER_RUN = 2 * 1024 * 1024
_TERMINAL_BYTE_RESERVE = 1024
_EVENT_TYPES = {
    event_type: event_class
    for event_type, event_class in (
        ("run_started", RunStarted),
        ("plan_proposed", PlanProposed),
        ("artifact_produced", ArtifactProduced),
        ("check_completed", CheckCompleted),
        ("approval_required", ApprovalRequired),
        ("action_completed", ActionCompleted),
        ("evidence_added", EvidenceAdded),
        ("run_completed", RunCompleted),
        ("run_failed", RunFailed),
        ("run_cancelled", RunCancelled),
    )
}


class RuntimeLedgerUnavailable(RuntimeError):
    """Lỗi storage an toàn, không chứa URI hoặc credential."""


class RuntimeLedgerSensitiveData(RuntimeError):
    """Từ chối event còn chứa field nhạy cảm chưa redact."""


def _contains_sensitive_data(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _is_sensitive_key(key) and item not in (None, "[REDACTED]"):
                return True
            if _contains_sensitive_data(item):
                return True
        return False
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return any(_contains_sensitive_data(item) for item in value)
    return False


def _event_document(event: RuntimeEvent) -> dict[str, Any]:
    document = runtime_event_to_dict(event)
    if event.compatibility_event is not None:
        document["compatibility_event"] = copy.deepcopy(
            dict(event.compatibility_event)
        )
    if _contains_sensitive_data(document):
        raise RuntimeLedgerSensitiveData(
            "Runtime event contains unredacted sensitive data"
        )
    return document


def _event_from_document(document: Mapping[str, Any]) -> RuntimeEvent:
    if _contains_sensitive_data(document):
        raise RuntimeLedgerUnavailable("Runtime event ledger is unavailable")
    payload = copy.deepcopy(dict(document))
    event_type = payload.pop("type", None)
    event_class = _EVENT_TYPES.get(event_type)
    if event_class is None:
        raise RuntimeLedgerUnavailable("Runtime event ledger is unavailable")
    try:
        return event_class(**payload)
    except (TypeError, ValueError):
        raise RuntimeLedgerUnavailable("Runtime event ledger is unavailable") from None


def _record_from_document(document: Mapping[str, Any]) -> _RunRecord:
    try:
        events = [_event_from_document(item) for item in document.get("events", [])]
        record = _RunRecord(
            owner_id=str(document["owner_id"]),
            run_id=str(document["_id"]),
            command_id=str(document["command_id"]),
            fingerprint=str(document["fingerprint"]),
            events=events,
            stored_bytes=int(document.get("stored_bytes", 0)),
            needs_reconciliation=document.get("status") == "needs_reconciliation",
        )
    except (KeyError, TypeError, ValueError):
        raise RuntimeLedgerUnavailable("Runtime event ledger is unavailable") from None
    expected_sequences = list(range(1, len(record.events) + 1))
    actual_sequences = [event.sequence for event in record.events]
    terminal_events = [
        event
        for event in record.events
        if isinstance(event, RunCompleted | RunFailed | RunCancelled)
    ]
    status = document.get("status")
    terminal_is_valid = (
        status == "terminal"
        and len(terminal_events) == 1
        and record.events
        and terminal_events[0] is record.events[-1]
        and document.get("terminal_type") == terminal_events[0].type
    )
    running_is_valid = status in {"running", "resuming"} and not terminal_events
    awaiting_is_valid = (
        status == "awaiting_approval"
        and not terminal_events
        and bool(record.events)
        and isinstance(record.events[-1], ApprovalRequired)
    )
    reconciliation_is_valid = (
        status == "needs_reconciliation" and not terminal_events
    )
    if (
        actual_sequences != expected_sequences
        or document.get("event_count") != len(record.events)
        or document.get("next_sequence") != len(record.events) + 1
        or not (
            terminal_is_valid
            or running_is_valid
            or awaiting_is_valid
            or reconciliation_is_valid
        )
    ):
        raise RuntimeLedgerUnavailable("Runtime event ledger is unavailable")
    if terminal_is_valid or awaiting_is_valid or reconciliation_is_valid:
        record.completed.set()
    return record


class MongoRuntimeEventStore:
    """Ledger Mongo với append atomic và local waiter cho duplicate cùng process."""

    def __init__(
        self,
        client: MongoClient,
        *,
        db_name: str,
        collection_name: str,
        retention_days: int = 30,
        max_events_per_run: int = _DEFAULT_MAX_EVENTS_PER_RUN,
        max_event_bytes_per_run: int = _DEFAULT_MAX_EVENT_BYTES_PER_RUN,
    ) -> None:
        if not isinstance(db_name, str) or not db_name.strip():
            raise ValueError("db_name must not be empty")
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("collection_name must not be empty")
        if retention_days < 1:
            raise ValueError("retention_days must be positive")
        if max_events_per_run < 2:
            raise ValueError("max_events_per_run must allow start and terminal events")
        if max_event_bytes_per_run < _TERMINAL_BYTE_RESERVE * 2:
            raise ValueError("max_event_bytes_per_run is too small")
        self._client = client
        self._collection = client[db_name][collection_name]
        self._retention = timedelta(days=retention_days)
        self._max_events_per_run = max_events_per_run
        self._max_event_bytes_per_run = max_event_bytes_per_run
        self._active_records: dict[str, _RunRecord] = {}
        self._active_commands: dict[tuple[str, str], _CommandRecord] = {}
        self._lock = threading.RLock()
        try:
            self._collection.create_index(
                [("owner_id", ASCENDING), ("command_id", ASCENDING)],
                unique=True,
                name="uq_runtime_owner_command",
            )
            self._collection.create_index(
                [("owner_id", ASCENDING), ("commands.command_id", ASCENDING)],
                unique=True,
                name="uq_runtime_owner_all_commands",
                partialFilterExpression={
                    "commands.command_id": {"$exists": True}
                },
            )
            self._collection.create_index(
                [("expires_at", ASCENDING)],
                expireAfterSeconds=0,
                name="ttl_runtime_terminal",
            )
        except PyMongoError:
            raise RuntimeLedgerUnavailable(
                "Runtime event ledger is unavailable"
            ) from None

    @classmethod
    def connect(
        cls,
        mongodb_uri: str,
        *,
        db_name: str = "hagent_journey",
        collection_name: str = "runtime_runs",
        retention_days: int = 30,
        max_events_per_run: int = _DEFAULT_MAX_EVENTS_PER_RUN,
        max_event_bytes_per_run: int = _DEFAULT_MAX_EVENT_BYTES_PER_RUN,
        server_selection_timeout_ms: int = 2000,
    ) -> MongoRuntimeEventStore:
        client = connect_mongo_client(
            mongodb_uri,
            error_type=RuntimeLedgerUnavailable,
            unavailable_message="Runtime event ledger is unavailable",
            server_selection_timeout_ms=server_selection_timeout_ms,
        )
        return cls(
            client,
            db_name=db_name,
            collection_name=collection_name,
            retention_days=retention_days,
            max_events_per_run=max_events_per_run,
            max_event_bytes_per_run=max_event_bytes_per_run,
        )

    def close(self) -> None:
        self._client.close()

    def begin(
        self,
        command: StartTurn,
        *,
        owner_id: str,
    ) -> tuple[_RunRecord, bool]:
        if not isinstance(owner_id, str) or not owner_id.strip() or len(owner_id) > 256:
            raise ValueError("owner_id must be a non-empty bounded identifier")
        fingerprint = _command_fingerprint(command)
        with self._lock:
            active = self._active_records.get(command.run_id)
            if active is not None:
                if active.owner_id != owner_id:
                    raise RuntimeAccessDenied()
                if (
                    active.command_id != command.command_id
                    or active.fingerprint != fingerprint
                ):
                    raise RuntimeCommandConflict()
                return active, False
            document = {
                "_id": command.run_id,
                "owner_id": owner_id,
                "command_id": command.command_id,
                "fingerprint": fingerprint,
                "status": "running",
                "next_sequence": 1,
                "event_count": 0,
                "stored_bytes": 0,
                "events": [],
                "commands": [
                    {
                        "command_id": command.command_id,
                        "fingerprint": fingerprint,
                        "command_type": "start_turn",
                        "status": "running",
                        "start_sequence": 1,
                    }
                ],
                "created_at": datetime.now(UTC),
                "updated_at": datetime.now(UTC),
            }
            try:
                self._collection.insert_one(document)
            except DuplicateKeyError:
                return self._load_duplicate(command, owner_id, fingerprint)
            except PyMongoError:
                raise RuntimeLedgerUnavailable(
                    "Runtime event ledger is unavailable"
                ) from None
            record = _record_from_document(document)
            self._active_records[command.run_id] = record
            return record, True

    def _load_duplicate(
        self,
        command: StartTurn,
        owner_id: str,
        fingerprint: str,
    ) -> tuple[_RunRecord, bool]:
        try:
            prior = self._collection.find_one(
                {"owner_id": owner_id, "command_id": command.command_id}
            )
            if prior is not None:
                if (
                    prior.get("_id") != command.run_id
                    or prior.get("fingerprint") != fingerprint
                ):
                    raise RuntimeCommandConflict()
                record = _record_from_document(prior)
                if prior.get("status") not in {"awaiting_approval", "terminal"}:
                    raise RuntimeCommandConflict()
                return record, False
            by_run = self._collection.find_one({"_id": command.run_id})
        except PyMongoError:
            raise RuntimeLedgerUnavailable(
                "Runtime event ledger is unavailable"
            ) from None
        if by_run is None:
            raise RuntimeCommandConflict()
        if by_run.get("owner_id") != owner_id:
            raise RuntimeAccessDenied()
        raise RuntimeCommandConflict()

    def append(self, record: _RunRecord, event: RuntimeEvent) -> None:
        if record.needs_reconciliation:
            raise RuntimeCommandConflict()
        if event.run_id != record.run_id:
            raise RuntimeError("Runtime event identity does not match run")
        if event.command_id != record.command_id:
            self._assert_active_command(record, event.command_id)
        document = _event_document(event)
        event_bytes = _event_storage_size(event)
        is_terminal = isinstance(event, RunCompleted | RunFailed | RunCancelled)
        event_limit = (
            self._max_events_per_run
            if is_terminal
            else self._max_events_per_run - 1
        )
        byte_limit = self._max_event_bytes_per_run
        if not is_terminal:
            byte_limit -= _TERMINAL_BYTE_RESERVE
        if event_bytes > byte_limit:
            raise RuntimeEventLimitExceeded()
        now = datetime.now(UTC)
        update: dict[str, Any] = {
            "$push": {"events": document},
            "$inc": {
                "next_sequence": 1,
                "event_count": 1,
                "stored_bytes": event_bytes,
            },
            "$set": {"updated_at": now},
        }
        if is_terminal:
            update["$set"].update(
                {
                    "status": "terminal",
                    "terminal_type": event.type,
                    "expires_at": now + self._retention,
                }
            )
        elif isinstance(event, ApprovalRequired):
            update["$set"]["status"] = "awaiting_approval"
        else:
            update["$set"]["status"] = "running"
        query = {
            "_id": record.run_id,
            "owner_id": record.owner_id,
            "command_id": record.command_id,
            "fingerprint": record.fingerprint,
            "status": {"$in": ["running", "resuming", "awaiting_approval"]},
            "next_sequence": event.sequence,
            "event_count": {"$lt": event_limit},
            "stored_bytes": {"$lte": byte_limit - event_bytes},
        }
        try:
            result = self._collection.update_one(query, update)
            if result.modified_count != 1:
                current = self._collection.find_one({"_id": record.run_id})
                self._raise_append_conflict(current, event.sequence)
        except PyMongoError:
            raise RuntimeLedgerUnavailable(
                "Runtime event ledger is unavailable"
            ) from None
        with self._lock:
            if record.is_terminal:
                raise RuntimeError("Runtime run already has a terminal event")
            record.events.append(copy.deepcopy(event))
            record.stored_bytes += event_bytes

    @staticmethod
    def _raise_append_conflict(
        current: Mapping[str, Any] | None,
        sequence: int,
    ) -> None:
        if current is None:
            raise RuntimeRunNotFound()
        if current.get("status") == "terminal":
            raise RuntimeError("Runtime run already has a terminal event")
        if current.get("next_sequence") != sequence:
            raise RuntimeError("Runtime event sequence is not monotonic")
        raise RuntimeEventLimitExceeded()

    def finish(self, record: _RunRecord) -> None:
        now = datetime.now(UTC)
        try:
            result = self._collection.update_one(
                {
                    "_id": record.run_id,
                    "owner_id": record.owner_id,
                    "commands": {
                        "$elemMatch": {
                            "command_id": record.command_id,
                            "fingerprint": record.fingerprint,
                        }
                    },
                },
                {
                    "$set": {
                        "commands.$[item].status": "completed",
                        "commands.$[item].completed_at": now,
                        "updated_at": now,
                    }
                },
                array_filters=[{"item.command_id": record.command_id}],
            )
            if result.matched_count != 1:
                raise RuntimeCommandConflict()
        except PyMongoError:
            raise RuntimeLedgerUnavailable(
                "Runtime event ledger is unavailable"
            ) from None
        with self._lock:
            self._active_records.pop(record.run_id, None)
            record.completed.set()

    def abandon(self, record: _RunRecord) -> None:
        now = datetime.now(UTC)
        try:
            result = self._collection.update_one(
                {
                    "_id": record.run_id,
                    "owner_id": record.owner_id,
                    "commands": {
                        "$elemMatch": {
                            "command_id": record.command_id,
                            "fingerprint": record.fingerprint,
                            "status": "running",
                        }
                    },
                    "status": {"$in": ["running", "resuming"]},
                },
                {
                    "$set": {
                        "status": "needs_reconciliation",
                        "commands.$[item].status": "needs_reconciliation",
                        "commands.$[item].completed_at": now,
                        "updated_at": now,
                    }
                },
                array_filters=[{"item.command_id": record.command_id}],
            )
            if result.matched_count != 1:
                raise RuntimeCommandConflict()
        except PyMongoError:
            raise RuntimeLedgerUnavailable(
                "Runtime event ledger is unavailable"
            ) from None
        with self._lock:
            self._active_records.pop(record.run_id, None)
            record.needs_reconciliation = True
            record.completed.set()

    def claim_command(
        self,
        command: RuntimeCommand,
        *,
        owner_id: str,
    ) -> tuple[_RunRecord, _CommandRecord, bool]:
        if isinstance(command, StartTurn):
            raise TypeError("StartTurn must use begin")
        if not isinstance(owner_id, str) or not owner_id.strip() or len(owner_id) > 256:
            raise ValueError("owner_id must be a non-empty bounded identifier")
        fingerprint = _command_fingerprint(command)
        command_key = (owner_id, command.command_id)
        with self._lock:
            active = self._active_commands.get(command_key)
            if active is not None:
                if active.run_id != command.run_id or active.fingerprint != fingerprint:
                    raise RuntimeCommandConflict()
                record = self.find(command.run_id, owner_id=owner_id)
                return record, active, False
            try:
                document = self._collection.find_one({"_id": command.run_id})
                if document is None:
                    raise RuntimeRunNotFound()
                if document.get("owner_id") != owner_id:
                    raise RuntimeAccessDenied()
                prior_start = self._collection.find_one(
                    {"owner_id": owner_id, "command_id": command.command_id},
                    {"_id": 1},
                )
                if prior_start is not None:
                    raise RuntimeCommandConflict()
                start_sequence = int(document.get("next_sequence", 0))
                events = document.get("events", ())
                if (
                    isinstance(command, ResolveApproval)
                    and (
                        not events
                        or events[-1].get("type") != "approval_required"
                        or events[-1].get("approval_id") != command.approval_id
                    )
                ):
                    raise RuntimeCommandConflict()
                command_document = {
                    "command_id": command.command_id,
                    "fingerprint": fingerprint,
                    "command_type": type(command).__name__,
                    "status": "running",
                    "start_sequence": start_sequence,
                    "created_at": datetime.now(UTC),
                }
                claim_query: dict[str, Any] = {
                    "_id": command.run_id,
                    "owner_id": owner_id,
                    "status": "awaiting_approval",
                    "next_sequence": start_sequence,
                    "commands.command_id": {"$ne": command.command_id},
                }
                if isinstance(command, ResolveApproval):
                    last_event_index = start_sequence - 2
                    claim_query[
                        f"events.{last_event_index}.type"
                    ] = "approval_required"
                    claim_query[
                        f"events.{last_event_index}.approval_id"
                    ] = command.approval_id
                result = self._collection.update_one(
                    claim_query,
                    {
                        "$push": {"commands": command_document},
                        "$set": {
                            "status": "resuming",
                            "updated_at": datetime.now(UTC),
                        },
                    },
                )
            except DuplicateKeyError:
                return self._load_command_claim(command, owner_id, fingerprint)
            except PyMongoError:
                raise RuntimeLedgerUnavailable(
                    "Runtime event ledger is unavailable"
                ) from None
            if result.modified_count != 1:
                return self._load_command_claim(command, owner_id, fingerprint)
            record = self.find(command.run_id, owner_id=owner_id)
            claimed = _CommandRecord(
                owner_id=owner_id,
                run_id=command.run_id,
                command_id=command.command_id,
                fingerprint=fingerprint,
                start_sequence=start_sequence,
            )
            self._active_commands[command_key] = claimed
            return record, claimed, True

    def _load_command_claim(
        self,
        command: RuntimeCommand,
        owner_id: str,
        fingerprint: str,
    ) -> tuple[_RunRecord, _CommandRecord, bool]:
        try:
            document = self._collection.find_one(
                {"owner_id": owner_id, "commands.command_id": command.command_id}
            )
        except PyMongoError:
            raise RuntimeLedgerUnavailable(
                "Runtime event ledger is unavailable"
            ) from None
        if document is None:
            self.find(command.run_id, owner_id=owner_id)
            raise RuntimeCommandConflict() from None
        if document.get("_id") != command.run_id:
            raise RuntimeCommandConflict()
        command_document = next(
            (
                item
                for item in document.get("commands", ())
                if item.get("command_id") == command.command_id
            ),
            None,
        )
        if (
            not isinstance(command_document, Mapping)
            or command_document.get("fingerprint") != fingerprint
        ):
            raise RuntimeCommandConflict()
        record = _record_from_document(document)
        claimed = _CommandRecord(
            owner_id=owner_id,
            run_id=command.run_id,
            command_id=command.command_id,
            fingerprint=fingerprint,
            start_sequence=int(command_document.get("start_sequence", 0)),
        )
        if command_document.get("status") == "completed":
            claimed.completed.set()
            return record, claimed, False
        with self._lock:
            active = self._active_commands.get((owner_id, command.command_id))
            if active is not None:
                return record, active, False
        raise RuntimeCommandConflict()

    def _assert_active_command(self, record: _RunRecord, command_id: str) -> None:
        try:
            command = self._collection.find_one(
                {
                    "_id": record.run_id,
                    "owner_id": record.owner_id,
                    "commands": {
                        "$elemMatch": {
                            "command_id": command_id,
                            "status": "running",
                        }
                    },
                },
                {"_id": 1},
            )
        except PyMongoError:
            raise RuntimeLedgerUnavailable(
                "Runtime event ledger is unavailable"
            ) from None
        if command is None:
            raise RuntimeCommandConflict()

    def finish_command(self, command: _CommandRecord) -> None:
        now = datetime.now(UTC)
        try:
            result = self._collection.update_one(
                {
                    "_id": command.run_id,
                    "owner_id": command.owner_id,
                    "commands": {
                        "$elemMatch": {
                            "command_id": command.command_id,
                            "fingerprint": command.fingerprint,
                            "status": "running",
                        }
                    },
                },
                {
                    "$set": {
                        "commands.$[item].status": "completed",
                        "commands.$[item].completed_at": now,
                        "updated_at": now,
                    }
                },
                array_filters=[
                    {
                        "item.command_id": command.command_id,
                        "item.fingerprint": command.fingerprint,
                    }
                ],
            )
            if result.matched_count != 1:
                raise RuntimeCommandConflict()
        except PyMongoError:
            raise RuntimeLedgerUnavailable(
                "Runtime event ledger is unavailable"
            ) from None
        with self._lock:
            self._active_commands.pop((command.owner_id, command.command_id), None)
            command.completed.set()

    def abandon_command(self, command: _CommandRecord) -> None:
        now = datetime.now(UTC)
        try:
            result = self._collection.update_one(
                {
                    "_id": command.run_id,
                    "owner_id": command.owner_id,
                    "commands": {
                        "$elemMatch": {
                            "command_id": command.command_id,
                            "fingerprint": command.fingerprint,
                            "status": "running",
                        }
                    },
                    "status": {"$in": ["running", "resuming"]},
                },
                {
                    "$set": {
                        "status": "needs_reconciliation",
                        "commands.$[item].status": "needs_reconciliation",
                        "commands.$[item].completed_at": now,
                        "updated_at": now,
                    }
                },
                array_filters=[
                    {
                        "item.command_id": command.command_id,
                        "item.fingerprint": command.fingerprint,
                    }
                ],
            )
            if result.matched_count != 1:
                raise RuntimeCommandConflict()
        except PyMongoError:
            raise RuntimeLedgerUnavailable(
                "Runtime event ledger is unavailable"
            ) from None
        with self._lock:
            self._active_commands.pop((command.owner_id, command.command_id), None)
            command.needs_reconciliation = True
            command.completed.set()

    def command_snapshot(
        self,
        record: _RunRecord,
        command: _CommandRecord,
    ) -> list[RuntimeEvent]:
        fresh = self.find(record.run_id, owner_id=record.owner_id)
        return [
            copy.deepcopy(event)
            for event in fresh.events
            if event.command_id == command.command_id
            and event.sequence >= command.start_sequence
        ]

    def snapshot(
        self,
        record: _RunRecord,
        *,
        after_sequence: int = 0,
    ) -> list[RuntimeEvent]:
        if after_sequence < 0:
            raise ValueError("after_sequence must not be negative")
        fresh = self.find(record.run_id, owner_id=record.owner_id)
        return [
            copy.deepcopy(event)
            for event in fresh.events
            if event.sequence > after_sequence
        ]

    def find(self, run_id: str, *, owner_id: str) -> _RunRecord:
        try:
            document = self._collection.find_one({"_id": run_id})
        except PyMongoError:
            raise RuntimeLedgerUnavailable(
                "Runtime event ledger is unavailable"
            ) from None
        if document is None:
            raise RuntimeRunNotFound()
        if document.get("owner_id") != owner_id:
            raise RuntimeAccessDenied()
        return _record_from_document(document)
