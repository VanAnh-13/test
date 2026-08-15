"""Regression cho runtime event ledger bền vững và owner-scoped."""

from __future__ import annotations

import asyncio
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest
from bson import BSON
from pymongo import MongoClient

from hagent.agent.runtime import (
    LegacyGraphRuntime,
    RequestScope,
    RunCompleted,
    RunStarted,
    RuntimeAccessDenied,
    RuntimeCommandConflict,
    RuntimeEventLimitExceeded,
    StartTurn,
)


async def _collect(stream):
    return [event async for event in stream]


def test_mongodb_ledger_outage_is_fail_closed_and_repr_safe():
    from hagent.agent.runtime import (
        MongoRuntimeEventStore,
        RuntimeLedgerUnavailable,
    )

    with pytest.raises(RuntimeLedgerUnavailable, match="unavailable") as exc_info:
        MongoRuntimeEventStore.connect(
            "mongodb://user:ledger-secret@127.0.0.1:1/test",
            server_selection_timeout_ms=50,
        )

    assert "ledger-secret" not in str(exc_info.value)


@pytest.fixture
def mongo_ledger():
    from hagent.agent.runtime import MongoRuntimeEventStore

    uri = os.getenv("HAGENT_TEST_MONGODB_URI")
    if not uri:
        pytest.skip("HAGENT_TEST_MONGODB_URI chưa được cấu hình")
    db_name = f"hagent_runtime_ledger_test_{uuid.uuid4().hex}"
    store = MongoRuntimeEventStore.connect(
        uri,
        db_name=db_name,
        collection_name="runs",
        retention_days=30,
    )
    try:
        yield store, uri, db_name
    finally:
        store.close()
        client = MongoClient(uri, serverSelectionTimeoutMS=2000, tz_aware=True)
        try:
            client.drop_database(db_name)
        finally:
            client.close()


@pytest.mark.asyncio
async def test_terminal_run_replays_after_store_recreation_and_is_owner_scoped(
    mongo_ledger,
):
    from hagent.agent.runtime import MongoRuntimeEventStore

    store, uri, db_name = mongo_ledger
    source_calls = 0

    async def source(command, scope):
        nonlocal source_calls
        source_calls += 1
        yield {"type": "token", "content": f"draft {scope.credential}"}
        yield {
            "type": "done",
            "response": {"message": "ok", "api_key": "provider-secret"},
        }

    command = StartTurn(
        message="audit",
        run_id="mongo-ledger-run",
        command_id="mongo-ledger-command",
    )
    scope = RequestScope(
        principal_id="owner-1",
        credential="runtime-ledger-sentinel",
    )
    first_runtime = LegacyGraphRuntime(event_source=source, event_store=store)
    first_events = await _collect(first_runtime.dispatch(command, scope=scope))
    store.close()

    recreated_store = MongoRuntimeEventStore.connect(
        uri,
        db_name=db_name,
        collection_name="runs",
        retention_days=30,
    )
    try:
        recreated_runtime = LegacyGraphRuntime(
            event_source=source,
            event_store=recreated_store,
        )
        repeated = await _collect(recreated_runtime.dispatch(command, scope=scope))
        replayed = await _collect(
            recreated_runtime.replay(
                command.run_id,
                after_sequence=1,
                scope=scope,
            )
        )
        with pytest.raises(RuntimeAccessDenied):
            await _collect(
                recreated_runtime.replay(
                    command.run_id,
                    after_sequence=0,
                    scope=RequestScope(principal_id="other-owner"),
                )
            )
        with pytest.raises(RuntimeCommandConflict):
            await _collect(
                recreated_runtime.dispatch(
                    StartTurn(
                        message="changed",
                        run_id=command.run_id,
                        command_id=command.command_id,
                    ),
                    scope=scope,
                )
            )
        with pytest.raises(RuntimeCommandConflict):
            await _collect(
                recreated_runtime.dispatch(
                    StartTurn(
                        message=command.message,
                        run_id="different-run",
                        command_id=command.command_id,
                    ),
                    scope=scope,
                )
            )

        assert repeated == first_events
        assert replayed == first_events[1:]
        assert source_calls == 1
        assert isinstance(first_events[-1], RunCompleted)

        client = MongoClient(uri, serverSelectionTimeoutMS=2000, tz_aware=True)
        try:
            collection = client[db_name]["runs"]
            document = collection.find_one({"_id": command.run_id})
            raw_bson = BSON.encode(document)
            ttl_indexes = [
                value
                for value in collection.index_information().values()
                if value.get("expireAfterSeconds") == 0
            ]
        finally:
            client.close()
        assert document["status"] == "terminal"
        assert document["expires_at"] > datetime.now(UTC)
        assert 29 <= (document["expires_at"] - datetime.now(UTC)).days <= 30
        assert ttl_indexes
        assert b"runtime-ledger-sentinel" not in raw_bson
        assert b"provider-secret" not in raw_bson
    finally:
        recreated_store.close()


def test_atomic_sequence_terminal_and_sensitive_payload_guards(mongo_ledger):
    from hagent.agent.runtime import RuntimeLedgerSensitiveData

    store, _, _ = mongo_ledger
    command = StartTurn(
        message="audit",
        run_id="atomic-ledger-run",
        command_id="atomic-ledger-command",
    )
    record, is_new = store.begin(command, owner_id="owner-1")
    assert is_new

    with pytest.raises(RuntimeLedgerSensitiveData):
        store.append(
            record,
            RunStarted(
                run_id=record.run_id,
                command_id=record.command_id,
                sequence=1,
                created_at=datetime.now(UTC).isoformat(),
                metadata={"token": "must-not-persist"},
            ),
        )

    started = RunStarted(
        run_id=record.run_id,
        command_id=record.command_id,
        sequence=1,
        created_at=datetime.now(UTC).isoformat(),
        metadata={"stage": "audit"},
    )
    store.append(record, started)
    terminal = RunCompleted(
        run_id=record.run_id,
        command_id=record.command_id,
        sequence=2,
        created_at=datetime.now(UTC).isoformat(),
        result={"status": "completed"},
    )

    def append_terminal():
        try:
            store.append(record, terminal)
            return "stored"
        except RuntimeError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: append_terminal(), range(2)))

    assert sorted(outcomes) == ["rejected", "stored"]
    assert [event.sequence for event in store.snapshot(record)] == [1, 2]
    assert record.is_terminal


def test_corrupted_or_unknown_event_document_is_rejected(mongo_ledger):
    from hagent.agent.runtime import RuntimeLedgerUnavailable

    store, uri, db_name = mongo_ledger
    client = MongoClient(uri, serverSelectionTimeoutMS=2000, tz_aware=True)
    try:
        client[db_name]["runs"].insert_one(
            {
                "_id": "corrupted-ledger-run",
                "owner_id": "owner-1",
                "command_id": "corrupted-ledger-command",
                "fingerprint": "a" * 64,
                "status": "terminal",
                "terminal_type": "unknown_event",
                "next_sequence": 2,
                "event_count": 1,
                "stored_bytes": 10,
                "events": [
                    {
                        "type": "unknown_event",
                        "run_id": "corrupted-ledger-run",
                        "command_id": "corrupted-ledger-command",
                        "sequence": 1,
                        "created_at": datetime.now(UTC).isoformat(),
                    }
                ],
            }
        )
    finally:
        client.close()

    with pytest.raises(RuntimeLedgerUnavailable, match="unavailable"):
        store.find("corrupted-ledger-run", owner_id="owner-1")


def test_event_count_and_byte_limits_reserve_terminal_capacity(mongo_ledger):
    from hagent.agent.runtime import MongoRuntimeEventStore, PlanProposed

    _, uri, db_name = mongo_ledger
    store = MongoRuntimeEventStore.connect(
        uri,
        db_name=db_name,
        collection_name="bounded_runs",
        max_events_per_run=2,
        max_event_bytes_per_run=2048,
    )
    try:
        command = StartTurn(
            message="bounded",
            run_id="bounded-ledger-run",
            command_id="bounded-ledger-command",
        )
        record, _ = store.begin(command, owner_id="owner-1")
        store.append(
            record,
            RunStarted(
                run_id=record.run_id,
                command_id=record.command_id,
                sequence=1,
                created_at=datetime.now(UTC).isoformat(),
            ),
        )
        with pytest.raises(RuntimeEventLimitExceeded):
            store.append(
                record,
                PlanProposed(
                    run_id=record.run_id,
                    command_id=record.command_id,
                    sequence=2,
                    created_at=datetime.now(UTC).isoformat(),
                    plan={"payload": "small"},
                ),
            )
        store.append(
            record,
            RunCompleted(
                run_id=record.run_id,
                command_id=record.command_id,
                sequence=2,
                created_at=datetime.now(UTC).isoformat(),
                result={"status": "limited"},
            ),
        )

        large_command = StartTurn(
            message="large",
            run_id="large-ledger-run",
            command_id="large-ledger-command",
        )
        large_record, _ = store.begin(large_command, owner_id="owner-1")
        with pytest.raises(RuntimeEventLimitExceeded):
            store.append(
                large_record,
                RunStarted(
                    run_id=large_record.run_id,
                    command_id=large_record.command_id,
                    sequence=1,
                    created_at=datetime.now(UTC).isoformat(),
                    metadata={"payload": "x" * 4096},
                ),
            )
    finally:
        store.close()


@pytest.mark.asyncio
async def test_concurrent_duplicate_uses_one_source_invocation(mongo_ledger):
    store, _, _ = mongo_ledger
    source_started = asyncio.Event()
    release_source = asyncio.Event()
    source_calls = 0

    async def source(command, scope):
        nonlocal source_calls
        source_calls += 1
        source_started.set()
        await release_source.wait()
        yield {"type": "done", "response": {"message": "ok"}}

    runtime = LegacyGraphRuntime(event_source=source, event_store=store)
    command = StartTurn(
        message="audit",
        run_id="concurrent-ledger-run",
        command_id="concurrent-ledger-command",
    )
    scope = RequestScope(principal_id="owner-1")
    first = asyncio.create_task(_collect(runtime.dispatch(command, scope=scope)))
    await asyncio.wait_for(source_started.wait(), timeout=2)
    second = asyncio.create_task(_collect(runtime.dispatch(command, scope=scope)))
    await asyncio.sleep(0)
    release_source.set()
    first_events, second_events = await asyncio.gather(first, second)

    assert source_calls == 1
    assert first_events == second_events


@pytest.mark.asyncio
async def test_journey_runtime_accepts_durable_event_store(mongo_ledger):
    from hagent.agent.capabilities.catalog import CapabilityCatalog
    from hagent.agent.capabilities.models import CapabilityDescriptor
    from hagent.agent.journey.runtime_adapter import JourneyAuditRuntime

    class DatasetAdapter:
        async def invoke(self, capability_id, arguments, *, scope):
            return {
                "_id": arguments["dataset_id"],
                "columns": ["feature", "target"],
                "target": "target",
                "missingness": {},
                "class_balance": {},
            }

    descriptor = CapabilityDescriptor(
        id="automl.dataset.inspect@1",
        input_schema={
            "type": "object",
            "required": ["dataset_id"],
            "properties": {"dataset_id": {"type": "string"}},
            "additionalProperties": False,
        },
        output_schema={"type": "object"},
        effect="read",
        required_scopes=frozenset({"automl.dataset.read"}),
        provider_id="mongo-ledger-fake",
    )
    catalog = CapabilityCatalog()
    catalog.register_provider("mongo-ledger-fake", [descriptor], DatasetAdapter())
    store, _, _ = mongo_ledger
    runtime = JourneyAuditRuntime(
        capability_snapshot=catalog.snapshot(),
        event_store=store,
    )
    scope = RequestScope(
        principal_id="owner-1",
        credential="journey-ledger-sentinel",
        services={"scopes": ("automl.dataset.read",)},
    )
    command = StartTurn(
        message="Audit dataset dataset-1 target target",
        run_id="journey-ledger-run",
        command_id="journey-ledger-command",
    )

    dispatched = await _collect(runtime.dispatch(command, scope=scope))
    replayed = await _collect(
        runtime.replay(command.run_id, after_sequence=2, scope=scope)
    )

    assert dispatched[-1].type == "run_completed"
    assert replayed == dispatched[2:]
    _, uri, db_name = mongo_ledger
    client = MongoClient(uri, serverSelectionTimeoutMS=2000)
    try:
        document = client[db_name]["runs"].find_one({"_id": command.run_id})
        raw_bson = BSON.encode(document)
    finally:
        client.close()
    assert b"journey-ledger-sentinel" not in raw_bson
