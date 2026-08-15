"""Regression qua seam AgentRuntime cho approval bền vững của journey."""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest
from bson import BSON
from langgraph.checkpoint.memory import InMemorySaver
from pymongo import MongoClient

from hagent.agent.capabilities.catalog import CapabilityCatalog
from hagent.agent.capabilities.models import CapabilityDescriptor
from hagent.agent.journey.persistence import create_journey_persistence
from hagent.agent.journey.runtime_adapter import JourneyAuditRuntime
from hagent.agent.runtime import (
    ApprovalRequired,
    RequestScope,
    ResolveApproval,
    RunCompleted,
    RuntimeAccessDenied,
    RuntimeCommandConflict,
    StartTurn,
)


class _DatasetAdapter:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, capability_id, arguments, *, scope):
        self.calls += 1
        return {
            "_id": arguments["dataset_id"],
            "columns": ["feature", "target"],
            "target": "target",
            "missingness": {},
            "class_balance": {"0": 0.5, "1": 0.5},
            "leakage_risks": [],
        }


def _snapshot(adapter: _DatasetAdapter):
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
        provider_id="experiment-runtime-fake",
    )
    catalog = CapabilityCatalog()
    catalog.register_provider("experiment-runtime-fake", [descriptor], adapter)
    return catalog.snapshot()


def _scope(owner: str = "owner-1", credential: str = "runtime-approval-sentinel"):
    return RequestScope(
        principal_id=owner,
        credential=credential,
        services={
            "scopes": ("automl.dataset.read",),
            "max_training_jobs": 5,
        },
    )


async def _collect(stream):
    return [event async for event in stream]


@pytest.mark.asyncio
async def test_start_waits_for_approval_then_resolve_completes_same_run_once():
    adapter = _DatasetAdapter()
    runtime = JourneyAuditRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=InMemorySaver(),
    )
    start = StartTurn(
        message="Train dataset dataset-1 target target",
        run_id="approval-runtime-run",
        command_id="approval-runtime-start",
    )

    initial_events = await _collect(runtime.dispatch(start, scope=_scope()))

    assert isinstance(initial_events[-1], ApprovalRequired)
    assert not any(isinstance(event, RunCompleted) for event in initial_events)
    assert [event.sequence for event in initial_events] == list(
        range(1, len(initial_events) + 1)
    )
    assert [
        event.artifact_type
        for event in initial_events
        if event.type == "artifact_produced"
    ] == ["DatasetAudit", "ExperimentSpec"]
    assert adapter.calls == 1

    approval = ResolveApproval(
        run_id=start.run_id,
        approval_id=initial_events[-1].approval_id,
        approved=True,
        command_id="approval-runtime-resolve",
    )
    resolved_events = await _collect(runtime.dispatch(approval, scope=_scope()))
    duplicate_events = await _collect(runtime.dispatch(approval, scope=_scope()))
    replayed = await _collect(
        runtime.replay(start.run_id, after_sequence=0, scope=_scope())
    )

    assert resolved_events == duplicate_events
    assert isinstance(resolved_events[-1], RunCompleted)
    assert resolved_events[-1].result["status"] == "approved"
    assert replayed == initial_events + resolved_events
    assert [event.sequence for event in replayed] == list(range(1, len(replayed) + 1))
    assert adapter.calls == 1
    assert "runtime-approval-sentinel" not in repr(replayed)


@pytest.mark.asyncio
async def test_wrong_owner_cannot_resolve_approval():
    adapter = _DatasetAdapter()
    runtime = JourneyAuditRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=InMemorySaver(),
    )
    start = StartTurn(
        message="Train dataset dataset-1 target target",
        run_id="wrong-owner-approval-run",
        command_id="wrong-owner-approval-start",
    )
    initial_events = await _collect(runtime.dispatch(start, scope=_scope()))

    with pytest.raises(RuntimeAccessDenied):
        await _collect(
            runtime.dispatch(
                ResolveApproval(
                    run_id=start.run_id,
                    approval_id=initial_events[-1].approval_id,
                    approved=True,
                    command_id="wrong-owner-approval-resolve",
                ),
                scope=_scope(owner="other-owner"),
            )
        )

    assert adapter.calls == 1


@pytest.mark.asyncio
async def test_audit_only_with_checkpointer_keeps_compatible_terminal_flow():
    adapter = _DatasetAdapter()
    runtime = JourneyAuditRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=InMemorySaver(),
    )
    events = await _collect(
        runtime.dispatch(
            StartTurn(
                message="Audit dataset dataset-1 target target",
                run_id="checkpointed-audit-run",
                command_id="checkpointed-audit-start",
            ),
            scope=_scope(),
        )
    )

    assert isinstance(events[-1], RunCompleted)
    assert events[-1].result["status"] == "completed"
    assert not any(isinstance(event, ApprovalRequired) for event in events)
    assert [
        event.artifact_type for event in events if event.type == "artifact_produced"
    ] == ["DatasetAudit"]
    assert adapter.calls == 1


@pytest.mark.asyncio
async def test_reject_approval_completes_without_mutation():
    adapter = _DatasetAdapter()
    runtime = JourneyAuditRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=InMemorySaver(),
    )
    start = StartTurn(
        message="Train dataset dataset-1 target target",
        run_id="reject-approval-runtime-run",
        command_id="reject-approval-runtime-start",
    )
    initial_events = await _collect(runtime.dispatch(start, scope=_scope()))
    rejected = await _collect(
        runtime.dispatch(
            ResolveApproval(
                run_id=start.run_id,
                approval_id=initial_events[-1].approval_id,
                approved=False,
                command_id="reject-approval-runtime-resolve",
            ),
            scope=_scope(),
        )
    )

    assert rejected[-1].result["status"] == "rejected"
    assert adapter.calls == 1


@pytest.mark.asyncio
async def test_edit_emits_superseding_artifact_and_new_approval():
    adapter = _DatasetAdapter()
    runtime = JourneyAuditRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=InMemorySaver(),
    )
    start = StartTurn(
        message="Train dataset dataset-1 target target",
        run_id="edit-approval-runtime-run",
        command_id="edit-approval-runtime-start",
    )
    initial_events = await _collect(runtime.dispatch(start, scope=_scope()))
    first_artifact = next(
        event.artifact
        for event in initial_events
        if event.type == "artifact_produced" and event.artifact_type == "ExperimentSpec"
    )

    edited_events = await _collect(
        runtime.dispatch(
            ResolveApproval(
                run_id=start.run_id,
                approval_id=initial_events[-1].approval_id,
                approved=False,
                command_id="edit-approval-runtime-resolve",
                response={
                    "decision": "edit",
                    "changes": {"metric": "f1", "max_training_jobs": 2},
                },
            ),
            scope=_scope(),
        )
    )
    edited_artifact = next(
        event.artifact
        for event in edited_events
        if event.type == "artifact_produced" and event.artifact_type == "ExperimentSpec"
    )

    assert edited_artifact["version"] == 2
    assert edited_artifact["supersedes"] == first_artifact["artifact_id"]
    assert edited_artifact["metric"] == "f1"
    assert isinstance(edited_events[-1], ApprovalRequired)
    assert edited_events[-1].approval_id != initial_events[-1].approval_id
    assert adapter.calls == 1


@pytest.mark.asyncio
async def test_unknown_sensitive_approval_field_is_not_persisted_or_emitted():
    saver = InMemorySaver()
    adapter = _DatasetAdapter()
    runtime = JourneyAuditRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=saver,
    )
    start = StartTurn(
        message="Train dataset dataset-1 target target",
        run_id="sensitive-approval-runtime-run",
        command_id="sensitive-approval-runtime-start",
    )
    initial_events = await _collect(runtime.dispatch(start, scope=_scope()))
    command = ResolveApproval(
        run_id=start.run_id,
        approval_id=initial_events[-1].approval_id,
        approved=True,
        command_id="sensitive-approval-runtime-resolve",
        response={"token": "resume-command-secret"},
    )

    resolved = await _collect(runtime.dispatch(command, scope=_scope()))

    assert resolved[-1].result["status"] == "invalid_approval"
    assert "resume-command-secret" not in repr(resolved)
    assert "resume-command-secret" not in repr(list(saver.storage.values()))


@pytest.mark.asyncio
async def test_concurrent_duplicate_approval_replays_one_command_event_set():
    adapter = _DatasetAdapter()
    runtime = JourneyAuditRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=InMemorySaver(),
    )
    start = StartTurn(
        message="Train dataset dataset-1 target target",
        run_id="concurrent-approval-runtime-run",
        command_id="concurrent-approval-runtime-start",
    )
    initial_events = await _collect(runtime.dispatch(start, scope=_scope()))
    command = ResolveApproval(
        run_id=start.run_id,
        approval_id=initial_events[-1].approval_id,
        approved=True,
        command_id="concurrent-approval-runtime-resolve",
    )

    first, second = await asyncio.gather(
        _collect(runtime.dispatch(command, scope=_scope())),
        _collect(runtime.dispatch(command, scope=_scope())),
    )

    assert first == second
    assert len(first) == 1
    assert isinstance(first[0], RunCompleted)
    replayed = await _collect(
        runtime.replay(start.run_id, after_sequence=0, scope=_scope())
    )
    assert sum(isinstance(event, RunCompleted) for event in replayed) == 1


@pytest.mark.asyncio
async def test_duplicate_approval_command_id_with_changed_payload_conflicts():
    adapter = _DatasetAdapter()
    runtime = JourneyAuditRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=InMemorySaver(),
    )
    start = StartTurn(
        message="Train dataset dataset-1 target target",
        run_id="approval-conflict-run",
        command_id="approval-conflict-start",
    )
    initial_events = await _collect(runtime.dispatch(start, scope=_scope()))
    approved = ResolveApproval(
        run_id=start.run_id,
        approval_id=initial_events[-1].approval_id,
        approved=True,
        command_id="approval-conflict-command",
    )
    await _collect(runtime.dispatch(approved, scope=_scope()))

    with pytest.raises(RuntimeCommandConflict):
        await _collect(
            runtime.dispatch(
                ResolveApproval(
                    run_id=start.run_id,
                    approval_id=initial_events[-1].approval_id,
                    approved=False,
                    command_id=approved.command_id,
                ),
                scope=_scope(),
            )
        )


@pytest.mark.asyncio
async def test_stale_approval_id_is_denied_before_graph_resume():
    adapter = _DatasetAdapter()
    runtime = JourneyAuditRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=InMemorySaver(),
    )
    start = StartTurn(
        message="Train dataset dataset-1 target target",
        run_id="stale-runtime-approval-run",
        command_id="stale-runtime-approval-start",
    )
    initial_events = await _collect(runtime.dispatch(start, scope=_scope()))

    with pytest.raises(RuntimeCommandConflict):
        await _collect(
            runtime.dispatch(
                ResolveApproval(
                    run_id=start.run_id,
                    approval_id="approval-stale",
                    approved=True,
                    command_id="stale-runtime-approval-resolve",
                ),
                scope=_scope(),
            )
        )

    replayed = await _collect(
        runtime.replay(start.run_id, after_sequence=0, scope=_scope())
    )
    assert replayed == initial_events
    assert isinstance(replayed[-1], ApprovalRequired)


@pytest.mark.asyncio
async def test_closing_event_consumer_does_not_create_business_cancellation():
    adapter = _DatasetAdapter()
    runtime = JourneyAuditRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=InMemorySaver(),
    )
    start = StartTurn(
        message="Train dataset dataset-1 target target",
        run_id="consumer-disconnect-run",
        command_id="consumer-disconnect-start",
    )
    stream = runtime.dispatch(start, scope=_scope())

    first_event = await anext(stream)
    waiting_duplicate = asyncio.create_task(
        _collect(runtime.dispatch(start, scope=_scope()))
    )
    await asyncio.sleep(0)
    assert not waiting_duplicate.done()
    await stream.aclose()
    with pytest.raises(RuntimeCommandConflict):
        await waiting_duplicate
    replayed = await _collect(
        runtime.replay(start.run_id, after_sequence=0, scope=_scope())
    )

    assert replayed == [first_event]
    assert not any(
        event.type in {"run_cancelled", "run_completed"} for event in replayed
    )
    with pytest.raises(RuntimeCommandConflict):
        await _collect(runtime.dispatch(start, scope=_scope()))


@pytest.mark.asyncio
async def test_closing_approval_stream_marks_command_for_reconciliation():
    adapter = _DatasetAdapter()
    runtime = JourneyAuditRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=InMemorySaver(),
    )
    start = StartTurn(
        message="Train dataset dataset-1 target target",
        run_id="approval-disconnect-run",
        command_id="approval-disconnect-start",
    )
    initial_events = await _collect(runtime.dispatch(start, scope=_scope()))
    edit = ResolveApproval(
        run_id=start.run_id,
        approval_id=initial_events[-1].approval_id,
        approved=False,
        command_id="approval-disconnect-edit",
        response={"decision": "edit", "changes": {"metric": "f1"}},
    )
    stream = runtime.dispatch(edit, scope=_scope())

    first_edit_event = await anext(stream)
    waiting_duplicate = asyncio.create_task(
        _collect(runtime.dispatch(edit, scope=_scope()))
    )
    await asyncio.sleep(0)
    assert not waiting_duplicate.done()
    await stream.aclose()
    with pytest.raises(RuntimeCommandConflict):
        await waiting_duplicate

    assert first_edit_event.type == "artifact_produced"
    with pytest.raises(RuntimeCommandConflict):
        await _collect(runtime.dispatch(edit, scope=_scope()))
    replayed = await _collect(
        runtime.replay(start.run_id, after_sequence=0, scope=_scope())
    )
    assert replayed[-1] == first_edit_event
    assert not any(event.type == "run_cancelled" for event in replayed)


@pytest.mark.asyncio
async def test_mongo_restart_replays_waiting_run_and_resolves_once():
    uri = os.getenv("HAGENT_TEST_MONGODB_URI")
    if not uri:
        pytest.skip("HAGENT_TEST_MONGODB_URI chưa được cấu hình")
    from hagent.agent.runtime import MongoRuntimeEventStore

    db_name = f"hagent_experiment_runtime_{uuid.uuid4().hex}"
    adapter = _DatasetAdapter()
    snapshot = _snapshot(adapter)
    start = StartTurn(
        message="Train dataset dataset-1 target target",
        run_id="mongo-approval-runtime-run",
        command_id="mongo-approval-runtime-start",
    )
    first_persistence = create_journey_persistence(
        mode="mongodb",
        mongodb_uri=uri,
        db_name=db_name,
        checkpoint_collection_name="checkpoints",
        writes_collection_name="checkpoint_writes",
    )
    first_store = MongoRuntimeEventStore.connect(
        uri,
        db_name=db_name,
        collection_name="runs",
    )
    try:
        first_runtime = JourneyAuditRuntime(
            capability_snapshot=snapshot,
            event_store=first_store,
            checkpointer=first_persistence.checkpointer,
        )
        initial_events = await _collect(first_runtime.dispatch(start, scope=_scope()))
        assert isinstance(initial_events[-1], ApprovalRequired)
    finally:
        first_store.close()
        first_persistence.close()

    recreated_persistence = create_journey_persistence(
        mode="mongodb",
        mongodb_uri=uri,
        db_name=db_name,
        checkpoint_collection_name="checkpoints",
        writes_collection_name="checkpoint_writes",
    )
    recreated_store = MongoRuntimeEventStore.connect(
        uri,
        db_name=db_name,
        collection_name="runs",
    )
    try:
        recreated_runtime = JourneyAuditRuntime(
            capability_snapshot=snapshot,
            event_store=recreated_store,
            checkpointer=recreated_persistence.checkpointer,
        )
        repeated_start = await _collect(
            recreated_runtime.dispatch(start, scope=_scope())
        )
        assert repeated_start == initial_events

        approval = ResolveApproval(
            run_id=start.run_id,
            approval_id=initial_events[-1].approval_id,
            approved=True,
            command_id="mongo-approval-runtime-resolve",
            response={},
        )
        resolved, repeated = await asyncio.gather(
            _collect(
                recreated_runtime.dispatch(
                    approval,
                    scope=_scope(credential="mongo-runtime-approval-sentinel"),
                )
            ),
            _collect(
                recreated_runtime.dispatch(
                    approval,
                    scope=_scope(credential="mongo-runtime-approval-sentinel"),
                )
            ),
        )

        assert resolved == repeated
        assert resolved[-1].result["status"] == "approved"
        assert adapter.calls == 1
        client = MongoClient(uri, serverSelectionTimeoutMS=2000, tz_aware=True)
        try:
            document = client[db_name]["runs"].find_one({"_id": start.run_id})
            raw_bson = BSON.encode(document)
        finally:
            client.close()
        approval_command = next(
            item
            for item in document["commands"]
            if item["command_id"] == approval.command_id
        )
        assert document["status"] == "terminal"
        assert approval_command["status"] == "completed"
        assert len(approval_command["fingerprint"]) == 64
        assert "response" not in approval_command
        assert b"mongo-runtime-approval-sentinel" not in raw_bson
    finally:
        recreated_store.close()
        recreated_persistence.close()
        client = MongoClient(uri, serverSelectionTimeoutMS=2000)
        try:
            client.drop_database(db_name)
        finally:
            client.close()


@pytest.mark.asyncio
async def test_mongo_crash_after_command_claim_fails_closed_without_resume():
    uri = os.getenv("HAGENT_TEST_MONGODB_URI")
    if not uri:
        pytest.skip("HAGENT_TEST_MONGODB_URI chưa được cấu hình")
    from hagent.agent.runtime import MongoRuntimeEventStore

    db_name = f"hagent_approval_claim_crash_{uuid.uuid4().hex}"
    adapter = _DatasetAdapter()
    start = StartTurn(
        message="Train dataset dataset-1 target target",
        run_id="claim-crash-run",
        command_id="claim-crash-start",
    )
    persistence = create_journey_persistence(
        mode="mongodb",
        mongodb_uri=uri,
        db_name=db_name,
        checkpoint_collection_name="checkpoints",
        writes_collection_name="checkpoint_writes",
    )
    store = MongoRuntimeEventStore.connect(
        uri,
        db_name=db_name,
        collection_name="runs",
    )
    try:
        runtime = JourneyAuditRuntime(
            capability_snapshot=_snapshot(adapter),
            event_store=store,
            checkpointer=persistence.checkpointer,
        )
        initial_events = await _collect(runtime.dispatch(start, scope=_scope()))
        approval = ResolveApproval(
            run_id=start.run_id,
            approval_id=initial_events[-1].approval_id,
            approved=True,
            command_id="claim-crash-resolve",
            response={"token": "claim-crash-command-sentinel"},
        )
        _, _, is_new = store.claim_command(approval, owner_id="owner-1")
        assert is_new
    finally:
        store.close()
        persistence.close()

    recreated_persistence = create_journey_persistence(
        mode="mongodb",
        mongodb_uri=uri,
        db_name=db_name,
        checkpoint_collection_name="checkpoints",
        writes_collection_name="checkpoint_writes",
    )
    recreated_store = MongoRuntimeEventStore.connect(
        uri,
        db_name=db_name,
        collection_name="runs",
    )
    try:
        recreated_runtime = JourneyAuditRuntime(
            capability_snapshot=_snapshot(adapter),
            event_store=recreated_store,
            checkpointer=recreated_persistence.checkpointer,
        )
        with pytest.raises(RuntimeCommandConflict):
            await _collect(recreated_runtime.dispatch(approval, scope=_scope()))

        client = MongoClient(uri, serverSelectionTimeoutMS=2000, tz_aware=True)
        try:
            document = client[db_name]["runs"].find_one({"_id": start.run_id})
            raw_bson = BSON.encode(document)
        finally:
            client.close()
        assert document["status"] == "resuming"
        assert "terminal_type" not in document
        assert not any(
            event.type.startswith("run_") and event.type != "run_started"
            for event in initial_events
        )
        assert b"claim-crash-command-sentinel" not in raw_bson
    finally:
        recreated_store.close()
        recreated_persistence.close()
        client = MongoClient(uri, serverSelectionTimeoutMS=2000)
        try:
            client.drop_database(db_name)
        finally:
            client.close()


def test_mongo_command_index_allows_legacy_documents_without_command_array():
    uri = os.getenv("HAGENT_TEST_MONGODB_URI")
    if not uri:
        pytest.skip("HAGENT_TEST_MONGODB_URI chưa được cấu hình")
    from hagent.agent.runtime import MongoRuntimeEventStore

    db_name = f"hagent_index_migration_{uuid.uuid4().hex}"
    client = MongoClient(uri, serverSelectionTimeoutMS=2000, tz_aware=True)
    try:
        collection = client[db_name]["runs"]
        collection.insert_many(
            [
                {
                    "_id": "legacy-run-1",
                    "owner_id": "owner-1",
                    "command_id": "legacy-command-1",
                },
                {
                    "_id": "legacy-run-2",
                    "owner_id": "owner-1",
                    "command_id": "legacy-command-2",
                },
            ]
        )
        store = MongoRuntimeEventStore.connect(
            uri,
            db_name=db_name,
            collection_name="runs",
        )
        try:
            index = collection.index_information()["uq_runtime_owner_all_commands"]
            assert index["partialFilterExpression"] == {
                "commands.command_id": {"$exists": True}
            }
        finally:
            store.close()
    finally:
        client.drop_database(db_name)
        client.close()


@pytest.mark.asyncio
async def test_mongo_disconnect_mid_approval_requires_reconciliation_after_restart():
    uri = os.getenv("HAGENT_TEST_MONGODB_URI")
    if not uri:
        pytest.skip("HAGENT_TEST_MONGODB_URI chưa được cấu hình")
    from hagent.agent.runtime import MongoRuntimeEventStore

    db_name = f"hagent_disconnect_{uuid.uuid4().hex}"
    adapter = _DatasetAdapter()
    snapshot = _snapshot(adapter)
    persistence = create_journey_persistence(
        mode="mongodb",
        mongodb_uri=uri,
        db_name=db_name,
        checkpoint_collection_name="checkpoints",
        writes_collection_name="checkpoint_writes",
    )
    store = MongoRuntimeEventStore.connect(
        uri,
        db_name=db_name,
        collection_name="runs",
    )
    start = StartTurn(
        message="Train dataset dataset-1 target target",
        run_id="mongo-disconnect-run",
        command_id="mongo-disconnect-start",
    )
    try:
        runtime = JourneyAuditRuntime(
            capability_snapshot=snapshot,
            event_store=store,
            checkpointer=persistence.checkpointer,
        )
        initial_events = await _collect(runtime.dispatch(start, scope=_scope()))
        edit = ResolveApproval(
            run_id=start.run_id,
            approval_id=initial_events[-1].approval_id,
            approved=False,
            command_id="mongo-disconnect-edit",
            response={"decision": "edit", "changes": {"metric": "f1"}},
        )
        stream = runtime.dispatch(edit, scope=_scope())
        first_edit_event = await anext(stream)
        await stream.aclose()
        assert first_edit_event.type == "artifact_produced"
    finally:
        store.close()
        persistence.close()

    recreated_persistence = create_journey_persistence(
        mode="mongodb",
        mongodb_uri=uri,
        db_name=db_name,
        checkpoint_collection_name="checkpoints",
        writes_collection_name="checkpoint_writes",
    )
    recreated_store = MongoRuntimeEventStore.connect(
        uri,
        db_name=db_name,
        collection_name="runs",
    )
    try:
        recreated_runtime = JourneyAuditRuntime(
            capability_snapshot=snapshot,
            event_store=recreated_store,
            checkpointer=recreated_persistence.checkpointer,
        )
        with pytest.raises(RuntimeCommandConflict):
            await _collect(recreated_runtime.dispatch(edit, scope=_scope()))
        client = MongoClient(uri, serverSelectionTimeoutMS=2000, tz_aware=True)
        try:
            document = client[db_name]["runs"].find_one({"_id": start.run_id})
        finally:
            client.close()
        edit_command = next(
            item
            for item in document["commands"]
            if item["command_id"] == edit.command_id
        )
        assert document["status"] == "needs_reconciliation"
        assert edit_command["status"] == "needs_reconciliation"
        assert "terminal_type" not in document
    finally:
        recreated_store.close()
        recreated_persistence.close()
        client = MongoClient(uri, serverSelectionTimeoutMS=2000)
        try:
            client.drop_database(db_name)
        finally:
            client.close()
