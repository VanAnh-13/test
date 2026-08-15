"""Regression cho CancelRun owner-scoped của durable JourneyRuntime."""

from __future__ import annotations

import os
import uuid

import pytest
from bson import BSON
from langgraph.checkpoint.memory import InMemorySaver
from pymongo import MongoClient

from hagent.agent.capabilities.catalog import CapabilityCatalog
from hagent.agent.capabilities.models import CapabilityDescriptor
from hagent.agent.journey.persistence import create_journey_persistence
from hagent.agent.journey.runtime_adapter import JourneyRuntime
from hagent.agent.runtime import (
    ApprovalRequired,
    CancelRun,
    RequestScope,
    ResolveApproval,
    RunCancelled,
    RuntimeAccessDenied,
    RuntimeCommandConflict,
    StartTurn,
)


class _DatasetAdapter:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, _capability_id, arguments, *, scope):
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
        provider_id="cancel-test",
    )
    catalog = CapabilityCatalog()
    catalog.register_provider("cancel-test", [descriptor], adapter)
    return catalog.snapshot()


def _scope(owner_id: str = "owner-1") -> RequestScope:
    return RequestScope(
        principal_id=owner_id,
        credential="cancel-scope-credential-sentinel",
        services={"scopes": ("automl.dataset.read",)},
    )


async def _collect(stream):
    return [event async for event in stream]


async def _start_waiting_run(runtime: JourneyRuntime, *, suffix: str):
    command = StartTurn(
        message="Train dataset dataset-1 target target",
        run_id=f"cancel-run-{suffix}",
        command_id=f"cancel-start-{suffix}",
    )
    events = await _collect(runtime.dispatch(command, scope=_scope()))
    assert isinstance(events[-1], ApprovalRequired)
    return command, events


@pytest.mark.asyncio
async def test_cancel_is_terminal_idempotent_owner_scoped_and_does_not_resume_graph():
    adapter = _DatasetAdapter()
    runtime = JourneyRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=InMemorySaver(),
    )
    start, initial_events = await _start_waiting_run(runtime, suffix="memory")
    cancel = CancelRun(
        run_id=start.run_id,
        command_id="cancel-command-memory",
        reason="do-not-reflect-this-reason-sentinel",
    )

    cancelled = await _collect(runtime.dispatch(cancel, scope=_scope()))
    duplicate = await _collect(runtime.dispatch(cancel, scope=_scope()))
    replayed = await _collect(
        runtime.replay(start.run_id, after_sequence=0, scope=_scope())
    )

    assert cancelled == duplicate
    assert len(cancelled) == 1
    assert isinstance(cancelled[0], RunCancelled)
    assert cancelled[0].reason == "user_requested"
    assert replayed == initial_events + cancelled
    assert [event.sequence for event in replayed] == list(range(1, len(replayed) + 1))
    assert adapter.calls == 1
    assert "do-not-reflect-this-reason-sentinel" not in repr(replayed)
    assert "cancel-scope-credential-sentinel" not in repr(replayed)

    with pytest.raises(RuntimeAccessDenied):
        await _collect(runtime.dispatch(cancel, scope=_scope(owner_id="other-owner")))
    with pytest.raises(RuntimeCommandConflict):
        await _collect(
            runtime.dispatch(
                ResolveApproval(
                    run_id=start.run_id,
                    approval_id=initial_events[-1].approval_id,
                    approved=True,
                    command_id="approval-after-cancel",
                ),
                scope=_scope(),
            )
        )


@pytest.mark.asyncio
async def test_cancel_replays_after_mongo_runtime_restart_without_sentinel():
    uri = os.getenv("HAGENT_TEST_MONGODB_URI")
    if not uri:
        pytest.skip("HAGENT_TEST_MONGODB_URI chưa được cấu hình")
    from hagent.agent.runtime import MongoRuntimeEventStore

    db_name = f"hagent_journey_cancel_{uuid.uuid4().hex}"
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
    start = None
    cancel = None
    try:
        runtime = JourneyRuntime(
            capability_snapshot=snapshot,
            checkpointer=persistence.checkpointer,
            event_store=store,
        )
        start, _ = await _start_waiting_run(runtime, suffix="mongo")
        cancel = CancelRun(
            run_id=start.run_id,
            command_id="cancel-command-mongo",
            reason="mongo-cancel-reason-sentinel",
        )
        cancelled = await _collect(runtime.dispatch(cancel, scope=_scope()))
        assert isinstance(cancelled[-1], RunCancelled)
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
    client = MongoClient(uri, serverSelectionTimeoutMS=2000, tz_aware=True)
    try:
        recreated_runtime = JourneyRuntime(
            capability_snapshot=snapshot,
            checkpointer=recreated_persistence.checkpointer,
            event_store=recreated_store,
        )
        replayed = await _collect(
            recreated_runtime.replay(start.run_id, after_sequence=0, scope=_scope())
        )
        duplicate = await _collect(recreated_runtime.dispatch(cancel, scope=_scope()))
        document = client[db_name]["runs"].find_one({"_id": start.run_id})
        raw_document = BSON.encode(document)

        assert duplicate == [replayed[-1]]
        assert sum(isinstance(event, RunCancelled) for event in replayed) == 1
        assert replayed[-1].reason == "user_requested"
        assert b"mongo-cancel-reason-sentinel" not in raw_document
        assert b"cancel-scope-credential-sentinel" not in raw_document
    finally:
        recreated_store.close()
        recreated_persistence.close()
        client.drop_database(db_name)
        client.close()
