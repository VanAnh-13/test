"""Regression cho production composition root của JourneyRuntime."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pymongo import MongoClient

from hagent.agent.capabilities.native import HAutoMLNativeAdapter
from hagent.agent.runtime import (
    ApprovalRequired,
    LegacyGraphRuntime,
    RequestScope,
    RunCompleted,
    StartTurn,
)


def test_runtime_factory_is_owned_by_runtime_package():
    from hagent.agent.runtime import create_agent_runtime
    from hagent.agent.runtime.factory import AgentRuntimeHandle

    agent_dir = Path(__file__).parents[1] / "hagent" / "agent"

    assert create_agent_runtime.__module__ == "hagent.agent.runtime.factory"
    assert AgentRuntimeHandle.__module__ == "hagent.agent.runtime.factory"
    assert not (agent_dir / "runtime_factory.py").exists()


class _NativeInvokers:
    def __init__(self) -> None:
        self.calls = []

    async def list(self, arguments):
        self.calls.append(("list", dict(arguments)))
        return [{"_id": "dataset-1"}]

    async def inspect(self, arguments):
        self.calls.append(("inspect", dict(arguments)))
        return {
            "_id": arguments["dataset_id"],
            "columns": ["feature", "target"],
            "target": "target",
            "missingness": {},
            "class_balance": {"0": 0.5, "1": 0.5},
            "leakage_risks": [],
        }

    async def start(self, arguments):
        self.calls.append(("start", dict(arguments)))
        return {"status": "success", "job_id": "job-1"}

    async def lookup(self, arguments):
        self.calls.append(("lookup", dict(arguments)))
        return {"found": False}

    async def results(self, arguments):
        self.calls.append(("results", dict(arguments)))
        return {"status": "running", "job_id": "job-1"}

    def adapter(self) -> HAutoMLNativeAdapter:
        return HAutoMLNativeAdapter(
            list_invoker=self.list,
            inspect_invoker=self.inspect,
            training_start_invoker=self.start,
            training_lookup_invoker=self.lookup,
            training_results_invoker=self.results,
        )


def _scope() -> RequestScope:
    return RequestScope(
        principal_id="owner-1",
        credential="runtime-factory-credential-sentinel",
        services={
            "scopes": (
                "automl.dataset.read",
                "automl.training.read",
                "automl.training.write",
            )
        },
    )


async def _collect(stream):
    return [event async for event in stream]


def test_legacy_factory_has_no_durable_resources_and_close_is_idempotent():
    from hagent.agent.runtime.factory import create_agent_runtime

    handle = create_agent_runtime(mode="legacy")

    assert isinstance(handle.runtime, LegacyGraphRuntime)
    assert handle.mode == "legacy"
    assert handle.capability_snapshot_digest is None
    handle.close()
    handle.close()


def test_memory_journey_requires_explicit_opt_in():
    from hagent.agent.runtime.factory import (
        AgentRuntimeFactoryError,
        create_agent_runtime,
    )

    with pytest.raises(AgentRuntimeFactoryError, match="unavailable"):
        create_agent_runtime(mode="journey", persistence_mode="memory")


@pytest.mark.asyncio
async def test_memory_journey_freezes_native_catalog_and_runs_audit():
    from hagent.agent.journey.artifact_store import InMemoryArtifactMetadataStore
    from hagent.agent.journey.runtime_adapter import JourneyRuntime
    from hagent.agent.runtime.factory import create_agent_runtime

    invokers = _NativeInvokers()
    handle = create_agent_runtime(
        mode="journey",
        persistence_mode="memory",
        allow_memory=True,
        native_adapter=invokers.adapter(),
    )
    command = StartTurn(
        message="Audit dataset dataset-1 target target",
        run_id="runtime-factory-memory-run",
        command_id="runtime-factory-memory-command",
    )
    try:
        events = await _collect(handle.runtime.dispatch(command, scope=_scope()))
    finally:
        handle.close()

    assert isinstance(handle.runtime, JourneyRuntime)
    assert isinstance(handle._artifact_store, InMemoryArtifactMetadataStore)
    artifacts = handle._artifact_store.list_for_run(
        owner_id="owner-1",
        run_id=command.run_id,
    )
    assert len(artifacts) >= 1
    assert isinstance(handle.capability_snapshot_digest, str)
    assert len(handle.capability_snapshot_digest) == 64
    assert isinstance(events[-1], RunCompleted | ApprovalRequired)
    assert [item[0] for item in invokers.calls] == ["list", "inspect"]
    assert "runtime-factory-credential-sentinel" not in repr(events)


def test_mongo_outage_and_partial_construction_are_repr_safe(monkeypatch):
    from hagent.agent.runtime import RuntimeLedgerUnavailable
    from hagent.agent.runtime import factory as runtime_factory
    from hagent.agent.runtime.factory import AgentRuntimeFactoryError

    closed = []
    fake_persistence = SimpleNamespace(
        checkpointer=InMemorySaver(),
        close=lambda: closed.append("persistence"),
    )
    monkeypatch.setattr(
        runtime_factory,
        "create_journey_persistence",
        lambda **_kwargs: fake_persistence,
    )

    def fail_ledger(*_args, **_kwargs):
        raise RuntimeLedgerUnavailable("Runtime ledger unavailable")

    monkeypatch.setattr(
        runtime_factory.MongoRuntimeEventStore,
        "connect",
        fail_ledger,
    )
    uri = "mongodb://user:runtime-factory-secret@127.0.0.1:1/test"

    with pytest.raises(AgentRuntimeFactoryError, match="unavailable") as exc_info:
        runtime_factory.create_agent_runtime(
            mode="journey",
            mongodb_uri=uri,
        )

    assert closed == ["persistence"]
    assert "runtime-factory-secret" not in str(exc_info.value)


def test_artifact_store_failure_closes_prior_durable_resources(monkeypatch):
    from hagent.agent.journey.artifact_store import ArtifactMetadataUnavailable
    from hagent.agent.runtime import factory as runtime_factory
    from hagent.agent.runtime.factory import AgentRuntimeFactoryError

    closed = []
    fake_persistence = SimpleNamespace(
        checkpointer=InMemorySaver(),
        close=lambda: closed.append("persistence"),
    )
    fake_event_store = SimpleNamespace(close=lambda: closed.append("event"))
    monkeypatch.setattr(
        runtime_factory,
        "create_journey_persistence",
        lambda **_kwargs: fake_persistence,
    )
    monkeypatch.setattr(
        runtime_factory.MongoRuntimeEventStore,
        "connect",
        lambda *_args, **_kwargs: fake_event_store,
    )

    def fail_artifact_store(*_args, **_kwargs):
        raise ArtifactMetadataUnavailable("Artifact metadata store is unavailable")

    monkeypatch.setattr(
        runtime_factory.MongoArtifactMetadataStore,
        "connect",
        fail_artifact_store,
    )

    with pytest.raises(AgentRuntimeFactoryError, match="unavailable"):
        runtime_factory.create_agent_runtime(
            mode="journey",
            mongodb_uri="mongodb://user:factory-secret@127.0.0.1:1/test",
        )

    assert closed == ["event", "persistence"]


@pytest.mark.asyncio
async def test_mongo_factory_recreates_runtime_and_replays_owner_events():
    uri = os.getenv("HAGENT_TEST_MONGODB_URI")
    if not uri:
        pytest.skip("HAGENT_TEST_MONGODB_URI chưa được cấu hình")
    from hagent.agent.runtime.factory import create_agent_runtime

    db_name = f"hagent_runtime_factory_{uuid.uuid4().hex}"
    invokers = _NativeInvokers()
    command = StartTurn(
        message="Audit dataset dataset-1 target target",
        run_id="runtime-factory-mongo-run",
        command_id="runtime-factory-mongo-command",
    )
    first = create_agent_runtime(
        mode="journey",
        mongodb_uri=uri,
        db_name=db_name,
        native_adapter=invokers.adapter(),
    )
    try:
        dispatched = await _collect(first.runtime.dispatch(command, scope=_scope()))
    finally:
        first.close()

    recreated = create_agent_runtime(
        mode="journey",
        mongodb_uri=uri,
        db_name=db_name,
        native_adapter=invokers.adapter(),
    )
    try:
        artifacts = recreated._artifact_store.list_for_run(
            owner_id="owner-1",
            run_id=command.run_id,
        )
        replayed = await _collect(
            recreated.runtime.replay(command.run_id, after_sequence=1, scope=_scope())
        )
        recreated.close()
        recreated.close()
        assert len(artifacts) >= 1
    finally:
        recreated.close()
        client = MongoClient(uri, serverSelectionTimeoutMS=2000, tz_aware=True)
        try:
            client.drop_database(db_name)
        finally:
            client.close()

    assert replayed == dispatched[1:]
    assert "runtime-factory-credential-sentinel" not in repr(replayed)
