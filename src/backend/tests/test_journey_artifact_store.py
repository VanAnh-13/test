"""Regression cho artifact metadata store tách khỏi runtime event retention."""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from pymongo import MongoClient

from hagent.agent.runtime import (
    ArtifactProduced,
    InMemoryRuntimeEventStore,
    RequestScope,
    RunCompleted,
    RuntimeAccessDenied,
    StartTurn,
)


def _artifact_event(
    *,
    owner_id: str = "owner-1",
    run_id: str = "run-1",
    artifact_id: str = "artifact-1",
    sequence: int = 2,
) -> ArtifactProduced:
    return ArtifactProduced(
        run_id=run_id,
        command_id="command-1",
        sequence=sequence,
        created_at="2026-08-11T12:00:00+00:00",
        artifact_type="DatasetAudit",
        artifact={
            "artifact_id": artifact_id,
            "owner_id": owner_id,
            "run_id": run_id,
            "version": 1,
            "lineage": [],
            "evidence": [{"kind": "schema", "source": "dataset-1"}],
        },
    )


def test_memory_store_is_idempotent_immutable_owner_scoped_and_retained():
    from hagent.agent.journey.artifact_store import (
        ArtifactMetadataConflict,
        InMemoryArtifactMetadataStore,
    )

    store = InMemoryArtifactMetadataStore(retention_days=180)
    event = _artifact_event()

    store.put(owner_id="owner-1", event=event)
    store.put(owner_id="owner-1", event=event)
    records = store.list_for_run(owner_id="owner-1", run_id="run-1")

    assert len(records) == 1
    assert records[0].artifact_id == "artifact-1"
    assert records[0].expires_at - records[0].created_at == timedelta(days=180)
    mutable_copy = dict(records[0].payload)
    mutable_copy["version"] = 99
    assert (
        store.list_for_run(owner_id="owner-1", run_id="run-1")[0].payload["version"]
        == 1
    )

    changed = _artifact_event()
    changed.artifact["version"] = 2
    with pytest.raises(ArtifactMetadataConflict):
        store.put(owner_id="owner-1", event=changed)
    with pytest.raises(RuntimeAccessDenied):
        store.list_for_run(owner_id="other-owner", run_id="run-1")

    terminal_at = datetime(2026, 8, 12, 8, tzinfo=UTC)
    store.seal_run(owner_id="owner-1", run_id="run-1", terminal_at=terminal_at)
    sealed = store.list_for_run(owner_id="owner-1", run_id="run-1")[0]
    assert sealed.expires_at == terminal_at + timedelta(days=180)
    assert "artifact-1" not in repr(sealed)
    store.seal_run(owner_id="owner-1", run_id="run-1", terminal_at=terminal_at)
    with pytest.raises(ArtifactMetadataConflict):
        store.seal_run(
            owner_id="owner-1",
            run_id="run-1",
            terminal_at=terminal_at + timedelta(seconds=1),
        )


def test_memory_store_rejects_identity_mismatch_and_sensitive_metadata():
    from hagent.agent.journey.artifact_store import (
        ArtifactMetadataSensitiveData,
        InMemoryArtifactMetadataStore,
    )

    store = InMemoryArtifactMetadataStore()
    with pytest.raises(ValueError, match="identity"):
        store.put(owner_id="other-owner", event=_artifact_event())

    sensitive = _artifact_event(artifact_id="artifact-sensitive")
    sensitive.artifact["api_key"] = "credential-sentinel"
    with pytest.raises(ArtifactMetadataSensitiveData, match="sensitive") as exc_info:
        store.put(owner_id="owner-1", event=sensitive)
    assert "credential-sentinel" not in str(exc_info.value)


class _DatasetAdapter:
    async def invoke(self, capability_id, arguments, *, scope):
        del capability_id, scope
        return {
            "_id": arguments["dataset_id"],
            "columns": ["feature", "target"],
            "target": "target",
            "missingness": {},
            "class_balance": {"yes": 0.5, "no": 0.5},
            "leakage_risks": [],
        }


def _snapshot():
    from hagent.agent.capabilities.catalog import CapabilityCatalog
    from hagent.agent.capabilities.models import CapabilityDescriptor

    descriptor = CapabilityDescriptor(
        id="automl.dataset.inspect@1",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        effect="read",
        required_scopes=frozenset({"automl.dataset.read"}),
        provider_id="artifact-test",
    )
    catalog = CapabilityCatalog()
    catalog.register_provider("artifact-test", [descriptor], _DatasetAdapter())
    return catalog.snapshot()


@pytest.mark.asyncio
async def test_runtime_persists_artifact_before_event_and_seals_before_terminal():
    from hagent.agent.journey.artifact_store import InMemoryArtifactMetadataStore
    from hagent.agent.journey.runtime_adapter import JourneyRuntime

    order = []

    class OrderedArtifactStore(InMemoryArtifactMetadataStore):
        def put(self, *, owner_id, event):
            order.append(("put", event.type))
            return super().put(owner_id=owner_id, event=event)

        def seal_run(self, *, owner_id, run_id, terminal_at):
            order.append(("seal", "terminal"))
            return super().seal_run(
                owner_id=owner_id,
                run_id=run_id,
                terminal_at=terminal_at,
            )

    class OrderedEventStore(InMemoryRuntimeEventStore):
        def append(self, record, event):
            order.append(("event", event.type))
            return super().append(record, event)

    artifact_store = OrderedArtifactStore()
    runtime = JourneyRuntime(
        capability_snapshot=_snapshot(),
        event_store=OrderedEventStore(),
        artifact_store=artifact_store,
    )
    command = StartTurn(
        message="Audit dataset dataset-1 target target",
        run_id="runtime-artifact-run",
        command_id="runtime-artifact-command",
    )
    scope = RequestScope(
        principal_id="owner-1",
        credential="runtime-artifact-credential",
        services={"scopes": ("automl.dataset.read",)},
    )

    events = [event async for event in runtime.dispatch(command, scope=scope)]

    assert isinstance(events[-1], RunCompleted)
    assert order.index(("put", "artifact_produced")) < order.index(
        ("event", "artifact_produced")
    )
    assert order.index(("seal", "terminal")) < order.index(("event", "run_completed"))
    records = artifact_store.list_for_run(
        owner_id="owner-1",
        run_id="runtime-artifact-run",
    )
    assert len(records) == 1
    produced = next(event for event in events if isinstance(event, ArtifactProduced))
    assert records[0].payload["artifact_id"] == produced.artifact["artifact_id"]
    assert "runtime-artifact-credential" not in repr(records)


@pytest.fixture
def mongo_artifact_store():
    from hagent.agent.journey.artifact_store import MongoArtifactMetadataStore

    uri = os.getenv("HAGENT_TEST_MONGODB_URI")
    if not uri:
        pytest.skip("HAGENT_TEST_MONGODB_URI chưa được cấu hình")
    db_name = f"hagent_artifact_test_{uuid.uuid4().hex}"
    store = MongoArtifactMetadataStore.connect(
        uri,
        db_name=db_name,
        collection_name="artifacts",
        retention_days=180,
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


def test_mongo_store_restarts_with_owner_scope_ttl_and_no_secret(
    mongo_artifact_store,
):
    from hagent.agent.journey.artifact_store import (
        ArtifactMetadataConflict,
        MongoArtifactMetadataStore,
    )

    store, uri, db_name = mongo_artifact_store
    event = _artifact_event()
    store.put(owner_id="owner-1", event=event)
    store.close()

    recreated = MongoArtifactMetadataStore.connect(
        uri,
        db_name=db_name,
        collection_name="artifacts",
        retention_days=180,
    )
    try:
        recreated.put(owner_id="owner-1", event=event)
        records = recreated.list_for_run(owner_id="owner-1", run_id="run-1")
        assert len(records) == 1
        with pytest.raises(RuntimeAccessDenied):
            recreated.list_for_run(owner_id="other-owner", run_id="run-1")

        terminal_at = datetime.now(UTC)
        recreated.seal_run(
            owner_id="owner-1",
            run_id="run-1",
            terminal_at=terminal_at,
        )
        with pytest.raises(ArtifactMetadataConflict, match="already sealed"):
            recreated.seal_run(
                owner_id="owner-1",
                run_id="run-1",
                terminal_at=terminal_at + timedelta(seconds=1),
            )
        raw = recreated._collection.find_one({"artifact_id": "artifact-1"})
        assert raw is not None
        assert 179 <= (raw["expires_at"] - terminal_at).days <= 180
        assert "credential-sentinel" not in repr(raw)
        indexes = list(recreated._collection.list_indexes())
        assert any(
            item.get("name") == "ttl_journey_artifact"
            and item.get("expireAfterSeconds") == 0
            for item in indexes
        )
        assert any(
            item.get("name") == "uq_journey_owner_run_artifact"
            and item.get("unique") is True
            for item in indexes
        )
    finally:
        recreated.close()


def test_mongo_outage_is_fail_closed_and_repr_safe():
    from hagent.agent.journey.artifact_store import (
        ArtifactMetadataUnavailable,
        MongoArtifactMetadataStore,
    )

    with pytest.raises(ArtifactMetadataUnavailable, match="unavailable") as exc_info:
        MongoArtifactMetadataStore.connect(
            "mongodb://user:artifact-secret@127.0.0.1:1/test",
            server_selection_timeout_ms=50,
        )

    assert "artifact-secret" not in str(exc_info.value)
