from __future__ import annotations

import importlib.metadata
import os
import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from hagent.agent.runtime import RequestScope, StartTurn


def test_langgraph_checkpoint_dependency_versions_are_pinned_and_importable():
    expected_versions = {
        "langgraph": "1.2.9",
        "langgraph-checkpoint": "4.1.1",
        "langgraph-checkpoint-mongodb": "0.4.0",
        "pymongo": "4.16.0",
    }

    assert {
        package: importlib.metadata.version(package) for package in expected_versions
    } == expected_versions


class _DatasetAdapter:
    def __init__(self):
        self.calls = 0

    async def invoke(self, capability_id, arguments, *, scope):
        self.calls += 1
        return {
            "_id": arguments["dataset_id"],
            "columns": ["feature", "target"],
            "target": "target",
            "missingness": {},
            "class_balance": {},
        }


def _snapshot(adapter):
    from hagent.agent.capabilities.catalog import CapabilityCatalog
    from hagent.agent.capabilities.models import CapabilityDescriptor

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
        provider_id="checkpoint-fake",
    )
    catalog = CapabilityCatalog()
    catalog.register_provider("checkpoint-fake", [descriptor], adapter)
    return catalog.snapshot()


def _scope(owner="owner-1"):
    return RequestScope(
        principal_id=owner,
        credential="checkpoint-sentinel",
        services={
            "scopes": ("automl.dataset.read",),
            "service_handle": object(),
        },
    )


async def _collect(stream):
    return [event async for event in stream]


def test_thread_config_is_owner_scoped_hashed_and_versioned():
    from hagent.agent.journey.persistence import (
        JOURNEY_CHECKPOINT_NAMESPACE,
        JOURNEY_DURABILITY,
        journey_checkpoint_config,
        journey_thread_id,
    )

    assert journey_thread_id("owner-1", "run-1") == (
        "ef7423f70fcd162210ff9276b4c9eb7d3af06cc014fb8867772ec755482d72ac"
    )
    assert journey_thread_id("owner-2", "run-1") != journey_thread_id(
        "owner-1",
        "run-1",
    )
    config = journey_checkpoint_config(principal_id="owner-1", run_id="run-1")

    assert config == {
        "configurable": {
            "thread_id": journey_thread_id("owner-1", "run-1"),
            "checkpoint_ns": JOURNEY_CHECKPOINT_NAMESPACE,
        }
    }
    assert JOURNEY_CHECKPOINT_NAMESPACE == "journey-v1"
    assert JOURNEY_DURABILITY == "sync"
    assert "owner-1" not in repr(config)
    assert "run-1" not in repr(config)


def test_memory_persistence_requires_explicit_dev_test_permission():
    from hagent.agent.journey.persistence import (
        JourneyPersistenceError,
        create_journey_persistence,
    )

    with pytest.raises(JourneyPersistenceError, match="explicitly allowed"):
        create_journey_persistence(mode="memory")

    handle = create_journey_persistence(mode="memory", allow_memory=True)
    try:
        assert isinstance(handle.checkpointer, InMemorySaver)
        assert handle.mode == "memory"
    finally:
        handle.close()


def test_mongodb_mode_is_fail_closed_and_uri_is_repr_safe():
    from hagent.agent.journey.persistence import (
        JourneyPersistenceError,
        create_journey_persistence,
    )

    with pytest.raises(JourneyPersistenceError, match="URI is required"):
        create_journey_persistence(mode="mongodb", mongodb_uri=None)
    with pytest.raises(JourneyPersistenceError, match="unavailable") as exc_info:
        create_journey_persistence(
            mode="mongodb",
            mongodb_uri="mongodb://user:secret@127.0.0.1:1/test",
            server_selection_timeout_ms=50,
        )
    assert "secret" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_compiled_graph_writes_checkpoint_with_versioned_namespace():
    from hagent.agent.journey.graph import compile_audit_graph, initial_audit_state
    from hagent.agent.journey.persistence import journey_checkpoint_config
    from hagent.agent.runtime.context import GraphRequestContext

    checkpointer = InMemorySaver()
    graph = compile_audit_graph(checkpointer=checkpointer)
    result = await graph.ainvoke(
        initial_audit_state(
            message="Audit dataset dataset-1 target target",
            run_id="direct-checkpoint-run",
        ),
        config=journey_checkpoint_config(
            principal_id="owner-1",
            run_id="direct-checkpoint-run",
        ),
        context=GraphRequestContext(
            principal_id="owner-1",
            credential="checkpoint-sentinel",
            services={"scopes": ("automl.dataset.read",)},
            capability_snapshot=_snapshot(_DatasetAdapter()),
        ),
        durability="sync",
    )

    assert result["result"]["status"] == "completed"
    stored = list(
        checkpointer.list(
            journey_checkpoint_config(
                principal_id="owner-1",
                run_id="direct-checkpoint-run",
            )
        )
    )
    assert stored


def test_checkpoint_serializer_restores_only_allowlisted_journey_types():
    from hagent.agent.journey.artifacts import EvidenceRef
    from hagent.agent.journey.persistence import JourneyCheckpointSerializer

    serializer = JourneyCheckpointSerializer()
    evidence = EvidenceRef(
        evidence_id="evidence-1",
        source="dataset_profile",
        content_hash="a" * 64,
        summary="Schema profile",
    )

    restored = serializer.loads_typed(serializer.dumps_typed(evidence))
    untrusted = {
        "__hagent_journey_checkpoint_type_v1__": "os.system",
        "value": {"command": "whoami"},
    }
    restored_untrusted = serializer.loads_typed(serializer.dumps_typed(untrusted))

    assert restored == evidence
    assert restored_untrusted == untrusted


@pytest.mark.asyncio
async def test_runtime_checkpoint_survives_runtime_recreation_without_authority_data():
    from hagent.agent.journey.runtime_adapter import JourneyAuditRuntime

    checkpointer = InMemorySaver()
    adapter = _DatasetAdapter()
    runtime = JourneyAuditRuntime(
        capability_snapshot=_snapshot(adapter),
        checkpointer=checkpointer,
    )
    command = StartTurn(
        message="Audit dataset dataset-1 target target",
        run_id="checkpoint-run",
        command_id="checkpoint-command",
    )
    events = await _collect(runtime.dispatch(command, scope=_scope()))
    recreated = JourneyAuditRuntime(
        capability_snapshot=_snapshot(_DatasetAdapter()),
        checkpointer=checkpointer,
    )
    state = await recreated.get_checkpoint_state(
        run_id="checkpoint-run",
        scope=_scope(),
    )
    other_owner_state = await recreated.get_checkpoint_state(
        run_id="checkpoint-run",
        scope=_scope(owner="other-owner"),
    )

    assert events[-1].type == "run_completed"
    assert state["result"]["status"] == "completed"
    assert other_owner_state == {}
    assert adapter.calls == 1
    assert "checkpoint-sentinel" not in repr(state)
    assert "service_handle" not in repr(state)
    checkpoint_tuples = list(
        checkpointer.list(
            {
                "configurable": {
                    "thread_id": (
                        "c04d52f49a4fb051699eb7bf990ed3ab24e56bcf17efcffc70929a9991abe2f7"
                    ),
                    "checkpoint_ns": "journey-v1",
                }
            }
        )
    )
    assert "checkpoint-sentinel" not in repr(checkpoint_tuples)
    assert "service_handle" not in repr(checkpoint_tuples)


def test_persisted_node_names_are_a_compatibility_contract():
    from hagent.agent.journey.graph import build_audit_graph

    assert set(build_audit_graph().nodes) == {
        "contract_checker",
        "dataset_profiler",
        "finalize",
        "interpret",
        "policy_checker",
        "statistical_checker",
    }


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("HAGENT_TEST_MONGODB_URI"),
    reason="HAGENT_TEST_MONGODB_URI chưa được cấu hình",
)
async def test_real_mongodb_saver_persists_restart_and_owner_isolation():
    from bson import BSON
    from pymongo import MongoClient

    from hagent.agent.journey.persistence import create_journey_persistence
    from hagent.agent.journey.runtime_adapter import JourneyAuditRuntime

    uri = os.environ["HAGENT_TEST_MONGODB_URI"]
    db_name = f"hagent_checkpoint_test_{uuid.uuid4().hex}"
    client = MongoClient(uri, serverSelectionTimeoutMS=2000)
    handle = None
    try:
        handle = create_journey_persistence(
            mode="mongodb",
            mongodb_uri=uri,
            db_name=db_name,
            checkpoint_collection_name="checkpoints",
            writes_collection_name="checkpoint_writes",
            server_selection_timeout_ms=2000,
        )
        adapter = _DatasetAdapter()
        runtime = JourneyAuditRuntime(
            capability_snapshot=_snapshot(adapter),
            checkpointer=handle.checkpointer,
        )
        await _collect(
            runtime.dispatch(
                StartTurn(
                    message="Audit dataset mongo-dataset target target",
                    run_id="mongo-restart-run",
                    command_id="mongo-restart-command",
                ),
                scope=_scope(),
            )
        )
        recreated = JourneyAuditRuntime(
            capability_snapshot=_snapshot(_DatasetAdapter()),
            checkpointer=handle.checkpointer,
        )
        state = await recreated.get_checkpoint_state(
            run_id="mongo-restart-run",
            scope=_scope(),
        )
        foreign_state = await recreated.get_checkpoint_state(
            run_id="mongo-restart-run",
            scope=_scope(owner="other-owner"),
        )
        raw_documents = [
            *client[db_name]["checkpoints"].find({}),
            *client[db_name]["checkpoint_writes"].find({}),
        ]
        raw_bson = b"".join(BSON.encode(document) for document in raw_documents)

        assert state["result"]["status"] == "completed"
        assert foreign_state == {}
        assert adapter.calls == 1
        assert b"checkpoint-sentinel" not in raw_bson
        assert b"service_handle" not in raw_bson
        assert raw_documents
    finally:
        if handle is not None:
            handle.close()
        client.drop_database(db_name)
        client.close()
