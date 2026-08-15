"""Contract test cho interface package ``hagent.agent.runtime``."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_runtime_is_package_with_explicit_contract_surface():
    from hagent.agent import runtime

    public_names = {
        "ActionCompleted",
        "AgentRuntime",
        "ApprovalRequired",
        "ArtifactProduced",
        "CancelRun",
        "CheckCompleted",
        "EvidenceAdded",
        "GraphRequestContext",
        "LegacyGraphRuntime",
        "PlanProposed",
        "RequestScope",
        "ResolveApproval",
        "RunCancelled",
        "RunCompleted",
        "RunFailed",
        "RunStarted",
        "RuntimeCommand",
        "RuntimeEvent",
        "StartTurn",
        "get_agent_runtime",
        "runtime_event_to_dict",
        "set_agent_runtime",
    }

    assert hasattr(runtime, "__path__")
    assert Path(runtime.__file__).name == "__init__.py"
    assert public_names <= set(runtime.__all__)
    assert runtime.StartTurn.__module__ == "hagent.agent.runtime.contracts"
    assert runtime.GraphRequestContext.__module__ == "hagent.agent.runtime.context"
    assert not Path(runtime.__file__).parent.with_suffix(".py").exists()
    assert not (Path(runtime.__file__).parent.parent / "context.py").exists()
    for private_ledger_name in (
        "_CommandRecord",
        "_RunRecord",
        "_command_fingerprint",
        "_event_storage_size",
        "_is_sensitive_key",
    ):
        assert hasattr(runtime, private_ledger_name)


@pytest.mark.asyncio
async def test_runtime_package_keeps_legacy_dispatch_and_replay_behavior():
    from hagent.agent.runtime import (
        LegacyGraphRuntime,
        RequestScope,
        RunCompleted,
        StartTurn,
    )

    async def event_source(command, scope):
        assert scope.principal_id == "owner-1"
        yield {"type": "done", "response": {"message": command.message}}

    runtime = LegacyGraphRuntime(event_source=event_source)
    command = StartTurn(
        message="package migration",
        run_id="runtime-package-run",
        command_id="runtime-package-command",
    )
    scope = RequestScope(principal_id="owner-1")

    events = [event async for event in runtime.dispatch(command, scope=scope)]
    replayed = [
        event
        async for event in runtime.replay(
            command.run_id,
            after_sequence=1,
            scope=scope,
        )
    ]

    assert isinstance(events[-1], RunCompleted)
    assert events[-1].result == {"message": command.message}
    assert replayed == events[1:]


def test_runtime_store_uses_canonical_package_module():
    from hagent.agent.runtime import MongoRuntimeEventStore

    assert MongoRuntimeEventStore.__module__ == "hagent.agent.runtime.store"
    assert not (
        Path(__file__).parents[1] / "hagent" / "agent" / "runtime_store.py"
    ).exists()


def test_orchestration_state_uses_canonical_package_module():
    from hagent.agent.orchestration import AutoMLState, DatasetContext, JobContext

    assert AutoMLState.__module__ == "hagent.agent.orchestration.state"
    assert DatasetContext.__module__ == "hagent.agent.orchestration.state"
    assert JobContext.__module__ == "hagent.agent.orchestration.state"
    assert not (Path(__file__).parents[1] / "hagent" / "agent" / "state.py").exists()


def test_registry_uses_canonical_package_module():
    import hagent.agent.orchestration.registry as canonical_registry

    assert canonical_registry.AgentRegistry.__module__ == (
        "hagent.agent.orchestration.registry"
    )
    assert not (Path(__file__).parents[1] / "hagent" / "agent" / "registry.py").exists()


def test_coordinator_uses_canonical_package_module():
    import hagent.agent.orchestration.coordinator as canonical_coordinator

    assert canonical_coordinator.coordinator_node.__module__ == (
        "hagent.agent.orchestration.coordinator"
    )
    assert not (
        Path(__file__).parents[1] / "hagent" / "agent" / "coordinator.py"
    ).exists()


def test_graph_uses_canonical_package_module():
    import hagent.agent.orchestration.graph as canonical_graph

    assert canonical_graph.run_agent.__module__ == "hagent.agent.orchestration.graph"
    assert not (Path(__file__).parents[1] / "hagent" / "agent" / "graph.py").exists()
