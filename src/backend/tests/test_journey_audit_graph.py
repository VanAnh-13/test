from __future__ import annotations

import re

import pytest

from hagent.agent.runtime import RequestScope, StartTurn


class _DatasetAdapter:
    def __init__(self, output=None, error=None):
        self.output = output or {
            "_id": "dataset-1",
            "columns": ["feature", "target"],
            "target": "target",
            "missingness": {"feature": 0.0, "target": 0.0},
            "class_balance": {"yes": 0.5, "no": 0.5},
        }
        self.error = error
        self.calls = []

    async def invoke(self, capability_id, arguments, *, scope):
        self.calls.append((capability_id, dict(arguments), scope))
        if self.error is not None:
            raise self.error
        return self.output


def _snapshot(adapter):
    from hagent.agent.capabilities.catalog import CapabilityCatalog
    from hagent.agent.capabilities.models import CapabilityDescriptor

    descriptor = CapabilityDescriptor(
        id="automl.dataset.inspect@1",
        input_schema={
            "type": "object",
            "required": ["dataset_id"],
            "properties": {"dataset_id": {"type": "string", "minLength": 1}},
            "additionalProperties": False,
        },
        output_schema={"type": "object"},
        effect="read",
        required_scopes=frozenset({"automl.dataset.read"}),
        provider_id="fake-dataset",
    )
    catalog = CapabilityCatalog()
    catalog.register_provider("fake-dataset", [descriptor], adapter)
    return catalog.snapshot()


def _scope(owner="owner-1", credential="runtime-sentinel"):
    return RequestScope(
        principal_id=owner,
        credential=credential,
        services={
            "scopes": ("automl.dataset.read",),
            "max_training_jobs": 2,
        },
    )


async def _collect(stream):
    return [event async for event in stream]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "dataset_id", "target"),
    [
        ("Audit dataset dataset-en target target", "dataset-en", "target"),
        ("Kiểm tra dataset dataset-vi cột mục tiêu target", "dataset-vi", "target"),
    ],
)
async def test_audit_graph_interprets_bilingual_goal_without_persisting_context(
    message,
    dataset_id,
    target,
):
    from hagent.agent.journey.graph import build_audit_graph, initial_audit_state
    from hagent.agent.runtime.context import GraphRequestContext

    adapter = _DatasetAdapter()
    snapshot = _snapshot(adapter)
    initial = initial_audit_state(
        message=message,
        run_id="run-direct",
        owner_id="state-owner-spoof",
    )
    result = (
        await build_audit_graph()
        .compile()
        .ainvoke(
            initial,
            context=GraphRequestContext(
                principal_id="owner-1",
                credential="runtime-sentinel",
                services={"scopes": ("automl.dataset.read",)},
                capability_snapshot=snapshot,
            ),
        )
    )

    assert result["goal"]["dataset_id"] == dataset_id
    assert result["goal"]["target_column"] == target
    assert result["artifact"].owner_id == "owner-1"
    assert result["artifact"].dataset_id == dataset_id
    assert [verdict.checker for verdict in result["verdicts"]] == [
        "contract",
        "statistical",
        "policy",
    ]
    assert "runtime-sentinel" not in repr(result)
    assert not {"credential", "services", "capability_snapshot"} & set(result)
    assert [call[0] for call in adapter.calls] == ["automl.dataset.inspect@1"]


@pytest.mark.asyncio
async def test_audit_runtime_emits_monotonic_replayable_events_and_no_mutation():
    from hagent.agent.journey.runtime_adapter import JourneyAuditRuntime
    from hagent.agent.runtime import RunCompleted

    adapter = _DatasetAdapter()
    runtime = JourneyAuditRuntime(capability_snapshot=_snapshot(adapter))
    command = StartTurn(
        message="Audit dataset dataset-1 target target",
        run_id="run-audit",
        command_id="command-audit",
    )

    events = await _collect(runtime.dispatch(command, scope=_scope()))
    duplicate = await _collect(runtime.dispatch(command, scope=_scope()))
    replayed = await _collect(
        runtime.replay("run-audit", after_sequence=2, scope=_scope())
    )

    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert [event.type for event in duplicate] == [event.type for event in events]
    assert [event.sequence for event in replayed] == list(range(3, len(events) + 1))
    assert sum(isinstance(event, RunCompleted) for event in events) == 1
    assert events[-1].result["status"] == "completed"
    assert "runtime-sentinel" not in repr(events)
    assert [call[0] for call in adapter.calls] == ["automl.dataset.inspect@1"]
    assert not any("train" in call[0] or "predict" in call[0] for call in adapter.calls)


@pytest.mark.asyncio
async def test_missing_target_produces_artifact_and_deterministic_blocker():
    from hagent.agent.journey.runtime_adapter import JourneyAuditRuntime

    adapter = _DatasetAdapter(
        output={"_id": "dataset-1", "columns": ["feature"], "missingness": {}}
    )
    runtime = JourneyAuditRuntime(capability_snapshot=_snapshot(adapter))

    events = await _collect(
        runtime.dispatch(
            StartTurn(
                message="Audit dataset dataset-1",
                run_id="run-missing-target",
                command_id="command-missing-target",
            ),
            scope=_scope(),
        )
    )

    check_events = [event for event in events if event.type == "check_completed"]
    assert any(
        finding["code"] == "TARGET_REQUIRED"
        for event in check_events
        for finding in event.details["findings"]
    )
    assert any(event.type == "artifact_produced" for event in events)
    assert events[-1].type == "run_completed"
    assert events[-1].result["status"] == "blocked"


@pytest.mark.asyncio
async def test_upstream_failure_is_safe_terminal_and_replay_is_owner_scoped():
    from hagent.agent.capabilities.models import CapabilityInvocationError
    from hagent.agent.journey.runtime_adapter import JourneyAuditRuntime
    from hagent.agent.runtime import RuntimeAccessDenied

    adapter = _DatasetAdapter(
        error=CapabilityInvocationError(
            "PROVIDER_FAILURE",
            "upstream leaked runtime-sentinel",
        )
    )
    runtime = JourneyAuditRuntime(capability_snapshot=_snapshot(adapter))
    events = await _collect(
        runtime.dispatch(
            StartTurn(
                message="Audit dataset dataset-1 target target",
                run_id="run-upstream-failure",
                command_id="command-upstream-failure",
            ),
            scope=_scope(),
        )
    )

    assert events[-1].type == "run_failed"
    assert events[-1].error_code == "PROVIDER_FAILURE"
    assert "runtime-sentinel" not in repr(events)
    assert (
        sum(
            event.type in {"run_failed", "run_completed", "run_cancelled"}
            for event in events
        )
        == 1
    )
    with pytest.raises(RuntimeAccessDenied):
        await _collect(
            runtime.replay(
                "run-upstream-failure",
                after_sequence=0,
                scope=_scope(owner="other-owner"),
            )
        )


def test_journey_state_contract_has_no_runtime_authority_fields():
    from hagent.agent.journey.state import JourneyAuditState

    fields = set(JourneyAuditState.__annotations__)

    assert not fields & {
        "credential",
        "token",
        "user_token",
        "services",
        "capability_snapshot",
    }
    assert re.fullmatch(r"[a-z_]+", "_".join(sorted(fields)))
