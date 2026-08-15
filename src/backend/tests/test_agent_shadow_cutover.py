"""Regression cho shadow cutover không tạo side effect thứ hai."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from hagent.agent.capabilities.native import HAutoMLNativeAdapter
from hagent.agent.runtime import (
    ApprovalRequired,
    ArtifactProduced,
    CancelRun,
    CheckCompleted,
    EvidenceAdded,
    RequestScope,
    ResolveApproval,
    RunCancelled,
    RunCompleted,
    RunStarted,
    RuntimeCommand,
    RuntimeEvent,
    StartTurn,
)
from hagent.agent.runtime.shadow import ShadowAgentRuntime


def test_shadow_runtime_is_owned_by_runtime_package():
    from hagent.agent import runtime

    agent_dir = Path(__file__).parents[1] / "hagent" / "agent"

    assert ShadowAgentRuntime.__module__ == "hagent.agent.runtime.shadow"
    assert runtime.ShadowAgentRuntime is ShadowAgentRuntime
    assert {
        "ReportSink",
        "RuntimeObservation",
        "ShadowAgentRuntime",
        "ShadowComparisonReport",
    } <= set(runtime.__all__)
    assert not (agent_dir / "shadow_runtime.py").exists()


def _identity(*, run_id: str, command_id: str, sequence: int) -> dict:
    return {
        "run_id": run_id,
        "command_id": command_id,
        "sequence": sequence,
        "created_at": "2026-08-09T00:00:00+00:00",
    }


class _PrimaryRuntime:
    def __init__(self) -> None:
        self.events: list[RuntimeEvent] = []

    async def dispatch(
        self,
        command: RuntimeCommand,
        *,
        scope: RequestScope,
    ) -> AsyncIterator[RuntimeEvent]:
        assert isinstance(command, StartTurn)
        assert scope.credential
        events: tuple[RuntimeEvent, ...] = (
            RunStarted(
                **_identity(
                    run_id=command.run_id, command_id=command.command_id, sequence=1
                )
            ),
            ArtifactProduced(
                **_identity(
                    run_id=command.run_id, command_id=command.command_id, sequence=2
                ),
                artifact_type="response_delta",
                artifact={"raw": "primary-output-secret"},
            ),
            EvidenceAdded(
                **_identity(
                    run_id=command.run_id, command_id=command.command_id, sequence=3
                ),
                evidence_type="tool_result",
                evidence={"raw": "primary-evidence-secret"},
            ),
            CheckCompleted(
                **_identity(
                    run_id=command.run_id, command_id=command.command_id, sequence=4
                ),
                checker="legacy_plan_executor",
                verdict="observed",
            ),
            RunCompleted(
                **_identity(
                    run_id=command.run_id, command_id=command.command_id, sequence=5
                ),
                result={
                    "status": "success",
                    "raw": "primary-result-secret",
                    "cost_metrics": {"total_tokens": 120, "total_cost": 0.25},
                },
            ),
        )
        self.events.extend(events)
        for event in events:
            yield event

    async def replay(
        self,
        run_id: str,
        *,
        after_sequence: int,
        scope: RequestScope,
    ) -> AsyncIterator[RuntimeEvent]:
        for event in self.events:
            if (
                event.run_id == run_id
                and event.sequence > after_sequence
                and scope.principal_id == "owner-1"
            ):
                yield event


class _ReadOnlyObserverRuntime:
    def __init__(self) -> None:
        self.commands: list[RuntimeCommand] = []

    async def dispatch(
        self,
        command: RuntimeCommand,
        *,
        scope: RequestScope,
    ) -> AsyncIterator[RuntimeEvent]:
        self.commands.append(command)
        if isinstance(command, ResolveApproval):
            raise AssertionError("Shadow không được resume approval")
        if isinstance(command, CancelRun):
            yield RunCancelled(
                **_identity(
                    run_id=command.run_id,
                    command_id=command.command_id,
                    sequence=6,
                ),
                reason="shadow_observation_complete",
            )
            return
        assert isinstance(command, StartTurn)
        yield RunStarted(
            **_identity(
                run_id=command.run_id, command_id=command.command_id, sequence=1
            )
        )
        yield ArtifactProduced(
            **_identity(
                run_id=command.run_id, command_id=command.command_id, sequence=2
            ),
            artifact_type="ExperimentSpec",
            artifact={"raw": "observer-artifact-secret"},
        )
        yield EvidenceAdded(
            **_identity(
                run_id=command.run_id, command_id=command.command_id, sequence=3
            ),
            evidence_type="dataset_profile",
            evidence={"raw": "observer-evidence-secret"},
        )
        yield CheckCompleted(
            **_identity(
                run_id=command.run_id, command_id=command.command_id, sequence=4
            ),
            checker="policy",
            verdict="passed",
        )
        yield ApprovalRequired(
            **_identity(
                run_id=command.run_id, command_id=command.command_id, sequence=5
            ),
            approval_id="approval-1",
            proposal={"raw": "observer-proposal-secret"},
        )

    async def replay(
        self,
        run_id: str,
        *,
        after_sequence: int,
        scope: RequestScope,
    ) -> AsyncIterator[RuntimeEvent]:
        if False:
            yield


class _FailingObserverRuntime:
    async def dispatch(
        self,
        command: RuntimeCommand,
        *,
        scope: RequestScope,
    ) -> AsyncIterator[RuntimeEvent]:
        raise RuntimeError("observer-exception-secret")
        if False:
            yield

    async def replay(
        self,
        run_id: str,
        *,
        after_sequence: int,
        scope: RequestScope,
    ) -> AsyncIterator[RuntimeEvent]:
        if False:
            yield


class _BlockingObserverRuntime:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.stopped = asyncio.Event()

    async def dispatch(
        self,
        command: RuntimeCommand,
        *,
        scope: RequestScope,
    ) -> AsyncIterator[RuntimeEvent]:
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.stopped.set()
        if False:
            yield


class _DelayedObserverRuntime:
    async def dispatch(
        self,
        command: RuntimeCommand,
        *,
        scope: RequestScope,
    ) -> AsyncIterator[RuntimeEvent]:
        assert isinstance(command, StartTurn)
        await asyncio.sleep(0.05)
        yield RunCompleted(
            **_identity(
                run_id=command.run_id,
                command_id=command.command_id,
                sequence=1,
            ),
            result={
                "status": "success",
                "cost_metrics": {"total_tokens": 60, "total_cost": 0.125},
            },
        )

    async def replay(
        self,
        run_id: str,
        *,
        after_sequence: int,
        scope: RequestScope,
    ) -> AsyncIterator[RuntimeEvent]:
        if False:
            yield


class _UnsafeLabelPrimaryRuntime(_PrimaryRuntime):
    async def dispatch(
        self,
        command: RuntimeCommand,
        *,
        scope: RequestScope,
    ) -> AsyncIterator[RuntimeEvent]:
        events: tuple[RuntimeEvent, ...] = (
            ArtifactProduced(
                **_identity(
                    run_id=command.run_id,
                    command_id=command.command_id,
                    sequence=1,
                ),
                artifact_type="credentialLeakLabel",
                artifact={},
            ),
            RunCompleted(
                **_identity(
                    run_id=command.run_id,
                    command_id=command.command_id,
                    sequence=2,
                ),
                result={
                    "status": "credentialLeakStatus",
                    "cost_metrics": {"total_tokens": 120, "total_cost": 0.25},
                },
            ),
        )
        self.events.extend(events)
        for event in events:
            yield event

    async def replay(
        self,
        run_id: str,
        *,
        after_sequence: int,
        scope: RequestScope,
    ) -> AsyncIterator[RuntimeEvent]:
        if False:
            yield


async def _collect(stream: AsyncIterator[RuntimeEvent]) -> list[RuntimeEvent]:
    return [event async for event in stream]


@pytest.mark.asyncio
async def test_shadow_returns_primary_stream_and_sanitizes_comparison_report():
    primary = _PrimaryRuntime()
    observer = _ReadOnlyObserverRuntime()
    reports = []
    runtime = ShadowAgentRuntime(
        primary=primary,
        observer=observer,
        report_sink=reports.append,
    )
    command = StartTurn(
        message="Train dataset secret-prompt",
        run_id="shadow-run-1",
        command_id="shadow-command-1",
    )
    scope = RequestScope(
        principal_id="owner-1",
        credential="shadow-credential-sentinel",
    )

    events = await _collect(runtime.dispatch(command, scope=scope))
    replayed = await _collect(
        runtime.replay(command.run_id, after_sequence=2, scope=scope)
    )

    assert events == primary.events
    assert replayed == primary.events[2:]
    assert [type(item) for item in observer.commands] == [StartTurn, CancelRun]
    assert len(reports) == 1
    report = reports[0]
    assert report.run_id == command.run_id
    assert report.primary.outcome == "completed:success"
    assert report.primary.artifact_types == ("response_delta",)
    assert report.primary.evidence_types == ("tool_result",)
    assert report.primary.checker_verdicts == ("legacy_plan_executor:observed",)
    assert report.primary.total_tokens == 120
    assert report.primary.total_cost == 0.25
    assert report.observer.outcome == "cancelled:shadow_observation_complete"
    assert report.observer.artifact_types == ("ExperimentSpec",)
    assert report.observer.evidence_types == ("dataset_profile",)
    assert report.observer.checker_verdicts == ("policy:passed",)
    assert report.observer.total_tokens is None
    assert report.observer.total_cost is None
    assert report.outcome_match is False
    serialized = repr(report)
    for secret in (
        command.message,
        scope.credential,
        "primary-output-secret",
        "primary-evidence-secret",
        "primary-result-secret",
        "observer-artifact-secret",
        "observer-evidence-secret",
        "observer-proposal-secret",
    ):
        assert secret not in serialized


@pytest.mark.asyncio
async def test_observer_failure_and_report_sink_failure_do_not_change_primary(
    caplog,
):
    primary = _PrimaryRuntime()
    reports = []

    def failing_sink(report):
        reports.append(report)
        raise RuntimeError("sink-exception-secret")

    runtime = ShadowAgentRuntime(
        primary=primary,
        observer=_FailingObserverRuntime(),
        report_sink=failing_sink,
    )
    command = StartTurn(
        message="Audit dataset private-prompt",
        run_id="shadow-failure-run",
        command_id="shadow-failure-command",
    )
    scope = RequestScope(
        principal_id="owner-1",
        credential="shadow-failure-credential",
    )

    events = await _collect(runtime.dispatch(command, scope=scope))

    assert events == primary.events
    assert reports[0].observer.outcome == "failed:observer_error"
    combined_log = caplog.text
    assert "observer-exception-secret" not in combined_log
    assert "sink-exception-secret" not in combined_log
    assert command.message not in combined_log
    assert scope.credential not in combined_log


@pytest.mark.asyncio
async def test_aclose_cancels_and_waits_for_active_observer():
    observer = _BlockingObserverRuntime()
    runtime = ShadowAgentRuntime(
        primary=_PrimaryRuntime(),
        observer=observer,
    )
    command = StartTurn(
        message="Audit dataset dataset-1",
        run_id="shadow-close-run",
        command_id="shadow-close-command",
    )
    scope = RequestScope(principal_id="owner-1", credential="credential")
    dispatch_task = asyncio.create_task(
        _collect(runtime.dispatch(command, scope=scope))
    )
    await observer.started.wait()

    await runtime.aclose()

    assert observer.stopped.is_set()
    with pytest.raises(asyncio.CancelledError):
        await dispatch_task
    with pytest.raises(RuntimeError, match="closed"):
        await _collect(runtime.dispatch(command, scope=scope))


@pytest.mark.asyncio
async def test_report_excludes_untrusted_labels_and_measures_branches_independently():
    reports = []
    runtime = ShadowAgentRuntime(
        primary=_UnsafeLabelPrimaryRuntime(),
        observer=_DelayedObserverRuntime(),
        report_sink=reports.append,
    )
    command = StartTurn(
        message="Audit dataset dataset-1",
        run_id="shadow-metrics-run",
        command_id="shadow-metrics-command",
    )

    await _collect(
        runtime.dispatch(
            command,
            scope=RequestScope(principal_id="owner-1", credential="credential"),
        )
    )

    report = reports[0]
    assert "credentialLeakLabel" not in repr(report)
    assert "credentialLeakStatus" not in repr(report)
    assert report.primary.artifact_types == ("unknown_artifact",)
    assert report.primary.outcome == "completed:completed"
    assert report.observer.latency_ms > report.primary.latency_ms * 5
    assert report.latency_ratio is not None
    assert report.latency_ratio > 5
    assert report.token_ratio == 0.5
    assert report.cost_ratio == 0.5


def test_handle_closes_storage_even_when_runtime_close_fails():
    from hagent.agent.runtime.factory import (
        AgentRuntimeFactoryError,
        AgentRuntimeHandle,
    )

    closed = []

    class _FailingCloseRuntime:
        def close(self):
            raise RuntimeError("runtime-close-secret")

    class _Resource:
        def close(self):
            closed.append("resource")

    handle = AgentRuntimeHandle(
        mode="shadow",
        runtime=_FailingCloseRuntime(),
        _event_store=_Resource(),
        _persistence=_Resource(),
    )

    with pytest.raises(AgentRuntimeFactoryError, match="could not close") as exc_info:
        handle.close()

    assert closed == ["resource", "resource"]
    assert "runtime-close-secret" not in str(exc_info.value)


class _FactoryInvokers:
    def __init__(self) -> None:
        self.read_calls: list[str] = []
        self.write_calls: list[str] = []

    async def list_datasets(self, _arguments):
        self.read_calls.append("list")
        return [{"_id": "dataset-1"}]

    async def inspect_dataset(self, arguments):
        self.read_calls.append("inspect")
        return {
            "_id": arguments["dataset_id"],
            "columns": ["feature", "target"],
            "target": "target",
            "missingness": {},
            "class_balance": {"0": 0.5, "1": 0.5},
            "leakage_risks": [],
        }

    async def start_training(self, _arguments):
        self.write_calls.append("training_start")
        return {"status": "success", "job_id": "forbidden-job"}

    async def lookup_training(self, _arguments):
        self.read_calls.append("training_lookup")
        return {"found": False}

    async def training_results(self, _arguments):
        self.read_calls.append("training_results")
        return {"status": "running", "job_id": "job-1"}

    def adapter(self) -> HAutoMLNativeAdapter:
        return HAutoMLNativeAdapter(
            list_invoker=self.list_datasets,
            inspect_invoker=self.inspect_dataset,
            training_start_invoker=self.start_training,
            training_lookup_invoker=self.lookup_training,
            training_results_invoker=self.training_results,
        )


@pytest.mark.asyncio
async def test_shadow_factory_uses_read_only_catalog_and_zero_mutation():
    from hagent.agent.runtime.factory import create_agent_runtime

    primary = _PrimaryRuntime()
    invokers = _FactoryInvokers()
    reports = []
    handle = create_agent_runtime(
        mode="shadow",
        persistence_mode="memory",
        allow_memory=True,
        native_adapter=invokers.adapter(),
        legacy_runtime=primary,
        shadow_report_sink=reports.append,
    )
    command = StartTurn(
        message="Train dataset dataset-1 target target",
        run_id="shadow-factory-run",
        command_id="shadow-factory-command",
    )
    scope = RequestScope(
        principal_id="owner-1",
        credential="shadow-factory-credential-sentinel",
        services={
            "scopes": (
                "automl.dataset.read",
                "automl.training.read",
                "automl.training.write",
                "automl.prediction.write",
            )
        },
    )
    try:
        events = await _collect(handle.runtime.dispatch(command, scope=scope))
        replayed = await _collect(
            handle.runtime.replay(command.run_id, after_sequence=0, scope=scope)
        )
    finally:
        handle.close()
        handle.close()

    assert handle.mode == "shadow"
    assert isinstance(handle.runtime, ShadowAgentRuntime)
    assert isinstance(handle.capability_snapshot_digest, str)
    assert events == primary.events
    assert replayed == primary.events
    assert invokers.read_calls == ["list", "inspect"]
    assert invokers.write_calls == []
    assert len(reports) == 1
    assert "shadow-factory-credential-sentinel" not in repr(reports[0])
