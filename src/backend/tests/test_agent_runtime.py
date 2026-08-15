"""Kiểm thử contract cho seam tương thích AgentRuntime công khai."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta

import pytest

from hagent.agent.runtime import (
    ActionCompleted,
    AgentRuntimeError,
    ArtifactProduced,
    CancelRun,
    EvidenceAdded,
    InMemoryRuntimeEventStore,
    LegacyGraphRuntime,
    PlanProposed,
    RequestScope,
    ResolveApproval,
    RunCancelled,
    RunCompleted,
    RunFailed,
    RunStarted,
    RuntimeAccessDenied,
    RuntimeCapacityExceeded,
    RuntimeCommandConflict,
    RuntimeCommandExpired,
    StartTurn,
    UnsupportedRuntimeCommand,
    build_start_turn,
    collect_runtime_result,
    runtime_event_to_dict,
    stream_legacy_events,
)


async def _collect(iterator):
    return [item async for item in iterator]


@pytest.mark.asyncio
async def test_tool_call_scope_injection_is_authoritative_and_concurrency_safe():
    from types import SimpleNamespace

    from hagent.agent.orchestration.graph import _inject_request_scope_into_tool_call
    from hagent.agent.runtime.context import GraphRequestContext

    class _Tool:
        def __init__(self):
            self.args = {"token": {}, "user_id": {}, "dataset_id": {}}

    class _Request:
        def __init__(self, *, state, runtime, tool_call):
            self.state = state
            self.runtime = runtime
            self.tool = _Tool()
            self.tool_call = tool_call

        def override(self, *, tool_call):
            return _Request(
                state=self.state,
                runtime=self.runtime,
                tool_call=tool_call,
            )

    async def invoke(owner, token):
        request = _Request(
            state={"user_id": "state-spoofed", "user_token": "state-token"},
            runtime=SimpleNamespace(
                context=GraphRequestContext(
                    principal_id=owner,
                    credential=token,
                )
            ),
            tool_call={
                "name": "get_dataset_info",
                "id": owner,
                "type": "tool_call",
                "args": {
                    "dataset_id": "dataset",
                    "user_id": "spoofed",
                    "token": "model-supplied-token",
                },
            },
        )

        async def execute(scoped_request):
            await asyncio.sleep(0)
            return dict(scoped_request.tool_call["args"])

        return await _inject_request_scope_into_tool_call(request, execute)

    first, second = await asyncio.gather(
        invoke("owner-a", "token-a"),
        invoke("owner-b", "token-b"),
    )

    assert first == {
        "dataset_id": "dataset",
        "user_id": "owner-a",
        "token": "token-a",
    }
    assert second == {
        "dataset_id": "dataset",
        "user_id": "owner-b",
        "token": "token-b",
    }


@pytest.mark.asyncio
async def test_missing_scope_credential_cannot_inherit_ambient_token(monkeypatch):
    from types import SimpleNamespace

    from hagent.agent.orchestration.graph import _inject_request_scope_into_tool_call
    from hagent.agent.runtime.context import GraphRequestContext

    class _Tool:
        def __init__(self):
            self.args = {"token": {}, "dataset_id": {}}

    class _Request:
        def __init__(self):
            self.tool = _Tool()
            self.state = {
                "user_id": "state-spoofed",
                "user_token": "state-secret",
            }
            self.runtime = SimpleNamespace(
                context=GraphRequestContext(
                    principal_id="owner",
                    credential=None,
                )
            )
            self.tool_call = {
                "name": "get_dataset_info",
                "id": "call-1",
                "type": "tool_call",
                "args": {
                    "dataset_id": "dataset",
                    "token": "model-supplied-token",
                },
            }

        def override(self, *, tool_call):
            raise AssertionError("unauthenticated tool request must not execute")

    executed = False

    async def execute(request):
        nonlocal executed
        executed = True

    monkeypatch.setenv("USER_TOKEN", "ambient-process-token")
    result = await _inject_request_scope_into_tool_call(_Request(), execute)

    assert not executed
    assert result.status == "error"
    payload = json.loads(result.content)
    assert payload["error"]["code"] == "AUTH_SCOPE_REQUIRED"
    assert "ambient-process-token" not in result.content
    assert "model-supplied-token" not in result.content


@pytest.mark.asyncio
async def test_dispatch_is_typed_monotonic_idempotent_and_replayable():
    invocations = 0

    async def source(command, scope):
        nonlocal invocations
        invocations += 1
        assert command.message == "train"
        assert scope.credential == "runtime-secret"
        yield {"type": "route", "agent": "coordinator"}
        yield {"type": "token", "content": "draft runtime-secret"}
        yield {
            "type": "tool_call",
            "tool": "list_datasets",
            "args": {"APIKey": "provider-key"},
        }
        yield {
            "type": "tool_result",
            "tool": "list_datasets",
            "output": {"rows": 3},
        }
        yield {
            "type": "done",
            "response": {
                "message": "complete",
                "privateKey": "private-material",
            },
        }
        yield {"type": "done", "response": {"message": "duplicate"}}

    runtime = LegacyGraphRuntime(event_source=source)
    command = StartTurn(
        command_id="command-1",
        run_id="run-1",
        message="train",
    )
    scope = RequestScope(
        principal_id="owner",
        credential="runtime-secret",
        trace_id="trace-1",
    )

    events = await _collect(runtime.dispatch(command, scope=scope))
    repeated = await _collect(runtime.dispatch(command, scope=scope))
    replayed = await _collect(runtime.replay("run-1", after_sequence=2, scope=scope))

    assert invocations == 1
    assert repeated == events
    assert replayed == events[2:]
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert isinstance(events[0], RunStarted)
    assert any(isinstance(event, ActionCompleted) for event in events)
    assert any(isinstance(event, ArtifactProduced) for event in events)
    assert any(isinstance(event, EvidenceAdded) for event in events)
    assert isinstance(events[-1], RunCompleted)
    assert (
        sum(
            isinstance(event, (RunCompleted, RunFailed, RunCancelled))
            for event in events
        )
        == 1
    )

    serialized = str([runtime_event_to_dict(event) for event in events])
    assert "runtime-secret" not in serialized
    assert "provider-key" not in serialized
    assert "private-material" not in serialized
    assert "[REDACTED]" in serialized


@pytest.mark.asyncio
async def test_concurrent_duplicate_command_executes_source_once():
    source_started = asyncio.Event()
    release_source = asyncio.Event()
    invocation_count = 0

    async def source(command, scope):
        nonlocal invocation_count
        invocation_count += 1
        source_started.set()
        await release_source.wait()
        yield {"type": "done", "response": {"message": "ok"}}

    runtime = LegacyGraphRuntime(event_source=source)
    command = StartTurn(command_id="same-command", run_id="same-run", message="go")
    scope = RequestScope(principal_id="owner")

    first = asyncio.create_task(_collect(runtime.dispatch(command, scope=scope)))
    await asyncio.wait_for(source_started.wait(), timeout=1)
    second = asyncio.create_task(_collect(runtime.dispatch(command, scope=scope)))
    await asyncio.sleep(0)
    release_source.set()
    first_events, second_events = await asyncio.gather(first, second)

    assert invocation_count == 1
    assert first_events == second_events


@pytest.mark.asyncio
async def test_evicted_command_is_rejected_instead_of_executed_again():
    invocation_count = 0

    async def source(command, scope):
        nonlocal invocation_count
        invocation_count += 1
        yield {"type": "done", "response": {"message": command.message}}

    runtime = LegacyGraphRuntime(
        event_source=source,
        event_store=InMemoryRuntimeEventStore(max_runs=1),
    )
    scope = RequestScope(principal_id="owner")
    first = StartTurn(command_id="command-1", run_id="run-1", message="first")
    second = StartTurn(command_id="command-2", run_id="run-2", message="second")

    await _collect(runtime.dispatch(first, scope=scope))
    await _collect(runtime.dispatch(second, scope=scope))
    with pytest.raises(RuntimeCommandExpired):
        await _collect(runtime.dispatch(first, scope=scope))

    assert invocation_count == 2


@pytest.mark.asyncio
async def test_evicted_run_id_cannot_be_reassigned_to_a_new_command():
    invocation_count = 0

    async def source(command, scope):
        nonlocal invocation_count
        invocation_count += 1
        yield {"type": "done", "response": {"message": command.message}}

    runtime = LegacyGraphRuntime(
        event_source=source,
        event_store=InMemoryRuntimeEventStore(max_runs=1),
    )
    scope = RequestScope(principal_id="owner")

    await _collect(
        runtime.dispatch(
            StartTurn(command_id="command-1", run_id="run-1", message="first"),
            scope=scope,
        )
    )
    await _collect(
        runtime.dispatch(
            StartTurn(command_id="command-2", run_id="run-2", message="second"),
            scope=scope,
        )
    )

    with pytest.raises(RuntimeCommandExpired):
        await _collect(
            runtime.dispatch(
                StartTurn(
                    command_id="command-3",
                    run_id="run-1",
                    message="must-not-run",
                ),
                scope=scope,
            )
        )

    assert invocation_count == 2


@pytest.mark.asyncio
async def test_replay_is_immutable_when_dispatch_or_replay_payload_is_mutated():
    async def source(command, scope):
        yield {
            "type": "plan",
            "title": "audit",
            "steps": [{"name": "inspect", "status": "pending"}],
        }
        yield {"type": "done", "response": {"message": "ok"}}

    runtime = LegacyGraphRuntime(event_source=source)
    scope = RequestScope(principal_id="owner")
    command = StartTurn(command_id="command", run_id="run", message="audit")

    dispatched = await _collect(runtime.dispatch(command, scope=scope))
    plan = next(event for event in dispatched if isinstance(event, PlanProposed))
    plan.plan["steps"][0]["status"] = "tampered"

    first_replay = await _collect(runtime.replay("run", after_sequence=0, scope=scope))
    replayed_plan = next(
        event for event in first_replay if isinstance(event, PlanProposed)
    )
    assert replayed_plan.plan["steps"][0]["status"] == "pending"

    replayed_plan.plan["steps"][0]["status"] = "tampered-again"
    second_replay = await _collect(runtime.replay("run", after_sequence=0, scope=scope))
    replayed_again = next(
        event for event in second_replay if isinstance(event, PlanProposed)
    )
    assert replayed_again.plan["steps"][0]["status"] == "pending"


@pytest.mark.asyncio
async def test_active_run_capacity_fails_closed():
    source_started = asyncio.Event()

    async def source(command, scope):
        source_started.set()
        await asyncio.Future()
        if False:
            yield {}

    runtime = LegacyGraphRuntime(
        event_source=source,
        event_store=InMemoryRuntimeEventStore(max_runs=1),
    )
    scope = RequestScope(principal_id="owner")
    active = asyncio.create_task(
        _collect(runtime.dispatch(StartTurn(message="active"), scope=scope))
    )
    await asyncio.wait_for(source_started.wait(), timeout=1)

    with pytest.raises(RuntimeCapacityExceeded):
        await _collect(runtime.dispatch(StartTurn(message="overflow"), scope=scope))

    active.cancel()
    with pytest.raises(asyncio.CancelledError):
        await active


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event_store", "payload"),
    [
        (InMemoryRuntimeEventStore(max_events_per_run=3), "small"),
        (
            InMemoryRuntimeEventStore(
                max_runs=1,
                max_event_bytes_per_run=2048,
            ),
            "x" * 4096,
        ),
    ],
    ids=["event-count", "event-bytes"],
)
async def test_per_run_event_limits_end_with_safe_terminal(event_store, payload):
    async def source(command, scope):
        while True:
            yield {"type": "token", "content": payload}

    runtime = LegacyGraphRuntime(event_source=source, event_store=event_store)
    events = await _collect(
        runtime.dispatch(
            StartTurn(message="bounded"),
            scope=RequestScope(principal_id="owner"),
        )
    )

    assert isinstance(events[-1], RunFailed)
    assert events[-1].error_code == "EVENT_LIMIT_EXCEEDED"
    assert len(events) == (3 if payload == "small" else 2)
    if payload != "small":
        next_events = await _collect(
            runtime.dispatch(
                StartTurn(message="next-admitted-run"),
                scope=RequestScope(principal_id="owner"),
            )
        )
        assert isinstance(next_events[-1], RunFailed)
        assert next_events[-1].error_code == "EVENT_LIMIT_EXCEEDED"


def test_runtime_ids_are_bounded_before_store_admission():
    with pytest.raises(ValueError, match="command_id"):
        StartTurn(message="invalid", command_id="x" * 129)
    with pytest.raises(ValueError, match="run_id"):
        StartTurn(message="invalid", run_id="contains spaces")
    with pytest.raises(ValueError, match="trace_id"):
        RequestScope(principal_id="owner", trace_id="x" * 129)


@pytest.mark.asyncio
async def test_future_deadline_cancels_blocked_source_and_records_terminal():
    source_closed = asyncio.Event()

    async def source(command, scope):
        try:
            await asyncio.Future()
            if False:
                yield {}
        finally:
            source_closed.set()

    runtime = LegacyGraphRuntime(event_source=source)
    events = await _collect(
        runtime.dispatch(
            StartTurn(message="deadline"),
            scope=RequestScope(
                principal_id="owner",
                deadline=datetime.now(UTC) + timedelta(milliseconds=20),
            ),
        )
    )

    assert source_closed.is_set()
    assert isinstance(events[-1], RunFailed)
    assert events[-1].error_code == "DEADLINE_EXCEEDED"


@pytest.mark.asyncio
async def test_plain_typed_runtime_completed_event_maps_to_legacy_done():
    class _TypedRuntime:
        async def dispatch(self, command, *, scope):
            yield RunStarted(
                run_id=command.run_id,
                command_id=command.command_id,
                sequence=1,
                created_at="2026-08-08T00:00:00+00:00",
            )
            yield PlanProposed(
                run_id=command.run_id,
                command_id=command.command_id,
                sequence=2,
                created_at="2026-08-08T00:00:00+00:00",
                plan={"type": "payload-must-not-win", "title": "typed plan"},
            )
            yield RunCompleted(
                run_id=command.run_id,
                command_id=command.command_id,
                sequence=3,
                created_at="2026-08-08T00:00:01+00:00",
                result={"message": "typed"},
            )

        async def replay(self, run_id, *, after_sequence, scope):
            if False:
                yield None

    events = await _collect(
        stream_legacy_events(
            _TypedRuntime(),
            StartTurn(message="typed"),
            scope=RequestScope(principal_id="owner"),
        )
    )

    assert events == [
        {"type": "plan", "title": "typed plan"},
        {"type": "done", "response": {"message": "typed"}},
    ]


@pytest.mark.asyncio
async def test_checker_identity_cannot_be_overridden_by_details_payload():
    class _TypedRuntime:
        async def dispatch(self, command, *, scope):
            from hagent.agent.runtime import CheckCompleted

            yield CheckCompleted(
                run_id=command.run_id,
                command_id=command.command_id,
                sequence=1,
                created_at="2026-08-08T00:00:00+00:00",
                checker="contract_checker",
                verdict="passed",
                details={
                    "checker": "forged_checker",
                    "verdict": "forged_verdict",
                    "reason": "schema_valid",
                },
            )

        async def replay(self, run_id, *, after_sequence, scope):
            if False:
                yield None

    events = await _collect(
        stream_legacy_events(
            _TypedRuntime(),
            StartTurn(message="typed"),
            scope=RequestScope(principal_id="owner"),
        )
    )

    assert events == [
        {
            "type": "plan_event",
            "event": {
                "checker": "contract_checker",
                "verdict": "passed",
                "reason": "schema_valid",
            },
        }
    ]


@pytest.mark.asyncio
async def test_command_id_conflict_and_principal_scoped_replay_fail_closed():
    async def source(command, scope):
        yield {"type": "done", "response": {"message": "ok"}}

    runtime = LegacyGraphRuntime(event_source=source)
    owner_scope = RequestScope(principal_id="owner")
    first = StartTurn(command_id="command", run_id="run", message="first")
    await _collect(runtime.dispatch(first, scope=owner_scope))

    conflicting = StartTurn(command_id="command", run_id="other", message="changed")
    with pytest.raises(RuntimeCommandConflict):
        await _collect(runtime.dispatch(conflicting, scope=owner_scope))

    with pytest.raises(RuntimeAccessDenied):
        await _collect(
            runtime.replay(
                "run",
                after_sequence=0,
                scope=RequestScope(principal_id="other-owner"),
            )
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_mode", ["exception", "incomplete"])
async def test_legacy_failure_has_one_safe_terminal(failure_mode):
    async def source(command, scope):
        if failure_mode == "exception":
            raise RuntimeError("credential runtime-secret must not leak")
        if False:
            yield {}

    runtime = LegacyGraphRuntime(event_source=source)
    command = StartTurn(message="fail")
    scope = RequestScope(principal_id="owner", credential="runtime-secret")

    events = await _collect(runtime.dispatch(command, scope=scope))

    assert isinstance(events[-1], RunFailed)
    assert sum(isinstance(event, RunFailed) for event in events) == 1
    assert "runtime-secret" not in str(runtime_event_to_dict(events[-1]))
    with pytest.raises(AgentRuntimeError) as exc:
        await collect_runtime_result(runtime, command, scope=scope)
    assert exc.value.code in {"LEGACY_RUNTIME_ERROR", "LEGACY_STREAM_INCOMPLETE"}


@pytest.mark.asyncio
async def test_legacy_error_payload_is_replaced_by_stable_safe_terminal():
    async def source(command, scope):
        yield {
            "type": "error",
            "error": {
                "code": "UPSTREAM_TIMEOUT",
                "message": "internal path and provider credential",
                "detail": {"debug": "sensitive"},
            },
        }

    runtime = LegacyGraphRuntime(event_source=source)
    events = await _collect(
        runtime.dispatch(
            StartTurn(message="fail"),
            scope=RequestScope(principal_id="owner"),
        )
    )

    terminal = events[-1]
    assert isinstance(terminal, RunFailed)
    assert terminal.error_code == "UPSTREAM_TIMEOUT"
    assert terminal.compatibility_event == {
        "type": "error",
        "error": {
            "code": "upstream_timeout",
            "message": "Agent runtime failed",
        },
    }
    assert "internal path" not in str(terminal)
    assert "sensitive" not in str(terminal)


@pytest.mark.asyncio
async def test_legacy_stream_cleanup_failure_is_logged_without_sensitive_detail(caplog):
    sensitive_detail = "cleanup failed with provider-token-secret"

    class _CloseFailureStream:
        def __init__(self):
            self.emitted = False

        async def __anext__(self):
            if self.emitted:
                raise StopAsyncIteration
            self.emitted = True
            return {"type": "done", "response": {"message": "ok"}}

        async def aclose(self):
            raise RuntimeError(sensitive_detail)

    def source(command, scope):
        return _CloseFailureStream()

    runtime = LegacyGraphRuntime(event_source=source)
    events = await _collect(
        runtime.dispatch(
            StartTurn(command_id="command", run_id="run", message="close"),
            scope=RequestScope(principal_id="owner", trace_id="trace-close"),
        )
    )

    assert isinstance(events[-1], RunCompleted)
    assert "LEGACY_STREAM_CLOSE_FAILED" in caplog.text
    assert "run_id=run" in caplog.text
    assert "trace_id=trace-close" in caplog.text
    assert sensitive_detail not in caplog.text


@pytest.mark.asyncio
async def test_cancelled_dispatch_is_recorded_for_owner_replay():
    source_started = asyncio.Event()

    async def source(command, scope):
        source_started.set()
        await asyncio.Future()
        if False:
            yield {}

    runtime = LegacyGraphRuntime(event_source=source)
    command = StartTurn(
        command_id="cancel-command", run_id="cancel-run", message="wait"
    )
    scope = RequestScope(principal_id="owner")
    task = asyncio.create_task(_collect(runtime.dispatch(command, scope=scope)))

    await asyncio.wait_for(source_started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    replayed = await _collect(
        runtime.replay("cancel-run", after_sequence=0, scope=scope)
    )
    repeated = await asyncio.wait_for(
        _collect(runtime.dispatch(command, scope=scope)),
        timeout=1,
    )
    assert isinstance(replayed[-1], RunCancelled)
    assert sum(isinstance(event, RunCancelled) for event in replayed) == 1
    assert repeated == replayed


@pytest.mark.asyncio
async def test_unsupported_approval_and_cancel_commands_are_explicit_failures():
    runtime = LegacyGraphRuntime()
    scope = RequestScope(principal_id="owner")

    with pytest.raises(UnsupportedRuntimeCommand):
        await _collect(
            runtime.dispatch(
                ResolveApproval(
                    run_id="run",
                    approval_id="approval",
                    approved=True,
                ),
                scope=scope,
            )
        )
    with pytest.raises(UnsupportedRuntimeCommand):
        await _collect(runtime.dispatch(CancelRun(run_id="run"), scope=scope))


@pytest.mark.asyncio
async def test_legacy_helpers_keep_credentials_outside_command_and_round_trip_result():
    captured = {}

    async def source(command, scope):
        captured["command"] = command
        captured.setdefault("scopes", []).append(scope)
        yield {"type": "route", "agent": "coordinator"}
        yield {"type": "done", "response": {"message": "ok", "route": "direct"}}

    runtime = LegacyGraphRuntime(event_source=source)
    command, scope = build_start_turn(
        "hello",
        user_id="owner",
        user_token="ephemeral-token",
        history=[{"role": "user", "content": "prior"}],
        mongo_client=object(),
        model_name="ci-mock",
    )

    result = await collect_runtime_result(runtime, command, scope=scope)
    legacy_events = await _collect(
        stream_legacy_events(
            LegacyGraphRuntime(event_source=source),
            StartTurn(message="stream"),
            scope=RequestScope(principal_id="owner"),
        )
    )

    assert result == {"message": "ok", "route": "direct"}
    assert [event["type"] for event in legacy_events] == ["route", "done"]
    assert "ephemeral-token" not in str(asdict(command))
    assert "ephemeral-token" not in repr(scope)
    assert captured["scopes"][0].credential == "ephemeral-token"
