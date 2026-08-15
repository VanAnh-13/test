"""Docker smoke for the public HAgent sync and SSE chat contracts.

The smoke uses real signup/login endpoints and the local OpenAI-compatible CI
mock. It never calls a paid model and never prints credentials.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

ALLOWED_EVENTS = frozenset(
    {
        "meta",
        "route",
        "phase",
        "plan",
        "plan_event",
        "surprise",
        "token",
        "tool_call",
        "tool_result",
        "done",
        "error",
    }
)
TERMINAL_EVENTS = frozenset({"done", "error"})
ROUTED_EVENTS = frozenset(
    {
        "phase",
        "plan",
        "plan_event",
        "surprise",
        "token",
        "tool_call",
        "tool_result",
    }
)
CHAT_RESPONSE_KEYS = frozenset(
    {
        "message",
        "conversation_id",
        "provider",
        "model",
        "route",
        "tool_outputs",
        "planning",
        "campaign",
        "hierarchy",
        "world_model",
        "evaluation",
        "execution_events",
        "execution_log",
        "revision_count",
        "cost_metrics",
    }
)


class E2EFailure(RuntimeError):
    """A contract assertion failed without exposing request credentials."""


@dataclass(frozen=True)
class E2EConfig:
    base_url: str = "http://localhost:5370"
    hagent_url: str = "http://localhost:5360"
    model: str = "ci-mock"
    request_timeout_seconds: float = 60.0
    abort_settle_seconds: float = 5.0

    @classmethod
    def from_env(cls) -> E2EConfig:
        return cls(
            base_url=os.getenv("BASE_URL", cls.base_url).rstrip("/"),
            hagent_url=os.getenv("HAGENT_URL", cls.hagent_url).rstrip("/"),
            model=os.getenv("E2E_MODEL", cls.model).strip(),
            request_timeout_seconds=_positive_env_float(
                "E2E_REQUEST_TIMEOUT_SECONDS", cls.request_timeout_seconds
            ),
            abort_settle_seconds=_nonnegative_env_float(
                "E2E_ABORT_SETTLE_SECONDS", cls.abort_settle_seconds
            ),
        )


@dataclass(frozen=True)
class SSEEvent:
    event: str
    event_id: int
    data: dict[str, Any]


@dataclass(frozen=True)
class SmokeReport:
    run_id: str
    model: str
    conversation_id: str
    event_types: tuple[str, ...]
    cleanup_count: int


def _positive_env_float(name: str, default: float) -> float:
    value = _nonnegative_env_float(name, default)
    if value <= 0:
        raise E2EFailure(f"{name} must be greater than zero")
    return value


def _nonnegative_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise E2EFailure(f"{name} must be numeric") from exc
    if value < 0:
        raise E2EFailure(f"{name} must not be negative")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise E2EFailure(message)


def _json_object(response: httpx.Response, stage: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise E2EFailure(f"{stage} returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise E2EFailure(f"{stage} returned a non-object JSON value")
    return payload


def _expect_status(
    response: httpx.Response,
    expected: set[int],
    stage: str,
) -> None:
    if response.status_code not in expected:
        raise E2EFailure(f"{stage} returned HTTP {response.status_code}")


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _chat_payload(
    message: str,
    model: str,
    conversation_id: str | None,
) -> dict[str, Any]:
    return {
        "message": message,
        "conversation_id": conversation_id,
        "context": {},
        "model": model,
    }


def _validate_chat_response(
    payload: dict[str, Any],
    *,
    model: str,
    marker: str,
    conversation_id: str | None = None,
) -> str:
    missing = sorted(CHAT_RESPONSE_KEYS - payload.keys())
    _require(not missing, f"chat response misses fields: {', '.join(missing)}")
    actual_id = payload.get("conversation_id")
    _require(isinstance(actual_id, str) and actual_id, "missing conversation_id")
    if conversation_id is not None:
        _require(actual_id == conversation_id, "conversation_id changed between turns")
    _require(payload.get("model") == model, "requested model was not preserved")
    _require(marker in str(payload.get("message") or ""), "response marker missing")
    return actual_id


async def _register_and_login(
    client: httpx.AsyncClient,
    config: E2EConfig,
    run_id: str,
    label: str,
) -> str:
    username = f"e2e_{label}_{run_id}".lower()
    password = f"ci-only-{uuid.uuid4().hex}"
    signup = await client.post(
        f"{config.base_url}/signup",
        json={
            "username": username,
            "email": f"{username}@example.com",
            "gender": "other",
            "date": "01/01/2026",
            "number": "0900000000",
            "fullName": f"CI E2E {label}",
            "password": password,
        },
    )
    _expect_status(signup, {200}, f"{label} signup")

    login = await client.post(
        f"{config.base_url}/login",
        json={"username": username, "password": password},
    )
    _expect_status(login, {200}, f"{label} login")
    token = _json_object(login, f"{label} login").get("access_token")
    _require(isinstance(token, str) and token, f"{label} login returned no token")
    return token


async def _assert_model_registered(
    client: httpx.AsyncClient,
    config: E2EConfig,
) -> None:
    response = await client.get(f"{config.hagent_url}/api/v1/chat/providers")
    _expect_status(response, {200}, "provider registry")
    payload = _json_object(response, "provider registry")
    models = {str(payload.get("default_model") or "")}
    providers = payload.get("providers")
    _require(isinstance(providers, list), "provider registry has no providers list")
    for provider in providers:
        if isinstance(provider, dict) and isinstance(provider.get("models"), list):
            models.update(str(model) for model in provider["models"])
    _require(config.model in models, f"model {config.model!r} is not registered")


async def _post_chat(
    client: httpx.AsyncClient,
    config: E2EConfig,
    token: str,
    message: str,
    conversation_id: str | None,
) -> dict[str, Any]:
    response = await client.post(
        f"{config.hagent_url}/api/v1/chat/",
        headers=_auth_headers(token),
        json=_chat_payload(message, config.model, conversation_id),
    )
    _expect_status(response, {200}, "sync chat")
    return _json_object(response, "sync chat")


def _parse_sse_frame(lines: list[str]) -> SSEEvent:
    fields: dict[str, str] = {}
    for line in lines:
        if line.startswith(":"):
            continue
        if ":" not in line:
            raise E2EFailure("malformed SSE field")
        key, raw_value = line.split(":", 1)
        if key not in {"event", "id", "data"} or key in fields:
            raise E2EFailure("SSE frame has unsupported or duplicate fields")
        fields[key] = raw_value.lstrip()
    _require(set(fields) == {"event", "id", "data"}, "incomplete SSE frame")
    _require(fields["data"] != "[DONE]", "legacy SSE sentinel is forbidden")
    try:
        event_id = int(fields["id"])
    except ValueError as exc:
        raise E2EFailure("SSE id is not an integer") from exc
    _require(event_id > 0, "SSE id must be positive")
    try:
        data = json.loads(fields["data"])
    except json.JSONDecodeError as exc:
        raise E2EFailure("SSE data is not JSON") from exc
    _require(isinstance(data, dict), "SSE data must be an object")
    event = fields["event"]
    _require(event in ALLOWED_EVENTS, f"unsupported SSE event {event!r}")
    _require(data.get("type") == event, "SSE event and data.type differ")
    return SSEEvent(event=event, event_id=event_id, data=data)


async def iter_sse_events(response: httpx.Response) -> AsyncIterator[SSEEvent]:
    frame_lines: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if frame_lines:
                yield _parse_sse_frame(frame_lines)
                frame_lines = []
            continue
        frame_lines.append(line)
    if frame_lines:
        yield _parse_sse_frame(frame_lines)


def validate_sse_sequence(events: list[SSEEvent]) -> dict[str, Any]:
    _require(bool(events), "SSE stream emitted no events")
    previous_id = 0
    terminals = 0
    route_seen = False
    for index, event in enumerate(events):
        _require(event.event_id > previous_id, "SSE ids are not strictly increasing")
        previous_id = event.event_id
        if event.event == "route":
            route_seen = True
        elif event.event in ROUTED_EVENTS:
            _require(route_seen, "SSE work event occurred before route")
        if event.event in TERMINAL_EVENTS:
            terminals += 1
            _require(index == len(events) - 1, "SSE terminal event is not last")
    _require(route_seen, "SSE stream emitted no route event")
    _require(terminals == 1, "SSE stream must contain exactly one terminal")
    terminal = events[-1]
    _require(terminal.event == "done", "SSE stream ended with error")
    response = terminal.data.get("response")
    _require(isinstance(response, dict), "done.response is not an object")
    return response


async def _complete_stream(
    client: httpx.AsyncClient,
    config: E2EConfig,
    token: str,
    conversation_id: str,
    message: str,
) -> tuple[list[SSEEvent], dict[str, Any]]:
    async with client.stream(
        "POST",
        f"{config.hagent_url}/api/v1/chat/stream",
        headers=_auth_headers(token),
        json=_chat_payload(message, config.model, conversation_id),
    ) as response:
        _expect_status(response, {200}, "SSE chat")
        _require(
            "text/event-stream" in response.headers.get("content-type", ""),
            "SSE content type missing",
        )
        _require(
            response.headers.get("x-conversation-id") == conversation_id,
            "SSE conversation header mismatch",
        )
        events = [event async for event in iter_sse_events(response)]
    return events, validate_sse_sequence(events)


async def _abort_stream(
    client: httpx.AsyncClient,
    config: E2EConfig,
    token: str,
    conversation_id: str,
    message: str,
) -> tuple[str, SSEEvent]:
    async with client.stream(
        "POST",
        f"{config.hagent_url}/api/v1/chat/stream",
        headers=_auth_headers(token),
        json=_chat_payload(message, config.model, conversation_id),
    ) as response:
        _expect_status(response, {200}, "abort SSE chat")
        response_conversation_id = response.headers.get("x-conversation-id", "")
        _require(
            response_conversation_id == conversation_id,
            "abort stream conversation header mismatch",
        )
        first_event: SSEEvent | None = None
        async for event in iter_sse_events(response):
            first_event = event
            break
        _require(first_event is not None, "abort stream emitted no initial event")
        _require(
            first_event.event not in TERMINAL_EVENTS, "abort stream finished too early"
        )
        return response_conversation_id, first_event


async def _get_history(
    client: httpx.AsyncClient,
    config: E2EConfig,
    token: str,
    conversation_id: str,
) -> dict[str, Any]:
    response = await client.get(
        f"{config.hagent_url}/api/v1/chat/conversation/{conversation_id}",
        headers=_auth_headers(token),
    )
    _expect_status(response, {200}, "conversation history")
    return _json_object(response, "conversation history")


def _message_contents(history: dict[str, Any], role: str | None = None) -> list[str]:
    messages = history.get("messages")
    _require(isinstance(messages, list), "history has no messages list")
    return [
        str(message.get("content") or "")
        for message in messages
        if isinstance(message, dict) and (role is None or message.get("role") == role)
    ]


async def run_smoke(
    config: E2EConfig | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    run_id: str | None = None,
) -> SmokeReport:
    config = config or E2EConfig.from_env()
    _require(bool(config.model), "E2E model must not be empty")
    run_id = run_id or uuid.uuid4().hex[:12]
    cleanup_targets: dict[tuple[str, str], None] = {}
    report: SmokeReport | None = None
    failure: BaseException | None = None

    timeout = httpx.Timeout(config.request_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
        try:
            await _assert_model_registered(client, config)
            owner_token = await _register_and_login(client, config, run_id, "owner")
            other_token = await _register_and_login(client, config, run_id, "other")

            seed_request = f"E2E_HISTORY_MARKER:{run_id}"
            seed_ack = f"E2E_HISTORY_SEEDED:{run_id}"
            conversation_id = f"e2e-owner-{run_id}"
            cleanup_targets[(owner_token, conversation_id)] = None
            first = await _post_chat(
                client, config, owner_token, seed_request, conversation_id
            )
            _validate_chat_response(
                first,
                model=config.model,
                marker=seed_ack,
                conversation_id=conversation_id,
            )

            probe_request = f"E2E_HISTORY_PROBE:{run_id}"
            owner_probe = await _post_chat(
                client, config, owner_token, probe_request, conversation_id
            )
            _validate_chat_response(
                owner_probe,
                model=config.model,
                marker=f"E2E_HISTORY_OK:{run_id}",
                conversation_id=conversation_id,
            )

            cleanup_targets[(other_token, conversation_id)] = None
            other_probe = await _post_chat(
                client, config, other_token, probe_request, conversation_id
            )
            _validate_chat_response(
                other_probe,
                model=config.model,
                marker=f"E2E_HISTORY_NONE:{run_id}",
                conversation_id=conversation_id,
            )
            other_history = await _get_history(
                client, config, other_token, conversation_id
            )
            _require(
                not any(
                    seed_request in item for item in _message_contents(other_history)
                ),
                "owner history leaked to another user",
            )

            stream_request = f"E2E_STREAM_TURN:{run_id}"
            stream_ack = f"E2E_STREAM_ACK:{run_id}"
            events, done_response = await _complete_stream(
                client,
                config,
                owner_token,
                conversation_id,
                stream_request,
            )
            _validate_chat_response(
                done_response,
                model=config.model,
                marker=stream_ack,
                conversation_id=conversation_id,
            )
            owner_history = await _get_history(
                client, config, owner_token, conversation_id
            )
            assistant_contents = _message_contents(owner_history, role="assistant")
            _require(
                sum(stream_ack in item for item in assistant_contents) == 1,
                "stream final assistant was not persisted exactly once",
            )

            abort_request = f"E2E_ABORT_TURN:{run_id}"
            abort_id = f"e2e-abort-{run_id}"
            cleanup_targets[(owner_token, abort_id)] = None
            abort_id, _ = await _abort_stream(
                client,
                config,
                owner_token,
                abort_id,
                abort_request,
            )
            await asyncio.sleep(config.abort_settle_seconds)
            abort_history = await _get_history(client, config, owner_token, abort_id)
            _require(
                any(
                    abort_request in item
                    for item in _message_contents(abort_history, "user")
                ),
                "aborted turn did not persist the user message",
            )
            _require(
                not _message_contents(abort_history, "assistant"),
                "aborted turn persisted an assistant response",
            )

            report = SmokeReport(
                run_id=run_id,
                model=config.model,
                conversation_id=conversation_id,
                event_types=tuple(event.event for event in events),
                cleanup_count=len(cleanup_targets),
            )
        except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001
            failure = exc

        cleanup_errors = []
        for token, conversation_id in cleanup_targets:
            try:
                response = await client.delete(
                    f"{config.hagent_url}/api/v1/chat/conversation/{conversation_id}",
                    headers=_auth_headers(token),
                )
                if response.status_code not in {200, 404}:
                    cleanup_errors.append(response.status_code)
            except httpx.HTTPError:
                cleanup_errors.append(-1)

        if failure is not None:
            raise failure.with_traceback(failure.__traceback__)
        if cleanup_errors:
            raise E2EFailure(
                f"conversation cleanup failed ({len(cleanup_errors)} target(s))"
            )

    _require(report is not None, "smoke ended without a report")
    return report


async def _main() -> int:
    config = E2EConfig.from_env()
    print("HAgent Docker contract smoke")
    print(f"toolkit={config.base_url} bridge={config.hagent_url} model={config.model}")
    try:
        report = await run_smoke(config)
    except (E2EFailure, httpx.HTTPError) as exc:
        detail = str(exc).replace("\n", " ")[:300]
        print(f"FAILED {type(exc).__name__}: {detail}")
        return 1
    print(
        "PASSED "
        f"run={report.run_id} events={','.join(report.event_types)} "
        f"cleaned={report.cleanup_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
