from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from scripts import e2e_docker_test as e2e

RUN_ID = "contract123"
OWNER_TOKEN = "owner-token"
OTHER_TOKEN = "other-token"
OWNER_CONVERSATION = f"e2e-owner-{RUN_ID}"
ABORT_CONVERSATION = f"e2e-abort-{RUN_ID}"


def _chat_response(message: str, conversation_id: str) -> dict[str, Any]:
    return {
        "message": message,
        "conversation_id": conversation_id,
        "sources": [],
        "suggestions": [],
        "provider": "hagent",
        "model": "ci-mock",
        "route": "coordinator",
        "tool_outputs": [],
        "plan_status": None,
        "selected_plan": None,
        "planning": None,
        "surprise": None,
        "cost_metrics": {"total_tokens": 3},
        "execution_events": [],
        "execution_log": [],
        "revision_count": 0,
        "world_model": None,
        "campaign": None,
        "campaign_status": None,
        "hierarchy": None,
        "hierarchy_status": None,
        "evaluation": None,
    }


def _frame(event: str, event_id: int, data: dict[str, Any]) -> bytes:
    return (f"event: {event}\nid: {event_id}\ndata: {json.dumps(data)}\n\n").encode()


class RecordingStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]):
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


class ContractBackend:
    def __init__(
        self,
        *,
        duplicate_terminal: bool = False,
        invalid_first_response: bool = False,
    ):
        self.duplicate_terminal = duplicate_terminal
        self.invalid_first_response = invalid_first_response
        self.histories: dict[tuple[str, str], list[dict[str, str]]] = {}
        self.deleted: list[tuple[str, str]] = []
        self.abort_stream: RecordingStream | None = None
        self.models: list[str] = []

    @staticmethod
    def _token(request: httpx.Request) -> str:
        prefix = "Bearer "
        value = request.headers.get("authorization", "")
        assert value.startswith(prefix)
        return value[len(prefix) :]

    @staticmethod
    def _body(request: httpx.Request) -> dict[str, Any]:
        return json.loads(request.content.decode())

    @staticmethod
    def _message(role: str, content: str) -> dict[str, str]:
        return {"role": role, "content": content}

    def _sync_chat(self, request: httpx.Request) -> httpx.Response:
        token = self._token(request)
        body = self._body(request)
        assert body["model"] == "ci-mock"
        self.models.append(body["model"])
        message = body["message"]
        conversation_id = body.get("conversation_id") or OWNER_CONVERSATION
        history = self.histories.setdefault((token, conversation_id), [])
        history.append(self._message("user", message))

        if message == f"E2E_HISTORY_MARKER:{RUN_ID}":
            answer = f"E2E_HISTORY_SEEDED:{RUN_ID}"
        elif message == f"E2E_HISTORY_PROBE:{RUN_ID}":
            answer = (
                f"E2E_HISTORY_OK:{RUN_ID}"
                if any(
                    item["content"] == f"E2E_HISTORY_MARKER:{RUN_ID}"
                    for item in history[:-1]
                )
                else f"E2E_HISTORY_NONE:{RUN_ID}"
            )
        else:  # pragma: no cover - handler guard
            raise AssertionError(f"unexpected sync message: {message}")

        history.append(self._message("assistant", answer))
        response = _chat_response(answer, conversation_id)
        if self.invalid_first_response and message == f"E2E_HISTORY_MARKER:{RUN_ID}":
            response["model"] = "wrong-model"
        return httpx.Response(200, json=response)

    def _stream_chat(self, request: httpx.Request) -> httpx.Response:
        token = self._token(request)
        body = self._body(request)
        assert body["model"] == "ci-mock"
        self.models.append(body["model"])
        message = body["message"]

        if message == f"E2E_STREAM_TURN:{RUN_ID}":
            conversation_id = body["conversation_id"]
            history = self.histories[(token, conversation_id)]
            answer = f"E2E_STREAM_ACK:{RUN_ID}"
            history.extend(
                [
                    self._message("user", message),
                    self._message("assistant", answer),
                ]
            )
            done = _chat_response(answer, conversation_id)
            chunks = [
                _frame("route", 1, {"type": "route", "agent": "coordinator"}),
                _frame("token", 2, {"type": "token", "content": "partial"}),
                _frame("done", 3, {"type": "done", "response": done}),
            ]
            if self.duplicate_terminal:
                chunks.append(_frame("done", 4, {"type": "done", "response": done}))
            return httpx.Response(
                200,
                headers={
                    "Content-Type": "text/event-stream",
                    "X-Conversation-Id": conversation_id,
                },
                stream=RecordingStream(chunks),
            )

        assert message == f"E2E_ABORT_TURN:{RUN_ID}"
        conversation_id = body["conversation_id"]
        self.histories[(token, conversation_id)] = [self._message("user", message)]
        self.abort_stream = RecordingStream(
            [_frame("route", 1, {"type": "route", "agent": "coordinator"})]
        )
        return httpx.Response(
            200,
            headers={
                "Content-Type": "text/event-stream",
                "X-Conversation-Id": conversation_id,
            },
            stream=self.abort_stream,
        )

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "GET" and path == "/api/v1/chat/providers":
            return httpx.Response(
                200,
                json={
                    "default_model": "ci-mock",
                    "providers": [
                        {"provider_id": "openai_compatible", "models": ["ci-mock"]}
                    ],
                },
            )

        if request.method == "POST" and path == "/signup":
            return httpx.Response(200, json={"_id": "user"})

        if request.method == "POST" and path == "/login":
            username = self._body(request)["username"]
            token = OWNER_TOKEN if "_owner_" in username else OTHER_TOKEN
            return httpx.Response(
                200,
                json={
                    "access_token": token,
                    "refresh_token": "unused",
                    "token_type": "bearer",
                },
            )

        if request.method == "POST" and path == "/api/v1/chat/":
            return self._sync_chat(request)

        if request.method == "POST" and path == "/api/v1/chat/stream":
            return self._stream_chat(request)

        if request.method == "GET" and path.startswith("/api/v1/chat/conversation/"):
            token = self._token(request)
            conversation_id = path.rsplit("/", 1)[-1]
            history = self.histories.get((token, conversation_id))
            if history is None:
                return httpx.Response(404, json={"detail": "missing"})
            return httpx.Response(
                200,
                json={"conversation_id": conversation_id, "messages": history},
            )

        if request.method == "DELETE" and path.startswith("/api/v1/chat/conversation/"):
            token = self._token(request)
            conversation_id = path.rsplit("/", 1)[-1]
            self.deleted.append((token, conversation_id))
            self.histories.pop((token, conversation_id), None)
            return httpx.Response(200, json={"status": "deleted"})

        raise AssertionError(f"unexpected request: {request.method} {request.url}")


def _config() -> e2e.E2EConfig:
    return e2e.E2EConfig(
        base_url="http://toolkit.test",
        hagent_url="http://bridge.test",
        model="ci-mock",
        request_timeout_seconds=5,
        abort_settle_seconds=0,
    )


@pytest.mark.asyncio
async def test_contract_smoke_covers_model_history_sse_persistence_abort_and_cleanup():
    backend = ContractBackend()
    report = await e2e.run_smoke(
        _config(),
        transport=httpx.MockTransport(backend),
        run_id=RUN_ID,
    )

    assert report.model == "ci-mock"
    assert report.event_types == ("route", "token", "done")
    assert report.cleanup_count == 3
    assert backend.models == ["ci-mock"] * 5
    assert set(backend.deleted) == {
        (OWNER_TOKEN, OWNER_CONVERSATION),
        (OTHER_TOKEN, OWNER_CONVERSATION),
        (OWNER_TOKEN, ABORT_CONVERSATION),
    }
    assert backend.abort_stream is not None
    assert backend.abort_stream.closed is True


@pytest.mark.asyncio
async def test_contract_smoke_cleans_up_when_sse_has_duplicate_terminal():
    backend = ContractBackend(duplicate_terminal=True)

    with pytest.raises(e2e.E2EFailure, match="terminal event is not last"):
        await e2e.run_smoke(
            _config(),
            transport=httpx.MockTransport(backend),
            run_id=RUN_ID,
        )

    assert set(backend.deleted) == {
        (OWNER_TOKEN, OWNER_CONVERSATION),
        (OTHER_TOKEN, OWNER_CONVERSATION),
    }


@pytest.mark.asyncio
async def test_contract_smoke_cleans_preallocated_id_when_first_response_is_invalid():
    backend = ContractBackend(invalid_first_response=True)

    with pytest.raises(e2e.E2EFailure, match="requested model was not preserved"):
        await e2e.run_smoke(
            _config(),
            transport=httpx.MockTransport(backend),
            run_id=RUN_ID,
        )

    assert backend.deleted == [(OWNER_TOKEN, OWNER_CONVERSATION)]


def test_sse_validation_rejects_non_monotonic_ids():
    response = _chat_response("answer", OWNER_CONVERSATION)
    events = [
        e2e.SSEEvent("route", 2, {"type": "route"}),
        e2e.SSEEvent("done", 2, {"type": "done", "response": response}),
    ]

    with pytest.raises(e2e.E2EFailure, match="strictly increasing"):
        e2e.validate_sse_sequence(events)


def test_sse_validation_rejects_work_before_route():
    response = _chat_response("answer", OWNER_CONVERSATION)
    events = [
        e2e.SSEEvent("token", 1, {"type": "token", "content": "early"}),
        e2e.SSEEvent("route", 2, {"type": "route", "agent": "coordinator"}),
        e2e.SSEEvent("done", 3, {"type": "done", "response": response}),
    ]

    with pytest.raises(e2e.E2EFailure, match="work event occurred before route"):
        e2e.validate_sse_sequence(events)


def _completion_text(response: httpx.Response) -> str:
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def test_mock_llm_emits_owner_scoped_history_markers(mock_llm_server):
    url = f"{mock_llm_server.api_base_url}/chat/completions"

    seeded = httpx.post(
        url,
        json={
            "model": "mock-model",
            "messages": [{"role": "user", "content": "E2E_HISTORY_MARKER:probe"}],
        },
        timeout=3,
    )
    assert _completion_text(seeded) == "E2E_HISTORY_SEEDED:probe"

    owner_probe = httpx.post(
        url,
        json={
            "model": "mock-model",
            "messages": [
                {"role": "user", "content": "E2E_HISTORY_MARKER:probe"},
                {"role": "assistant", "content": "E2E_HISTORY_SEEDED:probe"},
                {"role": "user", "content": "E2E_HISTORY_PROBE:probe"},
            ],
        },
        timeout=3,
    )
    assert _completion_text(owner_probe) == "E2E_HISTORY_OK:probe"

    other_probe = httpx.post(
        url,
        json={
            "model": "mock-model",
            "messages": [{"role": "user", "content": "E2E_HISTORY_PROBE:probe"}],
        },
        timeout=3,
    )
    assert _completion_text(other_probe) == "E2E_HISTORY_NONE:probe"


def test_mock_llm_emits_openai_compatible_text_stream(mock_llm_server):
    url = f"{mock_llm_server.api_base_url}/chat/completions"

    with httpx.stream(
        "POST",
        url,
        json={
            "model": "mock-model",
            "messages": [{"role": "user", "content": "E2E_STREAM_TURN:probe"}],
            "stream": True,
        },
        timeout=3,
    ) as response:
        response.raise_for_status()
        assert "text/event-stream" in response.headers["content-type"]
        data_lines = [
            line.removeprefix("data: ")
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]

    assert data_lines[-1] == "[DONE]"
    chunks = [json.loads(line) for line in data_lines[:-1]]
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
    streamed_text = "".join(
        str(choice["delta"].get("content") or "")
        for chunk in chunks
        for choice in chunk["choices"]
    )
    assert streamed_text == "E2E_STREAM_ACK:probe"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
