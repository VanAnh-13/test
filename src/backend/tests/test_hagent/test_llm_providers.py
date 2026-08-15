"""
Kiểm thử đơn vị cho các plugin LLM provider và LLMClient.

Tất cả kiểm thử đều mô phỏng provider/model — không cần API key thật.
Phạm vi kiểm thử:
  1. Hàm tạo get_provider() chọn đúng lớp provider
  2. LLMProvider retry logic (RetryableError → retry, non-retryable → fail fast)
  3. Đường gọi mô phỏng cho OpenAIProvider, AnthropicProvider, OllamaProvider
  4. Truyền Strategy vào LLMClient
  5. count_tokens fallback
  6. OllamaProvider raise ValueError khi thiếu base_url
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from hagent.agent.llm.client import LLMClient
from hagent.agent.llm.providers import (
    AnthropicProvider,
    LLMProvider,
    OllamaProvider,
    OpenAIProvider,
    get_provider,
)
from hagent.agent.llm.providers.base import RetryableError

# ── Hàm hỗ trợ ────────────────────────────────────────────────────────────────


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_config(
    name: str = "test-model",
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    api_key: str | None = "sk-test",
    base_url: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Any:
    """Tạo ModelConfig stub (không cần hagent.yaml)."""
    from hagent.agent.llm.config import ModelConfig

    return ModelConfig(
        name=name,
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
        max_tokens=256,
        is_default=False,
        extra=extra or {},
    )


# ── Provider giả lập ──────────────────────────────────────────────────────────


class _StubProvider(LLMProvider):
    """Provider giả để test base class logic (retry, count_tokens...)."""

    def __init__(self, responses=None, fail_times=0, **kwargs):
        super().__init__(**kwargs)
        self._responses = responses or [{"content": "hello"}]
        self._call_count = 0
        self._fail_times = fail_times

    def build_chat_model(self, callbacks=None, *, max_retries=0):
        return MagicMock(callbacks=callbacks, max_retries=max_retries)

    async def _generate_raw(self, messages, **kwargs):
        self._call_count += 1
        if self._call_count <= self._fail_times:
            raise RetryableError("transient", status_code=429)
        idx = min(self._call_count - 1, len(self._responses) - 1)
        return self._responses[idx]

    async def _stream_raw(self, messages, **kwargs) -> AsyncIterator[str]:
        self._call_count += 1
        if self._call_count <= self._fail_times:
            raise RetryableError("transient stream", status_code=503)
        for token in ["he", "ll", "o"]:
            yield token

    def _count_tokens_raw(self, messages):
        return sum(len(m.get("content", "")) for m in messages)


# ── Kiểm thử: hàm tạo get_provider() ─────────────────────────────────────────


class TestGetProviderFactory:
    def test_openai_provider_selected(self):
        cfg = _make_config(provider="openai")
        p = get_provider(cfg)
        assert isinstance(p, OpenAIProvider)

    def test_openai_compatible_maps_to_openai_provider(self):
        cfg = _make_config(
            provider="openai_compatible",
            base_url="http://localhost:8080/v1",
            api_key=None,
        )
        p = get_provider(cfg)
        assert isinstance(p, OpenAIProvider)

    def test_anthropic_provider_selected(self):
        cfg = _make_config(provider="anthropic", model="claude-3-5-haiku-20241022")
        p = get_provider(cfg)
        assert isinstance(p, AnthropicProvider)

    def test_ollama_provider_selected(self):
        cfg = _make_config(
            provider="ollama",
            model="llama3",
            api_key=None,
            base_url="http://localhost:11434",
        )
        p = get_provider(cfg)
        assert isinstance(p, OllamaProvider)

    def test_unknown_provider_raises_value_error(self):
        cfg = _make_config(provider="cohere")
        with pytest.raises(ValueError, match="cohere"):
            get_provider(cfg)

    def test_runtime_factory_delegates_to_provider_strategy(self):
        from hagent.agent.llm.config import _build_model

        cfg = _make_config(api_key="$REFAC005_RUNTIME_KEY")
        expected_model = object()
        strategy = MagicMock()
        strategy.max_retries = 3
        strategy.build_chat_model.return_value = expected_model
        callbacks = [object()]

        with patch(
            "hagent.agent.llm.providers.get_provider",
            return_value=strategy,
        ) as provider_factory:
            model = _build_model(
                "openai",
                cfg,
                "",
                0.25,
                512,
                callbacks=callbacks,
            )

        assert model is expected_model
        effective_config = provider_factory.call_args.args[0]
        assert effective_config.provider == "openai"
        assert effective_config.api_key == "$REFAC005_RUNTIME_KEY"
        assert effective_config.temperature == 0.25
        assert effective_config.max_tokens == 512
        strategy.build_chat_model.assert_called_once_with(
            callbacks=callbacks,
            max_retries=3,
        )


# ── Kiểm thử: logic retry của LLMProvider ────────────────────────────────────


class TestLLMProviderRetry:
    def test_generate_succeeds_first_try(self):
        provider = _StubProvider(responses=[{"content": "ok"}])
        result = _run(provider.generate([{"role": "user", "content": "hi"}]))
        assert result["content"] == "ok"
        assert provider._call_count == 1

    def test_generate_retries_on_retryable_error(self):
        # fail_times=2 → fail on attempt 1,2 → succeed on attempt 3
        provider = _StubProvider(
            responses=[{"content": "recovered"}],
            fail_times=2,
            max_retries=3,
            base_delay=0.0,
        )
        result = _run(provider.generate([{"role": "user", "content": "hi"}]))
        assert result["content"] == "recovered"
        assert provider._call_count == 3

    def test_generate_exhausts_retries_and_raises(self):
        provider = _StubProvider(
            fail_times=10,  # luôn lỗi
            max_retries=2,
            base_delay=0.0,
        )
        with pytest.raises(RuntimeError, match="retries"):
            _run(provider.generate([{"role": "user", "content": "hi"}]))

    def test_generate_non_retryable_raises_immediately(self):
        class _BrokenProvider(_StubProvider):
            async def _generate_raw(self, messages, **kwargs):
                raise ValueError("bad input — not retryable")

        provider = _BrokenProvider(max_retries=3, base_delay=0.0)
        with pytest.raises(RuntimeError, match="bad input"):
            _run(provider.generate([{"role": "user", "content": "hi"}]))
        # Phải fail ngay, không retry
        assert provider._call_count == 0

    @pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ConnectTimeout])
    def test_generate_retries_httpx_transport_errors(self, error_type):
        request = httpx.Request("POST", "https://example.test/v1/messages")

        class _TransportProvider(_StubProvider):
            async def _generate_raw(self, messages, **kwargs):
                self._call_count += 1
                if self._call_count == 1:
                    raise error_type("Lỗi kết nối tạm thời", request=request)
                return {"content": "đã phục hồi"}

        provider = _TransportProvider(max_retries=1, base_delay=0.0)

        result = _run(provider.generate([{"role": "user", "content": "xin chào"}]))

        assert result == {"content": "đã phục hồi"}
        assert provider._call_count == 2

    def test_recognizes_sdk_transport_errors_and_wrapped_causes(self):
        import anthropic
        import openai

        request = httpx.Request("POST", "https://example.test/v1/messages")
        provider = _StubProvider()
        sdk_errors = [
            openai.APIConnectionError(request=request),
            openai.APITimeoutError(request=request),
            anthropic.APIConnectionError(request=request),
            anthropic.APITimeoutError(request=request),
        ]

        assert all(provider._is_retryable_error(exc) for exc in sdk_errors)

        wrapped = RuntimeError("Lỗi đã được SDK bọc")
        wrapped.__cause__ = httpx.ConnectError(
            "Mất kết nối",
            request=request,
        )
        assert provider._is_retryable_error(wrapped)

    def test_stream_retries_on_retryable_error(self):
        provider = _StubProvider(fail_times=1, max_retries=2, base_delay=0.0)

        async def _collect():
            tokens = []
            async for t in provider.stream([{"role": "user", "content": "hi"}]):
                tokens.append(t)
            return tokens

        tokens = _run(_collect())
        assert tokens == ["he", "ll", "o"]
        assert provider._call_count == 2

    def test_stream_exhausts_retries_and_raises(self):
        provider = _StubProvider(fail_times=10, max_retries=1, base_delay=0.0)

        async def _collect():
            async for _ in provider.stream([{"role": "user", "content": "hi"}]):
                pass

        with pytest.raises(RuntimeError, match="retries"):
            _run(_collect())

    def test_stream_does_not_retry_after_emitting_output(self):
        class _InterruptedProvider(_StubProvider):
            async def _stream_raw(self, messages, **kwargs) -> AsyncIterator[str]:
                self._call_count += 1
                yield "partial"
                raise RetryableError("connection dropped", status_code=503)

        provider = _InterruptedProvider(max_retries=2, base_delay=0.0)

        async def _collect():
            tokens = []
            async for token in provider.stream([{"role": "user", "content": "hi"}]):
                tokens.append(token)
            return tokens

        with pytest.raises(RuntimeError, match="lặp token"):
            _run(_collect())
        assert provider._call_count == 1


# ── Kiểm thử: count_tokens ────────────────────────────────────────────────────


class TestCountTokens:
    def test_count_tokens_calls_raw(self):
        provider = _StubProvider()
        messages = [{"role": "user", "content": "hello world"}]
        count = provider.count_tokens(messages)
        assert count == len("hello world")

    def test_count_tokens_returns_minus_one_on_error(self):
        class _FailingProvider(_StubProvider):
            def _count_tokens_raw(self, messages):
                raise RuntimeError("no tiktoken")

        provider = _FailingProvider()
        assert provider.count_tokens([{"role": "user", "content": "x"}]) == -1

    @pytest.mark.parametrize(
        ("provider", "expected_name"),
        [
            (OpenAIProvider(_make_config()), "openai"),
            (
                AnthropicProvider(
                    _make_config(
                        provider="anthropic",
                        model="claude-3-5-haiku-20241022",
                    )
                ),
                "anthropic",
            ),
            (
                OllamaProvider(
                    _make_config(
                        provider="ollama",
                        model="llama3",
                        api_key=None,
                        base_url="http://localhost:11434",
                    )
                ),
                "ollama",
            ),
        ],
    )
    def test_provider_token_count_uses_model_counter(self, provider, expected_name):
        mock_model = MagicMock()
        mock_model.get_num_tokens_from_messages.return_value = 42
        messages = [{"role": "user", "content": "hello"}]

        with patch.object(provider, "build_chat_model", return_value=mock_model):
            assert provider.count_tokens(messages) == 42

        assert provider.provider_name == expected_name
        mock_model.get_num_tokens_from_messages.assert_called_once()


class TestBuiltInProviderSharedBehavior:
    @pytest.mark.parametrize(
        "provider",
        [
            OpenAIProvider(_make_config()),
            AnthropicProvider(
                _make_config(
                    provider="anthropic",
                    model="claude-3-5-haiku-20241022",
                )
            ),
            OllamaProvider(
                _make_config(
                    provider="ollama",
                    model="llama3",
                    api_key=None,
                    base_url="http://localhost:11434",
                )
            ),
        ],
    )
    def test_stream_forwards_callbacks_and_yields_only_non_empty_chunks(self, provider):
        callbacks = [object()]
        observed: dict[str, Any] = {}

        class _StreamingModel:
            async def astream(self, messages, **kwargs):
                observed["messages"] = messages
                observed["kwargs"] = kwargs
                for content in ("xin", "", " chào"):
                    yield MagicMock(content=content)

        async def _collect():
            chunks = []
            async for chunk in provider.stream(
                [{"role": "user", "content": "hello"}],
                callbacks=callbacks,
                request_timeout=5,
            ):
                chunks.append(chunk)
            return chunks

        with patch.object(
            provider,
            "build_chat_model",
            return_value=_StreamingModel(),
        ) as build_chat_model:
            chunks = _run(_collect())

        assert chunks == ["xin", " chào"]
        build_chat_model.assert_called_once_with(callbacks=callbacks)
        assert observed["kwargs"] == {"request_timeout": 5}
        assert len(observed["messages"]) == 1


# ── Kiểm thử: truyền Strategy vào LLMClient ──────────────────────────────────


class TestLLMClient:
    def test_client_delegates_generate_to_provider(self):
        stub = _StubProvider(responses=[{"content": "injected"}])
        client = LLMClient(stub)
        result = _run(client.generate([{"role": "user", "content": "test"}]))
        assert result["content"] == "injected"

    def test_client_delegates_stream_to_provider(self):
        stub = _StubProvider()
        client = LLMClient(stub)

        async def _collect():
            tokens = []
            async for t in client.stream([{"role": "user", "content": "test"}]):
                tokens.append(t)
            return tokens

        assert _run(_collect()) == ["he", "ll", "o"]

    def test_client_delegates_count_tokens(self):
        stub = _StubProvider()
        client = LLMClient(stub)
        assert client.count_tokens([{"role": "user", "content": "abc"}]) == 3

    def test_client_from_config_creates_correct_provider(self):
        cfg = _make_config(provider="openai")
        client = LLMClient.from_config(cfg)
        assert isinstance(client.provider, OpenAIProvider)
        assert client.provider_name == "openai"

    def test_client_repr(self):
        stub = _StubProvider()
        client = LLMClient(stub)
        assert "LLMClient" in repr(client)


# ── Kiểm thử: lời gọi mô phỏng OpenAIProvider ────────────────────────────────


class TestOpenAIProviderMock:
    @pytest.mark.parametrize(
        "provider_name", ["openai_compatible", "OPENAI_COMPATIBLE"]
    )
    def test_openai_compatible_requires_base_url(self, provider_name):
        cfg = _make_config(provider=provider_name, base_url=None, api_key=None)
        provider = OpenAIProvider(cfg)

        with pytest.raises(ValueError, match="base_url"):
            provider.build_chat_model()

    def test_openai_compatible_without_configured_key_uses_fallback(self):
        cfg = _make_config(
            provider="openai_compatible",
            base_url="http://localhost:8080/v1",
            api_key=None,
            extra={"api_key": "extra-key-must-not-win"},
        )
        provider = OpenAIProvider(cfg)

        with patch("langchain_openai.ChatOpenAI") as chat_openai:
            provider.build_chat_model()

        assert chat_openai.call_args.kwargs["api_key"] == "not-needed"

    def test_missing_configured_api_key_does_not_use_global_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "global-key-must-not-be-used")
        monkeypatch.delenv("REFAC005_OPENAI_KEY", raising=False)
        cfg = _make_config(provider="openai", api_key="$REFAC005_OPENAI_KEY")
        provider = OpenAIProvider(cfg)

        with pytest.raises(ValueError, match="không khả dụng"):
            provider.build_chat_model()

    def test_sdk_retries_are_disabled(self):
        cfg = _make_config(provider="openai")
        provider = OpenAIProvider(cfg)

        with patch("langchain_openai.ChatOpenAI") as chat_openai:
            provider.build_chat_model()

        assert chat_openai.call_args.kwargs["max_retries"] == 0

    def test_generate_builds_response_dict(self):
        cfg = _make_config(provider="openai", model="gpt-4o-mini")
        provider = OpenAIProvider(cfg, max_retries=0)

        mock_response = MagicMock()
        mock_response.content = "mocked content"
        mock_response.response_metadata = {"finish_reason": "stop"}

        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=mock_response)

        with patch.object(provider, "build_chat_model", return_value=mock_model):
            result = _run(provider.generate([{"role": "user", "content": "hello"}]))

        assert result["content"] == "mocked content"
        assert result["provider"] == "openai"
        assert result["finish_reason"] == "stop"

    def test_openai_compatible_preserves_openai_response_label(self):
        cfg = _make_config(
            provider="openai_compatible",
            base_url="http://localhost:8080/v1",
            api_key=None,
        )
        provider = OpenAIProvider(cfg, max_retries=0)
        mock_response = MagicMock(
            content="compatible response",
            response_metadata={"finish_reason": "stop"},
        )
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=mock_response)

        with patch.object(provider, "build_chat_model", return_value=mock_model):
            result = _run(provider.generate([{"role": "user", "content": "hello"}]))

        assert provider.provider_name == "openai_compatible"
        assert result["provider"] == "openai"

    def test_generate_retries_on_429(self):
        cfg = _make_config(provider="openai")
        provider = OpenAIProvider(cfg, max_retries=2, base_delay=0.0)

        call_count = 0

        async def _mock_ainvoke(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                exc = Exception("rate limited")
                exc.status_code = 429  # type: ignore[attr-defined]
                raise exc
            mock_r = MagicMock()
            mock_r.content = "ok"
            mock_r.response_metadata = {}
            return mock_r

        mock_model = MagicMock()
        mock_model.ainvoke = _mock_ainvoke

        with patch.object(provider, "build_chat_model", return_value=mock_model):
            result = _run(provider.generate([{"role": "user", "content": "hi"}]))

        assert result["content"] == "ok"
        assert call_count == 3


@pytest.mark.parametrize(
    ("provider_cls", "model", "chat_model_path"),
    [
        (OpenAIProvider, "gpt-4o-mini", "langchain_openai.ChatOpenAI"),
        (
            AnthropicProvider,
            "claude-3-5-haiku-20241022",
            "langchain_anthropic.ChatAnthropic",
        ),
    ],
)
def test_credentialed_providers_share_model_kwargs_override_order(
    provider_cls,
    model,
    chat_model_path,
):
    callbacks = [object()]
    cfg = _make_config(
        provider=provider_cls.__name__.replace("Provider", "").lower(),
        model=model,
        api_key="resolved-key",
        extra={
            "model": "extra-model",
            "temperature": 0.75,
            "max_tokens": 1024,
            "max_retries": 99,
            "api_key": "extra-key",
            "callbacks": ["extra-callback"],
        },
    )
    provider = provider_cls(cfg)

    with patch(chat_model_path) as chat_model:
        provider.build_chat_model(callbacks=callbacks, max_retries=3)

    assert chat_model.call_args.kwargs == {
        "model": "extra-model",
        "temperature": 0.75,
        "max_tokens": 1024,
        "max_retries": 3,
        "api_key": "resolved-key",
        "callbacks": callbacks,
    }


# ── Kiểm thử: lời gọi mô phỏng AnthropicProvider ─────────────────────────────


class TestAnthropicProviderMock:
    def test_missing_configured_api_key_does_not_use_global_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "global-key-must-not-be-used")
        monkeypatch.delenv("REFAC005_ANTHROPIC_KEY", raising=False)
        cfg = _make_config(
            provider="anthropic",
            model="claude-3-5-haiku-20241022",
            api_key="$REFAC005_ANTHROPIC_KEY",
        )
        provider = AnthropicProvider(cfg)

        with pytest.raises(ValueError, match="không khả dụng"):
            provider.build_chat_model()

    def test_sdk_retries_are_disabled(self):
        cfg = _make_config(provider="anthropic", model="claude-3-5-haiku-20241022")
        provider = AnthropicProvider(cfg)

        with patch("langchain_anthropic.ChatAnthropic") as chat_anthropic:
            provider.build_chat_model()

        assert chat_anthropic.call_args.kwargs["max_retries"] == 0

    def test_generate_returns_content(self):
        cfg = _make_config(provider="anthropic", model="claude-3-5-haiku-20241022")
        provider = AnthropicProvider(cfg, max_retries=0)

        mock_response = MagicMock()
        mock_response.content = "claude says hi"
        mock_response.response_metadata = {"stop_reason": "end_turn"}

        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=mock_response)

        with patch.object(provider, "build_chat_model", return_value=mock_model):
            result = _run(provider.generate([{"role": "user", "content": "hello"}]))

        assert result["content"] == "claude says hi"
        assert result["provider"] == "anthropic"
        assert result["finish_reason"] == "end_turn"

    def test_529_is_retryable(self):
        """Anthropic 529 (overloaded) phải được retry."""
        cfg = _make_config(provider="anthropic", model="claude-3-5-haiku-20241022")
        provider = AnthropicProvider(cfg, max_retries=1, base_delay=0.0)
        call_count = 0

        async def _mock_ainvoke(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            exc = Exception("overloaded")
            exc.status_code = 529  # type: ignore[attr-defined]
            raise exc

        mock_model = MagicMock()
        mock_model.ainvoke = _mock_ainvoke

        with (
            patch.object(provider, "build_chat_model", return_value=mock_model),
            pytest.raises(RuntimeError, match="retries"),
        ):
            _run(provider.generate([{"role": "user", "content": "hi"}]))

        assert call_count == 2  # 1 + 1 retry


# ── Kiểm thử: OllamaProvider ──────────────────────────────────────────────────


class TestOllamaProviderMock:
    def test_http_transport_thuc_hien_dung_so_lan_retry(self):
        import httpcore
        from httpcore._backends.mock import MockStream

        class _CountingBackend:
            def __init__(self) -> None:
                self.attempts = 0

            def connect_tcp(
                self,
                host,
                port,
                timeout=None,
                local_address=None,
                socket_options=None,
            ):
                self.attempts += 1
                if self.attempts <= 2:
                    raise httpcore.ConnectError("Lỗi kết nối tạm thời")
                return MockStream([b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK"])

            def connect_unix_socket(
                self,
                path,
                timeout=None,
                socket_options=None,
            ):
                raise AssertionError("Không dùng Unix socket trong kiểm thử này")

            def sleep(self, seconds):
                return None

        cfg = _make_config(
            provider="ollama",
            model="llama3",
            api_key=None,
            base_url="http://localhost:11434",
            extra={
                "max_retries": 99,
                "sync_client_kwargs": {"timeout": 5.0, "transport": object()},
                "async_client_kwargs": {"timeout": 5.0, "transport": object()},
            },
        )
        provider = OllamaProvider(cfg)

        with patch("langchain_ollama.ChatOllama") as chat_ollama:
            provider.build_chat_model(max_retries=2)

        model_kwargs = chat_ollama.call_args.kwargs
        assert "max_retries" not in model_kwargs
        assert model_kwargs["sync_client_kwargs"]["timeout"] == 5.0
        assert model_kwargs["async_client_kwargs"]["timeout"] == 5.0

        transport = model_kwargs["sync_client_kwargs"]["transport"]
        backend = _CountingBackend()
        transport._pool._network_backend = backend
        with httpx.Client(transport=transport) as client:
            response = client.get("http://example.test")

        assert response.status_code == 200
        assert backend.attempts == 3

    def test_missing_base_url_raises_value_error(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        cfg = _make_config(provider="ollama", base_url=None, api_key=None)
        provider = OllamaProvider(cfg, max_retries=0)

        async def _call():
            # _resolve_base_url phải raise trước khi gọi model
            result = await provider._generate_raw([{"role": "user", "content": "x"}])
            return result

        with pytest.raises(ValueError, match="base_url"):
            _run(_call())

    def test_generate_with_base_url(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
        cfg = _make_config(
            provider="ollama", model="llama3", api_key=None, base_url=None
        )
        provider = OllamaProvider(cfg, max_retries=0)

        mock_response = MagicMock()
        mock_response.content = "ollama response"
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=mock_response)

        with patch.object(provider, "build_chat_model", return_value=mock_model):
            result = _run(provider.generate([{"role": "user", "content": "hi"}]))

        assert result["content"] == "ollama response"
        assert result["provider"] == "ollama"
        assert result["finish_reason"] == "stop"
