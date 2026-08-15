"""
LLMProvider — lớp cơ sở trừu tượng cho tất cả plugin provider.

Logic thử lại và chờ lùi được đặt tại đây để các lớp con không bị trùng lặp.
Lớp con chỉ cần triển khai:
  _generate_raw()   — 1 LLM call, không retry
  _stream_raw()     — async generator, không retry
  _count_tokens_raw() — ước tính số token

Lý do thiết kế:
  - Mẫu chiến lược: ``LLMClient`` nhận đối tượng ``LLMProvider`` thay vì
    chuỗi if-elif theo tên provider.
  - ``generate()`` / ``stream()`` ở lớp cơ sở bọc ``_raw`` bằng cơ chế thử lại.
  - Thời gian chờ tăng theo cấp số nhân bằng Python thuần, không thêm dependency.
  - ``count_tokens()`` là tùy chọn; lớp con trả -1 nếu không hỗ trợ.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Self

import structlog

if TYPE_CHECKING:
    from hagent.agent.llm.config import ModelConfig

logger = structlog.get_logger(__name__)

# ── Giá trị retry mặc định ────────────────────────────────────────────────────

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BASE_DELAY = 1.0  # giây
_DEFAULT_MAX_DELAY = 30.0  # giây
_DEFAULT_BACKOFF_FACTOR = 2.0

# Mã trạng thái HTTP và ngoại lệ được phép thử lại.
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_RETRYABLE_SDK_ERRORS = frozenset(
    {
        ("openai", "APIConnectionError"),
        ("openai", "APITimeoutError"),
        ("anthropic", "APIConnectionError"),
        ("anthropic", "APITimeoutError"),
    }
)


# ── Ngoại lệ đánh dấu ─────────────────────────────────────────────────────────


class RetryableError(RuntimeError):
    """Ném từ _generate_raw/_stream_raw để báo cho lớp cơ sở thử lại."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


# ── Lớp cơ sở ─────────────────────────────────────────────────────────────────


class LLMProvider(ABC):
    """
    Lớp cơ sở trừu tượng cho các plugin LLM provider.

    Lớp con ghi đè:
      _generate_raw(messages, **kwargs) -> dict
      _stream_raw(messages, **kwargs)   -> AsyncIterator[str]
      _count_tokens_raw(messages)       -> int   (tùy chọn; trả về -1)

    API công khai (đã bọc retry):
      generate(messages, **kwargs) -> dict
      stream(messages, **kwargs)   -> AsyncIterator[str]
      count_tokens(messages)       -> int
    """

    def __init__(
        self,
        *,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        base_delay: float = _DEFAULT_BASE_DELAY,
        max_delay: float = _DEFAULT_MAX_DELAY,
        backoff_factor: float = _DEFAULT_BACKOFF_FACTOR,
    ) -> None:
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor

    retryable_status_codes = _RETRYABLE_STATUS_CODES

    # ── Phương thức trừu tượng (lớp con triển khai) ────────────────────────────

    @abstractmethod
    def build_chat_model(
        self,
        callbacks: list | None = None,
        *,
        max_retries: int = 0,
    ) -> Any:
        """Tạo model LangChain của provider cho đường tích hợp runtime."""

    @abstractmethod
    async def _generate_raw(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Gọi LLM một lần, không thử lại; phát sinh RetryableError khi cần."""

    @abstractmethod
    async def _stream_raw(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Trả token dạng luồng từ LLM, không thử lại; phát sinh RetryableError khi cần."""

    def _count_tokens_raw(self, messages: list[dict[str, str]]) -> int:
        """Ước tính số token. Lớp con có thể ghi đè để dùng tiktoken hoặc API đếm.
        Trả về -1 nếu provider không hỗ trợ đếm token.
        """
        return -1

    @staticmethod
    def _to_langchain_messages(messages: list[dict[str, str]]) -> list[Any]:
        """Chuyển message trung lập với provider sang message của LangChain."""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        converted = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                converted.append(SystemMessage(content=content))
            elif role == "assistant":
                converted.append(AIMessage(content=content))
            else:
                converted.append(HumanMessage(content=content))
        return converted

    # ── API công khai (đã bọc retry) ──────────────────────────────────────────

    async def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Gọi LLM với retry/backoff tự động.

        Kết quả:
            Từ điển có ít nhất khóa ``"content": str`` chứa nội dung trả lời của model.
        """
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await self._generate_raw(messages, **kwargs)
            except Exception as exc:
                if not self._is_retryable_error(exc):
                    raise RuntimeError(
                        f"{type(self).__name__}.generate failed: {exc}"
                    ) from exc
                last_exc = exc
                if attempt < self.max_retries:
                    delay = self._calc_delay(attempt)
                    logger.warning(
                        "%s.generate retry %d/%d sau %.1fs — %s",
                        type(self).__name__,
                        attempt + 1,
                        self.max_retries,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "%s.generate exhausted %d retries: %s",
                        type(self).__name__,
                        self.max_retries,
                        exc,
                    )
        raise RuntimeError(
            f"{type(self).__name__}.generate failed after {self.max_retries} retries"
        ) from last_exc

    async def stream(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Trả token dạng luồng với retry/backoff cho lỗi kết nối trước khi bắt đầu.

        Giá trị trả dần:
            Các token dạng chuỗi từ kết quả của model.
        """
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            emitted_token = False
            try:
                async for token in self._stream_raw(messages, **kwargs):
                    emitted_token = True
                    yield token
                return
            except Exception as exc:
                if emitted_token:
                    raise RuntimeError(
                        f"{type(self).__name__}.stream bị gián đoạn sau khi đã trả dữ liệu; "
                        "không retry để tránh lặp token"
                    ) from exc
                if not self._is_retryable_error(exc):
                    raise RuntimeError(
                        f"{type(self).__name__}.stream failed: {exc}"
                    ) from exc
                last_exc = exc
                if attempt < self.max_retries:
                    delay = self._calc_delay(attempt)
                    logger.warning(
                        "%s.stream retry %d/%d sau %.1fs — %s",
                        type(self).__name__,
                        attempt + 1,
                        self.max_retries,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "%s.stream exhausted %d retries: %s",
                        type(self).__name__,
                        self.max_retries,
                        exc,
                    )
                    break
        raise RuntimeError(
            f"{type(self).__name__}.stream failed after {self.max_retries} retries"
        ) from last_exc

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        """Trả về ước tính số token. -1 nếu không hỗ trợ."""
        try:
            return self._count_tokens_raw(messages)
        except Exception as exc:  # noqa: BLE001
            logger.debug("%s.count_tokens failed: %s", type(self).__name__, exc)
            return -1

    # ── Hàm hỗ trợ ────────────────────────────────────────────────────────────

    def _calc_delay(self, attempt: int) -> float:
        """Tính thời gian chờ tăng theo cấp số nhân và giới hạn trên."""
        delay = self.base_delay * (self.backoff_factor**attempt)
        return min(delay, self.max_delay)

    @staticmethod
    def _httpx_retry_client_kwargs(
        max_retries: int,
        *,
        sync_client_kwargs: dict[str, Any] | None = None,
        async_client_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Tạo transport HTTPX với một chính sách retry kết nối thống nhất."""
        import httpx

        sync_options = dict(sync_client_kwargs or {})
        async_options = dict(async_client_kwargs or {})
        sync_options["transport"] = httpx.HTTPTransport(retries=max_retries)
        async_options["transport"] = httpx.AsyncHTTPTransport(retries=max_retries)
        return {
            "sync_client_kwargs": sync_options,
            "async_client_kwargs": async_options,
        }

    def _is_retryable_error(self, exc: Exception) -> bool:
        """Nhận diện lỗi tạm thời, kể cả lỗi transport bị bọc bởi SDK."""
        import httpx

        current: BaseException | None = exc
        visited: set[int] = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            if isinstance(
                current,
                (RetryableError, ConnectionError, TimeoutError, httpx.TransportError),
            ):
                return True
            if getattr(current, "status_code", None) in self.retryable_status_codes:
                return True

            error_type = type(current)
            sdk_error = (error_type.__module__.split(".", 1)[0], error_type.__name__)
            if sdk_error in _RETRYABLE_SDK_ERRORS:
                return True
            current = current.__cause__ or current.__context__
        return False

    @property
    def provider_name(self) -> str:
        """Tên nhà cung cấp model dùng cho log và chỉ số đo lường."""
        return type(self).__name__.replace("Provider", "").lower()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"max_retries={self.max_retries}, "
            f"base_delay={self.base_delay})"
        )


class LangChainChatProvider(LLMProvider, ABC):
    """Adapter dùng chung cho lời gọi thô của các nhà cung cấp chat LangChain tích hợp sẵn."""

    _response_provider_name: str | None = None
    _finish_reason_key: str | None = "finish_reason"
    _default_finish_reason = "stop"

    def __init__(self, config: ModelConfig, **retry_kwargs: Any) -> None:
        super().__init__(**retry_kwargs)
        self._config = config

    @classmethod
    def from_config(cls, config: ModelConfig) -> Self:
        return cls(config)

    def _build_credentialed_model_kwargs(
        self,
        callbacks: list | None = None,
        *,
        max_retries: int = 0,
        fallback_api_key: str | None = None,
    ) -> dict[str, Any]:
        """Tạo bộ kwargs chung cho các client chat model cần thông tin xác thực."""
        api_key = self._config.resolve_api_key()
        if self._config.api_key and not api_key:
            raise ValueError(
                f"API key đã cấu hình cho model '{self._config.name}' không khả dụng"
            )

        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            **(self._config.extra or {}),
        }
        kwargs["max_retries"] = max_retries
        if api_key:
            kwargs["api_key"] = api_key
        elif fallback_api_key is not None:
            kwargs["api_key"] = fallback_api_key
        if callbacks:
            kwargs["callbacks"] = callbacks
        return kwargs

    def _prepare_model_call(
        self,
        messages: list[dict[str, str]],
        kwargs: dict[str, Any],
    ) -> tuple[Any, list[Any]]:
        langchain_messages = self._to_langchain_messages(messages)
        callbacks = kwargs.pop("callbacks", None)
        model = self.build_chat_model(callbacks=callbacks)
        return model, langchain_messages

    def _response_finish_reason(self, response: Any) -> str:
        if self._finish_reason_key is None:
            return self._default_finish_reason
        return getattr(response, "response_metadata", {}).get(
            self._finish_reason_key,
            self._default_finish_reason,
        )

    async def _generate_raw(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        model, langchain_messages = self._prepare_model_call(messages, kwargs)
        response = await model.ainvoke(langchain_messages, **kwargs)
        return {
            "content": response.content,
            "model": self._config.model,
            "provider": self._response_provider_name or self.provider_name,
            "finish_reason": self._response_finish_reason(response),
        }

    async def _stream_raw(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        model, langchain_messages = self._prepare_model_call(messages, kwargs)
        async for chunk in model.astream(langchain_messages, **kwargs):
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content

    def _count_tokens_raw(self, messages: list[dict[str, str]]) -> int:
        model = self.build_chat_model()
        return int(
            model.get_num_tokens_from_messages(self._to_langchain_messages(messages))
        )
