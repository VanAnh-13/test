"""
LLMClient — lớp bao áp dụng mẫu Strategy cho các plugin LLM provider.

Thay vì dùng chuỗi if-elif theo tên provider (như trong config.py cũ),
LLMClient nhận đối tượng ``LLMProvider`` và chuyển tiếp mọi lời gọi tới đó.

Thiết kế:
  - ``LLMClient`` là API ổn định, không phụ thuộc provider cụ thể.
  - Provider được truyền qua hàm khởi tạo (Dependency Injection).
  - Thay đổi provider không cần sửa LLMClient — chỉ cần tạo provider mới.
  - Dùng ``from_config(model_config)`` để tạo client tự động từ config.
  - Dùng ``from_name(model_name)`` để tạo từ tên model trong YAML.

Tương thích ngược:
  - ``create_chat_model()`` trong config.py vẫn hoạt động bình thường.
  - ``create_chat_model()`` dùng cùng registry provider cho môi trường chạy thực tế.
  - Thành phần gọi hiện tại (các node đồ thị) không cần thay đổi.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import structlog

from hagent.agent.llm.providers import LLMProvider, RetryableError, get_provider

logger = structlog.get_logger(__name__)

__all__ = [
    "LLMClient",
    "RetryableError",
]


class LLMClient:
    """
    LLM client dựa trên mẫu Strategy.

    Sử dụng:
        # Từ ModelConfig (phổ biến trong kiểm thử và điều phối):
        from hagent.agent.llm.config import get_default_model_config
        client = LLMClient.from_config(get_default_model_config())
        result = await client.generate([{"role": "user", "content": "Hello"}])

        # Từ tên model (cách gọi tiện dụng):
        client = LLMClient.from_name("openai-gpt4o-mini")
        async for token in client.stream(messages):
            logger.info("llm_stream_token", token=token)

        # Truyền provider trực tiếp (dùng khi kiểm thử):
        client = LLMClient(my_mock_provider)

    API công khai:
        generate(messages, **kwargs) -> dict  — dict kết quả có khóa "content"
        stream(messages, **kwargs)   -> AsyncIterator[str]  — luồng token
        count_tokens(messages)       -> int   — số token, -1 nếu không hỗ trợ
    """

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    # ── Các phương thức khởi tạo ─────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: Any) -> LLMClient:
        """Tạo client từ một đối tượng ModelConfig."""
        return cls(get_provider(config))

    @classmethod
    def from_name(cls, name: str) -> LLMClient:
        """Tạo client từ tên model trong YAML config."""
        from hagent.agent.llm.config import require_model_config

        config = require_model_config(name)
        return cls.from_config(config)

    @classmethod
    def default(cls) -> LLMClient:
        """Tạo client từ cấu hình model mặc định."""
        from hagent.agent.llm.config import get_default_model_config

        return cls.from_config(get_default_model_config())

    # ── API công khai ─────────────────────────────────────────────────────────

    async def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Gọi LLM và trả về dict kết quả.

        Kết quả:
            dict với ít nhất ``{"content": str}``

        Ngoại lệ:
            RuntimeError: sau khi hết retry.
        """
        logger.debug(
            "LLMClient.generate via %s (%d messages)",
            self._provider.provider_name,
            len(messages),
        )
        return await self._provider.generate(messages, **kwargs)

    async def stream(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream token từ LLM.

        Giá trị trả dần:
            Các đoạn token kiểu chuỗi.
        """
        logger.debug(
            "LLMClient.stream via %s (%d messages)",
            self._provider.provider_name,
            len(messages),
        )
        async for token in self._provider.stream(messages, **kwargs):
            yield token

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        """
        Ước tính số token. Trả về -1 nếu provider không hỗ trợ.
        """
        return self._provider.count_tokens(messages)

    # ── Siêu dữ liệu ──────────────────────────────────────────────────────────

    @property
    def provider(self) -> LLMProvider:
        """Trả về đối tượng provider đang được dùng."""
        return self._provider

    @property
    def provider_name(self) -> str:
        """Tên provider hiện tại."""
        return self._provider.provider_name

    def __repr__(self) -> str:
        return f"LLMClient(provider={self._provider!r})"
