"""OpenAI và OpenAI-compatible provider plugin."""

from __future__ import annotations

from typing import Any

from hagent.agent.llm.providers.base import LangChainChatProvider


class OpenAIProvider(LangChainChatProvider):
    """
    Provider cho OpenAI API (``provider: openai``) và các endpoint
    tương thích OpenAI (``provider: openai_compatible``).

    LangChain ChatOpenAI là phần triển khai phía sau — provider này là lớp bao mỏng
    cung cấp giao diện thống nhất với retry/backoff từ lớp cơ sở.
    """

    _response_provider_name = "openai"

    # ── Hàm hỗ trợ nội bộ ─────────────────────────────────────────────────────

    def build_chat_model(
        self,
        callbacks: list | None = None,
        *,
        max_retries: int = 0,
    ) -> Any:
        """Khởi tạo ChatOpenAI khi thực sự cần dùng."""
        from langchain_openai import ChatOpenAI

        kwargs = self._build_credentialed_model_kwargs(
            callbacks,
            max_retries=max_retries,
            fallback_api_key="not-needed" if self._config.base_url else None,
        )
        if (
            self._config.provider.lower() == "openai_compatible"
            and not self._config.base_url
        ):
            raise ValueError(
                f"Model '{self._config.name}' dùng openai_compatible "
                "nhưng thiếu base_url"
            )
        # openai_compatible dùng base_url tùy chỉnh
        if self._config.base_url:
            kwargs["base_url"] = self._config.base_url
        return ChatOpenAI(**kwargs)

    @property
    def provider_name(self) -> str:
        return self._config.provider  # "openai" hoặc "openai_compatible"
