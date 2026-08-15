"""Plugin provider cho Ollama cục bộ."""

from __future__ import annotations

import os
from typing import Any

from hagent.agent.llm.providers.base import LangChainChatProvider


class OllamaProvider(LangChainChatProvider):
    """
    Provider cho suy luận Ollama cục bộ (``provider: ollama``).

    base_url đọc từ YAML config hoặc env var OLLAMA_BASE_URL.
    """

    _finish_reason_key = None

    # ── Hàm hỗ trợ nội bộ ─────────────────────────────────────────────────────

    def _resolve_base_url(self) -> str:
        url = self._config.base_url or os.getenv("OLLAMA_BASE_URL", "")
        if not url:
            raise ValueError(
                f"Model '{self._config.name}' dùng provider ollama "
                f"nhưng thiếu base_url. Cấu hình trong YAML hoặc đặt OLLAMA_BASE_URL."
            )
        return url

    def build_chat_model(
        self,
        callbacks: list | None = None,
        *,
        max_retries: int = 0,
    ) -> Any:
        from langchain_ollama import ChatOllama

        extra = dict(self._config.extra or {})
        extra.pop("max_retries", None)
        sync_client_kwargs = extra.pop("sync_client_kwargs", None)
        async_client_kwargs = extra.pop("async_client_kwargs", None)
        kwargs: dict[str, Any] = {
            **extra,
            "model": self._config.model,
            "base_url": self._resolve_base_url(),
            "temperature": self._config.temperature,
            "num_predict": self._config.max_tokens,
        }
        kwargs.update(
            self._httpx_retry_client_kwargs(
                max_retries,
                sync_client_kwargs=sync_client_kwargs,
                async_client_kwargs=async_client_kwargs,
            )
        )
        if callbacks:
            kwargs["callbacks"] = callbacks
        return ChatOllama(**kwargs)

    @property
    def provider_name(self) -> str:
        return "ollama"
