"""Anthropic provider plugin."""

from __future__ import annotations

from typing import Any

from hagent.agent.llm.providers.base import LangChainChatProvider


class AnthropicProvider(LangChainChatProvider):
    """
    Provider cho Anthropic Claude API (``provider: anthropic``).

    LangChain ChatAnthropic là phần triển khai phía sau.
    """

    _finish_reason_key = "stop_reason"
    _default_finish_reason = "end_turn"

    # ── Hàm hỗ trợ nội bộ ─────────────────────────────────────────────────────

    def build_chat_model(
        self,
        callbacks: list | None = None,
        *,
        max_retries: int = 0,
    ) -> Any:
        from langchain_anthropic import ChatAnthropic

        kwargs = self._build_credentialed_model_kwargs(
            callbacks,
            max_retries=max_retries,
        )
        return ChatAnthropic(**kwargs)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    retryable_status_codes = LangChainChatProvider.retryable_status_codes | {529}
