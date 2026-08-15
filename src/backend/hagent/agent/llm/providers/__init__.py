"""
providers — gói plugin LLM provider.

API công khai:
  LLMProvider           — lớp cơ sở trừu tượng
  RetryableError        — tín hiệu cho lớp cơ sở thực hiện retry
  OpenAIProvider        — openai + openai_compatible
  AnthropicProvider     — anthropic
  OllamaProvider        — ollama (cục bộ)
  get_provider(config)  — hàm tạo: ModelConfig → LLMProvider
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hagent.agent.llm.providers.anthropic_provider import AnthropicProvider
from hagent.agent.llm.providers.base import LLMProvider, RetryableError
from hagent.agent.llm.providers.ollama_provider import OllamaProvider
from hagent.agent.llm.providers.openai_provider import OpenAIProvider

if TYPE_CHECKING:
    from hagent.agent.llm.config import ModelConfig

_PROVIDER_MAP = {
    "openai": OpenAIProvider,
    "openai_compatible": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
}

_SUPPORTED = frozenset(_PROVIDER_MAP)


def get_provider(config: ModelConfig) -> LLMProvider:
    """
    Tạo đối tượng LLMProvider từ ModelConfig.

    Dùng trong LLMClient để truyền provider theo mẫu Strategy.

    Ngoại lệ:
        ValueError: nếu provider không được hỗ trợ.
    """
    provider_key = config.provider.lower()
    cls = _PROVIDER_MAP.get(provider_key)
    if cls is None:
        raise ValueError(
            f"Provider '{config.provider}' không được hỗ trợ. "
            f"Hỗ trợ: {', '.join(sorted(_SUPPORTED))}."
        )
    return cls.from_config(config)


__all__ = [
    "AnthropicProvider",
    "LLMProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "RetryableError",
    "get_provider",
]
