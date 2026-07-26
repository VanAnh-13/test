"""
DeerFlow-AutoML — Multi-provider LLM Configuration.

Supports: OpenAI, Anthropic, Ollama, and any OpenAI-compatible endpoint.
Configuration is loaded from hagent.yaml via bridge/config.py.
KHÔNG hardcode bất kỳ giá trị nào — tất cả đọc từ YAML + env vars.

Reference: deerflow/config/app_config.py (model resolution logic)
"""

from __future__ import annotations

import contextvars
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

# ── Model config dataclass ───────────────────────────────


@dataclass
class ModelConfig:
    """Một mục cấu hình model — parse từ hagent.yaml ``llm.models[]``."""
    name: str
    provider: str  # openai | anthropic | ollama | openai_compatible
    model: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4096
    is_default: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def resolve_api_key(self) -> str | None:
        """Resolve API key — hỗ trợ ``$ENV_VAR`` và ``${ENV_VAR}`` syntax."""
        if not self.api_key:
            return None
        if self.api_key.startswith("$"):
            var_name = self.api_key.lstrip("$").strip("{}")
            return os.getenv(var_name, "")
        return self.api_key


# ── Config loader (delegates to bridge/config.py) ────────


def load_llm_configs() -> list[ModelConfig]:
    """Parse ``llm.models`` trong hagent.yaml thành danh sách ``ModelConfig``."""
    from hagent.bridge.config import get_llm_models

    models_raw = get_llm_models()

    if not models_raw:
        logger.info("Không tìm thấy llm.models trong config, dùng fallback từ env vars.")
        return [_fallback_config()]

    configs: list[ModelConfig] = []
    for entry in models_raw:
        configs.append(ModelConfig(
            name=entry.get("name", "unnamed"),
            provider=entry.get("provider", "openai"),
            model=entry.get("model", ""),
            api_key=entry.get("api_key"),
            base_url=entry.get("base_url"),
            temperature=entry.get("temperature", 0.0),
            max_tokens=entry.get("max_tokens", 4096),
            is_default=entry.get("default", False),
            extra=entry.get("extra", {}),
        ))

    return configs


def _fallback_config() -> ModelConfig:
    """Config mặc định khi YAML không có llm section — đọc hoàn toàn từ env vars."""
    return ModelConfig(
        name=os.getenv("LLM_DEFAULT_MODEL", "default"),
        provider=os.getenv("LLM_PROVIDER", ""),
        model=os.getenv("LLM_MODEL", ""),
        api_key=os.getenv("OPENAI_API_KEY", os.getenv("LLM_API_KEY", "")),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        is_default=True,
    )


def get_default_model_config() -> ModelConfig:
    """Trả về model config mặc định (``default: true`` hoặc model đầu tiên)."""
    from hagent.bridge.config import get_llm_config

    configs = load_llm_configs()
    default_name = get_llm_config().get("default_model", "")

    # Ưu tiên: default_model setting → model có default: true → model đầu tiên
    if default_name:
        for cfg in configs:
            if cfg.name == default_name:
                return cfg
        # Tên sai (typo LLM_DEFAULT_MODEL) mà âm thầm rơi về model khác là
        # thảm họa hai mặt: hóa đơn API bất ngờ, và thí nghiệm chạy nhầm model
        # không ai phát hiện. Phải fail to tiếng.
        raise ValueError(
            f"default_model={default_name!r} không khớp model nào trong cấu hình. "
            f"Các tên hợp lệ: {[c.name for c in configs]}"
        )

    for cfg in configs:
        if cfg.is_default:
            return cfg

    return configs[0]


def get_model_config_by_name(name: str) -> ModelConfig | None:
    """Tìm model config theo tên."""
    for cfg in load_llm_configs():
        if cfg.name == name:
            return cfg
    return None


def list_available_models() -> list[dict[str, str]]:
    """Liệt kê các models khả dụng (cho API endpoint)."""
    return [
        {"name": cfg.name, "provider": cfg.provider, "model": cfg.model}
        for cfg in load_llm_configs()
    ]


# ── Chat model factory ───────────────────────────────────

_SUPPORTED_PROVIDERS = {"openai", "anthropic", "ollama", "openai_compatible"}

# Model của RUN hiện tại (per-request) — contextvar như usage tracker:
# coordinator/subagents gọi create_chat_model() không tên sẽ dùng model này
_current_model_name: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "hagent_model_name", default=None
)


def set_current_model_name(name: str | None) -> contextvars.Token:
    return _current_model_name.set(name)


def get_current_model_name() -> str | None:
    return _current_model_name.get()


def reset_current_model_name(token: contextvars.Token) -> None:
    _current_model_name.reset(token)


def require_model_config(name: str) -> ModelConfig:
    """Resolve tên model, RAISE kèm danh sách hợp lệ nếu không tồn tại."""
    cfg = get_model_config_by_name(name)
    if cfg is None:
        raise ValueError(
            f"Model {name!r} không tồn tại trong cấu hình. "
            f"Các tên hợp lệ: {[c.name for c in load_llm_configs()]}"
        )
    return cfg


def create_chat_model(
    name: str | None = None,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    callbacks: list | None = None,
) -> BaseChatModel:
    """
    Tạo LangChain ChatModel từ config.

    Hỗ trợ 4 provider:
    - ``openai``: ChatOpenAI (OpenAI chính thức)
    - ``anthropic``: ChatAnthropic
    - ``ollama``: ChatOllama (local)
    - ``openai_compatible``: ChatOpenAI với custom base_url

    Tất cả config đọc từ hagent.yaml, KHÔNG hardcode.

    Reference: deerflow/models/__init__.py — create_chat_model()
    """
    # Ưu tiên: tên truyền tường minh → model per-request (contextvar) → default
    if not name:
        name = get_current_model_name()
    if name:
        config = get_model_config_by_name(name)
        if not config:
            logger.warning("Model '%s' không tìm thấy, dùng default.", name)
            config = get_default_model_config()
    else:
        config = get_default_model_config()

    temp = temperature if temperature is not None else config.temperature
    tokens = max_tokens if max_tokens is not None else config.max_tokens
    api_key = config.resolve_api_key()

    provider = config.provider.lower()
    if provider not in _SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Provider '{provider}' không được hỗ trợ. "
            f"Hỗ trợ: {', '.join(sorted(_SUPPORTED_PROVIDERS))}."
        )

    # Không truyền callbacks tường minh → tự nhặt usage tracker của run
    # hiện tại (contextvar). Callbacks đi vào CONSTRUCTOR để giữ nguyên class
    # model — bind_tools của coordinator/subagents vẫn hoạt động.
    if callbacks is None:
        from hagent.agent.middlewares.usage_tracker import (
            UsageTrackingCallback,
            get_current_tracker,
        )

        tracker = get_current_tracker()
        if tracker is not None:
            callbacks = [UsageTrackingCallback(tracker)]

    return _build_model(provider, config, api_key, temp, tokens, callbacks=callbacks)


def _build_model(
    provider: str,
    config: ModelConfig,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
    callbacks: list | None = None,
) -> BaseChatModel:
    """Internal: tạo model instance theo provider."""

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=callbacks,
            **(config.extra or {}),
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=callbacks,
            **(config.extra or {}),
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        # base_url đọc từ YAML config hoặc env var — KHÔNG hardcode
        ollama_url = config.base_url or os.getenv("OLLAMA_BASE_URL", "")
        if not ollama_url:
            raise ValueError(
                f"Model '{config.name}' dùng provider ollama "
                f"nhưng thiếu base_url. Cấu hình trong YAML hoặc đặt OLLAMA_BASE_URL."
            )
        return ChatOllama(
            model=config.model,
            base_url=ollama_url,
            temperature=temperature,
            num_predict=max_tokens,
            callbacks=callbacks,
            **(config.extra or {}),
        )

    # openai_compatible
    from langchain_openai import ChatOpenAI
    if not config.base_url:
        raise ValueError(
            f"Model '{config.name}' dùng provider openai_compatible "
            f"nhưng thiếu base_url trong config."
        )
    return ChatOpenAI(
        model=config.model,
        api_key=api_key or "not-needed",
        base_url=config.base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        callbacks=callbacks,
        **(config.extra or {}),
    )
