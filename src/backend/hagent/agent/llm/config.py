"""Cấu hình LLM đa nhà cung cấp cho OpenAI, Anthropic, Ollama và endpoint tương thích."""

from __future__ import annotations

import contextvars
import os
from dataclasses import dataclass, field, replace
from typing import Any

import structlog
from langchain_core.language_models.chat_models import BaseChatModel

logger = structlog.get_logger(__name__)

# ── Dataclass cấu hình model ─────────────────────────────


@dataclass
class ModelConfig:
    """Một mục cấu hình model được phân tích từ ``llm.models[]`` trong hagent.yaml."""

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
        """Phân giải API key, hỗ trợ cú pháp ``$ENV_VAR`` và ``${ENV_VAR}``."""
        if not self.api_key:
            return None
        if self.api_key.startswith("$"):
            var_name = self.api_key.lstrip("$").strip("{}")
            return os.getenv(var_name, "")
        return self.api_key


# ── Bộ tải cấu hình, ủy quyền cho bridge/config.py ───────


def load_llm_configs() -> list[ModelConfig]:
    """Phân tích ``llm.models`` trong hagent.yaml thành danh sách ``ModelConfig``."""
    from hagent.bridge.config import get_llm_models

    models_raw = get_llm_models()

    if not models_raw:
        logger.info(
            "Không tìm thấy llm.models trong config, dùng fallback từ env vars."
        )
        return [_fallback_config()]

    configs: list[ModelConfig] = []
    for entry in models_raw:
        configs.append(
            ModelConfig(
                name=entry.get("name", "unnamed"),
                provider=entry.get("provider", "openai"),
                model=entry.get("model", ""),
                api_key=entry.get("api_key"),
                base_url=entry.get("base_url"),
                temperature=entry.get("temperature", 0.0),
                max_tokens=entry.get("max_tokens", 4096),
                is_default=entry.get("default", False),
                extra=entry.get("extra", {}),
            )
        )

    return configs


def _fallback_config() -> ModelConfig:
    """Tạo cấu hình mặc định từ biến môi trường khi YAML không có phần llm."""
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
    """Trả về cấu hình model mặc định (``default: true`` hoặc model đầu tiên)."""
    from hagent.bridge.config import get_llm_config

    configs = load_llm_configs()
    default_name = get_llm_config().get("default_model", "")

    # Thứ tự ưu tiên: default_model → model có default: true → model đầu tiên.
    if default_name:
        for cfg in configs:
            if cfg.name == default_name:
                return cfg
        # Tên sai do gõ nhầm LLM_DEFAULT_MODEL mà âm thầm rơi về model khác là
        # thảm họa hai mặt: hóa đơn API bất ngờ, và thí nghiệm chạy nhầm model
        # không ai phát hiện. Phải báo lỗi rõ ràng.
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
    """Liệt kê các model khả dụng cho endpoint API."""
    return [
        {"name": cfg.name, "provider": cfg.provider, "model": cfg.model}
        for cfg in load_llm_configs()
    ]


# ── Hàm tạo model trò chuyện ─────────────────────────────

# Model của lượt chạy hiện tại (theo từng request) — contextvar như bộ theo dõi:
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
    """Phân giải tên model và phát sinh lỗi kèm danh sách hợp lệ nếu không tồn tại."""
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

    Hỗ trợ bốn nhà cung cấp:
    - ``openai``: ChatOpenAI (OpenAI chính thức)
    - ``anthropic``: ChatAnthropic
    - ``ollama``: ChatOllama (cục bộ)
    - ``openai_compatible``: ChatOpenAI với base_url tùy chỉnh

    Tất cả cấu hình đọc từ hagent.yaml, KHÔNG mã hóa cứng.

    """
    # Ưu tiên: tên truyền tường minh → model theo từng request (contextvar) → mặc định
    if not name:
        name = get_current_model_name()
    if name:
        config = get_model_config_by_name(name)
        if not config:
            logger.warning("Không tìm thấy model '%s', dùng model mặc định.", name)
            config = get_default_model_config()
    else:
        config = get_default_model_config()

    temp = temperature if temperature is not None else config.temperature
    tokens = max_tokens if max_tokens is not None else config.max_tokens
    api_key = config.resolve_api_key()

    provider = config.provider.lower()
    # Khi không truyền callback tường minh, tự lấy bộ theo dõi sử dụng của lượt chạy
    # hiện tại từ contextvar. Callback đi vào hàm khởi tạo để giữ nguyên lớp model,
    # nhờ đó bind_tools của agent điều phối và các sub-agent vẫn hoạt động.
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
    """Tạo model LangChain thông qua registry chiến lược nhà cung cấp."""
    from hagent.agent.llm.providers import get_provider

    configured_api_key = (
        config.api_key if config.api_key and config.api_key.startswith("$") else api_key
    )
    effective_config = replace(
        config,
        provider=provider,
        api_key=configured_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    strategy = get_provider(effective_config)
    return strategy.build_chat_model(
        callbacks=callbacks,
        max_retries=strategy.max_retries,
    )
