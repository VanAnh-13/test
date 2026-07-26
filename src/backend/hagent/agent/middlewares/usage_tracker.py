"""
Token/cost tracker cho LLM calls — Ollama (miễn phí) lẫn API thương mại.

Gắn UsageTrackingCallback vào callbacks của LangChain model; tracker cộng dồn
token theo model và quy ra USD theo bảng giá per-1M-token trong config
(llm.usage_tracking.pricing). Model không có trong bảng giá → cost 0 (local).
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)


class UsageTracker:
    """Cộng dồn usage theo model; thread-safe (callback có thể chạy đa luồng)."""

    def __init__(self, pricing: Dict[str, Dict[str, float]] | None = None):
        # pricing: {model: {"input_per_1m": usd, "output_per_1m": usd}}
        self.pricing = dict(pricing or {})
        self._lock = threading.Lock()
        self._by_model: Dict[str, Dict[str, float]] = {}

    def record(
        self,
        *,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        calls: int = 1,
    ) -> None:
        model = str(model or "unknown")
        with self._lock:
            entry = self._by_model.setdefault(
                model, {"input_tokens": 0, "output_tokens": 0, "calls": 0}
            )
            entry["input_tokens"] += max(0, int(input_tokens))
            entry["output_tokens"] += max(0, int(output_tokens))
            entry["calls"] += max(0, int(calls))

    def _cost_for(self, model: str, entry: Dict[str, float]) -> float:
        price = self.pricing.get(model)
        if not price:
            return 0.0
        return (
            entry["input_tokens"] * float(price.get("input_per_1m", 0.0))
            + entry["output_tokens"] * float(price.get("output_per_1m", 0.0))
        ) / 1_000_000.0

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            by_model = {
                model: {**entry, "cost_usd": self._cost_for(model, entry)}
                for model, entry in self._by_model.items()
            }
        return {
            "total_input_tokens": sum(e["input_tokens"] for e in by_model.values()),
            "total_output_tokens": sum(e["output_tokens"] for e in by_model.values()),
            "total_calls": sum(e["calls"] for e in by_model.values()),
            "total_cost_usd": sum(e["cost_usd"] for e in by_model.values()),
            "by_model": by_model,
        }

    def reset(self) -> None:
        with self._lock:
            self._by_model.clear()


def _usage_from_llm_result(response: Any) -> Optional[Dict[str, Any]]:
    """Rút (model, input, output) từ LLMResult — hỗ trợ cả hai convention."""
    model = None
    input_tokens = 0
    output_tokens = 0
    found = False

    llm_output = getattr(response, "llm_output", None) or {}
    if isinstance(llm_output, dict):
        model = llm_output.get("model_name") or llm_output.get("model")
        token_usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        if isinstance(token_usage, dict) and token_usage:
            input_tokens = int(
                token_usage.get("prompt_tokens")
                or token_usage.get("input_tokens")
                or 0
            )
            output_tokens = int(
                token_usage.get("completion_tokens")
                or token_usage.get("output_tokens")
                or 0
            )
            found = input_tokens > 0 or output_tokens > 0

    if not found:
        # Chuẩn mới: usage_metadata trên AIMessage trong generations
        for gens in getattr(response, "generations", None) or []:
            for gen in gens:
                msg = getattr(gen, "message", None)
                usage = getattr(msg, "usage_metadata", None)
                if isinstance(usage, dict) and usage:
                    input_tokens += int(usage.get("input_tokens") or 0)
                    output_tokens += int(usage.get("output_tokens") or 0)
                    found = True
                if model is None and msg is not None:
                    meta = getattr(msg, "response_metadata", None) or {}
                    model = meta.get("model_name") or meta.get("model")

    if not found:
        return None
    return {
        "model": str(model or "unknown"),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


class UsageTrackingCallback(BaseCallbackHandler):
    """LangChain callback — gắn vào model.with_config(callbacks=[...])."""

    def __init__(self, tracker: UsageTracker):
        self.tracker = tracker

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            usage = _usage_from_llm_result(response)
            if usage:
                self.tracker.record(
                    model=usage["model"],
                    input_tokens=usage["input_tokens"],
                    output_tokens=usage["output_tokens"],
                )
        except Exception as exc:
            logger.debug("usage tracking parse failed: %s", exc)


def create_usage_tracker(config: dict | None = None) -> Optional[UsageTracker]:
    """Tracker từ llm.usage_tracking trong hagent.yaml. None khi disabled."""
    cfg = config
    if cfg is None:
        try:
            from hagent.bridge.config import get_llm_config

            cfg = dict((get_llm_config() or {}).get("usage_tracking") or {})
        except Exception:
            cfg = {}
    cfg = dict(cfg or {})
    if not cfg.get("enabled", True):
        return None
    return UsageTracker(pricing=dict(cfg.get("pricing") or {}))
