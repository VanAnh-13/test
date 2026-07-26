"""
Tests cho UsageTracker / UsageTrackingCallback.
"""

from __future__ import annotations

import threading

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from hagent.agent.middlewares.usage_tracker import (
    UsageTracker,
    UsageTrackingCallback,
    create_usage_tracker,
)

PRICING = {
    "gpt-4o-mini": {"input_per_1m": 0.15, "output_per_1m": 0.60},
}


class TestUsageTracker:
    def test_record_and_summary(self):
        t = UsageTracker(pricing=PRICING)
        t.record(model="gpt-4o-mini", input_tokens=1_000_000, output_tokens=500_000)
        s = t.summary()
        assert s["total_input_tokens"] == 1_000_000
        assert s["total_output_tokens"] == 500_000
        assert s["total_calls"] == 1
        assert s["total_cost_usd"] == pytest.approx(0.15 + 0.30)

    def test_unknown_model_zero_cost_but_counted(self):
        t = UsageTracker(pricing=PRICING)
        t.record(model="qwen2.5:14b", input_tokens=1000, output_tokens=2000)
        s = t.summary()
        assert s["total_cost_usd"] == 0.0
        assert s["by_model"]["qwen2.5:14b"]["input_tokens"] == 1000

    def test_accumulates_across_calls(self):
        t = UsageTracker(pricing=PRICING)
        for _ in range(3):
            t.record(model="gpt-4o-mini", input_tokens=100, output_tokens=50)
        s = t.summary()
        assert s["by_model"]["gpt-4o-mini"]["calls"] == 3
        assert s["total_input_tokens"] == 300

    def test_reset(self):
        t = UsageTracker(pricing=PRICING)
        t.record(model="m", input_tokens=10, output_tokens=10)
        t.reset()
        assert t.summary()["total_calls"] == 0

    def test_thread_safety_smoke(self):
        t = UsageTracker()

        def worker():
            for _ in range(200):
                t.record(model="m", input_tokens=1, output_tokens=1)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert t.summary()["total_input_tokens"] == 1600

    def test_negative_clamped(self):
        t = UsageTracker()
        t.record(model="m", input_tokens=-5, output_tokens=10)
        assert t.summary()["total_input_tokens"] == 0


class TestCallback:
    def test_parses_llm_output_token_usage(self):
        t = UsageTracker(pricing=PRICING)
        cb = UsageTrackingCallback(t)
        result = LLMResult(
            generations=[[]],
            llm_output={
                "model_name": "gpt-4o-mini",
                "token_usage": {"prompt_tokens": 120, "completion_tokens": 30},
            },
        )
        cb.on_llm_end(result)
        s = t.summary()
        assert s["by_model"]["gpt-4o-mini"]["input_tokens"] == 120
        assert s["by_model"]["gpt-4o-mini"]["output_tokens"] == 30

    def test_parses_usage_metadata_from_generations(self):
        t = UsageTracker()
        cb = UsageTrackingCallback(t)
        msg = AIMessage(
            content="hi",
            usage_metadata={
                "input_tokens": 15,
                "output_tokens": 7,
                "total_tokens": 22,
            },
            response_metadata={"model_name": "qwen2.5:14b"},
        )
        result = LLMResult(
            generations=[[ChatGeneration(message=msg)]], llm_output={}
        )
        cb.on_llm_end(result)
        s = t.summary()
        assert s["by_model"]["qwen2.5:14b"]["input_tokens"] == 15
        assert s["by_model"]["qwen2.5:14b"]["output_tokens"] == 7

    def test_missing_usage_ignored(self):
        t = UsageTracker()
        cb = UsageTrackingCallback(t)
        cb.on_llm_end(LLMResult(generations=[[]], llm_output={}))
        assert t.summary()["total_calls"] == 0

    def test_never_raises(self):
        t = UsageTracker()
        cb = UsageTrackingCallback(t)
        cb.on_llm_end(object())  # LLMResult dị dạng
        assert t.summary()["total_calls"] == 0


class TestFactory:
    def test_disabled_returns_none(self):
        assert create_usage_tracker({"enabled": False}) is None

    def test_creates_with_pricing(self):
        t = create_usage_tracker({"pricing": PRICING})
        assert isinstance(t, UsageTracker)
        assert t.pricing == PRICING
