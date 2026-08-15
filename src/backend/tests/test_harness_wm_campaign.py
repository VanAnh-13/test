"""
Tests cho T6 — scenario harness WM-on campaign chạy qua run_graph_scenario
(hierarchy leaf, đúng đường production), khóa chuỗi event
campaign_outcome_surprise → campaign_extended.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from hagent.agent.execution.tool_runner import set_tool_invoker
from hagent.agent.harness.loader import load_scenarios_from_yaml
from hagent.agent.harness.runners.graph import run_graph_scenario
from hagent.world.predictor.outcome_head_v1 import train_outcome_head

SCENARIO_PATH = (
    Path(__file__).parent.parent
    / "hagent"
    / "agent"
    / "harness"
    / "scenarios"
    / "wm_campaign.yaml"
)


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _overconfident_head():
    """Head tin mọi config đạt ~0.99 σ nhỏ — điểm mock (0.71–0.88) sẽ gây
    zscore rất cao → kích hoạt vòng mở rộng."""
    rng = np.random.default_rng(0)
    algos = ["grid_search", "bayesian_search", "genetic_algorithm"]
    samples = [
        {
            "params": {
                "search_algorithm": str(rng.choice(algos)),
                "problem_type": "classification",
                "metric": "f1",
                "time_limit": int(rng.choice([180, 300, 600])),
            },
            "best_score": float(0.99 + rng.normal(0, 0.003)),
        }
        for _ in range(80)
    ]
    return train_outcome_head(
        samples, config={"use_latent": False, "hidden_dim": 24}, epochs=60, seed=0
    )


@pytest.fixture
def wm_ext_env(monkeypatch):
    """Bật gate + inject outcome model — hai điều kiện live của scenario."""
    import hagent.bridge.config as bridge_config
    from hagent.agent.campaign import wm_hooks

    head = _overconfident_head()
    monkeypatch.setattr(wm_hooks, "_default_outcome_model", lambda: head)

    real_get_campaign = bridge_config.get_campaign_config

    def patched():
        cfg = dict(real_get_campaign())
        cfg["surprise_extension"] = {
            "enabled": True,
            "max_rounds": 1,
            "n_extra": 2,
            "exploration_weight": 0.5,
        }
        # cô lập khỏi warm-start memory trên đĩa
        cfg["warm_start_top_k"] = 0
        return cfg

    monkeypatch.setattr(bridge_config, "get_campaign_config", patched)
    yield head
    set_tool_invoker(None)


class TestScenarioFile:
    def test_loads_with_expected_events(self):
        scenarios = load_scenarios_from_yaml(SCENARIO_PATH)
        assert len(scenarios) == 1
        sc = scenarios[0]
        assert sc.id == "wm_campaign_extension"
        assert "campaign_outcome_surprise" in sc.expect.event_types_include
        assert "campaign_extended" in sc.expect.event_types_include
        assert "wm_ext" in sc.tags


class TestGraphLayerEventChain:
    def test_event_chain_through_harness(self, wm_ext_env):
        """Chạy scenario qua ĐÚNG máy móc harness graph layer (hierarchy leaf
        như production) — chuỗi event phải xuất hiện và assertions pass."""
        sc = load_scenarios_from_yaml(SCENARIO_PATH)[0]
        result = run(run_graph_scenario(sc))

        assert "campaign_outcome_surprise" in result.event_types
        assert "campaign_extended" in result.event_types
        # graph runner đã tự chạy assert_expectations → success phản ánh
        # đầy đủ expect trong YAML (kể cả event_types_include)
        assert result.success, f"harness assertions fail: {result.reasons}"

    def test_without_gate_no_extension_event(self, monkeypatch):
        """Không bật gate (mặc định) → scenario này PHẢI fail assertion —
        chứng minh guard thật sự canh cơ chế chứ không pass rỗng."""
        from hagent.agent.campaign import wm_hooks

        head = _overconfident_head()
        monkeypatch.setattr(wm_hooks, "_default_outcome_model", lambda: head)
        try:
            sc = load_scenarios_from_yaml(SCENARIO_PATH)[0]
            result = run(run_graph_scenario(sc))
            assert "campaign_extended" not in result.event_types
            assert not result.success  # guard thật sự canh cơ chế
        finally:
            set_tool_invoker(None)
