"""
Tests cho T5 — vòng mở rộng campaign theo outcome surprise + surfacing.
"""

from __future__ import annotations

import asyncio

import numpy as np

from hagent.agent.campaign.builder import build_campaign, propose_extension_variants
from hagent.agent.campaign.nodes import _select_surprise
from hagent.agent.campaign.runner import campaign_step
from hagent.agent.campaign.schema import Campaign
from hagent.agent.execution.tool_runner import set_tool_invoker
from hagent.world.predictor.outcome_head_v1 import train_outcome_head

GOAL = {
    "goal_type": "train",
    "dataset_id": "ds1",
    "problem_type": "classification",
    "metric": "accuracy",
    "target_column": "target",
}
DATASET_META = {"n_rows": 1000, "n_cols": 12}
HEAD_CFG = {"use_latent": False, "hidden_dim": 24}


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _trained_head(mean=0.85, seed=0):
    """Head tin rằng mọi config đạt ~mean — điểm thật lệch xa → surprise cao."""
    rng = np.random.default_rng(seed)
    algos = ["grid_search", "bayesian_search", "genetic_algorithm"]
    samples = [
        {
            "params": {
                "search_algorithm": str(rng.choice(algos)),
                "problem_type": "classification",
                "metric": "accuracy",
                "time_limit": int(rng.choice([60, 180, 600])),
            },
            "dataset_meta": DATASET_META,
            "best_score": float(mean + rng.normal(0, 0.01)),
        }
        for _ in range(80)
    ]
    return train_outcome_head(samples, config=dict(HEAD_CFG), epochs=60, seed=seed)


def _ext_cfg(enabled=True, max_rounds=1, n_extra=2):
    return {
        "enabled": enabled,
        "max_rounds": max_rounds,
        "n_extra": n_extra,
        "exploration_weight": 0.5,
    }


def _patch_campaign_cfg(monkeypatch, ext):
    import hagent.bridge.config as bridge_config

    base = {
        "enabled": True,
        "n_job_candidates": 2,
        "max_concurrent_jobs": 4,
        "warm_start_top_k": 0,
        "search_algorithms": [
            "grid_search",
            "bayesian_search",
            "genetic_algorithm",
            "random_search",
            "successive_halving",
        ],
        "time_limit_options": [180, 300, 600],
        "surprise_extension": ext,
    }
    monkeypatch.setattr(bridge_config, "get_campaign_config", lambda: base)


def _fake_tools(score=0.2):
    """Job hoàn thành ngay với best_score cho trước (rất xa 0.85 model tin)."""
    counter = {"n": 0}

    async def fake(action_type, params):
        if action_type == "start_training":
            counter["n"] += 1
            return {"job_id": f"j{counter['n']}", "status": 0}
        if action_type == "get_job_info":
            return {
                "id": params.get("job_id"),
                "status": "completed",
                "best_model": "rf",
                "best_score": score,
            }
        return {}

    return fake


async def _run_to_terminal(camp, head, max_ticks=10):
    events: list = []
    ticks = 0
    while camp.status not in ("done", "failed") and ticks < max_ticks:
        camp = await campaign_step(
            camp,
            user_id="t5",
            user_token=None,
            world_model={"datasets": {"ds1": dict(DATASET_META)}},
            surprise_events=events,
            outcome_model=head,
        )
        ticks += 1
    return camp, events


class TestExtensionRound:
    def teardown_method(self):
        set_tool_invoker(None)

    def test_high_surprise_triggers_extension(self, monkeypatch):
        _patch_campaign_cfg(monkeypatch, _ext_cfg())
        head = _trained_head()
        set_tool_invoker(_fake_tools(score=0.2))  # xa 0.85 → zscore rất cao

        camp = run(
            build_campaign(
                GOAL,
                user_id="t5",
                config={
                    "n_job_candidates": 2,
                    "warm_start_top_k": 0,
                    "wm_variant_proposal": False,
                    "wm_rank_variants": False,
                },
                outcome_model=None,
            )
        )
        n_before = len(camp.variants)
        camp, events = run(_run_to_terminal(camp, head))

        ext_events = [e for e in events if e.get("type") == "campaign_extended"]
        assert len(ext_events) == 1
        assert camp.extension_rounds == 1
        assert len(camp.variants) == n_before + 2
        ext_variants = [v for v in camp.variants if v.source == "surprise_extension"]
        assert len(ext_variants) == 2
        # Mọi variant (kể cả mở rộng) đều hoàn tất và campaign kết thúc
        assert camp.status == "done"
        assert all(v.status == "completed" for v in camp.variants)
        assert ext_events[0]["trigger_zscore"] > 3.0

    def test_no_second_extension(self, monkeypatch):
        """max_rounds=1: điểm vẫn gây surprise nhưng không mở rộng lần 2."""
        _patch_campaign_cfg(monkeypatch, _ext_cfg(max_rounds=1))
        head = _trained_head()
        set_tool_invoker(_fake_tools(score=0.2))
        camp = run(
            build_campaign(
                GOAL,
                user_id="t5",
                config={
                    "n_job_candidates": 2,
                    "warm_start_top_k": 0,
                    "wm_variant_proposal": False,
                    "wm_rank_variants": False,
                },
                outcome_model=None,
            )
        )
        camp, events = run(_run_to_terminal(camp, head))
        assert camp.extension_rounds == 1
        assert len([e for e in events if e.get("type") == "campaign_extended"]) == 1
        assert camp.status == "done"

    def test_gate_off_no_extension(self, monkeypatch):
        _patch_campaign_cfg(monkeypatch, _ext_cfg(enabled=False))
        head = _trained_head()
        set_tool_invoker(_fake_tools(score=0.2))
        camp = run(
            build_campaign(
                GOAL,
                user_id="t5",
                config={
                    "n_job_candidates": 2,
                    "warm_start_top_k": 0,
                    "wm_variant_proposal": False,
                    "wm_rank_variants": False,
                },
                outcome_model=None,
            )
        )
        camp, events = run(_run_to_terminal(camp, head))
        assert camp.extension_rounds == 0
        assert not [e for e in events if e.get("type") == "campaign_extended"]

    def test_no_surprise_no_extension(self, monkeypatch):
        """Điểm đúng như model dự đoán → không mở rộng."""
        _patch_campaign_cfg(monkeypatch, _ext_cfg())
        head = _trained_head(mean=0.85)
        set_tool_invoker(_fake_tools(score=0.85))
        camp = run(
            build_campaign(
                GOAL,
                user_id="t5",
                config={
                    "n_job_candidates": 2,
                    "warm_start_top_k": 0,
                    "wm_variant_proposal": False,
                    "wm_rank_variants": False,
                },
                outcome_model=None,
            )
        )
        camp, events = run(_run_to_terminal(camp, head))
        assert camp.extension_rounds == 0
        assert camp.status == "done"

    def test_model_none_disables(self, monkeypatch):
        _patch_campaign_cfg(monkeypatch, _ext_cfg())
        set_tool_invoker(_fake_tools(score=0.2))
        camp = run(
            build_campaign(
                GOAL,
                user_id="t5",
                config={
                    "n_job_candidates": 2,
                    "warm_start_top_k": 0,
                    "wm_variant_proposal": False,
                    "wm_rank_variants": False,
                },
                outcome_model=None,
            )
        )

        async def go():
            events: list = []
            c = camp
            ticks = 0
            while c.status not in ("done", "failed") and ticks < 10:
                c = await campaign_step(
                    c,
                    user_id="t5",
                    user_token=None,
                    world_model={"datasets": {"ds1": dict(DATASET_META)}},
                    surprise_events=events,
                    outcome_model=None,
                )
                ticks += 1
            return c

        c = run(go())
        assert c.extension_rounds == 0
        assert c.status == "done"

    def test_extension_rounds_survives_dict_roundtrip(self):
        camp = Campaign(
            campaign_id="c1", goal=dict(GOAL), variants=[], extension_rounds=1
        )
        assert Campaign.from_dict(camp.to_dict()).extension_rounds == 1


class TestProposeExtensionVariants:
    def test_dedups_and_labels(self, monkeypatch):
        _patch_campaign_cfg(monkeypatch, _ext_cfg())

        async def make():
            return await build_campaign(
                GOAL,
                user_id="t5",
                config={
                    "n_job_candidates": 3,
                    "warm_start_top_k": 0,
                    "wm_variant_proposal": False,
                    "wm_rank_variants": False,
                },
                outcome_model=None,
            )

        camp = run(make())
        head = _trained_head()
        new = propose_extension_variants(
            camp,
            GOAL,
            dataset_meta=DATASET_META,
            outcome_model=head,
            n_extra=2,
        )
        assert 1 <= len(new) <= 2
        sigs = {
            (
                v.params.get("search_algorithm"),
                v.params.get("time_limit"),
                tuple(v.params.get("models") or []),
            )
            for v in camp.variants
        }
        for v in new:
            assert v.source == "surprise_extension"
            assert v.label.startswith("ext1_")
            assert (
                v.params.get("search_algorithm"),
                v.params.get("time_limit"),
                tuple(v.params.get("models") or []),
            ) not in sigs

    def test_fallback_without_model_uses_untried_algos(self, monkeypatch):
        _patch_campaign_cfg(monkeypatch, _ext_cfg())

        async def make():
            return await build_campaign(
                GOAL,
                user_id="t5",
                config={
                    "n_job_candidates": 2,
                    "warm_start_top_k": 0,
                    "wm_variant_proposal": False,
                    "wm_rank_variants": False,
                },
                outcome_model=None,
            )

        camp = run(make())
        tried = {v.params.get("search_algorithm") for v in camp.variants}
        new = propose_extension_variants(
            camp,
            GOAL,
            dataset_meta=DATASET_META,
            outcome_model=None,
            n_extra=2,
        )
        assert new
        for v in new:
            assert v.params.get("search_algorithm") not in tried


class TestSelectSurprise:
    def test_prefers_outcome_highest_zscore(self):
        buf = [
            {"type": "campaign_surprise", "surprise": {"value": 0.5, "level": "high"}},
            {
                "type": "campaign_outcome_surprise",
                "outcome": {"zscore": 2.0, "level": "medium"},
            },
            {
                "type": "campaign_outcome_surprise",
                "outcome": {"zscore": 5.0, "level": "high"},
            },
        ]
        picked = _select_surprise(buf)
        assert picked["kind"] == "outcome"
        assert picked["zscore"] == 5.0

    def test_fallback_latent_last(self):
        buf = [
            {"type": "campaign_surprise", "surprise": {"value": 0.1, "level": "low"}},
            {"type": "campaign_surprise", "surprise": {"value": 0.5, "level": "high"}},
        ]
        assert _select_surprise(buf)["value"] == 0.5

    def test_empty(self):
        assert _select_surprise([]) is None
