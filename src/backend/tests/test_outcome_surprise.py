"""
Tests cho outcome-space surprise: compute_outcome_surprise, wm_hooks helper,
và event campaign_outcome_surprise trong campaign_step.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from hagent.agent.campaign.builder import build_campaign
from hagent.agent.campaign.schema import CampaignVariant
from hagent.agent.campaign.wm_hooks import campaign_outcome_surprise
from hagent.agent.execution.tool_runner import set_tool_invoker
from hagent.world.predictor.outcome_head_v1 import (
    OutcomeHeadV1,
    OutcomePrediction,
    train_outcome_head,
)
from hagent.world.surprise import compute_outcome_surprise


def run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


HEAD_CFG = {"use_latent": False, "hidden_dim": 24}
DATASET_META = {"n_rows": 1000, "n_cols": 12}


def _trained_head(seed=0):
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(80):
        t = int(rng.choice([60, 180, 600]))
        samples.append(
            {
                "params": {
                    "search_algorithm": "grid_search",
                    "problem_type": "classification",
                    "metric": "accuracy",
                    "time_limit": t,
                },
                "dataset_meta": DATASET_META,
                "best_score": float(0.8 + rng.normal(0, 0.02)),
            }
        )
    return train_outcome_head(samples, config=dict(HEAD_CFG), epochs=60, seed=seed)


# ── compute_outcome_surprise ─────────────────────────────


class TestComputeOutcomeSurprise:
    def test_zscore_value(self):
        pred = OutcomePrediction(mean=0.8, std=0.1)
        r = compute_outcome_surprise(pred, 0.95)
        assert r.value == pytest.approx(1.5)
        assert r.predicted_dim == 1 and r.actual_dim == 1

    def test_levels_from_thresholds(self):
        pred = OutcomePrediction(mean=0.8, std=0.1)
        assert compute_outcome_surprise(pred, 0.85).level == "low"
        assert compute_outcome_surprise(pred, 0.96).level == "medium"
        assert compute_outcome_surprise(pred, 1.2).level == "high"

    def test_custom_thresholds(self):
        pred = OutcomePrediction(mean=0.8, std=0.1)
        cfg = {"outcome_thresholds": {"medium": 0.5, "high": 1.0}}
        assert compute_outcome_surprise(pred, 0.86, cfg).level == "medium"
        assert compute_outcome_surprise(pred, 0.95, cfg).level == "high"

    def test_zero_std_guarded(self):
        r = compute_outcome_surprise((0.8, 0.0), 0.9)
        assert np.isfinite(r.value)
        assert r.level == "high"

    def test_exact_prediction_low(self):
        r = compute_outcome_surprise({"mean": 0.9, "std": 0.05}, 0.9)
        assert r.value == 0.0
        assert r.level == "low"


# ── wm_hooks.campaign_outcome_surprise ───────────────────


def _variant(status="completed", best_score=0.8, time_limit=180):
    return CampaignVariant(
        variant_id="v1",
        label="test",
        params={
            "search_algorithm": "grid_search",
            "problem_type": "classification",
            "metric": "accuracy",
            "time_limit": time_limit,
        },
        status=status,
        best_score=best_score,
    )


class TestCampaignOutcomeSurprise:
    def test_completed_variant_gets_surprise(self):
        head = _trained_head()
        out = campaign_outcome_surprise(
            variant=_variant(best_score=0.8),
            dataset_meta=DATASET_META,
            outcome_model=head,
            surprise_config={"outcome_thresholds": {"medium": 1.5, "high": 3.0}},
        )
        assert out is not None
        assert out["actual_score"] == 0.8
        assert out["level"] in ("low", "medium", "high")
        assert out["predictor"] == "outcome_head_v1"
        assert out["predicted_std"] > 0

    def test_shocking_score_is_high(self):
        head = _trained_head()
        out = campaign_outcome_surprise(
            variant=_variant(best_score=0.1),  # rất xa 0.8 đã học
            dataset_meta=DATASET_META,
            outcome_model=head,
            surprise_config={"outcome_thresholds": {"medium": 1.5, "high": 3.0}},
        )
        assert out is not None
        assert out["level"] == "high"

    def test_running_variant_returns_none(self):
        out = campaign_outcome_surprise(
            variant=_variant(status="running"),
            outcome_model=_trained_head(),
        )
        assert out is None

    def test_missing_score_returns_none(self):
        out = campaign_outcome_surprise(
            variant=_variant(best_score=None),
            outcome_model=_trained_head(),
        )
        assert out is None

    def test_unready_model_returns_none(self):
        out = campaign_outcome_surprise(
            variant=_variant(),
            outcome_model=OutcomeHeadV1({}),
        )
        assert out is None


# ── Runner integration ───────────────────────────────────


GOAL = {
    "goal_type": "train",
    "dataset_id": "ds1",
    "problem_type": "classification",
    "metric": "accuracy",
    "target_column": "target",
}


class TestRunnerIntegration:
    def teardown_method(self):
        set_tool_invoker(None)

    def test_outcome_surprise_event_on_completion(self, monkeypatch, tmp_path):
        """Variant hoàn thành → event campaign_outcome_surprise xuất hiện."""
        head = _trained_head()
        ckpt = str(tmp_path / "outcome.npz")
        head.save(ckpt)

        # Trỏ default outcome model của wm_hooks về checkpoint vừa train
        from hagent.agent.campaign import wm_hooks

        monkeypatch.setattr(
            wm_hooks,
            "_default_outcome_model",
            lambda: OutcomeHeadV1(dict(HEAD_CFG, checkpoint_path=ckpt)),
        )

        async def fake(action_type, params):
            if action_type == "start_training":
                return {"job_id": "j1", "status": 0}
            if action_type == "get_job_info":
                return {
                    "id": params.get("job_id"),
                    "status": "completed",
                    "best_model": "rf",
                    "best_score": 0.82,
                }
            return {}

        set_tool_invoker(fake)

        from hagent.agent.campaign.runner import campaign_step

        camp = run(build_campaign(GOAL, user_id="u1", config={"n_job_candidates": 1}))
        events: list = []
        camp = run(
            campaign_step(
                camp,
                user_id="u1",
                user_token=None,
                world_model={"datasets": {"ds1": DATASET_META}},
                surprise_events=events,
            )
        )
        outcome_events = [
            e for e in events if e.get("type") == "campaign_outcome_surprise"
        ]
        assert len(outcome_events) == 1
        ev = outcome_events[0]
        assert ev["job_id"] == "j1"
        assert ev["outcome"]["actual_score"] == 0.82
        assert ev["outcome"]["level"] in ("low", "medium", "high")

    def test_no_event_while_running(self, monkeypatch):
        from hagent.agent.campaign import wm_hooks

        monkeypatch.setattr(wm_hooks, "_default_outcome_model", lambda: _trained_head())

        async def fake(action_type, params):
            if action_type == "start_training":
                return {"job_id": "j1", "status": 0}
            return {"id": params.get("job_id"), "status": "running"}

        set_tool_invoker(fake)
        from hagent.agent.campaign.runner import campaign_step

        camp = run(build_campaign(GOAL, user_id="u1", config={"n_job_candidates": 1}))
        events: list = []
        camp = run(
            campaign_step(
                camp,
                user_id="u1",
                user_token=None,
                world_model={},
                surprise_events=events,
            )
        )
        assert not [e for e in events if e.get("type") == "campaign_outcome_surprise"]

    def test_explicit_none_disables_events(self, monkeypatch):
        """campaign_step(outcome_model=None) tắt hẳn — kể cả khi default có model."""
        from hagent.agent.campaign import wm_hooks

        monkeypatch.setattr(wm_hooks, "_default_outcome_model", lambda: _trained_head())

        async def fake(action_type, params):
            if action_type == "start_training":
                return {"job_id": "j1", "status": 0}
            return {
                "id": params.get("job_id"),
                "status": "completed",
                "best_score": 0.82,
            }

        set_tool_invoker(fake)
        from hagent.agent.campaign.runner import campaign_step

        camp = run(build_campaign(GOAL, user_id="u1", config={"n_job_candidates": 1}))
        events: list = []
        run(
            campaign_step(
                camp,
                user_id="u1",
                user_token=None,
                world_model={},
                surprise_events=events,
                outcome_model=None,
            )
        )
        assert not [e for e in events if e.get("type") == "campaign_outcome_surprise"]

    def test_explicit_model_used_by_runner(self):
        """Model truyền trực tiếp vào campaign_step được dùng, không cần default."""
        head = _trained_head()

        async def fake(action_type, params):
            if action_type == "start_training":
                return {"job_id": "j1", "status": 0}
            return {
                "id": params.get("job_id"),
                "status": "completed",
                "best_score": 0.82,
            }

        set_tool_invoker(fake)
        from hagent.agent.campaign.runner import campaign_step

        camp = run(build_campaign(GOAL, user_id="u1", config={"n_job_candidates": 1}))
        events: list = []
        run(
            campaign_step(
                camp,
                user_id="u1",
                user_token=None,
                world_model={"datasets": {"ds1": DATASET_META}},
                surprise_events=events,
                outcome_model=head,
            )
        )
        outcome_events = [
            e for e in events if e.get("type") == "campaign_outcome_surprise"
        ]
        assert len(outcome_events) == 1

    def test_event_fires_once_per_completion(self, monkeypatch):
        """Tick thứ hai sau khi đã completed không phát thêm event."""
        from hagent.agent.campaign import wm_hooks

        monkeypatch.setattr(wm_hooks, "_default_outcome_model", lambda: _trained_head())

        async def fake(action_type, params):
            if action_type == "start_training":
                return {"job_id": "j1", "status": 0}
            return {
                "id": params.get("job_id"),
                "status": "completed",
                "best_score": 0.82,
            }

        set_tool_invoker(fake)
        from hagent.agent.campaign.runner import campaign_step

        camp = run(build_campaign(GOAL, user_id="u1", config={"n_job_candidates": 1}))
        events: list = []
        camp = run(
            campaign_step(
                camp,
                user_id="u1",
                user_token=None,
                world_model={},
                surprise_events=events,
            )
        )
        n_first = len(
            [e for e in events if e.get("type") == "campaign_outcome_surprise"]
        )
        camp = run(
            campaign_step(
                camp,
                user_id="u1",
                user_token=None,
                world_model={},
                surprise_events=events,
            )
        )
        n_second = len(
            [e for e in events if e.get("type") == "campaign_outcome_surprise"]
        )
        assert n_first == 1
        assert n_second == 1
