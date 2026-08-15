"""
Unit tests for configurable CEM parameters and campaign early stopping (REFAC-016).
"""

from __future__ import annotations

import pytest

from hagent.agent.campaign.builder import _campaign_planner
from hagent.agent.campaign.runner import check_early_stopping, run_campaign_tick
from hagent.agent.campaign.schema import Campaign, CampaignVariant


def test_configurable_cem_parameters() -> None:
    """_campaign_planner nhận các thông số CEM cấu hình từ dictionary."""
    custom_cfg = {
        "n_candidates": 16,
        "population_size": 16,
        "n_iterations": 4,
        "max_iterations": 4,
        "elite_fraction": 0.3,
        "smoothing": 0.2,
        "noise_std": 0.2,
        "exploration_weight": 0.25,
        "convergence_threshold": 0.002,
        "patience": 2,
        "search_algorithms": ["grid_search", "bayesian_search"],
        "time_limit_options": [120, 240],
    }
    planner = _campaign_planner(custom_cfg)
    assert planner is not None
    assert planner.n_candidates == 16
    assert planner.n_iterations == 4
    assert abs(planner.elite_fraction - 0.3) < 1e-6
    assert abs(planner.smoothing - 0.2) < 1e-6
    assert abs(planner.exploration_weight - 0.25) < 1e-6


def test_check_early_stopping_convergence() -> None:
    """check_early_stopping trả về True khi score hội tụ và False khi vẫn đang cải thiện."""
    v1 = CampaignVariant(
        variant_id="v1", label="v1", params={}, status="completed", best_score=0.850
    )
    v2 = CampaignVariant(
        variant_id="v2", label="v2", params={}, status="completed", best_score=0.851
    )
    v3 = CampaignVariant(
        variant_id="v3", label="v3", params={}, status="completed", best_score=0.852
    )

    campaign = Campaign(
        campaign_id="camp_converged",
        goal={"metric": "accuracy"},
        variants=[v1, v2, v3],
    )

    # 1. Với convergence_threshold = 0.005 và patience = 2:
    # Improvements: |0.851 - 0.850| = 0.001 < 0.005, |0.852 - 0.851| = 0.001 < 0.005 -> True
    assert (
        check_early_stopping(
            campaign, convergence_threshold=0.005, patience=2, higher_is_better=True
        )
        is True
    )

    # 2. Với convergence_threshold = 0.0005 -> False (vẫn lớn hơn ngưỡng 0.0005)
    assert (
        check_early_stopping(
            campaign, convergence_threshold=0.0005, patience=2, higher_is_better=True
        )
        is False
    )

    # 3. Khi score cải thiện mạnh -> False
    v3_big = CampaignVariant(
        variant_id="v3", label="v3", params={}, status="completed", best_score=0.920
    )
    campaign_improving = Campaign(
        campaign_id="camp_improving",
        goal={"metric": "accuracy"},
        variants=[v1, v2, v3_big],
    )
    assert (
        check_early_stopping(
            campaign_improving,
            convergence_threshold=0.005,
            patience=2,
            higher_is_better=True,
        )
        is False
    )


@pytest.mark.asyncio
async def test_campaign_tick_triggers_early_stopping() -> None:
    """run_campaign_tick tự động kích hoạt early stopping và hủy pending variants khi hội tụ."""
    v1 = CampaignVariant(
        variant_id="v1", label="v1", params={}, status="completed", best_score=0.850
    )
    v2 = CampaignVariant(
        variant_id="v2", label="v2", params={}, status="completed", best_score=0.851
    )
    v3 = CampaignVariant(
        variant_id="v3", label="v3", params={}, status="completed", best_score=0.852
    )
    v4_pending = CampaignVariant(
        variant_id="v4", label="v4", params={"dataset_id": "d1"}, status="pending"
    )

    campaign = Campaign(
        campaign_id="camp_tick_stop",
        goal={"metric": "accuracy"},
        variants=[v1, v2, v3, v4_pending],
        max_concurrent=0,  # Không submit thêm để test check
    )

    updated = await run_campaign_tick(
        campaign,
        user_id="u1",
        outcome_model=None,
    )

    assert getattr(updated, "early_stopped", False) is True
    assert v4_pending.status == "failed"
    assert "Early stopped" in str(v4_pending.error)
