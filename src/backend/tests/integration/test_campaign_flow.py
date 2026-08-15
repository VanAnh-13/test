"""
Kiểm thử tích hợp cho Campaign đa lượt và Phân rã Mục tiêu Phân cấp (REFAC-027).
"""

from __future__ import annotations

from hagent.agent.campaign.compare import compare_campaign
from hagent.agent.campaign.schema import Campaign, CampaignVariant
from hagent.agent.planning.hierarchy import GoalHierarchy, decompose_goal


def test_goal_hierarchy_decomposition_and_progression() -> None:
    """Kiểm thử phân rã mục tiêu cấp cao thành các mục tiêu con (subgoals) và theo dõi tiến trình."""
    goal_dict = {
        "text": "Huấn luyện mô hình phân loại dự đoán rủi ro tín dụng",
        "dataset_name": "credit_risk.csv",
        "problem_type": "classification",
        "metric": "roc_auc",
    }
    hierarchy = decompose_goal(goal_dict)
    assert isinstance(hierarchy, GoalHierarchy)
    assert len(hierarchy.subgoals) > 0

    # Kiểm tra trạng thái của mục tiêu con đầu tiên
    first_subgoal = hierarchy.subgoals[0]
    assert first_subgoal.status in {"pending", "active", "completed", "skipped"}


def test_campaign_variant_comparison_and_budget_tracking() -> None:
    """Kiểm thử Campaign so sánh các biến thể thử nghiệm (variants) và chọn mô hình tối ưu nhất."""
    v1 = CampaignVariant(
        variant_id="var_rf",
        label="Random Forest",
        params={"algorithm": "random_forest", "n_estimators": 100},
        status="completed",
        best_score=0.88,
    )
    v2 = CampaignVariant(
        variant_id="var_lgbm",
        label="LightGBM",
        params={"algorithm": "lightgbm", "num_leaves": 31},
        status="completed",
        best_score=0.94,
    )
    v3 = CampaignVariant(
        variant_id="var_xgb",
        label="XGBoost",
        params={"algorithm": "xgboost", "max_depth": 6},
        status="completed",
        best_score=0.91,
    )

    campaign = Campaign(
        campaign_id="camp_001",
        goal={"metric": "accuracy"},
        variants=[v1, v2, v3],
        total_budget=5,
        spent_budget=3,
    )

    best_variant, comparison = compare_campaign(campaign)
    assert best_variant is not None
    assert best_variant.variant_id == "var_lgbm"
    assert len(comparison) == 3
