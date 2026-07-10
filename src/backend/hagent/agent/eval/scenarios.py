"""
Fixed AutoML evaluation scenarios (tabular-focused).

Inspired by AutoML-Agent benchmark spirit (tabular classification/regression)
but tool-only against HAutoML-style actions — no code-gen.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvalScenario:
    """One offline-eval scenario."""

    id: str
    name: str
    message: str
    goal: Dict[str, Any]
    world_model: Dict[str, Any] = field(default_factory=dict)
    # Expected outcomes for success checks
    expect_goal_type: str = "train"
    expect_min_tools: int = 1
    expect_has_job: bool = True
    expect_metric: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def default_scenarios() -> List[EvalScenario]:
    """Built-in tabular scenarios (no external datasets required for mock mode)."""
    glass_wm = {
        "user_id": "eval_user",
        "datasets": {
            "ds_glass": {
                "id": "ds_glass",
                "name": "glass",
                "n_rows": 214,
                "n_cols": 10,
                "features": [
                    "RI",
                    "Na",
                    "Mg",
                    "Al",
                    "Si",
                    "K",
                    "Ca",
                    "Ba",
                    "Fe",
                    "Type",
                ],
                "target": "Type",
            }
        },
        "jobs": {},
        "active_dataset_id": "ds_glass",
    }

    osi_wm = {
        "user_id": "eval_user",
        "datasets": {
            "ds_osi": {
                "id": "ds_osi",
                "name": "online_shoppers",
                "n_rows": 12330,
                "n_cols": 18,
                "features": [
                    "Administrative",
                    "ProductRelated",
                    "BounceRates",
                    "ExitRates",
                    "PageValues",
                    "Revenue",
                ],
                "target": "Revenue",
            }
        },
        "jobs": {
            "hist_1": {
                "id": "hist_1",
                "status": "completed",
                "best_score": 0.82,
                "best_model": "rf",
                "config": {
                    "problem_type": "classification",
                    "search_algorithm": "bayesian_search",
                    "models": ["rf"],
                    "metric": "f1",
                },
            }
        },
        "active_dataset_id": "ds_osi",
    }

    house_wm = {
        "user_id": "eval_user",
        "datasets": {
            "ds_house": {
                "id": "ds_house",
                "name": "house_prices",
                "n_rows": 1460,
                "n_cols": 80,
                "features": ["OverallQual", "GrLivArea", "GarageCars", "SalePrice"],
                "target": "SalePrice",
            }
        },
        "jobs": {},
        "active_dataset_id": "ds_house",
    }

    return [
        EvalScenario(
            id="tab_clf_glass",
            name="Glass classification train",
            message=(
                "Train classification trên dataset ds_glass target Type, "
                "metric f1 trong 5 phút"
            ),
            goal={
                "goal_type": "train",
                "dataset_id": "ds_glass",
                "target_column": "Type",
                "problem_type": "classification",
                "metric": "f1",
                "constraints": {"time_limit": 300},
            },
            world_model=glass_wm,
            expect_goal_type="train",
            expect_min_tools=1,
            expect_has_job=True,
            expect_metric="f1",
            tags=["tabular", "classification"],
        ),
        EvalScenario(
            id="tab_clf_osi_warm",
            name="OSI classification with warm-start history",
            message=(
                "Huấn luyện model classification dataset ds_osi target Revenue metric f1"
            ),
            goal={
                "goal_type": "train",
                "dataset_id": "ds_osi",
                "target_column": "Revenue",
                "problem_type": "classification",
                "metric": "f1",
            },
            world_model=osi_wm,
            expect_has_job=True,
            expect_metric="f1",
            tags=["tabular", "classification", "warm_start"],
        ),
        EvalScenario(
            id="tab_reg_house",
            name="House price regression",
            message=(
                "Train regression dataset ds_house target SalePrice metric rmse 3 phút"
            ),
            goal={
                "goal_type": "train",
                "dataset_id": "ds_house",
                "target_column": "SalePrice",
                "problem_type": "regression",
                "metric": "rmse",
                "constraints": {"time_limit": 180},
            },
            world_model=house_wm,
            expect_has_job=True,
            expect_metric="rmse",
            tags=["tabular", "regression"],
        ),
        EvalScenario(
            id="tab_analyze_glass",
            name="Analyze glass dataset",
            message="Phân tích dataset ds_glass, xem features",
            goal={
                "goal_type": "analyze",
                "dataset_id": "ds_glass",
            },
            world_model=glass_wm,
            expect_goal_type="analyze",
            expect_min_tools=1,
            expect_has_job=False,
            tags=["tabular", "analyze"],
        ),
        EvalScenario(
            id="tab_list_datasets",
            name="List datasets",
            message="Liệt kê dataset của tôi",
            goal={"goal_type": "list"},
            world_model=glass_wm,
            expect_goal_type="list",
            expect_min_tools=0,
            expect_has_job=False,
            tags=["list"],
        ),
    ]


def scenarios_by_tags(
    tags: List[str] | None = None,
    scenario_ids: List[str] | None = None,
) -> List[EvalScenario]:
    all_s = default_scenarios()
    if scenario_ids:
        want = set(scenario_ids)
        all_s = [s for s in all_s if s.id in want]
    if tags:
        tagset = {t.lower() for t in tags}
        all_s = [s for s in all_s if tagset.intersection({t.lower() for t in s.tags})]
    return all_s
