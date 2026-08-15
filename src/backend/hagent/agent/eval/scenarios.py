"""
Fixed AutoML evaluation scenarios (tabular-focused).

Inspired by AutoML-Agent benchmark spirit (tabular classification/regression)
but tool-only against HAutoML-style actions — no code-gen.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

BASELINE_VERSION = "hagent-eval-v1"


@dataclass(frozen=True)
class ToolExpectation:
    """Expected semantic tool call and the output fields that prove it happened."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    evidence_keys: list[str] = field(default_factory=list)


@dataclass
class EvalScenario:
    """One offline-eval scenario."""

    id: str
    name: str
    message: str
    goal: dict[str, Any]
    world_model: dict[str, Any] = field(default_factory=dict)
    # Expected outcomes for success checks
    expect_goal_type: str = "train"
    expect_min_tools: int = 1
    expect_has_job: bool = True
    expect_metric: str | None = None
    tags: list[str] = field(default_factory=list)
    baseline_version: str | None = None
    turns: list[str] = field(default_factory=list)
    expect_goal: dict[str, Any] = field(default_factory=dict)
    expect_tool_calls: list[ToolExpectation] = field(default_factory=list)
    expect_outcome: str = "succeeded"
    allow_mutations: bool = True
    max_latency_seconds: float | None = None
    max_tokens: int | None = None
    mock_failures: dict[str, str] = field(default_factory=dict)
    legacy_expected_success: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def messages(self) -> list[str]:
        """Return the ordered user turns represented by this scenario."""
        return list(self.turns) if self.turns else [self.message]


def default_scenarios() -> list[EvalScenario]:
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


def baseline_scenarios() -> list[EvalScenario]:
    """Frozen behavior-rich cases for legacy/new runtime comparisons."""
    glass_world = {
        "user_id": "eval_user",
        "datasets": {
            "ds_glass": {
                "id": "ds_glass",
                "name": "glass",
                "features": ["RI", "Na", "Mg", "Type"],
                "target": "Type",
            }
        },
        "jobs": {},
        "active_dataset_id": "ds_glass",
    }
    train_goal = {
        "goal_type": "train",
        "dataset_id": "ds_glass",
        "target_column": "Type",
        "problem_type": "classification",
        "metric": "f1",
        "constraints": {"time_limit": 300},
    }
    def train_expectations(scenario_id: str) -> list[ToolExpectation]:
        return [
            ToolExpectation(
                name="start_training",
                arguments={
                    "user_id": "eval_user",
                    "dataset_id": "ds_glass",
                    "target_column": "Type",
                    "problem_type": "classification",
                    "metric": "f1",
                    "time_limit": 300,
                    "list_feature": ["RI", "Na", "Mg", "Type"],
                },
                evidence_keys=["job_id", "status"],
            ),
            ToolExpectation(
                name="get_job_info",
                arguments={"job_id": f"eval-job-{scenario_id}-1"},
                evidence_keys=["job_id", "status", "best_score"],
            ),
        ]

    return [
        EvalScenario(
            id="vi_train_complete",
            name="Vietnamese complete training request",
            message=(
                "Huấn luyện classification trên dataset ds_glass, target Type, "
                "metric f1 trong 5 phút"
            ),
            goal={},
            world_model=glass_world,
            expect_goal=dict(train_goal),
            expect_tool_calls=train_expectations("vi_train_complete"),
            expect_metric="f1",
            baseline_version=BASELINE_VERSION,
            legacy_expected_success=True,
            tags=["baseline", "vi", "train"],
        ),
        EvalScenario(
            id="en_analyze_dataset",
            name="English dataset analysis",
            message="Analyze dataset ds_glass and show its features",
            goal={},
            world_model=glass_world,
            expect_goal={"goal_type": "analyze", "dataset_id": "ds_glass"},
            expect_tool_calls=[
                ToolExpectation(
                    name="get_dataset_info",
                    arguments={"dataset_id": "ds_glass"},
                    evidence_keys=["dataset_id", "features"],
                )
            ],
            expect_goal_type="analyze",
            expect_has_job=False,
            allow_mutations=False,
            baseline_version=BASELINE_VERSION,
            legacy_expected_success=True,
            tags=["baseline", "en", "analyze"],
        ),
        EvalScenario(
            id="vi_train_multiturn",
            name="Vietnamese multi-turn target completion",
            message="Target là Type, dùng metric f1 và chạy trong 5 phút",
            turns=[
                "Hãy huấn luyện classification trên dataset ds_glass",
                "Target là Type, dùng metric f1 và chạy trong 5 phút",
            ],
            goal={},
            world_model=glass_world,
            expect_goal=dict(train_goal),
            expect_tool_calls=train_expectations("vi_train_multiturn"),
            baseline_version=BASELINE_VERSION,
            legacy_expected_success=False,
            tags=["baseline", "vi", "multi_turn", "train"],
        ),
        EvalScenario(
            id="en_train_missing_target",
            name="English request missing target",
            message="Train a classification model on dataset ds_glass",
            goal={},
            world_model=glass_world,
            expect_goal={
                "goal_type": "train",
                "dataset_id": "ds_glass",
                "problem_type": "classification",
            },
            expect_min_tools=0,
            expect_has_job=False,
            expect_outcome="needs_input",
            allow_mutations=False,
            baseline_version=BASELINE_VERSION,
            legacy_expected_success=True,
            tags=["baseline", "en", "missing_info", "train"],
        ),
        EvalScenario(
            id="vi_analyze_upstream_failure",
            name="Vietnamese deterministic upstream failure",
            message="Phân tích dataset ds_glass và mô tả các feature",
            goal={},
            world_model=glass_world,
            expect_goal={"goal_type": "analyze", "dataset_id": "ds_glass"},
            expect_tool_calls=[
                ToolExpectation(
                    name="get_dataset_info",
                    arguments={"dataset_id": "ds_glass"},
                )
            ],
            expect_goal_type="analyze",
            expect_has_job=False,
            expect_outcome="upstream_failure",
            allow_mutations=False,
            mock_failures={"get_dataset_info": "UPSTREAM_UNAVAILABLE"},
            baseline_version=BASELINE_VERSION,
            legacy_expected_success=True,
            tags=["baseline", "vi", "upstream_failure", "analyze"],
        ),
    ]


def scenarios_by_tags(
    tags: list[str] | None = None,
    scenario_ids: list[str] | None = None,
) -> list[EvalScenario]:
    all_s = default_scenarios()
    if scenario_ids:
        want = set(scenario_ids)
        all_s = [s for s in all_s if s.id in want]
    if tags:
        tagset = {t.lower() for t in tags}
        all_s = [s for s in all_s if tagset.intersection({t.lower() for t in s.tags})]
    return all_s
