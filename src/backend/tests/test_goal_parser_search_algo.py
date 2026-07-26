"""
Tests cho T8 — nhận diện thuật toán HPO từ text + builder ghim variant yêu cầu.
"""

from __future__ import annotations

import asyncio

import pytest

from hagent.agent.campaign.builder import build_campaign
from hagent.agent.planning.goal_parser import _detect_search_algorithm, parse_goal

BUILD_CFG = {
    "n_job_candidates": 3,
    "warm_start_top_k": 0,
    "wm_variant_proposal": False,
    "wm_rank_variants": False,
}


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestDetectSearchAlgorithm:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("hãy dùng successive halving để train", "successive_halving"),
            ("use successive-halving please", "successive_halving"),
            ("train với halving cho nhanh", "successive_halving"),
            ("dùng random search nhé", "random_search"),
            ("tìm kiếm ngẫu nhiên trên dataset này", "random_search"),
            ("optimize with bayesian optimization", "bayesian_search"),
            ("dùng bayes cho tôi", "bayesian_search"),
            ("train bằng genetic algorithm", "genetic_algorithm"),
            ("dùng thuật toán di truyền", "genetic_algorithm"),
            ("chạy tiến hóa thử xem", "genetic_algorithm"),
            ("grid search toàn bộ", "grid_search"),
            ("vét cạn lưới tham số", "grid_search"),
        ],
    )
    def test_detects(self, text, expected):
        assert _detect_search_algorithm(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "train một model cho tôi",
            "hãy chọn thuật toán tốt nhất",
            "kết quả ra sao rồi",
            # từ đơn mơ hồ không được bắt
            "cho tôi xem ga tàu",
        ],
    )
    def test_no_false_positive(self, text):
        assert _detect_search_algorithm(text) is None

    def test_parse_goal_populates_constraint(self):
        goal = parse_goal("train ds_glass target Type bằng successive halving")
        assert goal["constraints"]["search_algorithm"] == "successive_halving"

    def test_parse_goal_absent_when_not_mentioned(self):
        goal = parse_goal("train ds_glass target Type cho tôi")
        assert "search_algorithm" not in (goal.get("constraints") or {})

    def test_explicit_constraint_not_overridden(self):
        goal = parse_goal(
            "train bằng random search",
            default_user_constraints={"search_algorithm": "grid_search"},
        )
        assert goal["constraints"]["search_algorithm"] == "grid_search"


class TestBuilderPinsRequested:
    def test_requested_algo_pinned_first(self):
        goal = {
            "goal_type": "train",
            "dataset_id": "ds1",
            "problem_type": "classification",
            "metric": "accuracy",
            "target_column": "target",
            "constraints": {"search_algorithm": "successive_halving"},
        }
        camp = run(
            build_campaign(goal, user_id="t8_no_mem", config=BUILD_CFG, outcome_model=None)
        )
        assert camp.variants[0].source == "requested"
        assert camp.variants[0].params["search_algorithm"] == "successive_halving"
        # các slot còn lại vẫn đa dạng hóa
        assert len(camp.variants) == 3

    def test_no_constraint_no_pin(self):
        goal = {
            "goal_type": "train",
            "dataset_id": "ds1",
            "problem_type": "classification",
            "metric": "accuracy",
            "target_column": "target",
        }
        camp = run(
            build_campaign(goal, user_id="t8_no_mem", config=BUILD_CFG, outcome_model=None)
        )
        assert all(v.source != "requested" for v in camp.variants)
