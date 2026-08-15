"""Kiểm thử vòng lặp plan executor và reviser không dùng HAutoML thật."""

from __future__ import annotations

import asyncio

import pytest

from hagent.agent.execution.plan_executor import (
    plan_executor_node,
    plan_executor_route,
)
from hagent.agent.execution.reviser import (
    _patch_plan_for_error,
    reviser_node,
    reviser_route,
)
from hagent.agent.execution.tool_runner import set_tool_invoker
from hagent.agent.orchestration.graph import (
    _should_run_plan_executor,
    coordinator_route,
)


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _reset_invoker(monkeypatch):
    set_tool_invoker(None)
    # Tắt campaign và hierarchy để kiểm thử tập trung vào executor.
    monkeypatch.setattr(
        "hagent.agent.orchestration.graph._campaign_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        "hagent.agent.orchestration.graph._hierarchy_live_enabled",
        lambda: False,
    )
    yield
    set_tool_invoker(None)


def _base_state(**kwargs):
    plan = {
        "plan_id": "p1",
        "title": "train:ds",
        "steps": [
            {
                "action": {
                    "type": "get_dataset_info",
                    "params": {"dataset_id": "ds1"},
                }
            },
            {
                "action": {
                    "type": "start_training",
                    "params": {
                        "dataset_id": "ds1",
                        "problem_type": "classification",
                        "target_column": "target",
                    },
                }
            },
        ],
    }
    state = {
        "messages": [],
        "user_id": "u1",
        "user_token": "tok",
        "world_model": {
            "user_id": "u1",
            "datasets": {
                "ds1": {
                    "id": "ds1",
                    "name": "glass",
                    "features": ["a", "b", "target"],
                    "n_rows": 10,
                    "n_cols": 3,
                }
            },
            "jobs": {},
            "active_dataset_id": "ds1",
        },
        "goal": {
            "goal_type": "train",
            "dataset_id": "ds1",
            "problem_type": "classification",
            "target_column": "target",
        },
        "selected_plan": plan,
        "plan_status": "ready",
        "plan_step_index": 0,
        "revision_count": 0,
        "execution_log": [],
        "execution_events": [],
        "cost_metrics": {},
    }
    state.update(kwargs)
    return state


class TestCoordinatorPlanRoute:
    def test_should_run_executor(self):
        st = _base_state()
        assert _should_run_plan_executor(st) is True

    def test_should_not_run_for_respond(self):
        st = _base_state(goal={"goal_type": "respond"})
        # Dù còn step, goal respond vẫn phải bỏ qua executor.
        assert _should_run_plan_executor(st) is False

    def test_coordinator_route_prefers_executor(self):
        class Msg:
            tool_calls = None

        st = _base_state(messages=[Msg()])
        assert coordinator_route(st) == "plan_executor"


class TestPlanExecutor:
    def test_execute_one_step_ok(self):
        async def fake(action_type, params):
            return {"id": "ds1", "name": "glass", "features": ["a", "b", "target"]}

        set_tool_invoker(fake)
        st = _base_state()
        out = run(plan_executor_node(st))
        assert out["plan_status"] == "executing"
        assert out["plan_step_index"] == 1
        assert out["cost_metrics"]["steps_executed"] == 1
        assert plan_executor_route({**st, **out}) == "plan_executor"

    def test_execute_until_done(self):
        async def fake(action_type, params):
            if action_type == "get_dataset_info":
                return {"id": "ds1", "features": ["a", "b", "target"]}
            if action_type == "start_training":
                return {"job_id": "j1", "status": "starting"}
            return {}

        set_tool_invoker(fake)
        st = _base_state()
        out1 = run(plan_executor_node(st))
        st2 = {**st, **out1}
        # Messages dùng list reducer nên thay mới trong unit test.
        st2["messages"] = []
        out2 = run(plan_executor_node(st2))
        assert out2["plan_status"] == "done"
        assert plan_executor_route({**st2, **out2}) == "synthesize"

    def test_validate_fail_triggers_revise(self):
        st = _base_state()
        # Target không hợp lệ phải kích hoạt revise.
        st["selected_plan"]["steps"] = [
            {
                "action": {
                    "type": "start_training",
                    "params": {
                        "dataset_id": "ds1",
                        "problem_type": "classification",
                        "target_column": "NOPE",
                    },
                }
            }
        ]

        async def fake(a, p):
            return {}

        set_tool_invoker(fake)
        out = run(plan_executor_node(st))
        assert out["plan_status"] == "need_revise"
        assert plan_executor_route({**st, **out}) == "reviser"

    def test_tool_error_triggers_revise(self):
        async def fake(a, p):
            return {"error": "backend down"}

        set_tool_invoker(fake)
        st = _base_state()
        out = run(plan_executor_node(st))
        assert out["plan_status"] == "need_revise"
        assert "TOOL_REPORTED_ERROR" in (out.get("last_step_error") or "")
        assert "backend down" not in str(out)

    def test_tool_exception_is_redacted_before_reaching_agent_state(self, caplog):
        sensitive_detail = "provider failed with sensitive-token-value"

        async def fake(a, p):
            raise RuntimeError(sensitive_detail)

        set_tool_invoker(fake)
        out = run(plan_executor_node(_base_state()))

        assert out["plan_status"] == "need_revise"
        assert "TOOL_INVOCATION_FAILED" in str(out.get("last_step_error"))
        assert sensitive_detail not in str(out)
        assert "action=get_dataset_info" in caplog.text
        assert "type=RuntimeError" in caplog.text
        assert sensitive_detail not in caplog.text

    def test_request_scope_overrides_model_authority(self, monkeypatch):
        captured = []

        class _ListDatasetsTool:
            args = {"user_id": {}, "token": {}}

            async def ainvoke(self, params):
                captured.append(dict(params))
                return {"datasets": []}

        monkeypatch.setattr(
            "hagent.agent.orchestration.registry.get_tool_map",
            lambda: {"list_datasets": _ListDatasetsTool()},
        )
        st = _base_state(user_id="owner", user_token="request-token")
        st["selected_plan"]["steps"] = [
            {
                "action": {
                    "type": "list_datasets",
                    "params": {
                        "user_id": "spoofed-owner",
                        "token": "model-token",
                    },
                }
            }
        ]

        out = run(plan_executor_node(st))

        assert out["plan_status"] == "done"
        assert captured == [{"user_id": "owner", "token": "request-token"}]
        assert "spoofed-owner" not in str(out)
        assert "model-token" not in str(out)
        assert "request-token" not in str(out)

    def test_missing_request_credential_fails_closed_before_tool_call(
        self,
        monkeypatch,
    ):
        captured = []

        class _ListDatasetsTool:
            args = {"user_id": {}, "token": {}}

            async def ainvoke(self, params):
                captured.append(dict(params))
                return {"datasets": []}

        monkeypatch.setenv("USER_TOKEN", "ambient-process-token")
        monkeypatch.setattr(
            "hagent.agent.orchestration.registry.get_tool_map",
            lambda: {"list_datasets": _ListDatasetsTool()},
        )
        st = _base_state(user_id="owner", user_token=None)
        st["selected_plan"]["steps"] = [
            {
                "action": {
                    "type": "list_datasets",
                    "params": {
                        "user_id": "spoofed-owner",
                        "token": "model-token",
                    },
                }
            }
        ]

        out = run(plan_executor_node(st))

        assert out["plan_status"] == "need_revise"
        assert captured == []
        assert "AUTH_SCOPE_REQUIRED" in str(out.get("last_step_error"))
        assert "spoofed-owner" not in str(out)
        assert "model-token" not in str(out)
        assert "ambient-process-token" not in str(out)


class TestReviser:
    def test_patch_missing_dataset(self):
        plan = {
            "plan_id": "p",
            "steps": [
                {"action": {"type": "start_training", "params": {}}},
            ],
        }
        patched = _patch_plan_for_error(
            plan, "dataset_id required", {"dataset_id": "ds1"}
        )
        assert patched is not None
        types = [(s.get("action") or {}).get("type") for s in patched["steps"]]
        assert "list_datasets" in types or "get_dataset_info" in types

    def test_reviser_budget(self):
        st = _base_state(
            revision_count=2,
            last_step_error="boom",
            plan_status="need_revise",
        )
        # Mặc định tối đa 2 lần; lần thứ ba phải fail.
        out = run(reviser_node(st))
        assert out["plan_status"] == "failed"
        assert reviser_route({**st, **out}) == "synthesize"

    def test_reviser_patch_and_retry(self):
        st = _base_state(
            revision_count=0,
            last_step_error="target_column='x' not in dataset features",
            plan_status="need_revise",
        )
        out = run(reviser_node(st))
        assert out["plan_status"] == "ready"
        assert out["plan_step_index"] == 0
        assert out["revision_count"] == 1
        assert reviser_route({**st, **out}) == "plan_executor"


class TestEnrichParams:
    def test_enrich_start_training(self):
        from hagent.agent.execution.tool_runner import enrich_params

        p = enrich_params(
            "start_training",
            {},
            user_id="u1",
            user_token="t",
            goal={
                "dataset_id": "ds1",
                "problem_type": "classification",
                "target_column": "y",
                "metric": "f1",
                "constraints": {"time_limit": 60},
            },
            world_model={},
        )
        assert p["user_id"] == "u1"
        assert p["dataset_id"] == "ds1"
        assert p["target_column"] == "y"
        assert p["time_limit"] == 60
