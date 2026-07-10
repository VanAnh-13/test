"""Offline layer — wraps Phase 7 eval modes with richer expectations."""

from __future__ import annotations

import time
from typing import List, Optional

from hagent.agent.eval import runner as eval_runner
from hagent.agent.eval.scenarios import EvalScenario
from hagent.agent.execution.tool_runner import set_tool_invoker
from hagent.agent.harness.assertions import assert_expectations, has_job_signal
from hagent.agent.harness.mock_env import make_mock_tool_invoker
from hagent.agent.harness.schema import AgentRunResult, AgentScenario


def _to_eval_scenario(s: AgentScenario) -> EvalScenario:
    return EvalScenario(
        id=s.id,
        name=s.name,
        message=s.message,
        goal=dict(s.goal),
        world_model=dict(s.world_model),
        expect_goal_type=s.expect.goal_type or s.expect_goal_type or "train",
        expect_min_tools=s.expect.tools_called_min or s.expect_min_tools,
        expect_has_job=bool(
            s.expect.has_job if s.expect.has_job is not None else s.expect_has_job
        ),
        expect_metric=s.expect_metric,
        tags=list(s.tags),
    )


async def run_offline_scenario(
    scenario: AgentScenario,
    mode: str,
) -> AgentRunResult:
    invoker = make_mock_tool_invoker(scenario)
    set_tool_invoker(invoker)
    t0 = time.time()
    try:
        # Reuse Phase 7 mode runners
        eval_sc = _to_eval_scenario(scenario)
        mode = mode.lower()
        if mode == "single_shot":
            out = await eval_runner.run_single_shot(eval_sc, user_id=scenario.user_id)
        elif mode == "plan_executor":
            out = await eval_runner.run_plan_executor_mode(
                eval_sc, user_id=scenario.user_id
            )
        elif mode == "campaign":
            out = await eval_runner.run_campaign_mode(eval_sc, user_id=scenario.user_id)
        elif mode == "hierarchical":
            out = await eval_runner.run_hierarchical_mode(
                eval_sc, user_id=scenario.user_id
            )
        else:
            raise ValueError(f"Unknown offline mode: {mode}")

        elapsed = time.time() - t0
        tools = int(out.get("tools_called") or 0)
        # synthetic tool names for offline modes
        tool_names: List[str] = []
        if out.get("has_job") or (scenario.expect.has_job or scenario.expect_has_job):
            if mode != "list" and (
                scenario.goal.get("goal_type") == "train"
                or scenario.expect_has_job
                or scenario.expect.has_job
            ):
                tool_names.append("start_training")
        if tools and not tool_names:
            tool_names = ["tool"] * min(tools, 5)

        # hierarchical returns skipped count etc.
        has_job = bool(out.get("has_job"))
        ok, reasons = assert_expectations(
            scenario.expect,
            tools=tool_names if tool_names else (["start_training"] * tools if tools else []),
            has_job=has_job,
            goal_type=out.get("goal_type") or scenario.goal.get("goal_type"),
            plan_status=out.get("plan_status"),
            campaign_status=out.get("campaign_status"),
            hierarchy_status=out.get("hierarchy_status"),
            elapsed=elapsed,
            wm=scenario.world_model,
        )

        # Fall back to Phase 7 judge if expect mostly empty
        if scenario.expect.tools_called_min == 0 and scenario.expect.has_job is None:
            from hagent.agent.eval.metrics import judge_success

            ok2, reasons2 = judge_success(
                eval_sc,
                tools_called=tools,
                has_job=has_job,
                goal_type=out.get("goal_type"),
                plan_status=out.get("plan_status"),
                campaign_status=out.get("campaign_status"),
                mode=mode,
            )
            ok, reasons = ok2, reasons2

        cost = out.get("cost_metrics") or {}
        return AgentRunResult(
            scenario_id=scenario.id,
            layer="offline",
            mode=mode,
            success=ok,
            reasons=reasons,
            elapsed_seconds=round(elapsed, 4),
            tools_called=tools,
            tool_names=tool_names,
            steps_executed=int(cost.get("steps_executed") or 0),
            revisions=int(cost.get("revisions") or 0),
            campaign_variants=int(cost.get("campaign_variants") or 0),
            campaign_completed=int(cost.get("campaign_completed") or 0),
            best_job_id=out.get("best_job_id"),
            plan_status=out.get("plan_status"),
            campaign_status=out.get("campaign_status"),
            hierarchy_status=out.get("hierarchy_status"),
            hierarchy_depth=int(out.get("hierarchy_depth") or 0),
            cost_metrics=cost,
            extra={k: out[k] for k in ("hierarchy", "evaluation") if k in out},
        )
    finally:
        set_tool_invoker(None)
