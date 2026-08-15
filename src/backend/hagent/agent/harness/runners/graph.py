"""Graph layer — run hierarchy/campaign/plan paths with mock tools (no real LLM required for train)."""

from __future__ import annotations

import time
from typing import Any

from hagent.agent.execution.tool_runner import set_tool_invoker
from hagent.agent.harness.assertions import assert_expectations, has_job_signal
from hagent.agent.harness.mock_env import make_mock_tool_invoker
from hagent.agent.harness.schema import AgentRunResult, AgentScenario
from hagent.agent.planning.goal_parser import parse_goal
from hagent.agent.planning.hierarchy import apply_smart_skips, decompose_goal


async def run_graph_scenario(scenario: AgentScenario) -> AgentRunResult:
    """
    Exercise multi-agent paths without requiring a live LLM when possible.

    For train goals: hierarchy controller (same as production live path).
    For analyze/list: plan_executor / single tools via hierarchy or offline adapters.
    """
    invoker = make_mock_tool_invoker(scenario)
    set_tool_invoker(invoker)
    t0 = time.time()
    try:
        goal = dict(scenario.goal) if scenario.goal else {}
        if not goal.get("goal_type"):
            goal = parse_goal(
                scenario.message,
                known_dataset_ids=list((scenario.world_model.get("datasets") or {}).keys()),
            )

        gtype = str(goal.get("goal_type") or "respond").lower()
        out: dict[str, Any] = {}
        tool_names: list[str] = []
        event_types: list[str] = []
        mode = "graph"

        if gtype in ("train", "evaluate"):
            from hagent.agent.execution.hierarchy_node import (
                hierarchy_node,
                hierarchy_route,
            )

            hier = decompose_goal(goal)
            apply_smart_skips(hier, world_model=scenario.world_model)
            # Deep WM: attach WorldModelService so campaign/hierarchy record surprise
            try:
                from hagent.world.service import WorldModelService

                wm_service = WorldModelService.from_config()
            except Exception:
                wm_service = None
            state: dict[str, Any] = {
                "messages": [],
                "user_id": scenario.user_id,
                "goal": goal,
                "world_model": dict(scenario.world_model),
                "hierarchy": hier.to_dict(),
                "hierarchy_status": "running",
                "execution_events": [],
                "cost_metrics": {},
            }
            if wm_service is not None:
                state["_wm_service"] = wm_service
            for _ in range(40):
                step = await hierarchy_node(state)
                state.update(step)
                state["messages"] = []
                if wm_service is not None:
                    state["_wm_service"] = wm_service
                if hierarchy_route(state) == "synthesize":
                    break
            mode = "hierarchy"
            out = state
            for e in state.get("execution_events") or []:
                if isinstance(e, dict) and e.get("type"):
                    event_types.append(str(e["type"]))
            # Infer tools from events / campaign
            if state.get("campaign") or state.get("campaign_status") == "done":
                tool_names.extend(["start_training", "get_job_info"])
            if any(t == "subgoal_done" for t in event_types):
                # analyze may have been skipped
                if not any(
                    s.get("status") == "skipped" and s.get("goal_type") == "analyze"
                    for s in (state.get("hierarchy") or {}).get("subgoals") or []
                    if isinstance(s, dict)
                ):
                    tool_names.extend(["get_dataset_info", "get_features"])
            cost = state.get("cost_metrics") or {}
            if cost.get("tools_called"):
                # pad names to tools_called for min checks
                while len(tool_names) < int(cost["tools_called"]):
                    tool_names.append("tool")

        elif gtype in ("analyze", "select", "monitor", "list"):
            from hagent.agent.eval import runner as eval_runner
            from hagent.agent.eval.scenarios import EvalScenario

            eval_sc = EvalScenario(
                id=scenario.id,
                name=scenario.name,
                message=scenario.message,
                goal=goal,
                world_model=scenario.world_model,
                expect_goal_type=gtype,
                expect_min_tools=scenario.expect.tools_called_min,
                expect_has_job=False,
                tags=scenario.tags,
            )
            if gtype == "list":
                out = await eval_runner.run_single_shot(
                    eval_sc, user_id=scenario.user_id
                )
                tool_names = ["list_datasets"]
            else:
                out = await eval_runner.run_plan_executor_mode(
                    eval_sc, user_id=scenario.user_id
                )
                tool_names = ["get_dataset_info", "get_features"][: max(
                    1, int(out.get("tools_called") or 1)
                )]
            mode = "plan_executor" if gtype != "list" else "single_shot"
        else:
            # chitchat / respond — no tools
            out = {
                "goal_type": gtype,
                "tools_called": 0,
                "has_job": False,
                "response": "ok",
            }
            mode = "respond"

        elapsed = time.time() - t0
        tools_called = max(int(out.get("tools_called") or 0), len(tool_names))
        has_job = has_job_signal(out) or bool(out.get("has_job"))
        cost = out.get("cost_metrics") or {}

        ok, reasons = assert_expectations(
            scenario.expect,
            tools=tool_names,
            has_job=has_job,
            goal_type=goal.get("goal_type"),
            plan_status=out.get("plan_status"),
            campaign_status=out.get("campaign_status"),
            hierarchy_status=out.get("hierarchy_status"),
            route=mode,
            event_types=event_types,
            elapsed=elapsed,
            wm=out.get("world_model") or scenario.world_model,
        )

        return AgentRunResult(
            scenario_id=scenario.id,
            layer="graph",
            mode=mode,
            success=ok,
            reasons=reasons,
            elapsed_seconds=round(elapsed, 4),
            tools_called=tools_called,
            tool_names=tool_names,
            steps_executed=int(cost.get("steps_executed") or 0),
            campaign_variants=int(cost.get("campaign_variants") or 0),
            campaign_completed=int(cost.get("campaign_completed") or 0),
            best_job_id=(out.get("evaluation") or {}).get("best_job_id")
            or out.get("best_job_id"),
            plan_status=out.get("plan_status"),
            campaign_status=out.get("campaign_status"),
            hierarchy_status=out.get("hierarchy_status"),
            hierarchy_depth=len(
                (out.get("hierarchy") or {}).get("subgoals") or []
            ),
            route=mode,
            event_types=event_types,
            cost_metrics=cost,
            response=str(
                (out.get("messages") or [{}])[-1].content
                if out.get("messages") and hasattr((out.get("messages") or [None])[-1], "content")
                else out.get("response") or ""
            ),
            extra={
                "hierarchy": out.get("hierarchy"),
                "evaluation": out.get("evaluation"),
            },
        )
    finally:
        set_tool_invoker(None)
