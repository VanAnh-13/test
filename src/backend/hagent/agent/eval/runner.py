"""
Eval harness runner — offline modes with mockable tools.

Modes:
  - single_shot: one start_training (baseline, AutoML-Agent-like single attempt without multi-plan)
  - plan_executor: sequential plan steps (campaign disabled)
  - campaign: multi-candidate jobs (Phase 6)
  - hierarchical: decompose goal then campaign/executor on train leaf
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from hagent.agent.eval.metrics import ScenarioResult, judge_success, summarize
from hagent.agent.eval.scenarios import EvalScenario, scenarios_by_tags
from hagent.agent.execution.tool_runner import set_tool_invoker
from hagent.agent.planning.hierarchy import decompose_goal, subgoal_as_goal
from hagent.agent.planning.goal_parser import parse_goal

logger = logging.getLogger(__name__)


def _default_mock_tool_factory(scenario: EvalScenario) -> Callable:
    """Deterministic HAutoML-like tool responses for offline eval."""
    job_n = {"i": 0}
    scores = [0.71, 0.88, 0.80, 0.76]

    async def invoker(action_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        ds = (
            params.get("dataset_id")
            or scenario.goal.get("dataset_id")
            or scenario.world_model.get("active_dataset_id")
        )
        wm_ds = (scenario.world_model.get("datasets") or {}).get(ds or "", {})

        if action_type == "list_datasets":
            datasets = list((scenario.world_model.get("datasets") or {}).values())
            return {"datasets": datasets}
        if action_type in ("get_dataset_info", "preview_data", "get_features"):
            feats = wm_ds.get("features") or ["f1", "f2", "target"]
            return {
                "id": ds,
                "dataset_id": ds,
                "name": wm_ds.get("name", ds),
                "features": feats,
                "target": wm_ds.get("target"),
                "n_rows": wm_ds.get("n_rows", 100),
                "n_cols": wm_ds.get("n_cols", len(feats)),
            }
        if action_type in ("get_available_models", "get_metrics"):
            return {
                "problem_type": params.get("problem_type") or "classification",
                "models": ["rf", "lr", "xgb"],
                "metrics": ["accuracy", "f1", "rmse", "mae"],
            }
        if action_type == "start_training":
            job_n["i"] += 1
            jid = f"eval-job-{scenario.id}-{job_n['i']}"
            return {"job_id": jid, "status": "starting", "dataset_id": ds}
        if action_type == "get_job_info":
            jid = params.get("job_id") or "eval-job"
            # Map job index from id suffix
            try:
                idx = int(str(jid).rsplit("-", 1)[-1]) - 1
            except ValueError:
                idx = 0
            score = scores[idx % len(scores)]
            return {
                "id": jid,
                "job_id": jid,
                "status": "completed",
                "best_score": score,
                "best_model": f"model_{idx}",
                "metrics": {
                    scenario.goal.get("metric") or "f1": score,
                },
            }
        if action_type == "list_jobs":
            return {"jobs": []}
        if action_type == "get_world_state":
            return scenario.world_model
        return {"ok": True, "action": action_type}

    return invoker


async def run_single_shot(
    scenario: EvalScenario,
    *,
    user_id: str = "eval_user",
) -> Dict[str, Any]:
    """Baseline: one start_training call."""
    from hagent.agent.execution.tool_runner import enrich_params, invoke_tool

    goal = dict(scenario.goal)
    params = enrich_params(
        "start_training",
        {},
        user_id=user_id,
        user_token=None,
        goal=goal,
        world_model=scenario.world_model,
    )
    # Ensure required
    params.setdefault("dataset_id", goal.get("dataset_id"))
    params.setdefault("target_column", goal.get("target_column"))
    params.setdefault("problem_type", goal.get("problem_type") or "classification")

    tools = 0
    has_job = False
    best_score = None
    best_job = None

    if goal.get("goal_type") == "train" and params.get("dataset_id") and params.get(
        "target_column"
    ):
        payload = await invoke_tool("start_training", params)
        tools += 1
        if isinstance(payload, dict) and not payload.get("error"):
            jid = payload.get("job_id") or payload.get("id")
            has_job = bool(jid)
            best_job = jid
            info = await invoke_tool("get_job_info", {"job_id": jid})
            tools += 1
            if isinstance(info, dict):
                best_score = info.get("best_score")
    elif goal.get("goal_type") in ("analyze", "list"):
        if goal.get("goal_type") == "list":
            await invoke_tool("list_datasets", {"user_id": user_id})
        else:
            await invoke_tool(
                "get_dataset_info",
                {"dataset_id": goal.get("dataset_id") or "ds"},
            )
        tools += 1

    return {
        "tools_called": tools,
        "has_job": has_job,
        "best_score": best_score,
        "best_job_id": best_job,
        "goal_type": goal.get("goal_type"),
        "plan_status": "done" if tools else None,
        "campaign_status": None,
        "cost_metrics": {"tools_called": tools, "steps_executed": tools},
        "hierarchy_depth": 1,
    }


async def run_plan_executor_mode(
    scenario: EvalScenario,
    *,
    user_id: str = "eval_user",
) -> Dict[str, Any]:
    """Sequential plan via WorldModelService + plan_executor node."""
    from hagent.agent.execution.plan_executor import plan_executor_node
    from hagent.agent.planning.plan_adapter import plan_results_to_state_update
    from hagent.world.service import WorldModelService

    goal = dict(scenario.goal)
    wm = WorldModelService.from_config(
        {
            "encoder": {"backend": "structured_v1", "dim": 32},
            "predictor": {"backend": "tabular_transition_v1"},
            "planner": {
                "backend": "cem_lite",
                "horizon": 4,
                "n_candidates": 6,
                "n_return_plans": 1,
            },
            "surprise": {"metric": "l2", "thresholds": {"medium": 0.2, "high": 0.5}},
            "trajectory": {"enabled": False},
        }
    )
    obs = wm.observation_from_snapshot(
        scenario.world_model, user_id=user_id, goal=goal
    )
    plans = wm.plan(obs, goal)
    update = plan_results_to_state_update(plans, select_best=True)

    state: Dict[str, Any] = {
        "messages": [],
        "user_id": user_id,
        "goal": goal,
        "world_model": scenario.world_model,
        "selected_plan": update.get("selected_plan"),
        "plan_status": "ready",
        "plan_step_index": 0,
        "revision_count": 0,
        "execution_log": [],
        "execution_events": [],
        "cost_metrics": {},
        "_wm_service": wm,
    }

    # Cap steps
    for _ in range(12):
        out = await plan_executor_node(state)
        state.update(out)
        state["messages"] = []
        if state.get("plan_status") in ("done", "failed", "need_revise"):
            if state.get("plan_status") == "need_revise":
                # one soft skip: advance index
                state["plan_status"] = "executing"
                state["plan_step_index"] = int(state.get("plan_step_index") or 0) + 1
                state["last_step_error"] = None
                if state["plan_step_index"] >= len(
                    (state.get("selected_plan") or {}).get("steps") or []
                ):
                    state["plan_status"] = "done"
                    break
            else:
                break

    cost = state.get("cost_metrics") or {}
    log = state.get("execution_log") or []
    has_job = any(
        (e.get("action") == "start_training" and e.get("status") == "ok")
        for e in log
    )
    # Also check world model jobs
    jobs = (state.get("world_model") or {}).get("jobs") or {}
    if jobs:
        has_job = True

    return {
        "tools_called": int(cost.get("tools_called") or cost.get("steps_executed") or 0),
        "has_job": has_job,
        "best_score": None,
        "best_job_id": next(iter(jobs.keys()), None) if jobs else None,
        "goal_type": goal.get("goal_type"),
        "plan_status": state.get("plan_status"),
        "campaign_status": None,
        "cost_metrics": cost,
        "hierarchy_depth": 1,
    }


async def run_campaign_mode(
    scenario: EvalScenario,
    *,
    user_id: str = "eval_user",
) -> Dict[str, Any]:
    from hagent.agent.campaign.nodes import campaign_node, campaign_route

    goal = dict(scenario.goal)
    state: Dict[str, Any] = {
        "messages": [],
        "user_id": user_id,
        "goal": goal,
        "world_model": dict(scenario.world_model),
        "execution_events": [],
        "cost_metrics": {},
        "campaign_tick": 0,
    }

    for _ in range(20):
        out = await campaign_node(state)
        state.update(out)
        state["messages"] = []
        if campaign_route(state) == "synthesize":
            break

    cost = state.get("cost_metrics") or {}
    evaluation = state.get("evaluation") or {}
    return {
        "tools_called": int(
            cost.get("campaign_submitted") or 0
        )
        + int(cost.get("campaign_completed") or 0),  # rough: submit+poll
        "has_job": bool(evaluation.get("best_job_id") or cost.get("campaign_submitted")),
        "best_score": None,
        "best_job_id": evaluation.get("best_job_id"),
        "goal_type": goal.get("goal_type"),
        "plan_status": state.get("plan_status"),
        "campaign_status": state.get("campaign_status"),
        "cost_metrics": cost,
        "hierarchy_depth": 1,
        "evaluation": evaluation,
    }


async def run_hierarchical_mode(
    scenario: EvalScenario,
    *,
    user_id: str = "eval_user",
) -> Dict[str, Any]:
    """
    Live hierarchy controller (smart-skip + campaign train leaf).
    """
    from hagent.agent.execution.hierarchy_node import hierarchy_node, hierarchy_route
    from hagent.agent.planning.hierarchy import apply_smart_skips, decompose_goal

    hier = decompose_goal(dict(scenario.goal))
    apply_smart_skips(hier, world_model=scenario.world_model)
    depth = len(hier.subgoals)

    state: Dict[str, Any] = {
        "messages": [],
        "user_id": user_id,
        "goal": dict(scenario.goal),
        "world_model": dict(scenario.world_model),
        "hierarchy": hier.to_dict(),
        "hierarchy_status": "running",
        "execution_events": [],
        "cost_metrics": {},
        "campaign_tick": 0,
    }

    for _ in range(40):
        out = await hierarchy_node(state)
        state.update(out)
        state["messages"] = []
        if hierarchy_route(state) == "synthesize":
            break

    cost = state.get("cost_metrics") or {}
    evaluation = state.get("evaluation") or {}
    has_job = bool(
        evaluation.get("best_job_id")
        or cost.get("campaign_submitted")
        or (state.get("world_model") or {}).get("jobs")
    )
    return {
        "tools_called": int(cost.get("tools_called") or 0),
        "has_job": has_job,
        "best_score": None,
        "best_job_id": evaluation.get("best_job_id"),
        "goal_type": scenario.goal.get("goal_type"),
        "plan_status": state.get("plan_status"),
        "campaign_status": state.get("campaign_status"),
        "cost_metrics": cost,
        "hierarchy_depth": depth,
        "hierarchy": state.get("hierarchy"),
        "hierarchy_status": state.get("hierarchy_status"),
        "evaluation": evaluation,
        "skipped": sum(
            1
            for s in ((state.get("hierarchy") or {}).get("subgoals") or [])
            if isinstance(s, dict) and s.get("status") == "skipped"
        ),
    }


_MODE_RUNNERS = {
    "single_shot": run_single_shot,
    "plan_executor": run_plan_executor_mode,
    "campaign": run_campaign_mode,
    "hierarchical": run_hierarchical_mode,
}


async def run_scenario(
    scenario: EvalScenario,
    mode: str,
    *,
    user_id: str = "eval_user",
    tool_invoker: Callable | None = None,
) -> ScenarioResult:
    mode = mode.lower()
    if mode not in _MODE_RUNNERS:
        raise ValueError(f"Unknown mode {mode}. Choose from {list(_MODE_RUNNERS)}")

    invoker = tool_invoker or _default_mock_tool_factory(scenario)
    set_tool_invoker(invoker)
    t0 = time.time()
    try:
        # Prefer explicit goal; else parse message
        if not scenario.goal.get("goal_type"):
            scenario.goal = parse_goal(
                scenario.message,
                known_dataset_ids=list(
                    (scenario.world_model.get("datasets") or {}).keys()
                ),
            )

        out = await _MODE_RUNNERS[mode](scenario, user_id=user_id)
        elapsed = time.time() - t0
        cost = out.get("cost_metrics") or {}
        success, reasons = judge_success(
            scenario,
            tools_called=int(out.get("tools_called") or 0),
            has_job=bool(out.get("has_job")),
            goal_type=out.get("goal_type") or scenario.goal.get("goal_type"),
            plan_status=out.get("plan_status"),
            campaign_status=out.get("campaign_status"),
            mode=mode,
        )
        return ScenarioResult(
            scenario_id=scenario.id,
            mode=mode,
            success=success,
            reasons=reasons,
            elapsed_seconds=round(elapsed, 4),
            tools_called=int(out.get("tools_called") or 0),
            steps_executed=int(cost.get("steps_executed") or 0),
            revisions=int(cost.get("revisions") or 0),
            campaign_variants=int(cost.get("campaign_variants") or 0),
            campaign_completed=int(cost.get("campaign_completed") or 0),
            best_score=out.get("best_score"),
            best_job_id=out.get("best_job_id"),
            plan_status=out.get("plan_status"),
            campaign_status=out.get("campaign_status"),
            hierarchy_depth=int(out.get("hierarchy_depth") or 1),
            extra={
                k: out[k]
                for k in ("hierarchy", "evaluation")
                if k in out
            },
        )
    finally:
        set_tool_invoker(None)


async def run_eval_suite(
    *,
    modes: List[str] | None = None,
    tags: List[str] | None = None,
    scenario_ids: List[str] | None = None,
    user_id: str = "eval_user",
) -> Dict[str, Any]:
    """Run all scenarios × modes and return report dict."""
    modes = modes or ["single_shot", "plan_executor", "campaign", "hierarchical"]
    scenarios = scenarios_by_tags(tags=tags, scenario_ids=scenario_ids)
    results: List[ScenarioResult] = []

    for scenario in scenarios:
        for mode in modes:
            # Skip campaign/hierarchical for non-train if no job expected
            if mode in ("campaign",) and scenario.goal.get("goal_type") != "train":
                # still run lightweight — campaign may fail without target; skip
                if not scenario.expect_has_job:
                    continue
            try:
                r = await run_scenario(scenario, mode, user_id=user_id)
                results.append(r)
            except Exception as exc:
                logger.exception("Eval failed %s/%s", scenario.id, mode)
                results.append(
                    ScenarioResult(
                        scenario_id=scenario.id,
                        mode=mode,
                        success=False,
                        reasons=[f"exception: {exc}"],
                    )
                )

    summaries = summarize(results)
    return {
        "results": [r.to_dict() for r in results],
        "summaries": [s.to_dict() for s in summaries],
        "n_scenarios": len(scenarios),
        "modes": modes,
    }


def report_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# HAgent Eval Report (Phase 7)",
        "",
        f"Scenarios: {report.get('n_scenarios')}",
        f"Modes: {', '.join(report.get('modes') or [])}",
        "",
        "## Summary by mode",
        "",
        "| Mode | N | Success rate | Avg latency (s) | Avg tools | Avg revisions | Avg campaign done |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in report.get("summaries") or []:
        lines.append(
            f"| {s['mode']} | {s['n']} | {s['success_rate']:.0%} | "
            f"{s['avg_elapsed']:.3f} | {s['avg_tools']:.1f} | "
            f"{s['avg_revisions']:.1f} | {s['avg_campaign_completed']:.1f} |"
        )
    lines.extend(["", "## Per scenario", ""])
    for r in report.get("results") or []:
        status = "OK" if r.get("success") else "FAIL"
        lines.append(
            f"- **{r['scenario_id']}** / `{r['mode']}`: {status} "
            f"({r.get('elapsed_seconds')}s, tools={r.get('tools_called')}) "
            f"— {', '.join(r.get('reasons') or [])}"
        )
    return "\n".join(lines) + "\n"
