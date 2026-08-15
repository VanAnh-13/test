"""
Eval harness runner — offline modes with mockable tools.

Modes:
  - single_shot: one start_training (baseline, AutoML-Agent-like single attempt without multi-plan)
  - plan_executor: sequential plan steps (campaign disabled)
  - campaign: multi-candidate jobs (Phase 6)
  - hierarchical: decompose goal then campaign/executor on train leaf
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
from collections.abc import Callable
from typing import Any

import structlog

from hagent.agent.eval.metrics import (
    ScenarioResult,
    ToolCallTrace,
    evaluate_quality,
    judge_success,
    summarize,
)
from hagent.agent.eval.scenarios import (
    BASELINE_VERSION,
    EvalScenario,
    baseline_scenarios,
    scenarios_by_tags,
)
from hagent.agent.execution import tool_runner as tool_runner_module
from hagent.agent.execution.tool_runner import set_tool_invoker
from hagent.agent.planning.goal_parser import parse_goal

logger = structlog.get_logger(__name__)

_MUTATING_ACTIONS = frozenset(
    {
        "start_training",
        "cancel_job",
        "cancel_training",
        "predict_batch",
        "upload_dataset",
        "delete_dataset",
    }
)
_SENSITIVE_TRACE_KEY_ALIASES = frozenset(
    {
        "accesstoken",
        "apikey",
        "authorization",
        "bearer",
        "clientsecret",
        "cookie",
        "credential",
        "jwt",
        "otp",
        "password",
        "privatekey",
        "refreshtoken",
        "secret",
        "token",
    }
)
_EVAL_INVOKER_LOCK = threading.Lock()


class EvalUpstreamFailure(RuntimeError):
    """Deterministic upstream failure used by the offline fake adapter."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _is_sensitive_trace_key(key: Any) -> bool:
    compact = re.sub(r"[^a-z0-9]", "", str(key).casefold())
    if compact == "tokencount":
        return False
    return compact in _SENSITIVE_TRACE_KEY_ALIASES or compact.endswith(
        ("apikey", "password", "privatekey", "secret", "token")
    )


def _redact_trace_value(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if _is_sensitive_trace_key(key):
                redacted[str(key)] = "[REDACTED]"
            else:
                redacted[str(key)] = _redact_trace_value(item)
        return redacted
    if isinstance(value, list):
        return [_redact_trace_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_trace_value(item) for item in value)
    return value


def _payload_error_code(payload: Any) -> str | None:
    if not isinstance(payload, dict) or not payload.get("error"):
        return None
    error = payload["error"]
    if isinstance(error, dict):
        error = error.get("code") or error.get("type") or error.get("message")
    normalized = str(error or "").strip().upper().replace(" ", "_")
    if "TIMEOUT" in normalized or "TIMED_OUT" in normalized:
        return "UPSTREAM_TIMEOUT"
    if any(
        marker in normalized
        for marker in ("UNAVAILABLE", "CONNECTION", "CONNECT", "NETWORK")
    ):
        return "UPSTREAM_UNAVAILABLE"
    return "TOOL_ERROR"


async def _acquire_eval_invoker_lock() -> None:
    while not _EVAL_INVOKER_LOCK.acquire(blocking=False):
        await asyncio.sleep(0.001)


def _default_mock_tool_factory(scenario: EvalScenario) -> Callable:
    """Deterministic HAutoML-like tool responses for offline eval."""
    job_n = {"i": 0}
    scores = [0.71, 0.88, 0.80, 0.76]

    async def invoker(action_type: str, params: dict[str, Any]) -> dict[str, Any]:
        failure_code = scenario.mock_failures.get(action_type)
        if failure_code:
            return {"error": failure_code}

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
) -> dict[str, Any]:
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
) -> dict[str, Any]:
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

    state: dict[str, Any] = {
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
) -> dict[str, Any]:
    from hagent.agent.campaign.nodes import campaign_node, campaign_route

    goal = dict(scenario.goal)
    state: dict[str, Any] = {
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
) -> dict[str, Any]:
    """
    Live hierarchy controller (smart-skip + campaign train leaf).
    """
    from hagent.agent.execution.hierarchy_node import hierarchy_node, hierarchy_route
    from hagent.agent.planning.hierarchy import apply_smart_skips, decompose_goal

    hier = decompose_goal(dict(scenario.goal))
    apply_smart_skips(hier, world_model=scenario.world_model)
    depth = len(hier.subgoals)

    state: dict[str, Any] = {
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


def _token_count(cost: dict[str, Any]) -> int:
    for key in ("total_tokens", "token_count", "tokens"):
        value = cost.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return max(0, int(value))
    usage = cost.get("token_usage")
    if isinstance(usage, dict):
        for key in ("total_tokens", "token_count", "tokens"):
            value = usage.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return max(0, int(value))
        values = []
        for alternatives in (
            ("input_tokens", "prompt_tokens"),
            ("output_tokens", "completion_tokens"),
        ):
            for key in alternatives:
                value = usage.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    values.append(value)
                    break
        return max(0, int(sum(values)))
    return 0


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
    invocations: list[ToolCallTrace] = []

    async def observed_invoker(
        action_type: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        started = time.perf_counter()
        effect = "mutation" if action_type in _MUTATING_ACTIONS else "read"
        try:
            payload = await invoker(action_type, params)
        except Exception as exc:
            error_code = getattr(exc, "code", None)
            if not error_code and isinstance(exc, (TimeoutError, ConnectionError)):
                error_code = "UPSTREAM_UNAVAILABLE"
            invocations.append(
                ToolCallTrace(
                    name=action_type,
                    arguments=_redact_trace_value(dict(params)),
                    effect=effect,
                    outcome="failed",
                    error_code=str(error_code or "INTERNAL_ERROR"),
                    elapsed_seconds=round(time.perf_counter() - started, 6),
                )
            )
            raise

        payload_error_code = _payload_error_code(payload)
        trace_output = None
        if payload_error_code:
            trace_output = {"error": payload_error_code}
        elif isinstance(payload, dict):
            trace_output = _redact_trace_value(dict(payload))
        invocations.append(
            ToolCallTrace(
                name=action_type,
                arguments=_redact_trace_value(dict(params)),
                effect=effect,
                outcome="failed" if payload_error_code else "succeeded",
                output=trace_output,
                error_code=payload_error_code,
                elapsed_seconds=round(time.perf_counter() - started, 6),
            )
        )
        return payload

    await _acquire_eval_invoker_lock()
    previous_invoker = getattr(tool_runner_module, "_tool_invoker", None)
    set_tool_invoker(observed_invoker)
    t0 = time.perf_counter()
    try:
        # Prefer explicit goal; else parse message
        if not scenario.goal.get("goal_type"):
            scenario.goal = parse_goal(
                "\n".join(scenario.messages()),
                known_dataset_ids=list(
                    (scenario.world_model.get("datasets") or {}).keys()
                ),
            )

        outcome = "succeeded"
        try:
            out = await _MODE_RUNNERS[mode](scenario, user_id=user_id)
        except (EvalUpstreamFailure, TimeoutError, ConnectionError):
            out = {}
            outcome = "upstream_failure"

        elapsed = time.perf_counter() - t0
        cost = out.get("cost_metrics") or {}
        goal = dict(scenario.goal)
        has_mutation = any(call.effect == "mutation" for call in invocations)
        upstream_failed = any(
            call.outcome == "failed"
            and call.error_code in {"UPSTREAM_UNAVAILABLE", "UPSTREAM_TIMEOUT"}
            for call in invocations
        )
        invocation_failed = any(call.outcome == "failed" for call in invocations)
        if outcome == "succeeded" and upstream_failed:
            outcome = "upstream_failure"
        elif outcome == "succeeded" and invocation_failed:
            outcome = "failed"
        elif (
            outcome == "succeeded"
            and goal.get("goal_type") == "train"
            and (not goal.get("dataset_id") or not goal.get("target_column"))
            and not has_mutation
        ):
            outcome = "needs_input"
        elif outcome == "succeeded" and (
            out.get("plan_status") == "failed"
            or out.get("campaign_status") == "failed"
            or out.get("error")
        ):
            outcome = "failed"

        token_count = _token_count(cost)
        quality = evaluate_quality(
            scenario,
            actual_goal=goal,
            invocations=invocations,
            outcome=outcome,
            elapsed_seconds=round(elapsed, 4),
            token_count=token_count,
        )
        tools_called = max(int(out.get("tools_called") or 0), len(invocations))
        success, reasons = judge_success(
            scenario,
            tools_called=tools_called,
            has_job=bool(out.get("has_job")),
            goal_type=out.get("goal_type") or goal.get("goal_type"),
            plan_status=out.get("plan_status"),
            campaign_status=out.get("campaign_status"),
            mode=mode,
            quality=quality,
        )
        return ScenarioResult(
            scenario_id=scenario.id,
            mode=mode,
            success=success,
            reasons=reasons,
            elapsed_seconds=round(elapsed, 4),
            outcome=outcome,
            goal_exactness=quality.goal_exactness,
            argument_exactness=quality.argument_exactness,
            evidence_faithfulness=quality.evidence_faithfulness,
            unauthorized_side_effects=quality.unauthorized_side_effects,
            duplicate_mutations=quality.duplicate_mutations,
            token_count=quality.token_count,
            invocations=list(invocations),
            tools_called=tools_called,
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
        set_tool_invoker(previous_invoker)
        _EVAL_INVOKER_LOCK.release()


async def run_eval_suite(
    *,
    modes: list[str] | None = None,
    tags: list[str] | None = None,
    scenario_ids: list[str] | None = None,
    user_id: str = "eval_user",
) -> dict[str, Any]:
    """Run all scenarios × modes and return report dict."""
    modes = modes or ["single_shot", "plan_executor", "campaign", "hierarchical"]
    scenarios = scenarios_by_tags(tags=tags, scenario_ids=scenario_ids)
    results: list[ScenarioResult] = []

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


async def run_baseline_suite(
    *,
    mode: str = "single_shot",
    user_id: str = "eval_user",
) -> dict[str, Any]:
    """Run the frozen v1 matrix and verify the recorded legacy pass/fail profile."""
    scenarios = baseline_scenarios()
    results = [
        await run_scenario(scenario, mode, user_id=user_id) for scenario in scenarios
    ]
    expected = {
        scenario.id: bool(scenario.legacy_expected_success) for scenario in scenarios
    }
    observed = {result.scenario_id: result.success for result in results}
    summaries = summarize(results)
    return {
        "baseline_version": BASELINE_VERSION,
        "legacy_expectations_match": observed == expected,
        "expected_success": expected,
        "observed_success": observed,
        "results": [result.to_dict() for result in results],
        "summaries": [summary.to_dict() for summary in summaries],
        "n_scenarios": len(scenarios),
        "modes": [mode],
    }


def report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# HAgent Eval Report (Phase 7)",
        "",
        f"Scenarios: {report.get('n_scenarios')}",
        f"Modes: {', '.join(report.get('modes') or [])}",
        "",
        "## Summary by mode",
        "",
        "| Mode | N | Success rate | Goal | Args | Evidence | Avg latency (s) | Avg tokens | Unauthorized | Duplicates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in report.get("summaries") or []:
        lines.append(
            f"| {s['mode']} | {s['n']} | {s['success_rate']:.0%} | "
            f"{float(s.get('avg_goal_exactness', 1.0)):.0%} | "
            f"{float(s.get('avg_argument_exactness', 1.0)):.0%} | "
            f"{float(s.get('avg_evidence_faithfulness', 1.0)):.0%} | "
            f"{s['avg_elapsed']:.3f} | {float(s.get('avg_tokens', 0)):.1f} | "
            f"{int(s.get('unauthorized_side_effects', 0))} | "
            f"{int(s.get('duplicate_mutations', 0))} |"
        )
    lines.extend(["", "## Per scenario", ""])
    for r in report.get("results") or []:
        status = "OK" if r.get("success") else "FAIL"
        lines.append(
            f"- **{r['scenario_id']}** / `{r['mode']}`: {status} "
            f"(outcome={r.get('outcome', 'succeeded')}, {r.get('elapsed_seconds')}s, "
            f"tokens={r.get('token_count', 0)}, tools={r.get('tools_called')}, "
            f"goal={float(r.get('goal_exactness', 1.0)):.0%}, "
            f"args={float(r.get('argument_exactness', 1.0)):.0%}, "
            f"evidence={float(r.get('evidence_faithfulness', 1.0)):.0%}, "
            f"unauthorized={r.get('unauthorized_side_effects', 0)}, "
            f"duplicates={r.get('duplicate_mutations', 0)}) "
            f"— {', '.join(r.get('reasons') or [])}"
        )
    return "\n".join(lines) + "\n"
