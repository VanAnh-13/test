"""Pure assertion helpers for harness expectations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from hagent.agent.harness.schema import AgentRunResult, ExpectSpec


def _tools_from_result(result: dict[str, Any] | AgentRunResult) -> list[str]:
    if isinstance(result, AgentRunResult):
        if result.tool_names:
            return list(result.tool_names)
        return []
    names = result.get("tool_names")
    if names:
        return list(names)
    outs = result.get("tool_outputs") or []
    extracted = []
    for o in outs:
        if isinstance(o, dict):
            n = o.get("tool_name") or o.get("name") or o.get("tool")
            if n:
                extracted.append(str(n))
    return extracted


def _event_types(result: dict[str, Any] | AgentRunResult) -> list[str]:
    if isinstance(result, AgentRunResult):
        return list(result.event_types)
    events = result.get("execution_events") or []
    types = []
    for e in events:
        if isinstance(e, dict) and e.get("type"):
            types.append(str(e["type"]))
    return types


def assert_expectations(
    expect: ExpectSpec,
    *,
    tools: Sequence[str],
    has_job: bool,
    goal_type: str | None = None,
    plan_status: str | None = None,
    campaign_status: str | None = None,
    hierarchy_status: str | None = None,
    route: str | None = None,
    event_types: Sequence[str] | None = None,
    elapsed: float = 0.0,
    wm: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """Return (ok, reasons)."""
    reasons: list[str] = []
    ok = True
    tools = list(tools)
    n_tools = len(tools)
    event_types = list(event_types or [])

    if expect.goal_type and goal_type:
        if str(goal_type).lower() != str(expect.goal_type).lower():
            ok = False
            reasons.append(f"goal_type={goal_type} expected {expect.goal_type}")

    if n_tools < int(expect.tools_called_min or 0):
        ok = False
        reasons.append(f"tools_called={n_tools} < min {expect.tools_called_min}")

    max_t = expect.tools_called_max if expect.tools_called_max is not None else expect.max_tools
    if max_t is not None and n_tools > int(max_t):
        ok = False
        reasons.append(f"tools_called={n_tools} > max {max_t}")

    for t in expect.tools_include or []:
        if t not in tools:
            ok = False
            reasons.append(f"missing tool {t}")

    if expect.tools_order:
        # subsequence check
        it = iter(tools)
        for needed in expect.tools_order:
            for got in it:
                if got == needed:
                    break
            else:
                ok = False
                reasons.append(f"tools_order missing subsequence {expect.tools_order}")
                break

    if expect.has_job is True and not has_job:
        ok = False
        reasons.append("expected job but none found")
    if expect.has_job is False and has_job:
        ok = False
        reasons.append("expected no job but found one")

    if expect.plan_status and str(plan_status or "") != str(expect.plan_status):
        ok = False
        reasons.append(f"plan_status={plan_status} expected {expect.plan_status}")

    if expect.campaign_status and str(campaign_status or "") != str(expect.campaign_status):
        ok = False
        reasons.append(
            f"campaign_status={campaign_status} expected {expect.campaign_status}"
        )

    if expect.hierarchy_status and str(hierarchy_status or "") != str(
        expect.hierarchy_status
    ):
        ok = False
        reasons.append(
            f"hierarchy_status={hierarchy_status} expected {expect.hierarchy_status}"
        )

    if expect.route_in:
        allowed = {str(x).lower() for x in expect.route_in}
        r = str(route or "").lower()
        # also accept if hierarchy/campaign/plan in event types
        route_hits = allowed.intersection(
            {r, *(e.lower() for e in event_types)}
        )
        if not route_hits and r and r not in allowed:
            # soft: if any expected node name appears in events
            if not any(a in " ".join(event_types).lower() for a in allowed):
                ok = False
                reasons.append(f"route={route} not in {expect.route_in}")

    for et in expect.event_types_include or []:
        if et not in event_types:
            ok = False
            reasons.append(f"missing event type {et}")

    if expect.wm_has_job is True:
        jobs = (wm or {}).get("jobs") or {}
        if not jobs:
            ok = False
            reasons.append("world_model has no jobs")

    if expect.max_elapsed_seconds is not None and elapsed > float(
        expect.max_elapsed_seconds
    ):
        ok = False
        reasons.append(
            f"elapsed={elapsed:.3f}s > max {expect.max_elapsed_seconds}s"
        )

    if ok and not reasons:
        reasons.append("ok")
    return ok, reasons


def has_job_signal(result: dict[str, Any]) -> bool:
    if result.get("best_job_id"):
        return True
    if result.get("has_job"):
        return True
    wm = result.get("world_model") or {}
    if wm.get("jobs"):
        return True
    evaluation = result.get("evaluation") or {}
    if evaluation.get("best_job_id") or evaluation.get("job_ids"):
        return True
    for o in result.get("tool_outputs") or []:
        if not isinstance(o, dict):
            continue
        name = o.get("tool_name") or o.get("name")
        payload = o.get("payload") or o.get("content") or {}
        if name == "start_training":
            if isinstance(payload, dict) and (
                payload.get("job_id") or payload.get("id")
            ):
                return True
    log = result.get("execution_log") or []
    for e in log:
        if isinstance(e, dict) and e.get("action") == "start_training" and e.get(
            "status"
        ) == "ok":
            return True
    camp = result.get("campaign") or {}
    for v in camp.get("variants") or []:
        if isinstance(v, dict) and v.get("job_id"):
            return True
    return False
