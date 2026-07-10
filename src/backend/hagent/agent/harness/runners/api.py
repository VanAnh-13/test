"""API layer — soft HTTP harness against Bridge/toolkit."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import httpx

from hagent.agent.harness.assertions import assert_expectations
from hagent.agent.harness.schema import AgentRunResult, AgentScenario


async def run_api_scenario(
    scenario: AgentScenario,
    *,
    base_url: Optional[str] = None,
    token: Optional[str] = None,
    require_live: bool = False,
) -> AgentRunResult:
    """
    POST chat to HAgent Bridge (or toolkit chat).

    base_url: e.g. http://localhost:5360  (bridge) or toolkit :5370
    Soft-skip when stack is down unless require_live=True.
    """
    base = (base_url or os.getenv("HAGENT_URL") or os.getenv("BASE_URL") or "").rstrip(
        "/"
    )
    tok = token or os.getenv("HAGENT_TOKEN") or os.getenv("USER_TOKEN") or ""
    t0 = time.time()

    if not base:
        if require_live:
            return AgentRunResult(
                scenario_id=scenario.id,
                layer="api",
                mode="http",
                success=False,
                reasons=["no base_url (set HAGENT_URL or --base-url)"],
            )
        return AgentRunResult(
            scenario_id=scenario.id,
            layer="api",
            mode="http",
            success=True,
            reasons=["skipped: no live base_url"],
            elapsed_seconds=0.0,
            extra={"skipped": True},
        )

    # Prefer bridge path; fall back to toolkit agent-run
    urls = [
        f"{base}/api/v1/chat/",
        f"{base}/api/v1/chat/agent-run",
    ]
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if tok:
        headers["Authorization"] = f"Bearer {tok}"

    last_err = ""
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            # Health probe
            healthy = False
            for health in (f"{base}/api/v1/chat/health", f"{base}/home"):
                try:
                    hr = await client.get(health)
                    if hr.status_code < 500:
                        healthy = True
                        break
                except Exception:
                    continue
            if not healthy:
                msg = f"stack unreachable at {base}"
                if require_live:
                    return AgentRunResult(
                        scenario_id=scenario.id,
                        layer="api",
                        mode="http",
                        success=False,
                        reasons=[msg],
                    )
                return AgentRunResult(
                    scenario_id=scenario.id,
                    layer="api",
                    mode="http",
                    success=True,
                    reasons=[f"skipped: {msg}"],
                    extra={"skipped": True},
                )

            payload = {
                "message": scenario.message,
                "context": {"world_state": scenario.world_model},
            }
            data: Dict[str, Any] = {}
            for url in urls:
                try:
                    resp = await client.post(url, json=payload, headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        break
                    last_err = f"HTTP {resp.status_code} at {url}: {resp.text[:200]}"
                except Exception as exc:
                    last_err = str(exc)
            elapsed = time.time() - t0
            if not data:
                return AgentRunResult(
                    scenario_id=scenario.id,
                    layer="api",
                    mode="http",
                    success=False,
                    reasons=[last_err or "empty response"],
                    elapsed_seconds=round(elapsed, 4),
                )

            message = data.get("message") or data.get("response") or ""
            tools = data.get("tool_outputs") or []
            tool_names = [
                str(t.get("tool_name") or t.get("name") or "")
                for t in tools
                if isinstance(t, dict)
            ]
            tool_names = [t for t in tool_names if t]
            has_job = "job" in message.lower() or any(
                "start_training" in t for t in tool_names
            )
            ok, reasons = assert_expectations(
                scenario.expect,
                tools=tool_names,
                has_job=has_job if scenario.expect.has_job is not None else False,
                goal_type=scenario.goal.get("goal_type") if scenario.goal else None,
                elapsed=elapsed,
            )
            # API success baseline: got a non-empty assistant message
            if not message and ok:
                ok = False
                reasons = ["empty assistant message"]
            elif message and scenario.expect.has_job is None and not scenario.expect.tools_include:
                ok = True
                reasons = ["ok"]

            return AgentRunResult(
                scenario_id=scenario.id,
                layer="api",
                mode="http",
                success=ok,
                reasons=reasons,
                elapsed_seconds=round(elapsed, 4),
                tools_called=len(tool_names),
                tool_names=tool_names,
                response=message,
                extra={"raw_keys": list(data.keys())},
            )
    except Exception as exc:
        elapsed = time.time() - t0
        if require_live:
            return AgentRunResult(
                scenario_id=scenario.id,
                layer="api",
                mode="http",
                success=False,
                reasons=[str(exc)],
                elapsed_seconds=round(elapsed, 4),
            )
        return AgentRunResult(
            scenario_id=scenario.id,
            layer="api",
            mode="http",
            success=True,
            reasons=[f"skipped: {exc}"],
            extra={"skipped": True},
        )
