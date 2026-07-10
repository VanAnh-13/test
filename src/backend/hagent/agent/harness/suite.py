"""Run multi-layer harness suite."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from hagent.agent.harness.loader import load_all_scenarios
from hagent.agent.harness.reporter import build_report
from hagent.agent.harness.runners.api import run_api_scenario
from hagent.agent.harness.runners.graph import run_graph_scenario
from hagent.agent.harness.runners.offline import run_offline_scenario
from hagent.agent.harness.schema import AgentRunResult

logger = logging.getLogger(__name__)

OFFLINE_MODES = ("single_shot", "plan_executor", "campaign", "hierarchical")


async def run_harness_suite(
    *,
    layers: Optional[List[str]] = None,
    offline_modes: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    scenario_ids: Optional[List[str]] = None,
    api_base_url: Optional[str] = None,
    api_token: Optional[str] = None,
    require_live: bool = False,
) -> Dict[str, Any]:
    layers = [x.strip().lower() for x in (layers or ["offline", "graph"])]
    if "all" in layers:
        layers = ["offline", "graph", "api"]
    offline_modes = [
        m.strip().lower()
        for m in (offline_modes or list(OFFLINE_MODES))
    ]
    scenarios = load_all_scenarios(tags=tags, scenario_ids=scenario_ids)
    results: List[AgentRunResult] = []

    for scenario in scenarios:
        if "offline" in layers:
            for mode in offline_modes:
                # skip campaign for non-train
                if mode == "campaign" and scenario.goal.get("goal_type") not in (
                    None,
                    "train",
                ):
                    if scenario.goal.get("goal_type") and scenario.goal.get(
                        "goal_type"
                    ) != "train":
                        continue
                if mode == "campaign" and not (
                    scenario.expect.has_job or scenario.expect_has_job
                    or scenario.goal.get("goal_type") == "train"
                ):
                    continue
                # hierarchical multi-leaf only meaningful for train/evaluate roots
                if mode == "hierarchical" and scenario.goal.get("goal_type") not in (
                    "train",
                    "evaluate",
                    None,
                    "",
                ):
                    if scenario.goal.get("goal_type") in (
                        "analyze",
                        "list",
                        "monitor",
                        "select",
                        "respond",
                    ):
                        continue
                try:
                    results.append(await run_offline_scenario(scenario, mode))
                except Exception as exc:
                    logger.exception("offline %s/%s", scenario.id, mode)
                    results.append(
                        AgentRunResult(
                            scenario_id=scenario.id,
                            layer="offline",
                            mode=mode,
                            success=False,
                            reasons=[f"exception: {exc}"],
                        )
                    )

        if "graph" in layers:
            try:
                results.append(await run_graph_scenario(scenario))
            except Exception as exc:
                logger.exception("graph %s", scenario.id)
                results.append(
                    AgentRunResult(
                        scenario_id=scenario.id,
                        layer="graph",
                        mode="graph",
                        success=False,
                        reasons=[f"exception: {exc}"],
                    )
                )

        if "api" in layers:
            try:
                results.append(
                    await run_api_scenario(
                        scenario,
                        base_url=api_base_url,
                        token=api_token,
                        require_live=require_live,
                    )
                )
            except Exception as exc:
                logger.exception("api %s", scenario.id)
                results.append(
                    AgentRunResult(
                        scenario_id=scenario.id,
                        layer="api",
                        mode="http",
                        success=False,
                        reasons=[f"exception: {exc}"],
                    )
                )

    return build_report(
        results,
        layers=layers,
        offline_modes=offline_modes,
        n_scenarios=len(scenarios),
    )
