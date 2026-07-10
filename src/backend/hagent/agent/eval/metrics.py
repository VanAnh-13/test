"""Aggregate eval metrics across scenarios/modes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ScenarioResult:
    scenario_id: str
    mode: str
    success: bool
    reasons: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    tools_called: int = 0
    steps_executed: int = 0
    revisions: int = 0
    campaign_variants: int = 0
    campaign_completed: int = 0
    best_score: Optional[float] = None
    best_job_id: Optional[str] = None
    plan_status: Optional[str] = None
    campaign_status: Optional[str] = None
    hierarchy_depth: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModeSummary:
    mode: str
    n: int
    success_rate: float
    avg_elapsed: float
    avg_tools: float
    avg_revisions: float
    avg_campaign_completed: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def summarize(results: List[ScenarioResult]) -> List[ModeSummary]:
    by_mode: Dict[str, List[ScenarioResult]] = {}
    for r in results:
        by_mode.setdefault(r.mode, []).append(r)

    summaries: List[ModeSummary] = []
    for mode, rows in sorted(by_mode.items()):
        n = len(rows)
        if n == 0:
            continue
        summaries.append(
            ModeSummary(
                mode=mode,
                n=n,
                success_rate=sum(1 for r in rows if r.success) / n,
                avg_elapsed=sum(r.elapsed_seconds for r in rows) / n,
                avg_tools=sum(r.tools_called for r in rows) / n,
                avg_revisions=sum(r.revisions for r in rows) / n,
                avg_campaign_completed=sum(r.campaign_completed for r in rows) / n,
            )
        )
    return summaries


def judge_success(
    scenario,
    *,
    tools_called: int,
    has_job: bool,
    goal_type: str | None,
    plan_status: str | None,
    campaign_status: str | None,
    mode: str,
) -> tuple[bool, List[str]]:
    """Rule-based success criteria for offline harness."""
    reasons: List[str] = []
    ok = True

    if scenario.expect_goal_type and goal_type:
        if str(goal_type).lower() != str(scenario.expect_goal_type).lower():
            # soft for list/analyze if mode doesn't force goal
            if scenario.expect_goal_type in ("train",):
                ok = False
                reasons.append(
                    f"goal_type={goal_type} expected {scenario.expect_goal_type}"
                )

    if tools_called < int(scenario.expect_min_tools or 0):
        ok = False
        reasons.append(
            f"tools_called={tools_called} < min {scenario.expect_min_tools}"
        )

    if scenario.expect_has_job and mode in ("single_shot", "plan_executor", "campaign", "hierarchical"):
        if not has_job:
            ok = False
            reasons.append("expected training job but none created")

    if mode == "campaign" and scenario.expect_has_job:
        if campaign_status not in ("done", None) and campaign_status == "failed":
            ok = False
            reasons.append(f"campaign_status={campaign_status}")

    if mode == "plan_executor" and scenario.expect_has_job:
        if plan_status == "failed":
            ok = False
            reasons.append("plan_status=failed")

    if ok and not reasons:
        reasons.append("ok")
    return ok, reasons
