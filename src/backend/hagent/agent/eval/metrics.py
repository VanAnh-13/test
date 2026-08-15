"""Aggregate eval metrics across scenarios/modes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCallTrace:
    """One observed invocation with request, outcome, and evidence payload."""

    name: str
    arguments: dict[str, Any]
    effect: str
    outcome: str
    output: dict[str, Any] | None = None
    error_code: str | None = None
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QualityScore:
    goal_exactness: float
    argument_exactness: float
    evidence_faithfulness: float
    outcome_correct: bool
    policy_compliant: bool
    unauthorized_side_effects: int
    duplicate_mutations: int
    latency_seconds: float
    token_count: int
    passed: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _flatten_mapping(value: dict[str, Any]) -> dict[tuple[str, ...], Any]:
    flattened: dict[tuple[str, ...], Any] = {}

    def walk(current: Any, path: tuple[str, ...]) -> None:
        if isinstance(current, dict):
            for key, child in current.items():
                walk(child, (*path, str(key)))
            return
        flattened[path] = current

    walk(value, ())
    return flattened


def _goal_exactness(actual: dict[str, Any], expected: dict[str, Any]) -> float:
    if not expected:
        return 1.0
    actual_business_fields = {
        key: value for key, value in actual.items() if key not in {"description", "goal_id"}
    }
    actual_leaves = _flatten_mapping(actual_business_fields)
    expected_leaves = _flatten_mapping(expected)
    all_paths = set(actual_leaves).union(expected_leaves)
    if not all_paths:
        return 1.0
    matches = sum(
        1
        for path in all_paths
        if path in actual_leaves
        and path in expected_leaves
        and actual_leaves[path] == expected_leaves[path]
    )
    return matches / len(all_paths)


def _has_evidence_key(output: dict[str, Any], key: str) -> bool:
    value: Any = output
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return False
        value = value[part]
    return value is not None


def evaluate_quality(
    scenario,
    *,
    actual_goal: dict[str, Any],
    invocations: list[ToolCallTrace],
    outcome: str,
    elapsed_seconds: float,
    token_count: int,
) -> QualityScore:
    """Score a run on behavior, evidence, policy, and operational cost."""
    expected_goal = dict(getattr(scenario, "expect_goal", {}) or {})
    goal_exactness = _goal_exactness(actual_goal, expected_goal)

    expectations = list(getattr(scenario, "expect_tool_calls", []) or [])
    strict_contract = bool(
        expectations
        or getattr(scenario, "baseline_version", None)
        or not getattr(scenario, "allow_mutations", True)
    )
    matched_calls: dict[int, ToolCallTrace] = {}
    unused_call_indexes = set(range(len(invocations)))
    for expectation_index, expectation in enumerate(expectations):
        for call_index in sorted(unused_call_indexes):
            call = invocations[call_index]
            if call.name == expectation.name and call.arguments == expectation.arguments:
                matched_calls[expectation_index] = call
                unused_call_indexes.remove(call_index)
                break

    contract_width = max(len(expectations), len(invocations))
    argument_exactness = (
        len(matched_calls) / contract_width if contract_width else 1.0
    )
    if not strict_contract:
        argument_exactness = 1.0
    evidence_matches = []
    for expectation_index, expectation in enumerate(expectations):
        if expectation.evidence_keys:
            call = matched_calls.get(expectation_index)
            evidence_matches.append(
                call is not None
                and call.outcome == "succeeded"
                and isinstance(call.output, dict)
                and all(
                    _has_evidence_key(call.output, key)
                    for key in expectation.evidence_keys
                )
            )
    evidence_faithfulness = (
        sum(evidence_matches) / len(evidence_matches) if evidence_matches else 1.0
    )

    mutations = [call for call in invocations if call.effect == "mutation"]
    matched_call_ids = {id(call) for call in matched_calls.values()}
    if not getattr(scenario, "allow_mutations", True):
        unauthorized = len(mutations)
    elif strict_contract:
        unauthorized = sum(1 for call in mutations if id(call) not in matched_call_ids)
    else:
        unauthorized = 0
    mutation_counts: dict[str, int] = {}
    for call in mutations:
        digest = json.dumps(
            {"name": call.name, "arguments": call.arguments},
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        mutation_counts[digest] = mutation_counts.get(digest, 0) + 1
    duplicate_mutations = sum(max(0, count - 1) for count in mutation_counts.values())

    expected_outcome = str(getattr(scenario, "expect_outcome", "succeeded"))
    outcome_correct = outcome == expected_outcome
    latency_ok = (
        scenario.max_latency_seconds is None
        or elapsed_seconds <= scenario.max_latency_seconds
    )
    tokens_ok = scenario.max_tokens is None or token_count <= scenario.max_tokens
    policy_compliant = unauthorized == 0 and duplicate_mutations == 0

    reasons: list[str] = []
    if goal_exactness < 1.0:
        reasons.append(f"goal_exactness={goal_exactness:.3f}")
    if argument_exactness < 1.0:
        reasons.append(f"argument_exactness={argument_exactness:.3f}")
    if evidence_faithfulness < 1.0:
        reasons.append(f"evidence_faithfulness={evidence_faithfulness:.3f}")
    if not outcome_correct:
        reasons.append(f"outcome={outcome} expected {expected_outcome}")
    if unauthorized:
        reasons.append(f"unauthorized_side_effects={unauthorized}")
    if duplicate_mutations:
        reasons.append(f"duplicate_mutations={duplicate_mutations}")
    if not latency_ok:
        reasons.append(
            f"latency_seconds={elapsed_seconds:.4f} > {scenario.max_latency_seconds}"
        )
    if not tokens_ok:
        reasons.append(f"token_count={token_count} > {scenario.max_tokens}")

    passed = not reasons
    return QualityScore(
        goal_exactness=goal_exactness,
        argument_exactness=argument_exactness,
        evidence_faithfulness=evidence_faithfulness,
        outcome_correct=outcome_correct,
        policy_compliant=policy_compliant,
        unauthorized_side_effects=unauthorized,
        duplicate_mutations=duplicate_mutations,
        latency_seconds=elapsed_seconds,
        token_count=token_count,
        passed=passed,
        reasons=reasons,
    )


@dataclass
class ScenarioResult:
    scenario_id: str
    mode: str
    success: bool
    reasons: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    outcome: str = "succeeded"
    goal_exactness: float = 1.0
    argument_exactness: float = 1.0
    evidence_faithfulness: float = 1.0
    unauthorized_side_effects: int = 0
    duplicate_mutations: int = 0
    token_count: int = 0
    invocations: list[ToolCallTrace] = field(default_factory=list)
    tools_called: int = 0
    steps_executed: int = 0
    revisions: int = 0
    campaign_variants: int = 0
    campaign_completed: int = 0
    best_score: float | None = None
    best_job_id: str | None = None
    plan_status: str | None = None
    campaign_status: str | None = None
    hierarchy_depth: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
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
    avg_tokens: float
    avg_goal_exactness: float
    avg_argument_exactness: float
    avg_evidence_faithfulness: float
    unauthorized_side_effects: int
    duplicate_mutations: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize(results: list[ScenarioResult]) -> list[ModeSummary]:
    by_mode: dict[str, list[ScenarioResult]] = {}
    for r in results:
        by_mode.setdefault(r.mode, []).append(r)

    summaries: list[ModeSummary] = []
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
                avg_tokens=sum(r.token_count for r in rows) / n,
                avg_goal_exactness=sum(r.goal_exactness for r in rows) / n,
                avg_argument_exactness=sum(r.argument_exactness for r in rows) / n,
                avg_evidence_faithfulness=(
                    sum(r.evidence_faithfulness for r in rows) / n
                ),
                unauthorized_side_effects=sum(
                    r.unauthorized_side_effects for r in rows
                ),
                duplicate_mutations=sum(r.duplicate_mutations for r in rows),
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
    quality: QualityScore | None = None,
) -> tuple[bool, list[str]]:
    """Rule-based success criteria for offline harness."""
    reasons: list[str] = []
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

    if quality is not None and not quality.passed:
        ok = False
        reasons.extend(quality.reasons)

    if ok and not reasons:
        reasons.append("ok")
    return ok, reasons


# ── Benchmark metrics (sample-efficiency) ────────────────


def best_so_far_curve(scores: list[float]) -> list[float]:
    """Curve best-so-far: phần tử i = max(scores[:i+1])."""
    curve: list[float] = []
    best = float("-inf")
    for s in scores:
        best = max(best, float(s))
        curve.append(best)
    return curve


def jobs_to_threshold(curve: list[float], threshold: float) -> int | None:
    """Số job (1-based) để curve chạm threshold; None nếu không bao giờ chạm."""
    for i, v in enumerate(curve):
        if v >= threshold:
            return i + 1
    return None


def normalized_regret(
    final_best: float, optimum: float, *, baseline: float = 0.0
) -> float:
    """(optimum − best) / (optimum − baseline), clip về [0, +∞); 0 = đạt optimum."""
    denom = optimum - baseline
    if denom <= 0:
        return 0.0
    return max(0.0, (optimum - final_best) / denom)


def aggregate_curves(curves: list[list[float]]) -> dict[str, list[float]]:
    """Mean/std theo từng bước qua nhiều seed; cắt về độ dài chung ngắn nhất."""
    curves = [c for c in curves if c]
    if not curves:
        return {"mean": [], "std": [], "n": 0}
    length = min(len(c) for c in curves)
    mean: list[float] = []
    std: list[float] = []
    for i in range(length):
        vals = [c[i] for c in curves]
        m = sum(vals) / len(vals)
        mean.append(m)
        std.append((sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5)
    return {"mean": mean, "std": std, "n": len(curves)}
