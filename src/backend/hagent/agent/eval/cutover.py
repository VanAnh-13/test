"""Quality gate bất biến và fail-closed cho Journey runtime cutover."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Literal

from hagent.agent.runtime.shadow import ShadowComparisonReport

BudgetMetric = Literal["latency", "token", "cost"]
MAX_CUTOVER_RATIO = 1.25

_SAFE_LABEL = re.compile(r"[a-z0-9][a-z0-9._:-]{0,127}")
_BUDGET_METRICS: tuple[BudgetMetric, ...] = ("latency", "token", "cost")
_BLOCKER_ORDER = (
    "required_fixture_missing",
    "safety_gate_failed",
    "contract_gate_failed",
    "outcome_regressed",
    "unauthorized_side_effect",
    "duplicate_mutation",
    "latency_evidence_missing",
    "token_evidence_missing",
    "cost_evidence_missing",
    "latency_budget_exceeded",
    "token_budget_exceeded",
    "cost_budget_exceeded",
)


class CutoverEvidenceError(ValueError):
    """Lỗi contract đã khử dữ liệu, không phản chiếu input không tin cậy."""


def _validate_label(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _SAFE_LABEL.fullmatch(value) is None:
        raise CutoverEvidenceError(f"Invalid {field_name}")
    return value


def _validate_count(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CutoverEvidenceError(f"Invalid {field_name}")
    return value


def _validate_ratio(value: object, *, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CutoverEvidenceError(f"Invalid {field_name}")
    ratio = float(value)
    if not math.isfinite(ratio) or ratio < 0:
        raise CutoverEvidenceError(f"Invalid {field_name}")
    return ratio


@dataclass(frozen=True, slots=True)
class CutoverFixtureEvidence:
    """Evidence tối thiểu của một fixture, không giữ raw prompt hoặc event."""

    fixture_id: str = field(repr=False)
    safety_passed: bool
    contract_passed: bool
    outcome_not_worse: bool
    unauthorized_side_effects: int
    duplicate_mutations: int
    latency_ratio: float | None
    token_ratio: float | None
    cost_ratio: float | None

    def __post_init__(self) -> None:
        _validate_label(self.fixture_id, field_name="fixture id")
        for field_name in (
            "safety_passed",
            "contract_passed",
            "outcome_not_worse",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise CutoverEvidenceError(f"Invalid {field_name}")
        _validate_count(
            self.unauthorized_side_effects,
            field_name="unauthorized side effect count",
        )
        _validate_count(
            self.duplicate_mutations,
            field_name="duplicate mutation count",
        )
        for metric in _BUDGET_METRICS:
            value = _validate_ratio(
                getattr(self, f"{metric}_ratio"),
                field_name=f"{metric} ratio",
            )
            object.__setattr__(self, f"{metric}_ratio", value)


@dataclass(frozen=True, slots=True)
class BudgetException:
    """Ngoại lệ budget được định danh, không nhận lý do dạng free text."""

    fixture_id: str = field(repr=False)
    metric: BudgetMetric = field(repr=False)
    approved_by: str = field(repr=False)
    reason: str = field(repr=False)

    def __post_init__(self) -> None:
        _validate_label(self.fixture_id, field_name="budget fixture id")
        if self.metric not in _BUDGET_METRICS:
            raise CutoverEvidenceError("Invalid budget metric")
        _validate_label(self.approved_by, field_name="budget approver")
        _validate_label(self.reason, field_name="budget reason")


@dataclass(frozen=True, slots=True)
class CutoverGatePolicy:
    """Policy cố định ngưỡng 125%, chỉ cho phép ngoại lệ có audit label."""

    required_fixture_ids: tuple[str, ...] = field(repr=False)
    budget_exceptions: tuple[BudgetException, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.required_fixture_ids, tuple) or not all(
            isinstance(item, str) for item in self.required_fixture_ids
        ):
            raise CutoverEvidenceError("Invalid required fixture manifest")
        if not self.required_fixture_ids:
            raise CutoverEvidenceError("Required fixture manifest is empty")
        for fixture_id in self.required_fixture_ids:
            _validate_label(fixture_id, field_name="required fixture id")
        if len(self.required_fixture_ids) != len(set(self.required_fixture_ids)):
            raise CutoverEvidenceError("Duplicate required fixture id")
        if not isinstance(self.budget_exceptions, tuple) or not all(
            isinstance(item, BudgetException) for item in self.budget_exceptions
        ):
            raise CutoverEvidenceError("Invalid budget exception collection")
        keys = [(item.fixture_id, item.metric) for item in self.budget_exceptions]
        if len(keys) != len(set(keys)):
            raise CutoverEvidenceError("Duplicate Budget exception")
        required_ids = set(self.required_fixture_ids)
        if any(item.fixture_id not in required_ids for item in self.budget_exceptions):
            raise CutoverEvidenceError("Budget exception references unknown fixture")


@dataclass(frozen=True, slots=True)
class CutoverGateDecision:
    """Quyết định aggregate an toàn, không chứa fixture ID hoặc raw evidence."""

    approved: bool
    blocker_codes: tuple[str, ...]
    required_fixture_count: int
    passed_fixture_count: int
    safety_pass_rate: float
    contract_pass_rate: float
    unauthorized_side_effects: int
    duplicate_mutations: int
    max_latency_ratio: float | None
    max_token_ratio: float | None
    max_cost_ratio: float | None
    approved_exception_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.approved, bool):
            raise CutoverEvidenceError("Invalid cutover decision")
        if not isinstance(self.blocker_codes, tuple):
            raise CutoverEvidenceError("Invalid cutover blocker collection")
        expected_order = tuple(
            code for code in _BLOCKER_ORDER if code in set(self.blocker_codes)
        )
        if self.blocker_codes != expected_order:
            raise CutoverEvidenceError("Invalid cutover blocker collection")
        for field_name in (
            "required_fixture_count",
            "passed_fixture_count",
            "unauthorized_side_effects",
            "duplicate_mutations",
            "approved_exception_count",
        ):
            _validate_count(getattr(self, field_name), field_name=field_name)
        if self.passed_fixture_count > self.required_fixture_count:
            raise CutoverEvidenceError("Invalid cutover fixture aggregate")
        if self.approved_exception_count > self.required_fixture_count * len(
            _BUDGET_METRICS
        ):
            raise CutoverEvidenceError("Invalid cutover exception aggregate")
        for field_name in ("safety_pass_rate", "contract_pass_rate"):
            value = _validate_ratio(getattr(self, field_name), field_name=field_name)
            if value is None or value > 1:
                raise CutoverEvidenceError("Invalid cutover pass rate")
        for metric in _BUDGET_METRICS:
            _validate_ratio(
                getattr(self, f"max_{metric}_ratio"),
                field_name=f"maximum {metric} ratio",
            )
        if self.approved != (not self.blocker_codes):
            raise CutoverEvidenceError("Invalid cutover approval state")
        if (
            self.required_fixture_count == 0
            and "required_fixture_missing" not in self.blocker_codes
        ):
            raise CutoverEvidenceError("Invalid required fixture aggregate")
        if self.approved and (
            self.required_fixture_count == 0
            or self.passed_fixture_count != self.required_fixture_count
        ):
            raise CutoverEvidenceError("Invalid approved fixture aggregate")
        if self.approved and (
            self.safety_pass_rate != 1.0
            or self.contract_pass_rate != 1.0
            or self.unauthorized_side_effects != 0
            or self.duplicate_mutations != 0
        ):
            raise CutoverEvidenceError("Invalid approved safety aggregate")
        maximum_ratios = (
            self.max_latency_ratio,
            self.max_token_ratio,
            self.max_cost_ratio,
        )
        if self.approved and any(value is None for value in maximum_ratios):
            raise CutoverEvidenceError("Invalid approved budget aggregate")
        exceeded_metrics = sum(
            value is not None and value > MAX_CUTOVER_RATIO for value in maximum_ratios
        )
        if self.approved and (
            self.approved_exception_count < exceeded_metrics
            or (exceeded_metrics == 0 and self.approved_exception_count != 0)
        ):
            raise CutoverEvidenceError("Invalid approved exception aggregate")


def _max_ratio(
    fixtures: tuple[CutoverFixtureEvidence, ...],
    metric: BudgetMetric,
) -> float | None:
    values = [
        value
        for fixture in fixtures
        if (value := getattr(fixture, f"{metric}_ratio")) is not None
    ]
    return max(values) if values else None


def _validated_exceptions(
    fixtures: tuple[CutoverFixtureEvidence, ...],
    policy: CutoverGatePolicy,
) -> frozenset[tuple[str, BudgetMetric]]:
    required_by_id = {item.fixture_id: item for item in fixtures}
    approved: set[tuple[str, BudgetMetric]] = set()
    for exception in policy.budget_exceptions:
        fixture = required_by_id.get(exception.fixture_id)
        if fixture is None:
            raise CutoverEvidenceError("Budget exception references unknown fixture")
        ratio = getattr(fixture, f"{exception.metric}_ratio")
        if ratio is None or ratio <= MAX_CUTOVER_RATIO:
            raise CutoverEvidenceError("Budget exception is unused")
        approved.add((exception.fixture_id, exception.metric))
    return frozenset(approved)


def _fixture_passes(
    fixture: CutoverFixtureEvidence,
    approved_exceptions: frozenset[tuple[str, BudgetMetric]],
) -> bool:
    if not (
        fixture.safety_passed
        and fixture.contract_passed
        and fixture.outcome_not_worse
        and fixture.unauthorized_side_effects == 0
        and fixture.duplicate_mutations == 0
    ):
        return False
    for metric in _BUDGET_METRICS:
        ratio = getattr(fixture, f"{metric}_ratio")
        if ratio is None:
            return False
        if (
            ratio > MAX_CUTOVER_RATIO
            and (fixture.fixture_id, metric) not in approved_exceptions
        ):
            return False
    return True


def evaluate_cutover(
    evidence: tuple[CutoverFixtureEvidence, ...],
    *,
    policy: CutoverGatePolicy,
) -> CutoverGateDecision:
    """Đánh giá toàn bộ fixture bắt buộc bằng thứ tự blocker ổn định."""
    if not isinstance(evidence, tuple) or not all(
        isinstance(item, CutoverFixtureEvidence) for item in evidence
    ):
        raise CutoverEvidenceError("Invalid fixture evidence collection")
    fixture_ids = [item.fixture_id for item in evidence]
    if len(fixture_ids) != len(set(fixture_ids)):
        raise CutoverEvidenceError("Duplicate fixture evidence")

    gate_policy = policy
    if not isinstance(gate_policy, CutoverGatePolicy):
        raise CutoverEvidenceError("Invalid cutover policy")
    required_ids = set(gate_policy.required_fixture_ids)
    unexpected_ids = set(fixture_ids) - required_ids
    if unexpected_ids:
        raise CutoverEvidenceError("Unexpected fixture evidence")
    approved_exceptions = _validated_exceptions(evidence, gate_policy)
    required = evidence

    flags = {code: False for code in _BLOCKER_ORDER}
    if set(fixture_ids) != required_ids:
        flags["required_fixture_missing"] = True
    for fixture in required:
        if not fixture.safety_passed:
            flags["safety_gate_failed"] = True
        if not fixture.contract_passed:
            flags["contract_gate_failed"] = True
        if not fixture.outcome_not_worse:
            flags["outcome_regressed"] = True
        if fixture.unauthorized_side_effects:
            flags["unauthorized_side_effect"] = True
        if fixture.duplicate_mutations:
            flags["duplicate_mutation"] = True
        for metric in _BUDGET_METRICS:
            ratio = getattr(fixture, f"{metric}_ratio")
            if ratio is None:
                flags[f"{metric}_evidence_missing"] = True
            elif (
                ratio > MAX_CUTOVER_RATIO
                and (fixture.fixture_id, metric) not in approved_exceptions
            ):
                flags[f"{metric}_budget_exceeded"] = True

    blocker_codes = tuple(code for code in _BLOCKER_ORDER if flags[code])
    required_count = len(gate_policy.required_fixture_ids)
    passed_count = sum(_fixture_passes(item, approved_exceptions) for item in required)
    return CutoverGateDecision(
        approved=not blocker_codes,
        blocker_codes=blocker_codes,
        required_fixture_count=required_count,
        passed_fixture_count=passed_count,
        safety_pass_rate=(
            sum(item.safety_passed for item in required) / required_count
            if required_count
            else 0.0
        ),
        contract_pass_rate=(
            sum(item.contract_passed for item in required) / required_count
            if required_count
            else 0.0
        ),
        unauthorized_side_effects=sum(
            item.unauthorized_side_effects for item in required
        ),
        duplicate_mutations=sum(item.duplicate_mutations for item in required),
        max_latency_ratio=_max_ratio(required, "latency"),
        max_token_ratio=_max_ratio(required, "token"),
        max_cost_ratio=_max_ratio(required, "cost"),
        approved_exception_count=len(approved_exceptions),
    )


def evidence_from_shadow_report(
    report: ShadowComparisonReport,
    *,
    fixture_id: str,
    safety_passed: bool,
    unauthorized_side_effects: int,
    duplicate_mutations: int,
) -> CutoverFixtureEvidence:
    """Chuyển báo cáo shadow thành evidence bảo thủ mà không giữ raw report."""
    if not isinstance(report, ShadowComparisonReport):
        raise CutoverEvidenceError("Invalid shadow report")
    return CutoverFixtureEvidence(
        fixture_id=fixture_id,
        safety_passed=safety_passed,
        contract_passed=(
            report.artifact_match and report.evidence_match and report.checker_match
        ),
        outcome_not_worse=report.outcome_match,
        unauthorized_side_effects=unauthorized_side_effects,
        duplicate_mutations=duplicate_mutations,
        latency_ratio=report.latency_ratio,
        token_ratio=report.token_ratio,
        cost_ratio=report.cost_ratio,
    )
