"""Regression cho quality gate trước khi bật Journey runtime."""

from __future__ import annotations

import math

import pytest

from hagent.agent.eval.cutover import (
    BudgetException,
    CutoverEvidenceError,
    CutoverFixtureEvidence,
    CutoverGateDecision,
    CutoverGatePolicy,
    evaluate_cutover,
    evidence_from_shadow_report,
)
from hagent.agent.runtime.shadow import (
    RuntimeObservation,
    ShadowComparisonReport,
)


def _fixture(
    fixture_id: str = "vi.audit.required",
    **changes: object,
) -> CutoverFixtureEvidence:
    values: dict[str, object] = {
        "fixture_id": fixture_id,
        "safety_passed": True,
        "contract_passed": True,
        "outcome_not_worse": True,
        "unauthorized_side_effects": 0,
        "duplicate_mutations": 0,
        "latency_ratio": 1.1,
        "token_ratio": 1.0,
        "cost_ratio": 0.9,
    }
    values.update(changes)
    return CutoverFixtureEvidence(**values)


def _policy(
    *fixture_ids: str,
    exceptions: tuple[BudgetException, ...] = (),
) -> CutoverGatePolicy:
    return CutoverGatePolicy(
        required_fixture_ids=fixture_ids or ("vi.audit.required",),
        budget_exceptions=exceptions,
    )


def test_required_fixtures_pass_exact_safety_and_budget_gate() -> None:
    evidence = (
        _fixture(),
        _fixture("en.training.required", latency_ratio=1.25),
    )
    decision = evaluate_cutover(
        evidence,
        policy=_policy(*(item.fixture_id for item in evidence)),
    )

    assert decision.approved is True
    assert decision.blocker_codes == ()
    assert decision.required_fixture_count == 2
    assert decision.passed_fixture_count == 2
    assert decision.safety_pass_rate == 1.0
    assert decision.contract_pass_rate == 1.0
    assert decision.max_latency_ratio == 1.25
    assert decision.max_token_ratio == 1.0
    assert decision.max_cost_ratio == 0.9
    assert decision.approved_exception_count == 0


@pytest.mark.parametrize(
    ("changes", "blocker"),
    [
        ({"safety_passed": False}, "safety_gate_failed"),
        ({"contract_passed": False}, "contract_gate_failed"),
        ({"outcome_not_worse": False}, "outcome_regressed"),
        ({"unauthorized_side_effects": 1}, "unauthorized_side_effect"),
        ({"duplicate_mutations": 1}, "duplicate_mutation"),
    ],
)
def test_safety_contract_outcome_and_policy_blockers_are_independent(
    changes: dict[str, object],
    blocker: str,
) -> None:
    decision = evaluate_cutover((_fixture(**changes),), policy=_policy())

    assert decision.approved is False
    assert blocker in decision.blocker_codes


@pytest.mark.parametrize("metric", ["latency", "token", "cost"])
def test_missing_or_exceeded_budget_evidence_fails_closed(metric: str) -> None:
    field = f"{metric}_ratio"
    missing = evaluate_cutover((_fixture(**{field: None}),), policy=_policy())
    exceeded = evaluate_cutover(
        (_fixture(**{field: 1.250001}),),
        policy=_policy(),
    )

    assert f"{metric}_evidence_missing" in missing.blocker_codes
    assert f"{metric}_budget_exceeded" in exceeded.blocker_codes


def test_budget_exception_requires_exact_fixture_metric_approver_and_reason() -> None:
    fixture = _fixture(latency_ratio=1.4)
    policy = _policy(
        fixture.fixture_id,
        exceptions=(
            BudgetException(
                fixture_id=fixture.fixture_id,
                metric="latency",
                approved_by="owner.release.board",
                reason="cold_start.accepted",
            ),
        ),
    )

    decision = evaluate_cutover((fixture,), policy=policy)

    assert decision.approved is True
    assert decision.blocker_codes == ()
    assert decision.approved_exception_count == 1


def test_unknown_or_unused_budget_exception_is_rejected() -> None:
    unknown = BudgetException(
        fixture_id="unknown.fixture",
        metric="latency",
        approved_by="owner.release.board",
        reason="cold_start.accepted",
    )
    unused = BudgetException(
        fixture_id="vi.audit.required",
        metric="latency",
        approved_by="owner.release.board",
        reason="cold_start.accepted",
    )
    with pytest.raises(CutoverEvidenceError, match="Budget exception"):
        _policy("vi.audit.required", exceptions=(unknown,))
    with pytest.raises(CutoverEvidenceError, match="Budget exception"):
        evaluate_cutover(
            (_fixture(latency_ratio=1.0),),
            policy=_policy(exceptions=(unused,)),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fixture_id": "prompt has spaces and secret"},
        {"fixture_id": "x" * 129},
        {"unauthorized_side_effects": -1},
        {"duplicate_mutations": True},
        {"latency_ratio": math.nan},
        {"token_ratio": math.inf},
        {"cost_ratio": -0.1},
    ],
)
def test_invalid_fixture_evidence_is_rejected_without_echo(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(CutoverEvidenceError) as exc_info:
        _fixture(**kwargs)

    assert "secret" not in str(exc_info.value)


def test_empty_and_duplicate_required_evidence_fail_closed() -> None:
    empty = evaluate_cutover((), policy=_policy())

    assert empty.blocker_codes == ("required_fixture_missing",)
    with pytest.raises(CutoverEvidenceError, match="Duplicate fixture"):
        evaluate_cutover((_fixture(), _fixture()), policy=_policy())
    with pytest.raises(CutoverEvidenceError, match="Unexpected fixture"):
        evaluate_cutover(
            (_fixture(), _fixture("untrusted.fixture")),
            policy=_policy(),
        )


def test_exception_contract_rejects_unknown_metric_and_untrusted_labels() -> None:
    with pytest.raises(CutoverEvidenceError):
        BudgetException(
            fixture_id="vi.audit.required",
            metric="memory",
            approved_by="owner.release.board",
            reason="approved",
        )
    with pytest.raises(CutoverEvidenceError):
        BudgetException(
            fixture_id="vi.audit.required",
            metric="latency",
            approved_by="release board with spaces",
            reason="approved",
        )
    with pytest.raises(CutoverEvidenceError):
        BudgetException(
            fixture_id="vi.audit.required",
            metric="latency",
            approved_by="owner.release.board",
            reason="",
        )


def test_evidence_and_exception_repr_never_emit_safe_looking_labels() -> None:
    secret_like = "sk-live-lowercasecredential123456789"
    evidence = _fixture(fixture_id=secret_like)
    exception = BudgetException(
        fixture_id=secret_like,
        metric="latency",
        approved_by=secret_like,
        reason=secret_like,
    )
    policy = _policy(secret_like, exceptions=(exception,))

    assert secret_like not in repr(evidence)
    assert secret_like not in repr(exception)
    assert secret_like not in repr(policy)


@pytest.mark.parametrize("invalid_policy", [[], 0, False])
def test_falsey_invalid_policy_fails_closed(invalid_policy: object) -> None:
    with pytest.raises(CutoverEvidenceError, match="Invalid cutover policy"):
        evaluate_cutover((_fixture(),), policy=invalid_policy)


@pytest.mark.parametrize(
    "changes",
    [
        {"approved": True, "blocker_codes": ("contract_gate_failed",)},
        {"approved": False, "blocker_codes": ("secret-token",)},
        {"required_fixture_count": -1},
        {"passed_fixture_count": 2},
        {"safety_pass_rate": math.nan},
        {"safety_pass_rate": 0.0},
        {"contract_pass_rate": 1.1},
        {"contract_pass_rate": 0.0},
        {"unauthorized_side_effects": 1},
        {"duplicate_mutations": 1},
        {"max_latency_ratio": None},
        {"max_latency_ratio": math.inf},
        {"max_latency_ratio": 1.3},
        {"approved_exception_count": 1},
        {"approved_exception_count": 4},
    ],
)
def test_public_decision_contract_rejects_invalid_aggregate(
    changes: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "approved": True,
        "blocker_codes": (),
        "required_fixture_count": 1,
        "passed_fixture_count": 1,
        "safety_pass_rate": 1.0,
        "contract_pass_rate": 1.0,
        "unauthorized_side_effects": 0,
        "duplicate_mutations": 0,
        "max_latency_ratio": 1.0,
        "max_token_ratio": 1.0,
        "max_cost_ratio": 1.0,
        "approved_exception_count": 0,
    }
    values.update(changes)

    with pytest.raises(CutoverEvidenceError):
        CutoverGateDecision(**values)


def test_required_fixture_manifest_is_nonempty_unique_and_repr_safe() -> None:
    secret_like = "sk-live-lowercasecredential123456789"
    policy = _policy(secret_like)

    assert secret_like not in repr(policy)
    with pytest.raises(CutoverEvidenceError, match="manifest is empty"):
        CutoverGatePolicy(required_fixture_ids=())
    with pytest.raises(CutoverEvidenceError, match="Duplicate required fixture"):
        CutoverGatePolicy(required_fixture_ids=("same.fixture", "same.fixture"))


def test_zero_required_decision_must_report_missing_fixture() -> None:
    with pytest.raises(CutoverEvidenceError, match="required fixture aggregate"):
        CutoverGateDecision(
            approved=False,
            blocker_codes=("safety_gate_failed",),
            required_fixture_count=0,
            passed_fixture_count=0,
            safety_pass_rate=0.0,
            contract_pass_rate=0.0,
            unauthorized_side_effects=0,
            duplicate_mutations=0,
            max_latency_ratio=None,
            max_token_ratio=None,
            max_cost_ratio=None,
            approved_exception_count=0,
        )


def test_shadow_report_conversion_is_conservative_and_sanitized() -> None:
    primary = RuntimeObservation(
        outcome="completed:success",
        artifact_types=("DatasetAudit",),
        evidence_types=("dataset_profile",),
        checker_verdicts=("policy:passed",),
        latency_ms=100.0,
        total_tokens=100,
        total_cost=1.0,
        event_count=5,
    )
    observer = RuntimeObservation(
        outcome="completed:success",
        artifact_types=("DatasetAudit",),
        evidence_types=("dataset_profile",),
        checker_verdicts=("policy:passed",),
        latency_ms=120.0,
        total_tokens=110,
        total_cost=0.9,
        event_count=6,
    )
    report = ShadowComparisonReport(
        run_id="report-run",
        primary=primary,
        observer=observer,
        outcome_match=True,
        artifact_match=True,
        evidence_match=True,
        checker_match=True,
        latency_ratio=1.2,
        token_ratio=1.1,
        cost_ratio=0.9,
    )

    evidence = evidence_from_shadow_report(
        report,
        fixture_id="vi.audit.required",
        safety_passed=True,
        unauthorized_side_effects=0,
        duplicate_mutations=0,
    )

    assert evidence.contract_passed is True
    assert evidence.outcome_not_worse is True
    assert evaluate_cutover((evidence,), policy=_policy()).approved is True
    serialized = repr(evaluate_cutover((evidence,), policy=_policy()))
    assert "completed:success" not in serialized
    assert "report-run" not in serialized


def test_decision_blocker_order_is_stable_and_contains_only_aggregates() -> None:
    decision = evaluate_cutover(
        (
            _fixture(
                safety_passed=False,
                contract_passed=False,
                outcome_not_worse=False,
                unauthorized_side_effects=2,
                duplicate_mutations=3,
                latency_ratio=None,
                token_ratio=2.0,
                cost_ratio=3.0,
            ),
        ),
        policy=_policy(),
    )

    assert decision.blocker_codes == (
        "safety_gate_failed",
        "contract_gate_failed",
        "outcome_regressed",
        "unauthorized_side_effect",
        "duplicate_mutation",
        "latency_evidence_missing",
        "token_budget_exceeded",
        "cost_budget_exceeded",
    )
    assert decision.unauthorized_side_effects == 2
    assert decision.duplicate_mutations == 3
    assert "vi.audit.required" not in repr(decision)
