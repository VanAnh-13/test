from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest


def _evidence(evidence_id: str = "evidence-1"):
    from hagent.agent.journey import EvidenceRef

    return EvidenceRef(
        evidence_id=evidence_id,
        source="capability:automl.dataset.inspect@1",
        content_hash="a" * 64,
        summary="Schema và thống kê được trả từ native capability.",
    )


def _audit(**overrides):
    from hagent.agent.journey import DatasetAudit

    values = {
        "artifact_id": "audit-1",
        "owner_id": "owner-1",
        "run_id": "run-1",
        "status": "accepted",
        "evidence": (_evidence(),),
        "dataset_id": "dataset-1",
        "dataset_fingerprint": "b" * 64,
        "target_hypothesis": "target",
        "columns": ("feature", "target"),
        "missingness": {"feature": 0.0, "target": 0.0},
        "class_balance": {"yes": 0.5, "no": 0.5},
        "leakage_risks": (),
        "quality_blockers": (),
    }
    values.update(overrides)
    return DatasetAudit(**values)


def _spec(**overrides):
    from hagent.agent.journey import ExperimentSpec

    values = {
        "artifact_id": "spec-1",
        "owner_id": "owner-1",
        "run_id": "run-1",
        "status": "accepted",
        "lineage": ("audit-1",),
        "evidence": (_evidence("evidence-spec"),),
        "dataset_audit_id": "audit-1",
        "problem_type": "classification",
        "target_column": "target",
        "metric": "accuracy",
        "metric_direction": "maximize",
        "split_strategy": "stratified_holdout",
        "model_families": ("linear", "tree"),
        "max_training_jobs": 2,
        "baseline_value": 0.6,
        "acceptance_threshold": 0.7,
        "default_reasons": {},
    }
    values.update(overrides)
    return ExperimentSpec(**values)


def _training(**overrides):
    from hagent.agent.journey import TrainingRunSet

    values = {
        "artifact_id": "training-1",
        "owner_id": "owner-1",
        "run_id": "run-1",
        "status": "accepted",
        "lineage": ("spec-1",),
        "evidence": (_evidence("evidence-training"),),
        "experiment_spec_id": "spec-1",
        "config_hash": "c" * 64,
        "idempotency_key": "train-key-1",
        "job_ids": ("job-1",),
        "job_statuses": {"job-1": "completed"},
        "cost": 1.5,
        "reconciliation_status": "confirmed",
    }
    values.update(overrides)
    return TrainingRunSet(**values)


def _evaluation(**overrides):
    from hagent.agent.journey import EvaluationReport

    values = {
        "artifact_id": "evaluation-1",
        "owner_id": "owner-1",
        "run_id": "run-1",
        "status": "accepted",
        "lineage": ("training-1",),
        "evidence": (_evidence("evidence-evaluation"),),
        "training_run_set_id": "training-1",
        "metric": "accuracy",
        "metric_direction": "maximize",
        "metric_value": 0.8,
        "baseline_value": 0.6,
        "baseline_delta": 0.2,
        "cv_mean": 0.79,
        "variance": 0.01,
        "overfit_gap": 0.03,
        "calibration_error": 0.04,
        "rejection_reasons": (),
    }
    values.update(overrides)
    return EvaluationReport(**values)


def test_artifact_is_deeply_immutable_and_validates_core_contract():
    source_columns = ["feature", "target"]
    audit = _audit(
        columns=source_columns,
        leakage_risks=[],
        quality_blockers=[],
    )
    source_columns.append("late-column")

    assert audit.version == 1
    assert audit.lineage == ()
    assert audit.columns == ("feature", "target")
    assert audit.leakage_risks == ()
    with pytest.raises(FrozenInstanceError):
        audit.dataset_id = "changed"
    with pytest.raises(TypeError):
        audit.missingness["feature"] = 1.0
    with pytest.raises(ValueError, match="version"):
        _audit(version=0)
    with pytest.raises(ValueError, match="content_hash"):
        replace(_evidence(), content_hash="not-a-hash")


def test_ledger_is_append_only_and_revision_requires_new_id_and_next_version():
    from hagent.agent.journey import ArtifactLedger

    ledger = ArtifactLedger()
    original = _audit()
    ledger.append(original)
    revision = _audit(
        artifact_id="audit-2",
        version=2,
        status="draft",
        supersedes="audit-1",
        leakage_risks=("proxy_target",),
    )
    ledger.append(revision)

    assert ledger.get("audit-1") is original
    assert ledger.get("audit-2") is revision
    assert ledger.latest_revision("audit-1") is revision
    with pytest.raises(ValueError, match="already exists"):
        ledger.append(original)
    with pytest.raises(ValueError, match="version"):
        ledger.append(
            _audit(
                artifact_id="audit-3",
                version=4,
                supersedes="audit-2",
            )
        )
    with pytest.raises(ValueError, match="already has a revision"):
        ledger.append(
            _audit(
                artifact_id="audit-branch",
                version=2,
                supersedes="audit-1",
            )
        )


def test_ledger_rejects_missing_lineage_and_cross_type_supersedes():
    from hagent.agent.journey import ArtifactLedger

    ledger = ArtifactLedger()
    ledger.append(_audit())

    with pytest.raises(ValueError, match="lineage"):
        ledger.append(_spec(lineage=("missing-audit",)))
    with pytest.raises(ValueError, match="same artifact type"):
        ledger.append(
            _spec(
                artifact_id="spec-cross-type",
                version=2,
                supersedes="audit-1",
            )
        )


def test_ledger_accepts_full_six_artifact_lineage():
    from hagent.agent.journey import (
        ArtifactLedger,
        PredictionArtifact,
        ReleaseCandidate,
    )

    ledger = ArtifactLedger()
    audit = _audit()
    spec = _spec()
    training = _training()
    evaluation = _evaluation()
    release = ReleaseCandidate(
        artifact_id="release-1",
        owner_id="owner-1",
        run_id="run-1",
        status="accepted",
        lineage=("evaluation-1",),
        evidence=(_evidence("evidence-release"),),
        evaluation_report_id="evaluation-1",
        model_version="model-1",
        input_schema={"feature": "number"},
        decision_threshold=0.5,
        readiness_verdict="ready",
    )
    prediction = PredictionArtifact(
        artifact_id="prediction-1",
        owner_id="owner-1",
        run_id="run-1",
        status="accepted",
        lineage=("release-1",),
        evidence=(_evidence("evidence-prediction"),),
        release_candidate_id="release-1",
        model_input_hash="d" * 64,
        result_uri="minio://predictions/result.json",
        row_errors={},
        provenance={"model_version": "model-1"},
    )

    for artifact in (audit, spec, training, evaluation, release, prediction):
        ledger.append(artifact)

    assert len(ledger) == 6
    assert ledger.get("prediction-1").lineage == ("release-1",)


def test_contract_checker_blocks_missing_evidence_target_and_lineage():
    from hagent.agent.journey import ArtifactLedger, ContractChecker

    ledger = ArtifactLedger()
    invalid = _audit(
        status="draft",
        evidence=(),
        target_hypothesis=None,
        columns=("feature",),
    )
    verdict = ContractChecker().check(invalid, ledger=ledger)

    assert verdict.blocked
    assert {finding.code for finding in verdict.findings} >= {
        "EVIDENCE_REQUIRED",
        "TARGET_REQUIRED",
    }


def test_contract_checker_blocks_wrong_parent_artifact_type():
    from hagent.agent.journey import ArtifactLedger, ContractChecker, ReleaseCandidate

    ledger = ArtifactLedger()
    ledger.append(_audit())
    invalid_release = ReleaseCandidate(
        artifact_id="release-invalid-parent",
        owner_id="owner-1",
        run_id="run-1",
        lineage=("audit-1",),
        evidence=(_evidence("evidence-release-parent"),),
        evaluation_report_id="audit-1",
        model_version="model-1",
        input_schema={"feature": "number"},
        decision_threshold=0.5,
        readiness_verdict="ready",
    )

    verdict = ContractChecker().check(invalid_release, ledger=ledger)

    assert "LINEAGE_TYPE_MISMATCH" in {finding.code for finding in verdict.findings}


@pytest.mark.parametrize(
    ("metric", "direction"),
    [
        ("accuracy", "maximize"),
        ("f1", "maximize"),
        ("r2", "maximize"),
        ("rmse", "minimize"),
        ("mse", "minimize"),
        ("mae", "minimize"),
        ("log_loss", "minimize"),
    ],
)
def test_metric_registry_has_correct_direction(metric, direction):
    from hagent.agent.journey import metric_direction

    assert metric_direction(metric) == direction


def test_statistical_checker_blocks_leakage_invalid_split_budget_and_direction():
    from hagent.agent.journey import StatisticalChecker

    checker = StatisticalChecker()
    audit_verdict = checker.check(_audit(leakage_risks=("target_encoded_feature",)))
    spec_verdict = checker.check(
        _spec(
            metric="rmse",
            metric_direction="maximize",
            split_strategy="random_same_rows",
            max_training_jobs=0,
        )
    )

    assert {finding.code for finding in audit_verdict.findings} == {
        "LEAKAGE_RISK",
    }
    assert {finding.code for finding in spec_verdict.findings} >= {
        "METRIC_DIRECTION_MISMATCH",
        "INVALID_SPLIT_STRATEGY",
        "INVALID_TRAINING_BUDGET",
    }
    assert audit_verdict.blocked and spec_verdict.blocked


@pytest.mark.parametrize(
    ("report", "expected_delta"),
    [
        (_evaluation(), 0.2),
        (
            _evaluation(
                metric="rmse",
                metric_direction="minimize",
                metric_value=0.4,
                baseline_value=0.7,
                baseline_delta=0.3,
            ),
            0.3,
        ),
    ],
)
def test_statistical_checker_computes_baseline_delta_by_metric_direction(
    report,
    expected_delta,
):
    from hagent.agent.journey import StatisticalChecker

    verdict = StatisticalChecker().check(report)

    assert verdict.computed["baseline_delta"] == pytest.approx(expected_delta)
    assert "BASELINE_DELTA_MISMATCH" not in {
        finding.code for finding in verdict.findings
    }


def test_statistical_checker_blocks_delta_variance_and_overfit_regressions():
    from hagent.agent.journey import StatisticalChecker

    verdict = StatisticalChecker(max_variance=0.05, max_overfit_gap=0.1).check(
        _evaluation(
            baseline_delta=-0.2,
            variance=0.2,
            overfit_gap=0.3,
        )
    )

    assert {finding.code for finding in verdict.findings} >= {
        "BASELINE_DELTA_MISMATCH",
        "HIGH_VARIANCE",
        "OVERFIT_RISK",
    }
    assert verdict.blocked


def test_statistical_checker_blocks_consistent_baseline_regression_and_negative_variance():
    from hagent.agent.journey import StatisticalChecker

    verdict = StatisticalChecker().check(
        _evaluation(
            metric_value=0.5,
            baseline_value=0.6,
            baseline_delta=-0.1,
            variance=-0.01,
        )
    )

    assert {finding.code for finding in verdict.findings} >= {
        "NO_BASELINE_IMPROVEMENT",
        "INVALID_VARIANCE",
    }


def test_policy_checker_blocks_wrong_owner_scope_budget_and_unapproved_spec():
    from hagent.agent.journey import PolicyChecker, PolicyContext

    context = PolicyContext(
        owner_id="owner-1",
        granted_scopes=frozenset({"automl.dataset.read"}),
        max_training_jobs=1,
        approved_artifact_ids=frozenset(),
    )
    checker = PolicyChecker(context)
    wrong_owner = checker.check(_audit(owner_id="other-owner"))
    over_budget = checker.check(_spec(max_training_jobs=2))
    training = checker.check(_training())

    assert {finding.code for finding in wrong_owner.findings} >= {"OWNER_MISMATCH"}
    assert {finding.code for finding in over_budget.findings} >= {"BUDGET_EXCEEDED"}
    assert {finding.code for finding in training.findings} >= {
        "APPROVAL_REQUIRED",
        "SCOPE_DENIED",
    }


def test_policy_context_freezes_caller_owned_sets():
    from hagent.agent.journey import PolicyContext

    scopes = {"automl.dataset.read"}
    approvals = {"spec-1"}
    context = PolicyContext(
        owner_id="owner-1",
        granted_scopes=scopes,
        max_training_jobs=2,
        approved_artifact_ids=approvals,
    )
    scopes.clear()
    approvals.clear()

    assert context.granted_scopes == frozenset({"automl.dataset.read"})
    assert context.approved_artifact_ids == frozenset({"spec-1"})


def test_checker_aggregation_preserves_deterministic_blockers():
    from hagent.agent.journey import (
        CheckerVerdict,
        CheckFinding,
        merge_verdicts,
    )

    blocked = CheckerVerdict(
        checker="contract",
        findings=(
            CheckFinding(
                code="TARGET_REQUIRED",
                message="Thiếu target.",
                severity="blocker",
            ),
        ),
    )
    optimistic = CheckerVerdict(checker="critic", findings=())

    merged = merge_verdicts(blocked, optimistic)

    assert merged.blocked
    assert merged.passed is False
    assert [finding.code for finding in merged.findings] == ["TARGET_REQUIRED"]
