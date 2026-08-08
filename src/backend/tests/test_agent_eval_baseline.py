"""Behavior-rich offline baseline for the legacy HAgent eval runner."""

from __future__ import annotations

import asyncio

import pytest

from hagent.agent.eval import runner as runner_module
from hagent.agent.eval.metrics import ToolCallTrace, evaluate_quality, summarize
from hagent.agent.eval.runner import report_markdown, run_baseline_suite, run_scenario
from hagent.agent.eval.scenarios import (
    BASELINE_VERSION,
    EvalScenario,
    ToolExpectation,
    baseline_scenarios,
)


def _scenario(scenario_id: str) -> EvalScenario:
    return next(item for item in baseline_scenarios() if item.id == scenario_id)


def test_baseline_matrix_is_versioned_and_covers_required_dimensions():
    scenarios = baseline_scenarios()

    assert len({item.id for item in scenarios}) == len(scenarios)
    assert all(item.baseline_version == BASELINE_VERSION for item in scenarios)
    assert all(item.legacy_expected_success is not None for item in scenarios)
    assert {"vi", "en"} <= {tag for item in scenarios for tag in item.tags}
    assert any(len(item.turns) > 1 for item in scenarios)
    assert {"needs_input", "upstream_failure"} <= {
        item.expect_outcome for item in scenarios
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario_id",
    [item.id for item in baseline_scenarios()],
)
async def test_each_frozen_scenario_produces_the_expected_behavior(scenario_id):
    scenario = _scenario(scenario_id)
    result = await run_scenario(scenario, "single_shot")

    assert result.success is scenario.legacy_expected_success
    if result.success:
        assert result.goal_exactness == 1.0
    else:
        assert result.reasons


@pytest.mark.asyncio
async def test_frozen_baseline_suite_matches_recorded_legacy_profile():
    report = await run_baseline_suite()

    assert report["baseline_version"] == BASELINE_VERSION
    assert report["legacy_expectations_match"]
    assert report["n_scenarios"] == 5
    assert sum(report["observed_success"].values()) == 4


def test_quality_rejects_wrong_goal_arguments_and_missing_evidence():
    scenario = _scenario("vi_train_complete")
    trace = [
        ToolCallTrace(
            name="start_training",
            arguments={
                "dataset_id": "ds_glass",
                "target_column": "wrong_target",
                "problem_type": "classification",
                "metric": "f1",
            },
            effect="mutation",
            outcome="succeeded",
            output={"job_id": "job-1"},
            elapsed_seconds=0.02,
        )
    ]

    quality = evaluate_quality(
        scenario,
        actual_goal={
            "goal_type": "train",
            "dataset_id": "ds_glass",
            "target_column": "wrong_target",
            "problem_type": "classification",
            "metric": "f1",
            "constraints": {"time_limit": 300},
        },
        invocations=trace,
        outcome="succeeded",
        elapsed_seconds=0.04,
        token_count=17,
    )

    assert quality.goal_exactness < 1.0
    assert quality.argument_exactness == 0.0
    assert quality.evidence_faithfulness == 0.0
    assert quality.token_count == 17
    assert quality.latency_seconds == 0.04
    assert not quality.passed


def test_quality_counts_unauthorized_and_duplicate_mutations():
    scenario = EvalScenario(
        id="policy",
        name="Policy probe",
        message="inspect only",
        goal={"goal_type": "analyze", "dataset_id": "ds-1"},
        expect_goal={"goal_type": "analyze", "dataset_id": "ds-1"},
        expect_has_job=False,
        expect_min_tools=0,
        allow_mutations=False,
    )
    repeated = ToolCallTrace(
        name="start_training",
        arguments={"dataset_id": "ds-1", "target_column": "label"},
        effect="mutation",
        outcome="succeeded",
        output={"job_id": "job-1"},
    )

    quality = evaluate_quality(
        scenario,
        actual_goal=scenario.goal,
        invocations=[repeated, repeated],
        outcome="succeeded",
        elapsed_seconds=0.01,
        token_count=0,
    )

    assert quality.unauthorized_side_effects == 2
    assert quality.duplicate_mutations == 1
    assert not quality.policy_compliant
    assert not quality.passed


@pytest.mark.asyncio
async def test_legacy_runner_captures_arguments_evidence_latency_and_tokens():
    result = await run_scenario(_scenario("vi_train_complete"), "single_shot")

    assert result.success
    assert result.goal_exactness == 1.0
    assert result.argument_exactness == 1.0
    assert result.evidence_faithfulness == 1.0
    assert result.outcome == "succeeded"
    assert result.unauthorized_side_effects == 0
    assert result.duplicate_mutations == 0
    assert result.elapsed_seconds >= 0
    assert result.token_count >= 0
    assert any(call.name == "start_training" for call in result.invocations)


@pytest.mark.asyncio
async def test_missing_information_is_needs_input_without_mutation():
    result = await run_scenario(_scenario("en_train_missing_target"), "single_shot")

    assert result.success
    assert result.outcome == "needs_input"
    assert result.tools_called == 0
    assert result.unauthorized_side_effects == 0
    assert all(call.effect != "mutation" for call in result.invocations)


@pytest.mark.asyncio
async def test_upstream_failure_is_observed_without_fake_success():
    scenario = _scenario("vi_analyze_upstream_failure")
    result = await run_scenario(scenario, "single_shot")

    assert result.success
    assert result.outcome == "upstream_failure"
    assert result.best_job_id is None
    assert len(result.invocations) == 1
    assert result.invocations[0].outcome == "failed"
    assert result.invocations[0].error_code == "UPSTREAM_UNAVAILABLE"


def test_tool_expectation_requires_evidence_not_only_a_job_id():
    scenario = EvalScenario(
        id="evidence",
        name="Evidence probe",
        message="train",
        goal={"goal_type": "train"},
        expect_goal={"goal_type": "train"},
        expect_tool_calls=[
            ToolExpectation(
                name="start_training",
                arguments={"dataset_id": "ds-1"},
                evidence_keys=["job_id", "status"],
            )
        ],
    )
    quality = evaluate_quality(
        scenario,
        actual_goal=scenario.goal,
        invocations=[
            ToolCallTrace(
                name="start_training",
                arguments={"dataset_id": "ds-1"},
                effect="mutation",
                outcome="succeeded",
                output={"job_id": "job-1"},
            )
        ],
        outcome="succeeded",
        elapsed_seconds=0.01,
        token_count=0,
    )

    assert quality.argument_exactness == 1.0
    assert quality.evidence_faithfulness == 0.0
    assert not quality.passed


def test_argument_exactness_penalizes_an_extra_wrong_call():
    scenario = _scenario("vi_train_complete")
    correct = ToolCallTrace(
        name="start_training",
        arguments={
            "user_id": "eval_user",
            "dataset_id": "ds_glass",
            "target_column": "Type",
            "problem_type": "classification",
            "metric": "f1",
            "time_limit": 300,
            "list_feature": ["RI", "Na", "Mg", "Type"],
        },
        effect="mutation",
        outcome="succeeded",
        output={"job_id": "job-1", "status": "starting"},
    )
    wrong = ToolCallTrace(
        name="start_training",
        arguments={"dataset_id": "another-dataset"},
        effect="mutation",
        outcome="succeeded",
        output={"job_id": "job-2", "status": "starting"},
    )

    quality = evaluate_quality(
        scenario,
        actual_goal=scenario.expect_goal,
        invocations=[correct, wrong],
        outcome="succeeded",
        elapsed_seconds=0.01,
        token_count=0,
    )

    assert quality.argument_exactness == 0.5
    assert not quality.passed


def test_exact_arguments_and_policy_reject_unsafe_extras():
    scenario = EvalScenario(
        id="strict",
        name="Strict invocation contract",
        message="train",
        goal={"goal_type": "train"},
        expect_goal={"goal_type": "train"},
        expect_tool_calls=[
            ToolExpectation(
                name="start_training",
                arguments={"dataset_id": "ds-1", "target_column": "label"},
            )
        ],
        allow_mutations=True,
    )
    extra_argument = ToolCallTrace(
        name="start_training",
        arguments={
            "dataset_id": "ds-1",
            "target_column": "label",
            "delete_after_training": True,
        },
        effect="mutation",
        outcome="succeeded",
        output={"job_id": "job-1"},
    )
    unrelated_mutation = ToolCallTrace(
        name="delete_dataset",
        arguments={"dataset_id": "ds-1"},
        effect="mutation",
        outcome="succeeded",
        output={"status": "deleted"},
    )

    quality = evaluate_quality(
        scenario,
        actual_goal=scenario.goal,
        invocations=[extra_argument, unrelated_mutation],
        outcome="succeeded",
        elapsed_seconds=0.01,
        token_count=0,
    )

    assert quality.argument_exactness == 0.0
    assert quality.unauthorized_side_effects == 2
    assert not quality.policy_compliant
    assert not quality.passed


@pytest.mark.asyncio
async def test_summary_and_report_publish_quality_metrics():
    result = await run_scenario(_scenario("vi_train_complete"), "single_shot")
    summary = summarize([result])[0].to_dict()
    report = {
        "n_scenarios": 1,
        "modes": ["single_shot"],
        "summaries": [summary],
        "results": [result.to_dict()],
    }

    assert summary["avg_goal_exactness"] == 1.0
    assert summary["avg_argument_exactness"] == 1.0
    assert summary["avg_evidence_faithfulness"] == 1.0
    assert summary["unauthorized_side_effects"] == 0
    assert summary["duplicate_mutations"] == 0
    markdown = report_markdown(report)
    assert "| Goal | Args | Evidence |" in markdown
    assert "outcome=succeeded" in markdown
    assert "tokens=" in markdown


@pytest.mark.asyncio
async def test_invocation_trace_redacts_credentials():
    expected_job_id = "eval-job-vi_train_complete-1"

    async def invoker(action_type, params):
        if action_type == "start_training":
            return {
                "job_id": expected_job_id,
                "status": "starting",
                "access_token": "do-not-record",
                "accessToken": "camel-token",
                "apiKey": "camel-key",
                "APIKey": "acronym-key",
                "nested": {
                    "clientSecret": "camel-secret",
                    "privateKey": "private-key",
                    "jwt": "signed-token",
                    "Bearer": "bearer-token",
                },
                "token_count": 12,
            }
        return {
            "job_id": expected_job_id,
            "status": "completed",
            "best_score": 0.9,
        }

    result = await run_scenario(
        _scenario("vi_train_complete"),
        "single_shot",
        tool_invoker=invoker,
    )

    assert result.success
    start_call = next(call for call in result.invocations if call.name == "start_training")
    assert start_call.output["access_token"] == "[REDACTED]"
    assert start_call.output["accessToken"] == "[REDACTED]"
    assert start_call.output["apiKey"] == "[REDACTED]"
    assert start_call.output["APIKey"] == "[REDACTED]"
    assert start_call.output["nested"]["clientSecret"] == "[REDACTED]"
    assert start_call.output["nested"]["privateKey"] == "[REDACTED]"
    assert start_call.output["nested"]["jwt"] == "[REDACTED]"
    assert start_call.output["nested"]["Bearer"] == "[REDACTED]"
    assert start_call.output["token_count"] == 12
    serialized = str(result.to_dict())
    assert "do-not-record" not in serialized
    assert "camel-token" not in serialized
    assert "camel-key" not in serialized
    assert "acronym-key" not in serialized
    assert "camel-secret" not in serialized
    assert "private-key" not in serialized
    assert "signed-token" not in serialized
    assert "bearer-token" not in serialized


@pytest.mark.asyncio
async def test_error_dictionary_is_not_recorded_as_success():
    async def invoker(action_type, params):
        return {"error": "connection failed at C:\\internal token=do-not-record"}

    result = await run_scenario(
        _scenario("vi_analyze_upstream_failure"),
        "single_shot",
        tool_invoker=invoker,
    )

    assert result.success
    assert result.outcome == "upstream_failure"
    assert result.invocations[0].outcome == "failed"
    assert result.invocations[0].error_code == "UPSTREAM_UNAVAILABLE"
    assert result.invocations[0].output == {"error": "UPSTREAM_UNAVAILABLE"}
    assert "do-not-record" not in str(result.to_dict())


@pytest.mark.asyncio
async def test_concurrent_scenarios_do_not_mix_invokers_or_traces():
    def make_scenario(label: str) -> EvalScenario:
        return EvalScenario(
            id=f"concurrent-{label}",
            name=f"Concurrent {label}",
            message=f"analyze {label}",
            goal={"goal_type": "analyze", "dataset_id": label},
            expect_goal={"goal_type": "analyze", "dataset_id": label},
            expect_tool_calls=[
                ToolExpectation(
                    name="get_dataset_info",
                    arguments={"dataset_id": label},
                    evidence_keys=["dataset_id", "features", "marker"],
                )
            ],
            expect_goal_type="analyze",
            expect_has_job=False,
            allow_mutations=False,
        )

    def make_invoker(label: str):
        async def invoker(action_type, params):
            await asyncio.sleep(0.01)
            return {"dataset_id": label, "features": [], "marker": label}

        return invoker

    first, second = await asyncio.gather(
        run_scenario(
            make_scenario("ds-a"),
            "single_shot",
            tool_invoker=make_invoker("ds-a"),
        ),
        run_scenario(
            make_scenario("ds-b"),
            "single_shot",
            tool_invoker=make_invoker("ds-b"),
        ),
    )

    assert first.success and second.success
    assert first.invocations[0].output["marker"] == "ds-a"
    assert second.invocations[0].output["marker"] == "ds-b"


@pytest.mark.asyncio
async def test_canonical_cancel_job_is_an_unauthorized_mutation(monkeypatch):
    from hagent.agent.execution.tool_runner import invoke_tool

    async def cancel_mode(scenario, *, user_id):
        await invoke_tool("cancel_job", {"job_id": "job-1"})
        return {
            "tools_called": 1,
            "has_job": False,
            "goal_type": "monitor",
            "cost_metrics": {"tools_called": 1},
        }

    async def invoker(action_type, params):
        return {"job_id": "job-1", "status": "cancelled"}

    monkeypatch.setitem(runner_module._MODE_RUNNERS, "cancel_probe", cancel_mode)
    scenario = EvalScenario(
        id="cancel-policy",
        name="Cancel policy probe",
        message="check job",
        goal={"goal_type": "monitor"},
        expect_goal={"goal_type": "monitor"},
        expect_goal_type="monitor",
        expect_min_tools=0,
        expect_has_job=False,
        allow_mutations=False,
    )

    result = await run_scenario(
        scenario,
        "cancel_probe",
        tool_invoker=invoker,
    )

    assert not result.success
    assert result.invocations[0].name == "cancel_job"
    assert result.invocations[0].effect == "mutation"
    assert result.unauthorized_side_effects == 1


def test_report_keeps_legacy_result_dictionary_compatible():
    report = {
        "n_scenarios": 1,
        "modes": ["single_shot"],
        "summaries": [
            {
                "mode": "single_shot",
                "n": 1,
                "success_rate": 1.0,
                "avg_elapsed": 0.1,
                "avg_tools": 1.0,
                "avg_revisions": 0.0,
                "avg_campaign_completed": 0.0,
            }
        ],
        "results": [
            {
                "scenario_id": "legacy",
                "mode": "single_shot",
                "success": True,
                "elapsed_seconds": 0.1,
                "tools_called": 1,
                "reasons": ["ok"],
            }
        ],
    }

    markdown = report_markdown(report)

    assert "outcome=succeeded" in markdown
    assert "goal=100%" in markdown
