"""Read-only LangGraph cho DatasetAudit journey v1."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from hagent.agent.journey.checkers import (
    ContractChecker,
    PolicyChecker,
    PolicyContext,
    StatisticalChecker,
)
from hagent.agent.journey.dataset_profiler import (
    interpret_audit_goal,
    profile_dataset,
)
from hagent.agent.journey.experiment_designer import (
    approval_proposal,
    design_experiment,
    requests_experiment,
    valid_edit_changes,
)
from hagent.agent.journey.prediction_operator import (
    requests_deploy,
    requests_prediction,
    run_prediction,
)
from hagent.agent.journey.result_critic import (
    evaluate_training,
    finalize_release_candidate,
)
from hagent.agent.journey.state import JourneyAuditState
from hagent.agent.journey.training_operator import dispatch_training
from hagent.agent.runtime.context import GraphRequestContext


def initial_audit_state(
    *,
    message: str,
    run_id: str,
    owner_id: str | None = None,
    capability_snapshot_digest: str | None = None,
    training_enabled: bool = False,
    evaluation_enabled: bool = False,
    prediction_enabled: bool = False,
) -> JourneyAuditState:
    """Tạo state persist-safe; owner_id caller cung cấp bị bỏ qua có chủ đích."""
    _ = owner_id
    state: JourneyAuditState = {
        "message": message,
        "run_id": run_id,
        "verdicts": (),
    }
    if capability_snapshot_digest is not None:
        state["capability_snapshot_digest"] = capability_snapshot_digest
        state["training_enabled"] = bool(training_enabled)
        state["evaluation_enabled"] = bool(evaluation_enabled)
        state["prediction_enabled"] = bool(prediction_enabled)
    return state


def _request_context(runtime: Runtime[GraphRequestContext]) -> GraphRequestContext:
    context = runtime.context
    if not isinstance(context, GraphRequestContext):
        raise TypeError("LANGGRAPH_REQUEST_CONTEXT_REQUIRED")
    return context


async def _interpret_node(
    state: JourneyAuditState,
    runtime: Runtime[GraphRequestContext],
) -> dict[str, Any]:
    _request_context(runtime)
    goal = interpret_audit_goal(state["message"])
    if requests_experiment(state["message"]):
        goal["operation"] = "experiment"
    return {"goal": goal}


async def _profile_node(
    state: JourneyAuditState,
    runtime: Runtime[GraphRequestContext],
) -> dict[str, Any]:
    return await profile_dataset(state, context=_request_context(runtime))


def _route_after_profile(state: JourneyAuditState) -> str:
    return "finalize" if state.get("error_code") else "contract_checker"


async def _contract_node(state: JourneyAuditState) -> dict[str, Any]:
    artifact = state["artifact"]
    verdict = ContractChecker().check(artifact)
    return {"verdicts": tuple(state.get("verdicts", ())) + (verdict,)}


async def _statistical_node(state: JourneyAuditState) -> dict[str, Any]:
    artifact = state["artifact"]
    verdict = StatisticalChecker().check(artifact)
    return {"verdicts": tuple(state.get("verdicts", ())) + (verdict,)}


async def _policy_node(
    state: JourneyAuditState,
    runtime: Runtime[GraphRequestContext],
) -> dict[str, Any]:
    context = _request_context(runtime)
    raw_scopes = context.services.get("scopes", ())
    granted_scopes = (
        frozenset(item for item in raw_scopes if isinstance(item, str))
        if not isinstance(raw_scopes, str)
        else frozenset()
    )
    max_training_jobs = context.services.get("max_training_jobs", 0)
    if not isinstance(max_training_jobs, int) or isinstance(max_training_jobs, bool):
        max_training_jobs = 0
    policy = PolicyContext(
        owner_id=context.principal_id,
        granted_scopes=granted_scopes,
        max_training_jobs=max_training_jobs,
        approved_artifact_ids=frozenset(),
    )
    verdict = PolicyChecker(policy).check(state["artifact"])
    return {"verdicts": tuple(state.get("verdicts", ())) + (verdict,)}


async def _finalize_node(state: JourneyAuditState) -> dict[str, Any]:
    if state.get("error_code"):
        return {
            "result": {
                "status": "failed",
                "error_code": state["error_code"],
            }
        }
    verdicts = tuple(state.get("verdicts", ()))
    blocked = any(verdict.blocked for verdict in verdicts)
    return {
        "result": {
            "status": "blocked" if blocked else "completed",
            "artifact_id": state["artifact"].artifact_id,
            "checker_count": len(verdicts),
        }
    }


def build_audit_graph() -> StateGraph:
    graph = StateGraph(JourneyAuditState, context_schema=GraphRequestContext)
    graph.add_node("interpret", _interpret_node)
    graph.add_node("dataset_profiler", _profile_node)
    graph.add_node("contract_checker", _contract_node)
    graph.add_node("statistical_checker", _statistical_node)
    graph.add_node("policy_checker", _policy_node)
    graph.add_node("finalize", _finalize_node)
    graph.set_entry_point("interpret")
    graph.add_edge("interpret", "dataset_profiler")
    graph.add_conditional_edges(
        "dataset_profiler",
        _route_after_profile,
        {"contract_checker": "contract_checker", "finalize": "finalize"},
    )
    graph.add_edge("contract_checker", "statistical_checker")
    graph.add_edge("statistical_checker", "policy_checker")
    graph.add_edge("policy_checker", "finalize")
    graph.add_edge("finalize", END)
    return graph


def _route_after_audit_checks(state: JourneyAuditState) -> str:
    if any(verdict.blocked for verdict in state.get("verdicts", ())):
        return "finalize"
    return (
        "experiment_designer"
        if state.get("goal", {}).get("operation") == "experiment"
        else "finalize"
    )


async def _experiment_designer_node(state: JourneyAuditState) -> dict[str, Any]:
    spec = design_experiment(state["artifact"], state["message"])
    return {
        "experiment_spec": spec,
        "experiment_verdicts": (),
        "approval": approval_proposal(spec),
    }


async def _experiment_contract_node(state: JourneyAuditState) -> dict[str, Any]:
    verdict = ContractChecker().check(state["experiment_spec"])
    return {
        "experiment_verdicts": tuple(state.get("experiment_verdicts", ()))
        + (verdict,)
    }


async def _experiment_statistical_node(state: JourneyAuditState) -> dict[str, Any]:
    verdict = StatisticalChecker().check(state["experiment_spec"])
    return {
        "experiment_verdicts": tuple(state.get("experiment_verdicts", ()))
        + (verdict,)
    }


async def _experiment_policy_node(
    state: JourneyAuditState,
    runtime: Runtime[GraphRequestContext],
) -> dict[str, Any]:
    context = _request_context(runtime)
    raw_limit = context.services.get(
        "max_training_jobs", state["experiment_spec"].max_training_jobs
    )
    limit = raw_limit if isinstance(raw_limit, int) and not isinstance(raw_limit, bool) else 0
    verdict = PolicyChecker(
        PolicyContext(
            owner_id=context.principal_id,
            granted_scopes=frozenset(),
            max_training_jobs=limit,
            approved_artifact_ids=frozenset(),
        )
    ).check(state["experiment_spec"])
    return {
        "experiment_verdicts": tuple(state.get("experiment_verdicts", ()))
        + (verdict,)
    }


def _route_after_experiment_checks(state: JourneyAuditState) -> str:
    return (
        "finalize_experiment"
        if any(verdict.blocked for verdict in state.get("experiment_verdicts", ()))
        else "approval"
    )


def _approval_node(state: JourneyAuditState) -> dict[str, Any]:
    response = interrupt(state["approval"])
    if not isinstance(response, dict):
        return {"approval_decision": "invalid", "approval_response": {}}
    if set(response) - {"approval_id", "decision", "changes"}:
        return {"approval_decision": "invalid", "approval_response": {}}
    if response.get("approval_id") != state["approval"]["approval_id"]:
        return {"approval_decision": "stale", "approval_response": {}}
    expires_at = datetime.fromisoformat(state["approval"]["expires_at"])
    if datetime.now().astimezone() >= expires_at:
        return {"approval_decision": "expired", "approval_response": {}}
    decision = response.get("decision")
    if decision not in {"approve", "reject", "edit"}:
        decision = "invalid"
    if decision == "edit" and not valid_edit_changes(response.get("changes")):
        decision = "invalid"
    safe_response = (
        {"changes": dict(response["changes"])} if decision == "edit" else {}
    )
    return {"approval_decision": decision, "approval_response": safe_response}


def _route_after_approval(state: JourneyAuditState) -> str:
    return "revise_experiment" if state.get("approval_decision") == "edit" else "finalize_experiment"


def _route_after_training_approval(state: JourneyAuditState) -> str:
    if state.get("approval_decision") == "edit":
        return "revise_experiment"
    if state.get("approval_decision") == "approve":
        return "training_operator"
    return "finalize_experiment"


async def _training_operator_node(
    state: JourneyAuditState,
    runtime: Runtime[GraphRequestContext],
) -> dict[str, Any]:
    return await dispatch_training(state, context=_request_context(runtime))


def _route_after_training(state: JourneyAuditState) -> str:
    if (
        state.get("evaluation_enabled") is True
        and state.get("training_run_set") is not None
        and state.get("training_outcome")
        in {"submitted", "replayed", "reconciled"}
    ):
        return "evaluation_operator"
    return "end"


async def _evaluation_operator_node(
    state: JourneyAuditState,
    runtime: Runtime[GraphRequestContext],
) -> dict[str, Any]:
    return await evaluate_training(state, context=_request_context(runtime))


def _route_after_evaluation(state: JourneyAuditState) -> str:
    return (
        "evaluation_contract_checker"
        if state.get("evaluation_report") is not None
        else "end"
    )


async def _evaluation_contract_node(state: JourneyAuditState) -> dict[str, Any]:
    verdict = ContractChecker().check(state["evaluation_report"])
    return {
        "evaluation_verdicts": tuple(state.get("evaluation_verdicts", ()))
        + (verdict,)
    }


async def _evaluation_statistical_node(state: JourneyAuditState) -> dict[str, Any]:
    verdict = StatisticalChecker().check(state["evaluation_report"])
    return {
        "evaluation_verdicts": tuple(state.get("evaluation_verdicts", ()))
        + (verdict,)
    }


async def _evaluation_policy_node(
    state: JourneyAuditState,
    runtime: Runtime[GraphRequestContext],
) -> dict[str, Any]:
    context = _request_context(runtime)
    raw_scopes = context.services.get("scopes", ())
    scopes = (
        frozenset(item for item in raw_scopes if isinstance(item, str))
        if not isinstance(raw_scopes, str)
        else frozenset()
    )
    verdict = PolicyChecker(
        PolicyContext(
            owner_id=context.principal_id,
            granted_scopes=scopes,
            max_training_jobs=0,
            approved_artifact_ids=frozenset(),
        )
    ).check(state["evaluation_report"])
    return {
        "evaluation_verdicts": tuple(state.get("evaluation_verdicts", ()))
        + (verdict,)
    }


async def _release_candidate_node(
    state: JourneyAuditState,
    runtime: Runtime[GraphRequestContext],
) -> dict[str, Any]:
    return await finalize_release_candidate(
        state,
        context=_request_context(runtime),
    )


def _route_after_release_candidate(state: JourneyAuditState) -> str:
    release = state.get("release_candidate")
    if release is None or release.readiness_verdict != "ready":
        return "end"
    if requests_deploy(state["message"]):
        return "capability_unavailable"
    if not requests_prediction(state["message"]):
        return "end"
    return (
        "prediction_operator"
        if state.get("prediction_enabled") is True
        else "capability_unavailable"
    )


async def _capability_unavailable_node(state: JourneyAuditState) -> dict[str, Any]:
    capability = (
        "automl.deploy@1"
        if requests_deploy(state["message"])
        else "automl.prediction.batch@1"
    )
    return {
        "result": {
            "status": "capability_unavailable",
            "error_code": "CAPABILITY_UNAVAILABLE",
            "capability": capability,
        }
    }


async def _prediction_operator_node(
    state: JourneyAuditState,
    runtime: Runtime[GraphRequestContext],
) -> dict[str, Any]:
    return await run_prediction(state, context=_request_context(runtime))


def _route_after_prediction(state: JourneyAuditState) -> str:
    return (
        "prediction_contract_checker"
        if state.get("prediction_artifact") is not None
        else "end"
    )


async def _prediction_contract_node(state: JourneyAuditState) -> dict[str, Any]:
    verdict = ContractChecker().check(state["prediction_artifact"])
    return {
        "prediction_verdicts": tuple(state.get("prediction_verdicts", ()))
        + (verdict,)
    }


async def _prediction_policy_node(
    state: JourneyAuditState,
    runtime: Runtime[GraphRequestContext],
) -> dict[str, Any]:
    context = _request_context(runtime)
    raw_scopes = context.services.get("scopes", ())
    scopes = (
        frozenset(item for item in raw_scopes if isinstance(item, str))
        if not isinstance(raw_scopes, str)
        else frozenset()
    )
    release = state["release_candidate"]
    verdict = PolicyChecker(
        PolicyContext(
            owner_id=context.principal_id,
            granted_scopes=scopes,
            max_training_jobs=0,
            approved_artifact_ids=frozenset({release.artifact_id}),
        )
    ).check(state["prediction_artifact"])
    return {
        "prediction_verdicts": tuple(state.get("prediction_verdicts", ()))
        + (verdict,)
    }


async def _finalize_prediction_node(state: JourneyAuditState) -> dict[str, Any]:
    artifact = state["prediction_artifact"]
    blocked = any(
        verdict.blocked for verdict in state.get("prediction_verdicts", ())
    )
    finalized_artifact = replace(
        artifact,
        status="rejected" if blocked else "accepted",
    )
    return {
        "prediction_artifact": finalized_artifact,
        "result": {
            "status": "prediction_rejected" if blocked else "prediction_completed",
            "artifact_id": finalized_artifact.artifact_id,
            "result_uri": finalized_artifact.result_uri,
        }
    }


async def _revise_experiment_node(state: JourneyAuditState) -> dict[str, Any]:
    response = state.get("approval_response", {})
    changes = response.get("changes", {})
    if not isinstance(changes, dict):
        changes = {}
    spec = design_experiment(
        state["artifact"],
        state["message"],
        previous=state["experiment_spec"],
        changes=changes,
    )
    return {
        "experiment_spec": spec,
        "experiment_verdicts": (),
        "approval": approval_proposal(spec),
        "approval_decision": "",
        "approval_response": {},
    }


async def _finalize_experiment_node(state: JourneyAuditState) -> dict[str, Any]:
    if any(verdict.blocked for verdict in state.get("experiment_verdicts", ())):
        status = "blocked"
    else:
        status = {
            "approve": "approved",
            "reject": "rejected",
            "stale": "stale_approval",
            "expired": "approval_expired",
        }.get(state.get("approval_decision"), "invalid_approval")
    return {
        "result": {
            "status": status,
            "artifact_id": state["experiment_spec"].artifact_id,
            "version": state["experiment_spec"].version,
        }
    }


def _build_experiment_graph(
    *,
    include_training: bool,
    include_evaluation: bool = False,
    include_prediction: bool = False,
) -> StateGraph:
    graph = StateGraph(JourneyAuditState, context_schema=GraphRequestContext)
    graph.add_node("interpret", _interpret_node)
    graph.add_node("dataset_profiler", _profile_node)
    graph.add_node("contract_checker", _contract_node)
    graph.add_node("statistical_checker", _statistical_node)
    graph.add_node("policy_checker", _policy_node)
    graph.add_node("finalize", _finalize_node)
    graph.add_node("experiment_designer", _experiment_designer_node)
    graph.add_node("experiment_contract_checker", _experiment_contract_node)
    graph.add_node("experiment_statistical_checker", _experiment_statistical_node)
    graph.add_node("experiment_policy_checker", _experiment_policy_node)
    graph.add_node("approval", _approval_node)
    graph.add_node("revise_experiment", _revise_experiment_node)
    graph.add_node("finalize_experiment", _finalize_experiment_node)
    if include_training:
        graph.add_node("training_operator", _training_operator_node)
    if include_evaluation:
        graph.add_node("evaluation_operator", _evaluation_operator_node)
        graph.add_node("evaluation_contract_checker", _evaluation_contract_node)
        graph.add_node("evaluation_statistical_checker", _evaluation_statistical_node)
        graph.add_node("evaluation_policy_checker", _evaluation_policy_node)
        graph.add_node("release_candidate", _release_candidate_node)
        graph.add_node("capability_unavailable", _capability_unavailable_node)
    if include_prediction:
        graph.add_node("prediction_operator", _prediction_operator_node)
        graph.add_node("prediction_contract_checker", _prediction_contract_node)
        graph.add_node("prediction_policy_checker", _prediction_policy_node)
        graph.add_node("finalize_prediction", _finalize_prediction_node)
    graph.set_entry_point("interpret")
    graph.add_edge("interpret", "dataset_profiler")
    graph.add_conditional_edges(
        "dataset_profiler",
        _route_after_profile,
        {"contract_checker": "contract_checker", "finalize": "finalize"},
    )
    graph.add_edge("contract_checker", "statistical_checker")
    graph.add_edge("statistical_checker", "policy_checker")
    graph.add_conditional_edges(
        "policy_checker",
        _route_after_audit_checks,
        {"experiment_designer": "experiment_designer", "finalize": "finalize"},
    )
    graph.add_edge("experiment_designer", "experiment_contract_checker")
    graph.add_edge("experiment_contract_checker", "experiment_statistical_checker")
    graph.add_edge("experiment_statistical_checker", "experiment_policy_checker")
    graph.add_conditional_edges(
        "experiment_policy_checker",
        _route_after_experiment_checks,
        {"approval": "approval", "finalize_experiment": "finalize_experiment"},
    )
    if include_training:
        graph.add_conditional_edges(
            "approval",
            _route_after_training_approval,
            {
                "revise_experiment": "revise_experiment",
                "training_operator": "training_operator",
                "finalize_experiment": "finalize_experiment",
            },
        )
        if include_evaluation:
            graph.add_conditional_edges(
                "training_operator",
                _route_after_training,
                {"evaluation_operator": "evaluation_operator", "end": END},
            )
            graph.add_conditional_edges(
                "evaluation_operator",
                _route_after_evaluation,
                {"evaluation_contract_checker": "evaluation_contract_checker", "end": END},
            )
            graph.add_edge(
                "evaluation_contract_checker",
                "evaluation_statistical_checker",
            )
            graph.add_edge(
                "evaluation_statistical_checker",
                "evaluation_policy_checker",
            )
            graph.add_edge("evaluation_policy_checker", "release_candidate")
            release_routes = {
                "capability_unavailable": "capability_unavailable",
                "end": END,
            }
            if include_prediction:
                release_routes["prediction_operator"] = "prediction_operator"
            graph.add_conditional_edges(
                "release_candidate",
                _route_after_release_candidate,
                release_routes,
            )
            graph.add_edge("capability_unavailable", END)
            if include_prediction:
                graph.add_conditional_edges(
                    "prediction_operator",
                    _route_after_prediction,
                    {
                        "prediction_contract_checker": "prediction_contract_checker",
                        "end": END,
                    },
                )
                graph.add_edge(
                    "prediction_contract_checker",
                    "prediction_policy_checker",
                )
                graph.add_edge("prediction_policy_checker", "finalize_prediction")
                graph.add_edge("finalize_prediction", END)
        else:
            graph.add_edge("training_operator", END)
    else:
        graph.add_conditional_edges(
            "approval",
            _route_after_approval,
            {
                "revise_experiment": "revise_experiment",
                "finalize_experiment": "finalize_experiment",
            },
        )
    graph.add_edge("revise_experiment", "experiment_contract_checker")
    graph.add_edge("finalize", END)
    graph.add_edge("finalize_experiment", END)
    return graph


def build_experiment_graph() -> StateGraph:
    return _build_experiment_graph(include_training=False)


def build_training_graph() -> StateGraph:
    """Mở rộng graph experiment bằng training node, không đổi node cũ."""
    return _build_experiment_graph(include_training=True)


def build_evaluation_graph() -> StateGraph:
    """Mở rộng training graph bằng evaluation và release nodes."""
    return _build_experiment_graph(
        include_training=True,
        include_evaluation=True,
    )


def build_prediction_graph() -> StateGraph:
    """Mở rộng evaluation graph bằng schema-gated prediction nodes."""
    return _build_experiment_graph(
        include_training=True,
        include_evaluation=True,
        include_prediction=True,
    )


def compile_experiment_graph(*, checkpointer: Any):
    from hagent.agent.journey.persistence import prepare_journey_checkpointer

    if checkpointer is None:
        raise ValueError("Experiment approval requires a checkpointer")
    return build_experiment_graph().compile(
        checkpointer=prepare_journey_checkpointer(checkpointer)
    )


def compile_training_graph(*, checkpointer: Any):
    from hagent.agent.journey.persistence import prepare_journey_checkpointer

    if checkpointer is None:
        raise ValueError("Journey training requires a checkpointer")
    return build_training_graph().compile(
        checkpointer=prepare_journey_checkpointer(checkpointer)
    )


def compile_evaluation_graph(*, checkpointer: Any):
    from hagent.agent.journey.persistence import prepare_journey_checkpointer

    if checkpointer is None:
        raise ValueError("Journey evaluation requires a checkpointer")
    return build_evaluation_graph().compile(
        checkpointer=prepare_journey_checkpointer(checkpointer)
    )


def compile_prediction_graph(*, checkpointer: Any):
    from hagent.agent.journey.persistence import prepare_journey_checkpointer

    if checkpointer is None:
        raise ValueError("Journey prediction requires a checkpointer")
    return build_prediction_graph().compile(
        checkpointer=prepare_journey_checkpointer(checkpointer)
    )


def compile_audit_graph(*, checkpointer: Any | None = None):
    """Compile tại composition root để saver được inject, không đọc global env."""
    from hagent.agent.journey.persistence import prepare_journey_checkpointer

    return build_audit_graph().compile(
        checkpointer=prepare_journey_checkpointer(checkpointer)
    )
