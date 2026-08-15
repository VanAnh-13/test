"""Regression cho owner-scoped training results và native capability adapter."""

from __future__ import annotations

import hashlib
import os
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from hagent.agent.capabilities.catalog import CapabilityCatalog
from hagent.agent.capabilities.models import CapabilityDescriptor
from hagent.agent.journey.result_critic import (
    TRAINING_RESULTS_CAPABILITY_ID,
    evaluate_training,
)
from hagent.agent.journey.training_operator import (
    TRAINING_LOOKUP_CAPABILITY_ID,
    TRAINING_START_CAPABILITY_ID,
)
from hagent.agent.runtime import RequestScope
from hagent.agent.runtime.context import GraphRequestContext

# Module API thí nghiệm khởi tạo MinIO client khi import; test chỉ cần cấu hình giả.
os.environ.setdefault("MINIO_ENDPOINT", "127.0.0.1:9000")
os.environ.setdefault("MINIO_ACCESS_KEY", "test-access")
os.environ.setdefault("MINIO_SECRET_KEY", "test-secret")


class _Collection:
    def __init__(self, documents):
        self.documents = list(documents)
        self.queries = []

    async def find_one(self, query, projection=None):
        self.queries.append((dict(query), projection))
        for document in self.documents:
            if all(document.get(key) == value for key, value in query.items()):
                return dict(document)
        return None


class _Database:
    def __init__(self, documents):
        self.tbl_Job = _Collection(documents)


def _persisted_evaluation():
    return {
        "metric": "accuracy",
        "metric_value": 0.82,
        "baseline_value": 0.70,
        "train_metric": 0.85,
        "cv_mean": 0.82,
        "cv_variance": 0.0004,
        "calibration_error": None,
        "input_features": ["age", "income"],
        "model_storage": {
            "bucket_name": "models",
            "object_name": "owner-1/job-1/model.pkl",
        },
        "model_version": "job-1:1:1",
    }


@pytest.mark.asyncio
async def test_reconcile_hashes_key_with_authenticated_owner_and_returns_safe_state():
    from api.experiment import reconcile_training_job

    key = "action-key"
    document_id = (
        "training-idempotency:" + hashlib.sha256(f"owner-1\0{key}".encode()).hexdigest()
    )
    database = _Database(
        [
            {
                "_id": document_id,
                "user.id": "owner-1",
                "job_id": "job-1",
                "dispatch": {"status": "sent"},
            }
        ]
    )

    result = await reconcile_training_job(
        key,
        db=database,
        current_user={"_id": "owner-1"},
    )

    assert result == {
        "found": True,
        "job_id": "job-1",
        "dispatch_status": "sent",
        "cost": 0.0,
    }
    assert database.tbl_Job.queries[0][0] == {
        "_id": document_id,
        "user.id": "owner-1",
    }


@pytest.mark.asyncio
async def test_results_query_is_owner_scoped_and_returns_only_typed_evidence():
    from api.experiment import get_training_results

    database = _Database(
        [
            {
                "job_id": "job-1",
                "user.id": "owner-1",
                "status": 1,
                "evaluation": _persisted_evaluation(),
                "model": {"secret": b"raw-model"},
                "infor": "internal failure detail",
            }
        ]
    )

    result = await get_training_results(
        ["job-1"],
        db=database,
        current_user={"_id": "owner-1"},
    )

    assert result == {
        "status": "completed",
        "job_id": "job-1",
        "metric": "accuracy",
        "metric_value": 0.82,
        "baseline_value": 0.70,
        "train_metric": 0.85,
        "cv_mean": 0.82,
        "cv_variance": 0.0004,
        "calibration_error": None,
        "model_version": "job-1:1:1",
        "input_schema": {"features": ["age", "income"]},
        "decision_threshold": None,
    }
    assert database.tbl_Job.queries[0][0] == {
        "job_id": "job-1",
        "user.id": "owner-1",
    }
    assert "model" not in result
    assert "infor" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (0, {"status": "running", "job_id": "job-1"}),
        (
            -1,
            {"status": "failed", "job_id": "job-1", "failure_code": "TRAINING_FAILED"},
        ),
    ],
)
async def test_results_api_reports_pending_or_failure_without_internal_text(
    status,
    expected,
):
    from api.experiment import get_training_results

    database = _Database(
        [
            {
                "job_id": "job-1",
                "user.id": "owner-1",
                "status": status,
                "infor": "do not expose",
            }
        ]
    )

    result = await get_training_results(
        ["job-1"],
        db=database,
        current_user={"_id": "owner-1"},
    )

    assert result == expected


@pytest.mark.asyncio
async def test_completed_job_without_valid_evidence_fails_closed():
    from api.experiment import get_training_results

    database = _Database([{"job_id": "job-1", "user.id": "owner-1", "status": 1}])

    with pytest.raises(HTTPException) as exc_info:
        await get_training_results(
            ["job-1"],
            db=database,
            current_user={"_id": "owner-1"},
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["code"] == "TRAINING_EVIDENCE_UNAVAILABLE"


@pytest.mark.asyncio
async def test_results_api_hides_jobs_owned_by_another_user():
    from api.experiment import get_training_results

    database = _Database(
        [
            {
                "job_id": "job-1",
                "user.id": "owner-2",
                "status": 1,
                "evaluation": _persisted_evaluation(),
            }
        ]
    )

    with pytest.raises(HTTPException) as exc_info:
        await get_training_results(
            ["job-1"],
            db=database,
            current_user={"_id": "owner-1"},
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == {"code": "TRAINING_JOB_NOT_FOUND"}


@pytest.mark.asyncio
async def test_results_api_rejects_invalid_job_list_before_database():
    from api.experiment import get_training_results

    database = _Database([])

    with pytest.raises(HTTPException) as exc_info:
        await get_training_results(
            [],
            db=database,
            current_user={"_id": "owner-1"},
        )

    assert exc_info.value.status_code == 400
    assert database.tbl_Job.queries == []


@pytest.mark.asyncio
async def test_native_http_tools_use_reconcile_and_results_read_endpoints(monkeypatch):
    from hagent.agent.tools import automl_tools

    calls = []

    async def fake_get(path, **kwargs):
        calls.append(("get", path, kwargs))
        return {"found": False}

    async def fake_post(path, **kwargs):
        calls.append(("post", path, kwargs))
        return {"status": "running", "job_id": "job-1"}

    monkeypatch.setattr(automl_tools, "_api_get", fake_get)
    monkeypatch.setattr(automl_tools, "_api_post", fake_post)

    await automl_tools.lookup_training_job.ainvoke(
        {"idempotency_key": "action-key", "token": "request-secret"}
    )
    await automl_tools.get_training_results.ainvoke(
        {"job_ids": ["job-1"], "token": "request-secret"}
    )

    assert calls == [
        (
            "get",
            "/v2/auto/jobs/by-idempotency/action-key",
            {"token": "request-secret", "use_cache": False},
        ),
        (
            "post",
            "/v2/auto/jobs/results",
            {"data": ["job-1"], "token": "request-secret"},
        ),
    ]


@pytest.mark.asyncio
async def test_native_adapter_injects_owner_and_credential_for_journey_capabilities():
    from hagent.agent.capabilities.native import (
        HAutoMLNativeAdapter,
        native_journey_descriptors,
    )

    calls = []

    async def list_invoker(arguments):
        calls.append(("list", dict(arguments)))
        return []

    async def inspect_invoker(arguments):
        calls.append(("inspect", dict(arguments)))
        return {}

    async def start_invoker(arguments):
        calls.append(("start", dict(arguments)))
        return {"status": "success", "job_id": "job-1"}

    async def lookup_invoker(arguments):
        calls.append(("lookup", dict(arguments)))
        return {"found": True, "job_id": "job-1", "dispatch_status": "sent"}

    async def results_invoker(arguments):
        calls.append(("results", dict(arguments)))
        return {"status": "running", "job_id": "job-1"}

    adapter = HAutoMLNativeAdapter(
        list_invoker=list_invoker,
        inspect_invoker=inspect_invoker,
        training_start_invoker=start_invoker,
        training_lookup_invoker=lookup_invoker,
        training_results_invoker=results_invoker,
    )
    scope = RequestScope(
        principal_id="owner-1",
        credential="request-secret",
        services={
            "scopes": (
                "automl.dataset.read",
                "automl.training.read",
                "automl.training.write",
            )
        },
    )

    await adapter.invoke(
        TRAINING_START_CAPABILITY_ID,
        {
            "dataset_id": "dataset-1",
            "problem_type": "classification",
            "target_column": "target",
            "metric": "accuracy",
            "models": ["RandomForestClassifier"],
            "time_limit": 30,
            "list_feature": ["age"],
            "idempotency_key": "action-key",
        },
        scope=scope,
    )
    await adapter.invoke(
        TRAINING_LOOKUP_CAPABILITY_ID,
        {"idempotency_key": "action-key"},
        scope=scope,
    )
    await adapter.invoke(
        TRAINING_RESULTS_CAPABILITY_ID,
        {"job_ids": ["job-1"]},
        scope=scope,
    )

    assert {item.id for item in native_journey_descriptors()} >= {
        TRAINING_START_CAPABILITY_ID,
        TRAINING_LOOKUP_CAPABILITY_ID,
        TRAINING_RESULTS_CAPABILITY_ID,
    }
    assert calls == [
        (
            "start",
            {
                "dataset_id": "dataset-1",
                "problem_type": "classification",
                "target_column": "target",
                "metric": "accuracy",
                "models": ["RandomForestClassifier"],
                "time_limit": 30,
                "list_feature": ["age"],
                "idempotency_key": "action-key",
                "user_id": "owner-1",
                "token": "request-secret",
            },
        ),
        ("lookup", {"idempotency_key": "action-key", "token": "request-secret"}),
        ("results", {"job_ids": ["job-1"], "token": "request-secret"}),
    ]


class _ResultsAdapter:
    def __init__(self, output):
        self.output = output

    async def invoke(self, _capability_id, _arguments, *, scope):
        return dict(self.output)


def _results_context(output):
    descriptor = CapabilityDescriptor(
        id=TRAINING_RESULTS_CAPABILITY_ID,
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        effect="read",
        required_scopes=frozenset({"automl.training.read"}),
        provider_id="results-fake",
    )
    catalog = CapabilityCatalog()
    catalog.register_provider("results-fake", [descriptor], _ResultsAdapter(output))
    return GraphRequestContext(
        principal_id="owner-1",
        credential="request-secret",
        services={"scopes": ("automl.training.read",)},
        capability_snapshot=catalog.snapshot(),
    )


def _evaluation_state():
    return {
        "run_id": "run-1",
        "training_run_set": SimpleNamespace(
            job_ids=("job-1",),
            artifact_id="training-artifact-1",
        ),
        "experiment_spec": SimpleNamespace(metric="accuracy"),
    }


@pytest.mark.asyncio
async def test_evaluator_accepts_real_cv_aggregate_without_synthesizing_folds():
    output = {
        "status": "completed",
        "job_id": "job-1",
        "metric": "accuracy",
        "metric_value": 0.82,
        "baseline_value": 0.70,
        "train_metric": 0.85,
        "cv_mean": 0.82,
        "cv_variance": 0.0004,
        "calibration_error": None,
        "model_version": "job-1:1:1",
        "input_schema": {"features": ["age", "income"]},
        "decision_threshold": None,
    }

    result = await evaluate_training(
        _evaluation_state(),
        context=_results_context(output),
    )

    report = result["evaluation_report"]
    assert report.cv_mean == pytest.approx(0.82)
    assert report.variance == pytest.approx(0.0004)
    assert "cv_scores" not in result["release_metadata"]


@pytest.mark.asyncio
async def test_evaluator_rejects_inconsistent_fold_and_aggregate_evidence():
    output = {
        "status": "completed",
        "job_id": "job-1",
        "metric": "accuracy",
        "metric_value": 0.82,
        "baseline_value": 0.70,
        "train_metric": 0.85,
        "cv_scores": [0.8, 0.82, 0.84],
        "cv_mean": 0.9,
        "cv_variance": 0.2,
        "calibration_error": None,
        "model_version": "job-1:1:1",
        "input_schema": {"features": ["age", "income"]},
    }

    result = await evaluate_training(
        _evaluation_state(),
        context=_results_context(output),
    )

    assert result["result"] == {
        "status": "evaluation_failed",
        "error_code": "INVALID_EVALUATION_EVIDENCE",
    }


@pytest.mark.asyncio
async def test_evaluator_preserves_safe_training_failure_code():
    result = await evaluate_training(
        _evaluation_state(),
        context=_results_context(
            {
                "status": "failed",
                "job_id": "job-1",
                "failure_code": "TRAINING_FAILED",
            }
        ),
    )

    assert result["evaluation_error_code"] == "TRAINING_FAILED"
    assert result["result"] == {
        "status": "evaluation_failed",
        "error_code": "TRAINING_FAILED",
    }
