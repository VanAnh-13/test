"""Regression cho evaluation evidence từ engine đến Mongo job."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

os.environ.setdefault("MINIO_ENDPOINT", "localhost:9000")
os.environ.setdefault("MINIO_ACCESS_KEY", "test-access-key")
os.environ.setdefault("MINIO_SECRET_KEY", "test-secret-key")

from automl.engine import build_evaluation_evidence


class _ClassificationEstimator:
    classes_ = np.array([0, 1])

    def predict(self, _features):
        return np.array([0, 0, 1, 1])

    def predict_proba(self, _features):
        return np.array(
            [
                [0.9, 0.1],
                [0.7, 0.3],
                [0.4, 0.6],
                [0.2, 0.8],
            ]
        )


class _RegressionEstimator:
    def predict(self, _features):
        return np.array([0.0, 1.0])


def _evaluation_payload(metric="accuracy"):
    return {
        "metric": metric,
        "metric_value": 0.8,
        "baseline_value": 0.7,
        "train_metric": 0.85,
        "cv_mean": 0.8,
        "cv_variance": 0.01,
        "calibration_error": None,
    }


def test_classification_evidence_uses_real_predictions_and_cv_std():
    features = np.zeros((4, 1))
    target = np.array([0, 0, 0, 1])

    evidence = build_evaluation_evidence(
        estimator=_ClassificationEstimator(),
        features=features,
        target=target,
        metric="accuracy",
        cv_results={
            "mean_test_accuracy": [0.8],
            "std_test_accuracy": [0.1],
        },
        best_index=0,
        problem_type="classification",
    )

    assert evidence == {
        "metric": "accuracy",
        "metric_value": pytest.approx(0.8),
        "baseline_value": pytest.approx(0.75),
        "train_metric": pytest.approx(0.75),
        "cv_mean": pytest.approx(0.8),
        "cv_variance": pytest.approx(0.01),
        "calibration_error": None,
    }


def test_regression_evidence_preserves_minimize_metric_direction():
    features = np.zeros((2, 1))
    target = np.array([0.0, 2.0])

    evidence = build_evaluation_evidence(
        estimator=_RegressionEstimator(),
        features=features,
        target=target,
        metric="rmse",
        cv_results={
            "mean_test_rmse": [-0.8],
            "std_test_rmse": [0.2],
        },
        best_index=0,
        problem_type="regression",
    )

    assert evidence["metric"] == "rmse"
    assert evidence["metric_value"] == pytest.approx(0.8)
    assert evidence["baseline_value"] == pytest.approx(1.0)
    assert evidence["train_metric"] == pytest.approx(math.sqrt(0.5))
    assert evidence["cv_mean"] == pytest.approx(0.8)
    assert evidence["cv_variance"] == pytest.approx(0.04)


def test_auc_uses_real_probability_and_probability_baseline():
    evidence = build_evaluation_evidence(
        estimator=_ClassificationEstimator(),
        features=np.zeros((4, 1)),
        target=np.array([0, 0, 0, 1]),
        metric="auc",
        cv_results={"mean_test_auc": [0.9], "std_test_auc": [0.05]},
        best_index=0,
        problem_type="classification",
    )

    assert evidence["train_metric"] == pytest.approx(1.0)
    assert evidence["baseline_value"] == pytest.approx(0.5)
    assert evidence["cv_mean"] == pytest.approx(0.9)


@pytest.mark.parametrize(
    ("metric", "cv_results"),
    [
        (
            "unsupported",
            {"mean_test_unsupported": [0.5], "std_test_unsupported": [0.1]},
        ),
        (
            "accuracy",
            {"mean_test_accuracy": [float("nan")], "std_test_accuracy": [0.1]},
        ),
        ("accuracy", {"mean_test_accuracy": [0.5]}),
    ],
)
def test_evaluation_evidence_fails_closed_for_unsupported_or_invalid_cv(
    metric,
    cv_results,
):
    with pytest.raises(ValueError):
        build_evaluation_evidence(
            estimator=_ClassificationEstimator(),
            features=np.zeros((4, 1)),
            target=np.array([0, 0, 0, 1]),
            metric=metric,
            cv_results=cv_results,
            best_index=0,
            problem_type="classification",
        )


def test_worker_forwards_only_engine_evaluation_for_its_model(monkeypatch):
    from cluster import worker

    evaluation = _evaluation_payload()

    def fake_train_process(*_args, **_kwargs):
        return (
            0,
            _ClassificationEstimator(),
            0.8,
            {"depth": 2},
            [
                {
                    "model_id": 0,
                    "model_name": "Classifier",
                    "scores": {"accuracy": 0.8},
                    "evaluation": evaluation,
                }
            ],
            False,
        )

    class Queue:
        item = None

        def put(self, item):
            self.item = item

    queue = Queue()
    monkeypatch.setattr(worker, "train_process", fake_train_process)
    monkeypatch.setattr(
        worker.np,
        "load",
        lambda path, **_kwargs: (
            np.zeros((4, 1)) if path == "features.npy" else np.array([0, 0, 0, 1])
        ),
    )

    worker._training_worker(
        queue,
        "features.npy",
        "target.npy",
        ["accuracy"],
        "accuracy",
        {0: {"model": _ClassificationEstimator(), "params": {}}},
        "classification",
        "grid_search",
        30,
    )

    assert queue.item["success"] is True
    assert queue.item["evaluation"] == evaluation


@pytest.mark.asyncio
async def test_master_persists_selected_model_evidence_and_trusted_features(
    monkeypatch,
):
    from automl.v2 import master

    captured = {}

    class FakeMongoJob:
        def __init__(self, _db):
            pass

        async def update_success(self, job_id, payload):
            captured["job_id"] = job_id
            captured["payload"] = payload

        async def update_failure(self, *_args):
            raise AssertionError("Không được đánh dấu failure trong happy path")

    first = _evaluation_payload(metric="accuracy")
    second = {**_evaluation_payload(metric="accuracy"), "metric_value": 0.9}
    job_id = "job-evidence"
    master.state.job_tracker[job_id] = {
        "results": [
            {
                "success": True,
                "model_name": "ModelA",
                "score": 0.8,
                "scores": {"accuracy": 0.8},
                "best_params": {},
                "model": {"bucket_name": "temp", "object_name": "a.pkl"},
                "evaluation": first,
            },
            {
                "success": True,
                "model_name": "ModelB",
                "score": 0.9,
                "scores": {"accuracy": 0.9},
                "best_params": {"depth": 3},
                "model": {"bucket_name": "temp", "object_name": "b.pkl"},
                "evaluation": second,
            },
        ],
        "config": {
            "metric_sort": "accuracy",
            "list_feature": ["age", "income"],
        },
        "id_user": "owner-1",
        "completed_tasks": 2,
        "total_tasks": 2,
        "timed_out": False,
    }
    monkeypatch.setattr(master, "MongoJob", FakeMongoJob)
    monkeypatch.setattr(master.minIOStorage, "move_model", lambda **_kwargs: None)

    try:
        assert await master.reduce_results_for_job(job_id, object()) is True
    finally:
        master.state.job_tracker.pop(job_id, None)

    persisted = captured["payload"]["evaluation"]
    assert captured["job_id"] == job_id
    assert persisted["metric_value"] == 0.9
    assert persisted["input_features"] == ["age", "income"]
    assert persisted["model_version"] == "job-evidence:1:1"


@pytest.mark.asyncio
async def test_mongo_job_rejects_non_finite_evidence_before_update():
    from database.get_dataset import MongoJob

    class Collection:
        calls = []

        async def update_one(self, query, update):
            self.calls.append((query, update))
            return type("Result", (), {"matched_count": 1})()

    class Database:
        tbl_Job = Collection()

    valid_payload = {
        "owner_id": "owner-1",
        "best_model_id": 1,
        "best_model": "ModelB",
        "model": {"bucket_name": "models", "object_name": "model.pkl"},
        "best_params": {},
        "best_score": 0.9,
        "model_scores": [],
        "evaluation": {
            **_evaluation_payload(),
            "input_features": ["age"],
            "model_storage": {
                "bucket_name": "models",
                "object_name": "owner-1/job-1/model.pkl",
            },
            "model_version": "job-1:1:1",
        },
    }
    job = MongoJob(Database())

    await job.update_success("job-1", valid_payload)
    assert Database.tbl_Job.calls[0][0] == {
        "job_id": "job-1",
        "user.id": "owner-1",
    }
    assert Database.tbl_Job.calls[0][1]["$set"]["evaluation"]["metric"] == "accuracy"

    invalid_payload = {
        **valid_payload,
        "evaluation": {
            **valid_payload["evaluation"],
            "cv_variance": float("inf"),
        },
    }
    with pytest.raises(ValueError):
        await job.update_success("job-2", invalid_payload)
    assert len(Database.tbl_Job.calls) == 1
