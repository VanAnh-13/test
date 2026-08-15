"""Regression cho owner và idempotency của public training API."""

import asyncio
import importlib
import logging
import os
import uuid
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from bson import BSON, ObjectId
from fastapi import FastAPI
from pymongo import AsyncMongoClient
from pymongo.errors import DuplicateKeyError

from config.providers import get_db
from hagent.agent.tools import automl_tools
from users.routers import get_current_user

# Module API thí nghiệm khởi tạo MinIO client khi import; test chỉ cần cấu hình giả.
os.environ.setdefault("MINIO_ENDPOINT", "127.0.0.1:9000")
os.environ.setdefault("MINIO_ACCESS_KEY", "test-access")
os.environ.setdefault("MINIO_SECRET_KEY", "test-secret")
experiment = importlib.import_module("api.experiment")


OWNER_ID = "64b64b64b64b64b64b64b641"
OTHER_OWNER_ID = "64b64b64b64b64b64b64b642"
DATASET_ID = "64b64b64b64b64b64b64b643"


def _nested_value(document: dict, dotted_key: str):
    value = document
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


class FakeJobCollection:
    """Fake Mongo tối thiểu, giữ hành vi quan sát được của training API."""

    def __init__(self):
        self.documents: list[dict] = []

    async def find_one(self, query, projection=None):
        del projection
        for document in self.documents:
            if all(
                _nested_value(document, key) == value for key, value in query.items()
            ):
                return deepcopy(document)
        return None

    async def insert_one(self, document):
        await asyncio.sleep(0)
        document_id = document.get("_id")
        if any(existing.get("_id") == document_id for existing in self.documents):
            raise DuplicateKeyError("duplicate training idempotency key")
        stored = deepcopy(document)
        stored.setdefault("_id", f"job-document-{len(self.documents) + 1}")
        self.documents.append(stored)
        return SimpleNamespace(inserted_id=stored["_id"])

    async def update_one(self, query, update):
        for document in self.documents:
            if all(
                _nested_value(document, key) == value for key, value in query.items()
            ):
                for dotted_key, value in update.get("$set", {}).items():
                    target = document
                    parts = dotted_key.split(".")
                    for part in parts[:-1]:
                        target = target.setdefault(part, {})
                    target[parts[-1]] = value
                return SimpleNamespace(matched_count=1, modified_count=1)
        return SimpleNamespace(matched_count=0, modified_count=0)


def _database(*, dataset_owner: str = OWNER_ID):
    jobs = FakeJobCollection()
    return SimpleNamespace(
        tbl_User=SimpleNamespace(
            find_one=AsyncMock(return_value={"_id": OWNER_ID, "username": "owner"})
        ),
        tbl_Data=SimpleNamespace(
            find_one=AsyncMock(
                return_value={
                    "_id": DATASET_ID,
                    "dataName": "dataset",
                    "userId": dataset_owner,
                    "activate": 1,
                }
            )
        ),
        tbl_Job=jobs,
    )


def _training_payload(*, owner_id: str = OWNER_ID) -> dict:
    return {
        "id_data": DATASET_ID,
        "id_user": owner_id,
        "config": {
            "problem_type": "classification",
            "target": "label",
            "metric_sort": "accuracy",
            "list_feature": ["feature"],
        },
    }


async def _post_training(
    *,
    db,
    payload: dict,
    key: str,
    current_owner_id: str = OWNER_ID,
):
    app = FastAPI()
    app.include_router(experiment.exp)
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = lambda: {
        "_id": current_owner_id,
        "role": "user",
    }
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(
            "/v2/auto/jobs/training",
            json=payload,
            headers={"Idempotency-Key": key},
        )


@pytest.mark.asyncio
async def test_training_rejects_spoofed_user_before_database_or_kafka(monkeypatch):
    class DatabaseMustNotBeRead:
        def __getattr__(self, name):
            raise AssertionError(
                f"Không được đọc database khi owner bị giả mạo: {name}"
            )

    send_message = AsyncMock()
    monkeypatch.setattr(experiment, "send_message", send_message)

    response = await _post_training(
        db=DatabaseMustNotBeRead(),
        payload=_training_payload(owner_id=OTHER_OWNER_ID),
        key="training-owner-boundary-1",
    )

    assert response.status_code == 403
    assert response.json() == {
        "detail": {
            "code": "TRAINING_OWNER_MISMATCH",
            "message": "Không được tạo training job cho người dùng khác.",
        }
    }
    send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_training_rejects_dataset_owned_by_another_user(monkeypatch):
    jobs = FakeJobCollection()
    db = SimpleNamespace(
        tbl_User=SimpleNamespace(
            find_one=AsyncMock(return_value={"_id": OWNER_ID, "username": "owner"})
        ),
        tbl_Data=SimpleNamespace(
            find_one=AsyncMock(
                return_value={
                    "_id": DATASET_ID,
                    "dataName": "private-dataset",
                    "userId": OTHER_OWNER_ID,
                    "activate": 1,
                }
            )
        ),
        tbl_Job=jobs,
    )
    send_message = AsyncMock()
    monkeypatch.setattr(experiment, "send_message", send_message)

    response = await _post_training(
        db=db,
        payload=_training_payload(),
        key="training-dataset-boundary-1",
    )

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "TRAINING_DATASET_FORBIDDEN"
    assert jobs.documents == []
    send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_same_owner_key_and_payload_replays_one_job_and_one_kafka_message(
    monkeypatch,
):
    db = _database()
    send_message = AsyncMock()
    monkeypatch.setattr(experiment, "send_message", send_message)

    first = await _post_training(
        db=db,
        payload=_training_payload(),
        key="same-training-request-1",
    )
    replay = await _post_training(
        db=db,
        payload=_training_payload(),
        key="same-training-request-1",
    )

    assert first.status_code == replay.status_code == 200
    assert first.json()["job_id"] == replay.json()["job_id"]
    assert first.json()["replayed"] is False
    assert replay.json()["replayed"] is True
    assert len(db.tbl_Job.documents) == 1
    send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_replay_does_not_require_dataset_to_still_be_available(monkeypatch):
    db = _database()
    send_message = AsyncMock()
    monkeypatch.setattr(experiment, "send_message", send_message)

    first = await _post_training(
        db=db,
        payload=_training_payload(),
        key="durable-training-request-1",
    )
    db.tbl_Data.find_one.return_value = None
    replay = await _post_training(
        db=db,
        payload=_training_payload(),
        key="durable-training-request-1",
    )

    assert first.status_code == replay.status_code == 200
    assert replay.json()["job_id"] == first.json()["job_id"]
    assert replay.json()["replayed"] is True
    send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_same_key_is_isolated_between_authenticated_owners(monkeypatch):
    db = _database(dataset_owner="0")
    send_message = AsyncMock()
    monkeypatch.setattr(experiment, "send_message", send_message)

    owner_job = await _post_training(
        db=db,
        payload=_training_payload(owner_id=OWNER_ID),
        key="shared-owner-scoped-key",
        current_owner_id=OWNER_ID,
    )
    other_job = await _post_training(
        db=db,
        payload=_training_payload(owner_id=OTHER_OWNER_ID),
        key="shared-owner-scoped-key",
        current_owner_id=OTHER_OWNER_ID,
    )

    assert owner_job.status_code == other_job.status_code == 200
    assert owner_job.json()["job_id"] != other_job.json()["job_id"]
    assert len(db.tbl_Job.documents) == 2
    assert "shared-owner-scoped-key" not in str(db.tbl_Job.documents)
    assert send_message.await_count == 2


@pytest.mark.asyncio
async def test_missing_or_invalid_idempotency_key_is_rejected_before_database(
    monkeypatch,
):
    class DatabaseMustNotBeRead:
        def __getattr__(self, name):
            raise AssertionError(f"Header lỗi không được chạm database: {name}")

    send_message = AsyncMock()
    monkeypatch.setattr(experiment, "send_message", send_message)
    app = FastAPI()
    app.include_router(experiment.exp)
    app.dependency_overrides[get_db] = lambda: DatabaseMustNotBeRead()
    app.dependency_overrides[get_current_user] = lambda: {
        "_id": OWNER_ID,
        "role": "user",
    }
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        missing = await client.post(
            "/v2/auto/jobs/training",
            json=_training_payload(),
        )
        invalid = await client.post(
            "/v2/auto/jobs/training",
            json=_training_payload(),
            headers={"Idempotency-Key": "bad key with spaces"},
        )

    assert missing.status_code == invalid.status_code == 422
    send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_same_key_with_different_payload_returns_conflict(monkeypatch):
    db = _database()
    send_message = AsyncMock()
    monkeypatch.setattr(experiment, "send_message", send_message)
    changed_payload = _training_payload()
    changed_payload["config"]["metric_sort"] = "f1"

    first = await _post_training(
        db=db,
        payload=_training_payload(),
        key="conflicting-training-request-1",
    )
    conflict = await _post_training(
        db=db,
        payload=changed_payload,
        key="conflicting-training-request-1",
    )

    assert first.status_code == 200
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "TRAINING_IDEMPOTENCY_CONFLICT"
    assert len(db.tbl_Job.documents) == 1
    send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_concurrent_duplicate_requests_create_and_publish_once(monkeypatch):
    db = _database()
    send_message = AsyncMock()
    monkeypatch.setattr(experiment, "send_message", send_message)

    first, duplicate = await asyncio.gather(
        _post_training(
            db=db,
            payload=_training_payload(),
            key="concurrent-training-request-1",
        ),
        _post_training(
            db=db,
            payload=_training_payload(),
            key="concurrent-training-request-1",
        ),
    )

    assert first.status_code == duplicate.status_code == 200
    assert first.json()["job_id"] == duplicate.json()["job_id"]
    assert {first.json()["replayed"], duplicate.json()["replayed"]} == {
        False,
        True,
    }
    assert len(db.tbl_Job.documents) == 1
    send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_uncertain_kafka_outcome_is_not_republished_on_retry(
    monkeypatch,
    caplog,
):
    db = _database()
    upstream_detail = "kafka unavailable raw-detail-sentinel"
    send_message = AsyncMock(side_effect=ConnectionError(upstream_detail))
    monkeypatch.setattr(experiment, "send_message", send_message)
    caplog.set_level(logging.ERROR, logger="experiment")

    first = await _post_training(
        db=db,
        payload=_training_payload(),
        key="uncertain-training-request-1",
    )
    retry = await _post_training(
        db=db,
        payload=_training_payload(),
        key="uncertain-training-request-1",
    )

    assert first.status_code == 503
    assert first.json()["detail"]["code"] == "TRAINING_DISPATCH_UNCERTAIN"
    assert retry.status_code == 200
    assert retry.json()["status"] == "needs_reconciliation"
    assert retry.json()["job_id"] == first.json()["detail"]["job_id"]
    assert db.tbl_Job.documents[0]["dispatch"]["status"] == "needs_reconciliation"
    assert "uncertain-training-request-1" not in str(db.tbl_Job.documents[0])
    assert "uncertain-training-request-1" not in caplog.text
    assert upstream_detail not in caplog.text
    send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_training_does_not_fallback_to_legacy_mutation(monkeypatch):
    api_post = AsyncMock(side_effect=httpx.ConnectError("upstream unavailable"))
    monkeypatch.setattr(automl_tools, "_api_post", api_post)

    result = await automl_tools.start_training.ainvoke(
        {
            "user_id": OWNER_ID,
            "dataset_id": DATASET_ID,
            "problem_type": "classification",
            "target_column": "label",
            "list_feature": ["feature"],
            "idempotency_key": "journey-action-digest-1",
        }
    )

    assert '"error"' in result
    api_post.assert_awaited_once()
    assert api_post.await_args.kwargs["idempotency_key"] == "journey-action-digest-1"


@pytest.mark.asyncio
async def test_start_training_requires_action_idempotency_key(monkeypatch):
    api_post = AsyncMock()
    monkeypatch.setattr(automl_tools, "_api_post", api_post)

    schema = automl_tools.start_training.get_input_schema().model_json_schema()

    assert "idempotency_key" in schema["required"]
    api_post.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("HAGENT_TEST_MONGODB_URI"),
    reason="HAGENT_TEST_MONGODB_URI chưa được cấu hình",
)
async def test_real_mongodb_concurrent_requests_keep_one_job_and_no_raw_key(
    monkeypatch,
):
    uri = os.environ["HAGENT_TEST_MONGODB_URI"]
    db_name = f"training_idempotency_test_{uuid.uuid4().hex}"
    client = AsyncMongoClient(uri, serverSelectionTimeoutMS=2000)
    db = client[db_name]

    async def publish_once(*args, **kwargs):
        del args, kwargs
        await asyncio.sleep(0.05)

    send_message = AsyncMock(side_effect=publish_once)
    monkeypatch.setattr(experiment, "send_message", send_message)
    raw_key = "real-mongo-training-key-sentinel"
    try:
        await db.tbl_User.insert_one({"_id": ObjectId(OWNER_ID), "username": "owner"})
        await db.tbl_Data.insert_one(
            {
                "_id": ObjectId(DATASET_ID),
                "dataName": "dataset",
                "userId": OWNER_ID,
                "activate": 1,
            }
        )

        responses = await asyncio.gather(
            *(
                _post_training(
                    db=db,
                    payload=_training_payload(),
                    key=raw_key,
                )
                for _ in range(8)
            )
        )
        documents = await db.tbl_Job.find({}).to_list(length=None)
        raw_bson = b"".join(BSON.encode(document) for document in documents)
        assert {response.status_code for response in responses} == {200}
        assert len({response.json()["job_id"] for response in responses}) == 1
        assert sum(not response.json()["replayed"] for response in responses) == 1
        assert len(documents) == 1
        assert send_message.await_count == 1
        assert raw_key.encode() not in raw_bson
        assert str(documents[0]["_id"]).startswith("training-idempotency:")
    finally:
        await client.drop_database(db_name)
        await client.close()
