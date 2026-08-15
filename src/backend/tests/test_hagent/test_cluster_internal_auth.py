"""
Kiểm thử đơn vị cho cơ chế xác thực shared-secret giữa Master/Worker (AUDIT-002).

Bối cảnh lỗi: các endpoint nội bộ /task/get, /task/submit (automl/v2/master.py)
và /check-for-work, /cancel-task (cluster/worker.py) trước đây KHÔNG có bất kỳ
lớp xác thực nào — bất kỳ ai truy cập được mạng nội bộ (hoặc public nếu lộ ra)
đều có thể cướp/giả mạo task hoặc phá job đang chạy.
"""

from __future__ import annotations

import os

os.environ.setdefault("MINIO_ENDPOINT", "localhost:9000")
os.environ.setdefault("MINIO_ACCESS_KEY", "minioadmin")
os.environ.setdefault("MINIO_SECRET_KEY", "minioadmin")
os.environ.setdefault("DATABASE_URI", "mongodb://localhost:27017")
os.environ.setdefault("SECRET_KEY", "testsecretkey1234567890123456789012")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from automl.v2 import master
from cluster import worker


def _master_client() -> TestClient:
    app = FastAPI()
    app.include_router(master.master)
    return TestClient(app)


def _worker_client() -> TestClient:
    return TestClient(worker.app)


def test_master_task_get_rejects_missing_secret() -> None:
    client = _master_client()
    response = client.get("/task/get")
    assert response.status_code == 401


def test_master_task_get_rejects_wrong_secret() -> None:
    client = _master_client()
    response = client.get("/task/get", headers={"X-Cluster-Secret": "wrong-secret"})
    assert response.status_code == 401


def test_master_task_get_accepts_correct_secret() -> None:
    client = _master_client()
    response = client.get(
        "/task/get", headers={"X-Cluster-Secret": master.CLUSTER_SHARED_SECRET}
    )
    assert response.status_code == 200
    assert "task" in response.json()


def test_master_task_submit_rejects_missing_secret() -> None:
    client = _master_client()
    response = client.post("/task/submit", json={"job_id": "job-x", "model_name": "m"})
    assert response.status_code == 401


def test_worker_check_for_work_rejects_missing_secret() -> None:
    client = _worker_client()
    response = client.get("/check-for-work")
    assert response.status_code == 401


def test_worker_check_for_work_accepts_correct_secret() -> None:
    client = _worker_client()
    response = client.get(
        "/check-for-work", headers={"X-Cluster-Secret": worker.CLUSTER_SHARED_SECRET}
    )
    assert response.status_code == 200
    assert response.json() == {"status": "starting"}


def test_worker_cancel_task_rejects_missing_or_wrong_secret() -> None:
    client = _worker_client()
    assert client.post("/cancel-task", params={"task_id": "t1"}).status_code == 401
    assert (
        client.post(
            "/cancel-task",
            params={"task_id": "t1"},
            headers={"X-Cluster-Secret": "wrong-secret"},
        ).status_code
        == 401
    )


def test_worker_cancel_task_accepts_correct_secret() -> None:
    client = _worker_client()
    response = client.post(
        "/cancel-task",
        params={"task_id": "unknown-task"},
        headers={"X-Cluster-Secret": worker.CLUSTER_SHARED_SECRET},
    )
    assert response.status_code == 200
    assert response.json() == {"status": "not_found"}


def test_worker_health_endpoint_stays_public() -> None:
    """Health-check không phải endpoint nội bộ nhạy cảm nên vẫn không yêu cầu secret."""
    client = _worker_client()
    response = client.get("/health")
    assert response.status_code == 200
