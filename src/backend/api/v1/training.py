# ruff: noqa: B008
"""
Định tuyến API Khởi chạy và Quản lý Tiến trình Huấn luyện Mô hình (REFAC-025).
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Form,
    HTTPException,
    Query,
    Request,
    status,
)
from pymongo.asynchronous.database import AsyncDatabase

from api.deps import get_current_user, get_db, get_kafka_producer, get_minio_client
from database.repositories import JobRepository

router = APIRouter(tags=["Training"])

# Trạng thái khởi tạo của một job huấn luyện (0 = đang chờ xử lý / pending).
_JOB_STATUS_PENDING = 0


@router.post("/train")
@router.post("/training-file-local")
async def train_endpoint(
    request: Request,
    background_tasks: BackgroundTasks,
    id: str = Form(...),
    target: str = Form(...),
    features: str = Form(...),
    problem_type: str = Form(...),
    model: str = Form(...),
    time_limit: int = Form(...),
    metric: str = Form(...),
    minio: Any = Depends(get_minio_client),
    db: AsyncDatabase = Depends(get_db),
    producer: Any = Depends(get_kafka_producer),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Khởi tạo tiến trình huấn luyện mô hình AutoML từ form data."""
    username = current_user.get("username", "")
    job_id = str(uuid.uuid4())
    job_doc = {
        "job_id": job_id,
        "dataset_id": id,
        "target": target,
        "features": features,
        "problem_type": problem_type,
        "model": model,
        "time_limit": time_limit,
        "metric": metric,
        "username": username,
        "status": _JOB_STATUS_PENDING,
        "created_at": time.time(),
    }
    repo = JobRepository(db)
    created = await repo.create(job_doc)
    return {"status": "success", "job_id": job_id, "data": created}


@router.post("/train-json")
@router.post("/train-from-requestbody-json/")
async def train_json_endpoint(
    request: Request,
    background_tasks: BackgroundTasks,
    payload: dict[str, Any],
    minio: Any = Depends(get_minio_client),
    db: AsyncDatabase = Depends(get_db),
    producer: Any = Depends(get_kafka_producer),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Khởi tạo tiến trình huấn luyện mô hình AutoML từ payload JSON."""
    username = current_user.get("username", "")
    job_id = str(uuid.uuid4())
    job_doc = {
        "job_id": job_id,
        "payload": payload,
        "username": username,
        "status": _JOB_STATUS_PENDING,
        "created_at": time.time(),
    }
    repo = JobRepository(db)
    created = await repo.create(job_doc)
    return {"status": "success", "job_id": job_id, "data": created}


@router.post("/get-job-info")
async def get_job_info_endpoint(
    id: str = Query(...),
    db: AsyncDatabase = Depends(get_db),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Lấy thông tin chi tiết và tiến độ của một job huấn luyện."""
    username = current_user.get("username", "")
    repo = JobRepository(db)
    job = await repo.get_for_user(username, id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy tiến trình huấn luyện",
        )
    return job


@router.get("/get-all-jobs")
@router.get("/get-list-job-by-userId")
async def get_all_jobs_endpoint(
    db: AsyncDatabase = Depends(get_db),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> list[dict[str, Any]]:
    """Lấy toàn bộ lịch sử các job huấn luyện của người dùng hiện tại."""
    username = current_user.get("username", "")
    repo = JobRepository(db)
    return await repo.get_by_username(username)


@router.delete("/delete-job")
async def delete_job_endpoint(
    id: str = Query(...),
    db: AsyncDatabase = Depends(get_db),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Xóa một job huấn luyện và các artifact kết quả liên quan."""
    username = current_user.get("username", "")
    repo = JobRepository(db)
    deleted = await repo.delete_for_user(username, id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy job để xóa",
        )
    return {"status": "success", "message": "Đã xóa tiến trình huấn luyện"}
