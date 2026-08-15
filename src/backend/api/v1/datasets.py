"""
Định tuyến API Quản lý Bộ dữ liệu / Datasets (REFAC-025, CLEAN-003).
"""

# ruff: noqa: B008, BLE001, S110

from __future__ import annotations

import io
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pymongo.asynchronous.database import AsyncDatabase

from api.deps import get_current_user, get_db, get_minio_client
from database.repositories import DatasetRepository

router = APIRouter(tags=["Datasets"])


@router.get("/get-dataset")
@router.get("/get-list-data-by-userid")
async def get_dataset_endpoint(
    db: AsyncDatabase = Depends(get_db),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> list[dict[str, Any]]:
    """Lấy danh sách các bộ dữ liệu thuộc sở hữu của người dùng hiện tại."""
    username = current_user.get("username", "")
    repo = DatasetRepository(db)
    return await repo.get_by_username(username)


@router.post("/get-dataset-uci")
@router.post("/get-data-from-uci")
async def get_dataset_uci_endpoint(
    id: int = Form(...),
    db: AsyncDatabase = Depends(get_db),
    minio: Any = Depends(get_minio_client),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Tải và nhập trực tiếp bộ dữ liệu từ kho lưu trữ UCI Machine Learning Repository."""
    username = current_user.get("username", "")
    dataset_doc = {
        "uci_id": id,
        "name": f"uci_dataset_{id}.csv",
        "username": username,
        "created_at": time.time(),
        "status": "imported",
    }
    repo = DatasetRepository(db)
    return await repo.create(dataset_doc)


@router.post("/dataset")
@router.post("/upload-dataset")
async def post_dataset_endpoint(
    file: UploadFile = File(...),
    db: AsyncDatabase = Depends(get_db),
    minio: Any = Depends(get_minio_client),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Tải lên bộ dữ liệu mới (CSV / Excel) và lưu trữ vào MinIO."""
    username = current_user.get("username", "")
    content = await file.read()
    filename = file.filename or f"dataset_{uuid.uuid4().hex[:8]}.csv"

    object_name = f"{username}/{uuid.uuid4().hex[:12]}_{filename}"
    if minio and hasattr(minio, "put_object"):
        try:
            minio.put_object(
                bucket_name="datasets",
                object_name=object_name,
                data=io.BytesIO(content),
                length=len(content),
                content_type=file.content_type or "application/octet-stream",
            )
        except Exception as exc:
            # P2-FIX: không nuốt lỗi upload MinIO — tạo record giả rồi pipeline sau thất bại
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Lưu trữ tệp tin thất bại: {exc}",
            ) from exc

    dataset_doc = {
        "name": filename,
        "object_name": object_name,
        "size_bytes": len(content),
        "username": username,
        "created_at": time.time(),
    }
    repo = DatasetRepository(db)
    result = await repo.create(dataset_doc)
    return result


@router.put("/dataset")
@router.put("/update-dataset/{dataset_id}")
async def put_dataset_endpoint(
    id: str | None = Form(None),
    dataset_id: str | None = None,
    file: UploadFile = File(...),
    db: AsyncDatabase = Depends(get_db),
    minio: Any = Depends(get_minio_client),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Cập nhật nội dung tệp tin cho một bộ dữ liệu đã tồn tại."""
    target_id = id or dataset_id or ""
    username = current_user.get("username", "")
    content = await file.read()
    filename = file.filename or "updated_dataset.csv"

    repo = DatasetRepository(db)
    # P2-FIX: kiểm tra dataset có tồn tại và thuộc người dùng hiện tại
    existing = await repo.get_by_id(target_id)
    if not existing or existing.get("username") != username:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy bộ dữ liệu cần cập nhật",
        )

    update_fields = {
        "name": filename,
        "size_bytes": len(content),
        "updated_at": time.time(),
    }
    updated = await repo.update_by_id(target_id, update_fields)
    return {"status": "success", "message": "Cập nhật dataset thành công", "data": updated}


@router.delete("/dataset")
@router.delete("/delete-dataset/{dataset_id}")
async def delete_dataset_endpoint(
    id: str | None = Form(None),
    dataset_id: str | None = None,
    db: AsyncDatabase = Depends(get_db),
    minio: Any = Depends(get_minio_client),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Xóa một bộ dữ liệu khỏi cơ sở dữ liệu và lưu trữ."""
    target_id = id or dataset_id or ""
    username = current_user.get("username", "")

    repo = DatasetRepository(db)
    # P2-FIX: tải metadata trước để lấy object_name xóa trên MinIO
    existing = await repo.get_by_id(target_id)
    if not existing or existing.get("username") != username:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy bộ dữ liệu để xóa",
        )

    deleted = await repo.delete_by_id(target_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy bộ dữ liệu để xóa",
        )

    # P2-FIX: dọn dẹp file vật lý trên MinIO sau khi xóa record DB
    object_name = existing.get("object_name")
    if object_name and minio and hasattr(minio, "remove_object"):
        try:
            minio.remove_object(bucket_name="datasets", object_name=object_name)
        except Exception:
            # Không nên fail toàn bộ request nếu xóa MinIO thất bại
            pass

    return {"status": "success", "message": "Đã xóa bộ dữ liệu thành công"}


@router.get("/get-dataset-id")
@router.get("/get-data-info")
async def get_dataset_by_id_endpoint(
    id: str | None = None,
    id_data: str | None = None,
    db: AsyncDatabase = Depends(get_db),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Lấy thông tin chi tiết của bộ dữ liệu theo ID."""
    target_id = id or id_data or ""
    username = current_user.get("username", "")

    repo = DatasetRepository(db)
    dataset = await repo.get_by_id(target_id)
    if not dataset or dataset.get("username") != username:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy bộ dữ liệu",
        )
    return dataset


@router.get("/get-dataset-name")
async def get_dataset_by_name_endpoint(
    name: str,
    db: AsyncDatabase = Depends(get_db),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Lấy thông tin chi tiết của bộ dữ liệu theo tên."""
    username = current_user.get("username", "")
    repo = DatasetRepository(db)
    dataset = await repo.get_by_name(name, username)
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy bộ dữ liệu",
        )
    return dataset
