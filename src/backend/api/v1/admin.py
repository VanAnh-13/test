"""
Định tuyến API Quản trị Hệ thống / Admin (REFAC-025).
"""

# ruff: noqa: B008

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pymongo.asynchronous.database import AsyncDatabase

from api.deps import get_current_user, get_db
from database.repositories import DatasetRepository

router = APIRouter(tags=["Admin"])


@router.get("/get-all-dataset")
@router.get("/get-list-data-user")
async def get_all_dataset_admin_endpoint(
    db: AsyncDatabase = Depends(get_db),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> list[dict[str, Any]]:
    """Lấy danh sách tất cả các bộ dữ liệu của toàn bộ người dùng (chỉ dành cho tài khoản Admin)."""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Không có quyền truy cập",
        )
    repo = DatasetRepository(db)
    return await repo.list_all()
