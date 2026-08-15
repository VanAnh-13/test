"""
Định tuyến API Danh sách Thuật toán (REFAC-025).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Path, status

from automl.v2.master import get_models

router = APIRouter(tags=["Models"])


@router.get("/get-models")
async def get_models_endpoint(
    type: str = "classification",
) -> list[str]:
    """Lấy danh sách các thuật toán máy học khả dụng theo loại bài toán (truyền qua query param)."""
    return get_models(type)


@router.get("/api/v1/available-models/{problem_type}")
async def get_available_models_by_path_endpoint(
    problem_type: str = Path(..., description="classification hoặc regression"),
) -> dict[str, Any]:
    """Lấy danh sách các thuật toán máy học theo chuẩn RESTful path parameter."""
    if problem_type not in {"classification", "regression"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Loại bài toán không hợp lệ. Chỉ chấp nhận 'classification' hoặc 'regression'",
        )
    models_list = get_models(problem_type)
    return {
        "problem_type": problem_type,
        "models": models_list,
        "count": len(models_list),
    }
