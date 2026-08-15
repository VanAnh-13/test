"""
API v1 Router Tổng hợp (REFAC-025).

Gom toàn bộ các route handlers riêng biệt thành một APIRouter duy nhất.
"""

from __future__ import annotations

from fastapi import APIRouter

from api.v1.admin import router as admin_router
from api.v1.auth import router as auth_router
from api.v1.datasets import router as datasets_router
from api.v1.models import router as models_router
from api.v1.training import router as training_router
from api.v1.users import router as users_router

api_v1_router = APIRouter()

# Tích hợp các router chức năng
api_v1_router.include_router(auth_router)
api_v1_router.include_router(users_router)
api_v1_router.include_router(datasets_router)
api_v1_router.include_router(training_router)
api_v1_router.include_router(models_router)
api_v1_router.include_router(admin_router)

__all__ = ["api_v1_router"]
