"""
Định tuyến API Xác thực người dùng (REFAC-025).
"""

from __future__ import annotations

from users.routers import router as auth_router

router = auth_router

__all__ = ["router"]
