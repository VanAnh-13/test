"""
Định tuyến API Quản lý Người dùng (REFAC-025).
"""

# ruff: noqa: B008

from __future__ import annotations

from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from pymongo.asynchronous.database import AsyncDatabase

from api.deps import get_current_user, get_db
from users.engine import (
    UpdateUser,
    check_exits_username,
    get_list_user,
    handle_change_password,
    handle_contact,
    handle_delete_user,
    handle_get_avatar,
    handle_update_avatar,
    handle_update_user,
    user_helper,
)
from users.schema import ResetPasswordRequest

router = APIRouter(tags=["Users"])


@router.get("/users")
async def get_users(
    db: AsyncDatabase = Depends(get_db),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> list[dict[str, Any]]:
    """Lấy danh sách tất cả người dùng (yêu cầu quyền admin)."""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Không có quyền truy cập",
        )
    return await get_list_user(db)


@router.get("/users/")
async def get_user(
    username: str = Query(...),
    db: AsyncDatabase = Depends(get_db),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Lấy thông tin chi tiết của một người dùng theo username."""
    # P1-FIX: kiểm tra đúng logic — chỉ admin mới truy cập tài khoản người khác
    if current_user.get("role") != "admin" and current_user.get("username") != username:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Không có quyền truy cập",
        )
    # P0-FIX: thứ tự đúng là (username, db)
    user = await check_exits_username(username, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy người dùng",
        )
    return user_helper(user)


@router.put("/users/{username}")
@router.put("/update/{username}")
async def update_user(
    username: str,
    user_data: UpdateUser,
    db: AsyncDatabase = Depends(get_db),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Cập nhật thông tin tài khoản người dùng."""
    # P1-FIX: chỉ admin mới cập nhật tài khoản người khác
    if current_user.get("role") != "admin" and current_user.get("username") != username:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Không có quyền truy cập",
        )
    # P0-FIX: thứ tự đúng là (username, new_user, db)
    res = await handle_update_user(username, user_data, db)
    if res.get("status") == "error":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=res.get("message", "Cập nhật thông tin thất bại"),
        )
    return res


@router.delete("/users/{username}")
@router.delete("/delete/{username}")
async def delete_user(
    username: str,
    db: AsyncDatabase = Depends(get_db),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Xóa tài khoản người dùng khỏi hệ thống."""
    # P1-FIX: chỉ admin hoặc chính user đó mới xóa được
    if current_user.get("role") != "admin" and current_user.get("username") != username:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Không có quyền truy cập",
        )
    # P0-FIX: thứ tự đúng là (username, db)
    return await handle_delete_user(username, db)


@router.post("/change-password")
async def change_password(
    data: ResetPasswordRequest,
    db: AsyncDatabase = Depends(get_db),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Đổi mật khẩu người dùng."""
    # P0-FIX: cần nạp đối tượng user đầy đủ trước, thứ tự đúng: (user, current_pwd, new_pwd, db)
    username = current_user.get("username", "")
    user_doc = await check_exits_username(username, db)
    if not user_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy người dùng",
        )
    return await handle_change_password(user_doc, data.current_password, data.new_password, db)


@router.post("/user/avatar")
@router.post("/update_avatar")
async def update_avatar(
    file: UploadFile = File(...),
    db: AsyncDatabase = Depends(get_db),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Cập nhật ảnh đại diện người dùng."""
    # P0-FIX: thứ tự đúng là (username, avatar, db)
    return await handle_update_avatar(current_user.get("username"), file, db)


@router.get("/user/avatar")
@router.get("/get_avatar/{username}")
async def get_avatar(
    username: str | None = None,
    db: AsyncDatabase = Depends(get_db),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> Any:
    """Lấy ảnh đại diện hiện tại của người dùng."""
    target_user = username or current_user.get("username")
    # P0-FIX: thứ tự đúng là (username, db)
    return await handle_get_avatar(target_user, db)


@router.post("/contact")
async def post_contact(
    name: str = Form(...),
    email: str = Form(...),
    subject: str = Form(...),
    message: str = Form(...),
    db: AsyncDatabase = Depends(get_db),
) -> dict[str, Any]:
    """Gửi liên hệ / phản hồi tới ban quản trị."""
    # P0-FIX: handle_contact nhận 4 tham số (fullname, email, message, db) — bỏ subject
    # subject được ghép vào message để bảo toàn thông tin
    full_message = f"[{subject}] {message}" if subject else message
    return await handle_contact(name, email, full_message, db)
