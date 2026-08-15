"""AUDIT-002: Cluster Internal Auth — shared-secret giữa Master và Worker.

Bảo vệ các endpoint nội bộ (/task/get, /task/submit phía Master;
/check-for-work, /cancel-task phía Worker) khỏi truy cập không xác thực
(trước đây bất kỳ ai cũng gọi được, dẫn tới nguy cơ cướp/giả mạo task).

Module này là nguồn dùng chung duy nhất cho automl/v2/master.py và
cluster/worker.py — hai process riêng biệt phải cấu hình cùng một
CLUSTER_SHARED_SECRET.
"""

import os
import secrets

from dotenv import load_dotenv
from fastapi import Header, HTTPException

# Nạp .env tại đây vì module được import trước khi entrypoint (server, worker)
# chạy load_dotenv của riêng nó; secret phải sẵn sàng ngay lúc import.
load_dotenv()

_CLUSTER_SECRET_PLACEHOLDER = "change-me-cluster-shared-secret"
DEPLOY_MODE = os.getenv("DEPLOY_MODE", "development").strip().lower()
CLUSTER_SHARED_SECRET = os.getenv("CLUSTER_SHARED_SECRET", _CLUSTER_SECRET_PLACEHOLDER)
if (
    DEPLOY_MODE in {"private", "public"}
    and CLUSTER_SHARED_SECRET == _CLUSTER_SECRET_PLACEHOLDER
):
    raise RuntimeError(
        "[AUDIT-002] Biến môi trường 'CLUSTER_SHARED_SECRET' bắt buộc phải được cấu hình "
        f"(khác giá trị mặc định) khi DEPLOY_MODE='{DEPLOY_MODE}'. Secret này phải được "
        "chia sẻ giữa Master và mọi Worker để bảo vệ các endpoint nội bộ của cluster."
    )


def cluster_auth_headers() -> dict[str, str]:
    """Header đính kèm khi Master gọi Worker hoặc Worker gọi Master."""
    return {"X-Cluster-Secret": CLUSTER_SHARED_SECRET}


async def verify_cluster_secret(
    x_cluster_secret: str | None = Header(default=None, alias="X-Cluster-Secret"),
) -> None:
    """AUDIT-002 FIX: chỉ bên biết CLUSTER_SHARED_SECRET mới được gọi endpoint nội bộ."""
    if not x_cluster_secret or not secrets.compare_digest(
        x_cluster_secret, CLUSTER_SHARED_SECRET
    ):
        raise HTTPException(status_code=401, detail="Unauthorized cluster request")
