"""
Công cụ migration schema cho các tài liệu World Model đã lưu bền vững.

Nâng cấp tài liệu MongoDB cũ có schema_version < 1.0 hoặc chưa có phiên bản
lên schema chuẩn hiện tại là "1.0".
"""

from __future__ import annotations

from typing import Any

import structlog

from hagent.world.schema import CURRENT_SCHEMA_VERSION

logger = structlog.get_logger("hagent.world.schema_migration")


def migrate_world_state_doc(doc: dict[str, Any]) -> dict[str, Any]:
    """
    Nâng cấp tài liệu world_state MongoDB thô lên CURRENT_SCHEMA_VERSION.

    Xử lý:
    - Thiếu schema_version ở tài liệu 0.x cũ.
    - Phiên bản cũ như "0.1" hoặc "0.9".
    - Giá trị mặc định cho collection và metadata mới thêm.
    """
    if not isinstance(doc, dict):
        return doc

    migrated = dict(doc)
    doc_version = str(migrated.get("schema_version") or "0.0")

    if doc_version != CURRENT_SCHEMA_VERSION:
        logger.debug(
            "Migrating WorldState doc for user %s from version %s to %s",
            migrated.get("user_id"),
            doc_version,
            CURRENT_SCHEMA_VERSION,
        )
        migrated["schema_version"] = CURRENT_SCHEMA_VERSION

    # Bảo đảm đủ các trường mặc định bắt buộc của phiên bản 1.0.
    migrated.setdefault("datasets", {})
    migrated.setdefault("jobs", {})
    migrated.setdefault("goals", [])
    migrated.setdefault("plans", {})
    migrated.setdefault("phase", "idle")
    migrated.setdefault("cost_metrics", {})
    migrated.setdefault("active_plan_id", None)
    migrated.setdefault("active_dataset_id", None)
    migrated.setdefault("active_job_id", None)
    migrated.setdefault("active_goal", None)
    migrated.setdefault("last_verification", None)
    migrated.setdefault("last_surprise", None)

    return migrated


def migrate_trajectory_doc(doc: dict[str, Any]) -> dict[str, Any]:
    """
    Nâng cấp tài liệu trajectory MongoDB thô lên CURRENT_SCHEMA_VERSION.
    """
    if not isinstance(doc, dict):
        return doc

    migrated = dict(doc)
    doc_version = str(migrated.get("schema_version") or "0.0")

    if doc_version != CURRENT_SCHEMA_VERSION:
        migrated["schema_version"] = CURRENT_SCHEMA_VERSION

    # Gắn schema_version cho các tài liệu con hiện có.
    for sub_key in (
        "observation",
        "action",
        "next_observation",
        "z",
        "z_hat",
        "z_next",
        "surprise",
    ):
        sub_val = migrated.get(sub_key)
        if isinstance(sub_val, dict) and "schema_version" not in sub_val:
            migrated[sub_key] = {**sub_val, "schema_version": CURRENT_SCHEMA_VERSION}

    return migrated


def migrate(doc: dict[str, Any], doc_type: str = "world_state") -> dict[str, Any]:
    """
    Bộ điều phối migration schema tổng quát.

    Tham số:
        doc: Dict tài liệu từ MongoDB.
        doc_type: "world_state" hoặc "trajectory".

    Giá trị trả về:
        Dict tài liệu đã migration.
    """
    if doc_type == "trajectory":
        return migrate_trajectory_doc(doc)
    return migrate_world_state_doc(doc)
