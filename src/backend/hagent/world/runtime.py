"""
Các factory dùng chung cho runtime agent: WorldModelService và WorldStateStore.

Giữ phần liên kết Mongo bên ngoài các node graph; nơi gọi truyền client khi có.
"""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def build_wm_runtime(
    *,
    mongo_client: Any | None = None,
    db_name: str | None = None,
    world_model_config: dict | None = None,
) -> tuple[Any, Any | None]:
    """
    Trả về cặp (WorldModelService, WorldStateStore | None).

    WorldStateStore là None khi không có mongo_client ở chế độ ngoại tuyến hoặc unit test.
    """
    from hagent.world.service import WorldModelService

    wm = WorldModelService.from_config(
        world_model_config,
        mongo_client=mongo_client,
        db_name=db_name,
    )

    store = None
    if mongo_client is not None:
        try:
            from hagent.world.state_store import create_world_state_store

            store = create_world_state_store(mongo_client, db_name=db_name)
        except Exception as exc:  # noqa: BLE001 - kho World State là dependency tùy chọn
            logger.debug("WorldStateStore create failed: %s", exc)
    return wm, store


def try_mongo_from_db(db: Any) -> tuple[Any | None, str | None]:
    """Trích xuất (client, db_name) từ AsyncDatabase hoặc Database của pymongo."""
    client = getattr(db, "client", None)
    name = getattr(db, "name", None)
    return client, str(name) if name else None
