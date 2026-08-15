"""WorldStateStore lưu tài liệu world bền vững theo từng người dùng."""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any

import structlog

try:
    from pymongo import ReturnDocument
except ImportError:  # pragma: no cover

    class ReturnDocument:  # type: ignore[no-redef]
        AFTER = True
        BEFORE = False


from .schema import WorldState, utc_now
from .schema_migration import migrate_world_state_doc

logger = structlog.get_logger(__name__)

_WORLD_STATE_FIELDS = {f.name for f in dataclasses.fields(WorldState)}


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


def _world_state_from_doc(doc: dict[str, Any]) -> WorldState:
    migrated = migrate_world_state_doc(doc)
    clean_doc = {k: v for k, v in migrated.items() if k in _WORLD_STATE_FIELDS}
    return WorldState(**clean_doc)


class WorldStateStore:
    def __init__(
        self,
        client: Any,
        db_name: str,
        collection_name: str,
        ttl_seconds: int,
    ):
        self.client = client
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.ttl_seconds = ttl_seconds

    async def ensure_indexes(self):
        await _maybe_await(self.collection.create_index("user_id", unique=True))
        if self.ttl_seconds > 0:
            await _maybe_await(
                self.collection.create_index(
                    "updated_at", expireAfterSeconds=self.ttl_seconds
                )
            )

    async def ensure(self, user_id: str) -> None:
        """Bảo đảm người dùng có một tài liệu world state."""
        now = utc_now()
        await _maybe_await(
            self.collection.update_one(
                {"user_id": user_id},
                {
                    "$setOnInsert": {
                        "created_at": now,
                        "updated_at": now,
                        "datasets": {},
                        "jobs": {},
                        "goals": [],
                        "plans": {},
                        "phase": "idle",
                        "cost_metrics": {},
                    }
                },
                upsert=True,
            )
        )

    async def get(self, user_id: str) -> WorldState | None:
        doc = await _maybe_await(self.collection.find_one({"user_id": user_id}))
        if doc:
            return _world_state_from_doc(doc)
        return None

    async def get_snapshot(self, user_id: str) -> dict[str, Any] | None:
        """
        Snapshot sẵn sàng cho JSON để chat_router và middleware sử dụng.

        Trả về None nếu người dùng chưa có tài liệu.
        """
        state = await self.get(user_id)
        if state is None:
            return None
        return state.to_dict()

    async def upsert(self, user_id: str, patch: dict[str, Any]) -> WorldState | None:
        """Gộp các trường patch vào tài liệu world state của người dùng."""
        # Chỉ cho phép các trường đã biết và luôn đặt updated_at.
        allowed = {k: v for k, v in patch.items() if k in _WORLD_STATE_FIELDS}
        allowed["updated_at"] = utc_now()

        updated_doc = await _maybe_await(
            self.collection.find_one_and_update(
                {"user_id": user_id},
                {"$set": allowed},
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
        )
        if updated_doc:
            return _world_state_from_doc(updated_doc)
        return None

    async def apply_patch(
        self, user_id: str, patch: dict[str, Any]
    ) -> WorldState | None:
        """Alias có ngữ nghĩa rõ hơn cho middleware."""
        return await self.upsert(user_id, patch)


def create_world_state_store(
    client: Any,
    *,
    db_name: str | None = None,
    collection_name: str | None = None,
    ttl_seconds: int | None = None,
) -> WorldStateStore:
    """Factory lấy mọi giá trị mặc định từ cấu hình, không mã hóa cứng giá trị sản phẩm."""
    from hagent.bridge.config import get_mongodb_config, get_world_state_config

    mongo = get_mongodb_config()
    ws = get_world_state_config()
    return WorldStateStore(
        client=client,
        db_name=db_name or mongo["db_name"],
        collection_name=collection_name or ws["collection_name"],
        ttl_seconds=ttl_seconds if ttl_seconds is not None else ws["ttl_seconds"],
    )
