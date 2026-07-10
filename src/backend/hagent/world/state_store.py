"""WorldStateStore — durable per-user world documents."""

from __future__ import annotations

import dataclasses
import inspect
import logging
from typing import Any, Dict, Optional

try:
    from pymongo import ReturnDocument
except ImportError:  # pragma: no cover
    class ReturnDocument:  # type: ignore[no-redef]
        AFTER = True
        BEFORE = False

from .schema import WorldState, utc_now

logger = logging.getLogger(__name__)

_WORLD_STATE_FIELDS = {f.name for f in dataclasses.fields(WorldState)}


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


def _world_state_from_doc(doc: Dict[str, Any]) -> WorldState:
    clean_doc = {k: v for k, v in doc.items() if k in _WORLD_STATE_FIELDS}
    # Defaults for newly added fields when reading old documents
    clean_doc.setdefault("plans", {})
    clean_doc.setdefault("goals", [])
    clean_doc.setdefault("cost_metrics", {})
    clean_doc.setdefault("phase", "idle")
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
        """Ensure a world state document exists for user."""
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

    async def get(self, user_id: str) -> Optional[WorldState]:
        doc = await _maybe_await(self.collection.find_one({"user_id": user_id}))
        if doc:
            return _world_state_from_doc(doc)
        return None

    async def get_snapshot(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        JSON-ready snapshot for chat_router / middleware.

        Returns None if user has no document yet.
        """
        state = await self.get(user_id)
        if state is None:
            return None
        return state.to_dict()

    async def upsert(self, user_id: str, patch: Dict[str, Any]) -> Optional[WorldState]:
        """Merge patch fields into the user world state document."""
        patch_with_timestamp = {
            k: v for k, v in patch.items() if k in _WORLD_STATE_FIELDS or k == "updated_at"
        }
        # Allow only known fields (+ updated_at always set)
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

    async def apply_patch(self, user_id: str, patch: Dict[str, Any]) -> Optional[WorldState]:
        """Alias with clearer semantics for middleware."""
        return await self.upsert(user_id, patch)


def create_world_state_store(
    client: Any,
    *,
    db_name: str | None = None,
    collection_name: str | None = None,
    ttl_seconds: int | None = None,
) -> WorldStateStore:
    """Factory — all defaults from config, no hard-coded product values."""
    from hagent.bridge.config import get_mongodb_config, get_world_state_config

    mongo = get_mongodb_config()
    ws = get_world_state_config()
    return WorldStateStore(
        client=client,
        db_name=db_name or mongo["db_name"],
        collection_name=collection_name or ws["collection_name"],
        ttl_seconds=ttl_seconds if ttl_seconds is not None else ws["ttl_seconds"],
    )
