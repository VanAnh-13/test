"""MongoDB-backed, owner-scoped fact persistence."""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

from pymongo import ASCENDING, DESCENDING, AsyncMongoClient, ReturnDocument

from hagent.agent.memory import Fact, FactStore


class MongoFactStore(FactStore):
    """Persist facts under the compound owner identity ``(user_id, key)``."""

    def __init__(
        self,
        connect: str | None = None,
        *,
        db_name: str = "hagent",
        collection_name: str = "memory_facts",
        collection: Any | None = None,
    ) -> None:
        self._client = None
        if collection is not None:
            self._collection = collection
        else:
            if not connect:
                raise ValueError("MongoFactStore requires a MongoDB connection")
            uri = connect if "://" in connect else f"mongodb://{connect}"
            self._client = AsyncMongoClient(uri, serverSelectionTimeoutMS=5000)
            self._collection = self._client[db_name][collection_name]
        self._indexes_ready = False
        self._index_lock = asyncio.Lock()

    async def _ensure_indexes(self) -> None:
        if self._indexes_ready:
            return
        async with self._index_lock:
            if self._indexes_ready:
                return
            await self._collection.create_index(
                [("user_id", ASCENDING), ("key", ASCENDING)],
                unique=True,
                name="memory_user_key_unique",
            )
            self._indexes_ready = True

    async def save(self, user_id: str, fact: Fact) -> None:
        await self._ensure_indexes()
        owner = str(user_id)
        updated_at = time.time()
        fact.updated_at = updated_at
        document = fact.to_dict()
        created_at = document.pop("created_at")
        document.update({"user_id": owner, "key": fact.key})
        await self._collection.update_one(
            {"user_id": owner, "key": fact.key},
            {
                "$set": document,
                "$setOnInsert": {"created_at": created_at},
            },
            upsert=True,
        )

    async def get(self, user_id: str, key: str) -> Fact | None:
        await self._ensure_indexes()
        document = await self._collection.find_one_and_update(
            {"user_id": str(user_id), "key": key},
            {"$inc": {"access_count": 1}},
            return_document=ReturnDocument.AFTER,
        )
        return Fact.from_dict(document) if document else None

    async def search(
        self,
        user_id: str,
        *,
        category: str | None = None,
        query: str | None = None,
        limit: int = 20,
    ) -> list[Fact]:
        await self._ensure_indexes()
        bounded_limit = max(0, int(limit))
        if bounded_limit == 0:
            return []
        mongo_query: dict[str, Any] = {"user_id": str(user_id)}
        if category:
            mongo_query["category"] = category
        if query:
            mongo_query["content"] = {
                "$regex": re.escape(query),
                "$options": "i",
            }
        cursor = self._collection.find(mongo_query).sort("updated_at", DESCENDING)
        cursor = cursor.limit(bounded_limit)
        return [Fact.from_dict(document) async for document in cursor]

    async def get_all(self, user_id: str) -> list[Fact]:
        await self._ensure_indexes()
        cursor = self._collection.find({"user_id": str(user_id)}).sort(
            "updated_at", DESCENDING
        )
        return [Fact.from_dict(document) async for document in cursor]

    async def delete(self, user_id: str, key: str) -> bool:
        await self._ensure_indexes()
        result = await self._collection.delete_one(
            {"user_id": str(user_id), "key": key}
        )
        return bool(result.deleted_count)

    async def clear(self, user_id: str) -> int:
        await self._ensure_indexes()
        result = await self._collection.delete_many({"user_id": str(user_id)})
        return int(result.deleted_count)
