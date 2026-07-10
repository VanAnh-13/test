"""
Trajectory store — offline (o, a, o', z, ẑ, surprise) for JEPA-style learning later.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, List, Optional

from hagent.world.schema import (
    AutoMLAction,
    AutoMLObservation,
    LatentState,
    SurpriseResult,
    utc_now,
)

logger = logging.getLogger(__name__)


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


class TrajectoryStore:
    """Persist world-model transitions. In-memory fallback when no Mongo client."""

    def __init__(
        self,
        *,
        collection: Any | None = None,
        max_per_user: int = 5000,
        enabled: bool = True,
    ):
        self.collection = collection
        self.max_per_user = max_per_user
        self.enabled = enabled
        self._memory: Dict[str, List[Dict[str, Any]]] = {}

    async def append(
        self,
        *,
        user_id: str,
        observation: AutoMLObservation,
        action: AutoMLAction,
        next_observation: AutoMLObservation,
        z: LatentState,
        z_hat: LatentState,
        z_next: LatentState,
        surprise: SurpriseResult,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {}

        doc = {
            "user_id": user_id,
            "observation": observation.to_dict(),
            "action": action.to_dict(),
            "next_observation": next_observation.to_dict(),
            "z": z.to_dict(),
            "z_hat": z_hat.to_dict(),
            "z_next": z_next.to_dict(),
            "surprise": surprise.to_dict(),
            "created_at": utc_now().isoformat(),
        }

        if self.collection is not None:
            try:
                await _maybe_await(self.collection.insert_one(doc))
            except Exception as exc:
                logger.warning("Trajectory Mongo insert failed: %s", exc)
                self._append_memory(user_id, doc)
        else:
            self._append_memory(user_id, doc)
        return doc

    def _append_memory(self, user_id: str, doc: Dict[str, Any]) -> None:
        bucket = self._memory.setdefault(user_id, [])
        bucket.append(doc)
        if len(bucket) > self.max_per_user:
            del bucket[: len(bucket) - self.max_per_user]

    async def list_recent(
        self, user_id: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        if self.collection is not None:
            try:
                cursor = self.collection.find({"user_id": user_id}).sort(
                    "created_at", -1
                ).limit(limit)
                if inspect.isawaitable(cursor):
                    cursor = await cursor
                results = []
                async for doc in cursor:  # type: ignore[union-attr]
                    doc.pop("_id", None)
                    results.append(doc)
                return results
            except TypeError:
                # Sync cursor
                results = []
                for doc in cursor:
                    doc.pop("_id", None)
                    results.append(doc)
                return results
            except Exception as exc:
                logger.debug("Trajectory list fallback to memory: %s", exc)
        return list(reversed(self._memory.get(user_id, [])[-limit:]))
