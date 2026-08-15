"""
Kho trajectory ngoại tuyến (o, a, o', z, ẑ, surprise) cho học theo kiểu JEPA về sau.
"""

from __future__ import annotations

import inspect
from typing import Any

import structlog

from hagent.world.schema import (
    AutoMLAction,
    AutoMLObservation,
    LatentState,
    SurpriseResult,
    utc_now,
)
from hagent.world.schema_migration import migrate_trajectory_doc

logger = structlog.get_logger(__name__)


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


class TrajectoryStore:
    """Lưu bền vững transition của World Model; dùng bộ nhớ khi không có Mongo client."""

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
        self._memory: dict[str, list[dict[str, Any]]] = {}

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
    ) -> dict[str, Any]:
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
            "schema_version": "1.0",
        }

        if self.collection is not None:
            try:
                await _maybe_await(self.collection.insert_one(doc))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Trajectory Mongo insert failed: %s", exc)
                self._append_memory(user_id, doc)
        else:
            self._append_memory(user_id, doc)
        return doc

    def _append_memory(self, user_id: str, doc: dict[str, Any]) -> None:
        bucket = self._memory.setdefault(user_id, [])
        bucket.append(doc)
        if len(bucket) > self.max_per_user:
            del bucket[: len(bucket) - self.max_per_user]

    async def list_recent(self, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        if self.collection is not None:
            try:
                cursor = (
                    self.collection.find({"user_id": user_id})
                    .sort("created_at", -1)
                    .limit(limit)
                )
                if inspect.isawaitable(cursor):
                    cursor = await cursor
                results = []
                async for doc in cursor:  # type: ignore[union-attr]
                    doc.pop("_id", None)
                    results.append(migrate_trajectory_doc(doc))
                return results
            except TypeError:
                # Sync cursor
                results = []
                for doc in cursor:
                    doc.pop("_id", None)
                    results.append(migrate_trajectory_doc(doc))
                return results
            except Exception as exc:  # noqa: BLE001
                logger.debug("Trajectory list fallback to memory: %s", exc)
        return [
            migrate_trajectory_doc(d)
            for d in reversed(self._memory.get(user_id, [])[-limit:])
        ]

    async def list_all(
        self, *, user_id: str | None = None, limit: int = 10000
    ) -> list[dict[str, Any]]:
        """Tải trajectory để huấn luyện ngoại tuyến cho một hoặc mọi người dùng."""
        if self.collection is not None:
            try:
                query: dict[str, Any] = {}
                if user_id:
                    query["user_id"] = user_id
                cursor = self.collection.find(query).sort("created_at", -1).limit(limit)
                if inspect.isawaitable(cursor):
                    cursor = await cursor
                results: list[dict[str, Any]] = []
                try:
                    async for doc in cursor:  # type: ignore[union-attr]
                        doc.pop("_id", None)
                        results.append(migrate_trajectory_doc(doc))
                    return results
                except TypeError:
                    for doc in cursor:
                        doc.pop("_id", None)
                        results.append(migrate_trajectory_doc(doc))
                    return results
            except Exception as exc:  # noqa: BLE001
                logger.debug("Trajectory list_all fallback to memory: %s", exc)
        if user_id:
            return [
                migrate_trajectory_doc(d)
                for d in self._memory.get(user_id, [])[-limit:]
            ]
        all_docs: list[dict[str, Any]] = []
        for bucket in self._memory.values():
            all_docs.extend([migrate_trajectory_doc(d) for d in bucket])
        return all_docs[-limit:]


def create_trajectory_store(
    client: Any | None = None,
    *,
    db_name: str | None = None,
    collection_name: str | None = None,
    max_per_user: int | None = None,
    enabled: bool | None = None,
) -> TrajectoryStore:
    """
    Factory dùng collection Mongo khi có client, nếu không thì dùng bộ nhớ.

    Giá trị mặc định lấy từ world_model.trajectory trong hagent.yaml.
    """
    traj_cfg: dict[str, Any] = {}
    try:
        from hagent.bridge.config import get_mongodb_config, get_world_model_config

        wm = get_world_model_config()
        traj_cfg = dict(wm.get("trajectory") or {})
        mongo = get_mongodb_config()
        db_name = db_name or mongo.get("db_name")
        collection_name = (
            collection_name or traj_cfg.get("collection") or "world_trajectories"
        )
    except Exception:  # noqa: BLE001
        collection_name = collection_name or "world_trajectories"

    coll = None
    if client is not None and db_name and collection_name:
        try:
            coll = client[db_name][collection_name]
        except Exception as exc:  # noqa: BLE001
            logger.warning("TrajectoryStore Mongo bind failed: %s", exc)
            coll = None

    return TrajectoryStore(
        collection=coll,
        max_per_user=int(
            max_per_user
            if max_per_user is not None
            else traj_cfg.get("max_per_user", 5000)
        ),
        enabled=bool(enabled if enabled is not None else traj_cfg.get("enabled", True)),
    )
