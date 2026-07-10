"""
Shared factories for agent runtime: WorldModelService + WorldStateStore.

Keeps Mongo binding out of graph nodes; call sites pass client when available.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


def build_wm_runtime(
    *,
    mongo_client: Any | None = None,
    db_name: str | None = None,
    world_model_config: dict | None = None,
) -> Tuple[Any, Any | None]:
    """
    Returns (WorldModelService, WorldStateStore | None).

    WorldStateStore is None when no mongo_client (offline / unit tests).
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
        except Exception as exc:
            logger.debug("WorldStateStore create failed: %s", exc)
    return wm, store


def try_mongo_from_db(db: Any) -> Tuple[Any | None, str | None]:
    """Extract (client, db_name) from pymongo AsyncDatabase / Database."""
    client = getattr(db, "client", None)
    name = getattr(db, "name", None)
    return client, str(name) if name else None
