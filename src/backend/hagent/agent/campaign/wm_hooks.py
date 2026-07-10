"""World-model helpers for campaign ticks (surprise / latent update)."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from hagent.world.schema import AutoMLAction

logger = logging.getLogger(__name__)


async def campaign_wm_step(
    *,
    wm_service: Any | None,
    world_model: dict | None,
    user_id: str | None,
    action_type: str,
    params: dict | None = None,
    goal: dict | None = None,
    next_world_model: dict | None = None,
) -> Tuple[dict | None, dict | None]:
    """
    Encode → predict → surprise after a campaign tool action.

    Returns (surprise_dict | None, updated_world_model | None).
    """
    if wm_service is None:
        try:
            from hagent.world.service import WorldModelService

            wm_service = WorldModelService.from_config()
        except Exception:
            return None, next_world_model or world_model

    snap = dict(world_model or {"user_id": user_id or ""})
    next_snap = dict(next_world_model or snap)
    try:
        obs = wm_service.observation_from_snapshot(
            snap, user_id=user_id, goal=goal
        )
        next_obs = wm_service.observation_from_snapshot(
            next_snap, user_id=user_id, goal=goal
        )
        action = AutoMLAction(type=action_type, params=dict(params or {}))
        _, _, _, surprise = await wm_service.update(obs, action, next_obs)
        sdict = surprise.to_dict() if hasattr(surprise, "to_dict") else dict(surprise)
        next_snap["last_surprise"] = sdict
        return sdict, next_snap
    except Exception as exc:
        logger.debug("campaign_wm_step failed: %s", exc)
        return None, next_world_model or world_model


def blend_score_with_surprise(
    score: float,
    surprise: dict | None,
    *,
    weight: float = 0.05,
) -> float:
    """Slightly prefer lower surprise when scores are close (higher_is_better score)."""
    if not surprise:
        return score
    try:
        val = float(surprise.get("value") or 0.0)
    except (TypeError, ValueError):
        return score
    return score - weight * val
