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


def _default_outcome_model() -> Any | None:
    """Ensemble từ config nếu sẵn sàng, fallback single head; None nếu không có."""
    try:
        from hagent.bridge.config import get_world_model_config
        from hagent.world.predictor import create_outcome_ensemble, create_outcome_head

        wm_cfg = get_world_model_config()
        ens = create_outcome_ensemble(dict(wm_cfg.get("outcome_ensemble") or {}))
        if ens is not None and ens.is_ready:
            return ens
        head = create_outcome_head(dict(wm_cfg.get("outcome_head") or {}))
        if head is not None and head.is_ready:
            return head
    except Exception as exc:
        logger.debug("outcome model init failed: %s", exc)
    return None


def campaign_outcome_surprise(
    *,
    variant: Any,
    dataset_meta: dict | None = None,
    z: Any | None = None,
    outcome_model: Any | None = None,
    surprise_config: dict | None = None,
) -> Optional[dict]:
    """
    Outcome-space surprise cho variant vừa hoàn thành.

    Trả None khi: variant chưa completed / thiếu best_score / model chưa
    sẵn sàng — caller cứ gọi thoải mái, không cần gate trước.
    """
    status = getattr(variant, "status", None) or (
        variant.get("status") if isinstance(variant, dict) else None
    )
    best_score = getattr(variant, "best_score", None) if not isinstance(variant, dict) else variant.get("best_score")
    params = getattr(variant, "params", None) if not isinstance(variant, dict) else variant.get("params")
    if str(status) != "completed" or best_score is None:
        return None

    model = outcome_model if outcome_model is not None else _default_outcome_model()
    if model is None or not getattr(model, "is_ready", False):
        return None

    try:
        pred = model.predict(dict(params or {}), dataset_meta, z)
        if pred is None:
            return None
        cfg = surprise_config
        if cfg is None:
            try:
                from hagent.bridge.config import get_world_model_config

                cfg = dict(get_world_model_config().get("surprise") or {})
            except Exception:
                cfg = {}
        from hagent.world.surprise import compute_outcome_surprise

        result = compute_outcome_surprise(pred, float(best_score), cfg)
        return {
            "zscore": result.value,
            "level": result.level,
            "actual_score": float(best_score),
            "predicted_mean": pred.mean,
            "predicted_std": pred.std,
            "predictor": (pred.meta or {}).get("predictor"),
        }
    except Exception as exc:
        logger.debug("campaign_outcome_surprise failed: %s", exc)
        return None


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
