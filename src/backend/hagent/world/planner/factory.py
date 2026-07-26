"""Planner factory."""

from __future__ import annotations

from typing import Any

from hagent.world.planner.cem_lite import CEMLitePlanner


def create_planner(predictor: Any, config: dict | None = None) -> Any:
    cfg = dict(config or {})
    backend = str(cfg.get("backend") or "cem_lite").lower()
    if backend == "cem_lite":
        return CEMLitePlanner(predictor, cfg)
    raise ValueError(
        f"Unsupported world_model.planner.backend={backend!r}. Supported: cem_lite"
    )


def create_campaign_planner(config: dict | None = None) -> Any:
    """Campaign-config planner from world_model.campaign_planner. None when disabled."""
    cfg = dict(config or {})
    if not cfg.get("enabled", True):
        return None
    backend = str(cfg.get("backend") or "cem_config_v1").lower()
    if backend in ("cem_config_v1", "cem_config"):
        from hagent.world.planner.cem_config_v1 import CemConfigV1Planner

        return CemConfigV1Planner(cfg)
    if backend in ("cem_mpc_v1", "cem_mpc"):
        from hagent.world.planner.cem_mpc_v1 import CemMpcV1Planner

        return CemMpcV1Planner(cfg)
    raise ValueError(
        f"Unsupported world_model.campaign_planner.backend={backend!r}. "
        f"Supported: cem_config_v1, cem_mpc_v1"
    )
