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
