"""Predictor factory — backend from config only."""

from __future__ import annotations

from typing import Any

from hagent.world.predictor.tabular_transition_v1 import TabularTransitionV1Predictor


def create_predictor(config: dict | None = None) -> Any:
    cfg = dict(config or {})
    backend = str(cfg.get("backend") or "tabular_transition_v1").lower()
    if backend == "tabular_transition_v1":
        return TabularTransitionV1Predictor(cfg)
    raise ValueError(
        f"Unsupported world_model.predictor.backend={backend!r}. "
        f"Supported: tabular_transition_v1"
    )
