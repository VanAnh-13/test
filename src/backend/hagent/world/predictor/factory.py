"""Predictor factory — backend from config only."""

from __future__ import annotations

from typing import Any

from hagent.world.predictor.tabular_transition_v1 import TabularTransitionV1Predictor


def create_predictor(config: dict | None = None) -> Any:
    cfg = dict(config or {})
    backend = str(cfg.get("backend") or "tabular_transition_v1").lower()
    if backend == "tabular_transition_v1":
        return TabularTransitionV1Predictor(cfg)
    if backend in ("neural_jepa_v1", "neural_jepa", "jepa_v1"):
        from hagent.world.predictor.neural_jepa_v1 import NeuralJepaV1Predictor

        return NeuralJepaV1Predictor(cfg)
    raise ValueError(
        f"Unsupported world_model.predictor.backend={backend!r}. "
        f"Supported: tabular_transition_v1, neural_jepa_v1"
    )


def create_outcome_head(config: dict | None = None) -> Any:
    """Outcome head from world_model.outcome_head config. None when disabled."""
    cfg = dict(config or {})
    if not cfg.get("enabled", True):
        return None
    backend = str(cfg.get("backend") or "outcome_head_v1").lower()
    if backend in ("outcome_head_v1", "outcome_v1"):
        from hagent.world.predictor.outcome_head_v1 import OutcomeHeadV1

        return OutcomeHeadV1(cfg)
    raise ValueError(
        f"Unsupported world_model.outcome_head.backend={backend!r}. "
        f"Supported: outcome_head_v1"
    )
