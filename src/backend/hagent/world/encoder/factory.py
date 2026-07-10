"""Encoder factory — backend selected only from config."""

from __future__ import annotations

from typing import Any

from hagent.world.encoder.structured_v1 import StructuredV1Encoder


def create_encoder(config: dict | None = None) -> Any:
    """
    Create WorldEncoder from config.

    Config keys (world_model.encoder):
      backend: structured_v1 | ...
      dim, feature_extractors, ...
    """
    cfg = dict(config or {})
    backend = str(cfg.get("backend") or "structured_v1").lower()
    if backend == "structured_v1":
        return StructuredV1Encoder(cfg)
    raise ValueError(
        f"Unsupported world_model.encoder.backend={backend!r}. "
        f"Supported: structured_v1"
    )
