"""
Tabular transition predictor — action→latent delta from config table.

Transitions live in config (world_model.predictor.transitions), not hard-coded
if/else chains in call sites. Defaults provided only as fallback when config
omits the table (tests / minimal YAML).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

from hagent.world.schema import AutoMLAction, LatentState


# Fallback deltas when YAML omits transitions (keyed by action type).
# Values are per-index offsets applied after action fingerprint mix.
_FALLBACK_TRANSITIONS: Dict[str, Dict[str, Any]] = {
    "list_datasets": {"scale": 0.05, "slots": {"dataset_signal": 0.15}},
    "get_dataset_info": {"scale": 0.04, "slots": {"dataset_signal": 0.12}},
    "get_features": {"scale": 0.04, "slots": {"feature_signal": 0.2}},
    "preview_data": {"scale": 0.03, "slots": {"dataset_signal": 0.08}},
    "get_available_models": {"scale": 0.03, "slots": {"model_signal": 0.15}},
    "get_metrics": {"scale": 0.03, "slots": {"model_signal": 0.1}},
    "start_training": {"scale": 0.12, "slots": {"job_pending": 0.35, "train_signal": 0.25}},
    "get_job_info": {"scale": 0.08, "slots": {"job_status": 0.2}},
    "list_jobs": {"scale": 0.05, "slots": {"job_status": 0.12}},
    "check_system_health": {"scale": 0.01, "slots": {}},
    "get_world_state": {"scale": 0.02, "slots": {}},
}

# Fixed slot indices into latent vector (documented; overridable via config.slot_indices)
_DEFAULT_SLOT_INDICES: Dict[str, int] = {
    "dataset_signal": 0,
    "feature_signal": 1,
    "model_signal": 2,
    "job_pending": 3,
    "job_status": 4,
    "train_signal": 5,
}


def _action_fingerprint(action: AutoMLAction, dim: int) -> List[float]:
    """Small deterministic perturbation from action type + param keys."""
    seed = action.type + "|" + ",".join(sorted(action.params.keys()))
    out = [0.0] * dim
    for i, ch in enumerate(seed.encode("utf-8")):
        out[i % dim] += (ch / 255.0 - 0.5) * 0.02
    return out


class TabularTransitionV1Predictor:
    """
    ẑ' = normalize(z + scale * fingerprint + slot_deltas(action_type)).
    """

    def __init__(self, config: dict | None = None):
        self.config = dict(config or {})
        transitions = self.config.get("transitions")
        self.transitions: Dict[str, Dict[str, Any]] = (
            dict(transitions) if isinstance(transitions, dict) else dict(_FALLBACK_TRANSITIONS)
        )
        slots = self.config.get("slot_indices")
        self.slot_indices: Dict[str, int] = (
            dict(slots) if isinstance(slots, dict) else dict(_DEFAULT_SLOT_INDICES)
        )
        self.default_scale = float(self.config.get("default_scale", 0.03))

    def predict(self, z: LatentState, action: AutoMLAction) -> LatentState:
        dim = z.dim
        vec = list(z.vector)
        if len(vec) < dim:
            vec = vec + [0.0] * (dim - len(vec))
        vec = vec[:dim]

        spec = self.transitions.get(action.type) or {
            "scale": self.default_scale,
            "slots": {},
        }
        scale = float(spec.get("scale", self.default_scale))
        slots: Dict[str, float] = dict(spec.get("slots") or {})

        fp = _action_fingerprint(action, dim)
        for i in range(dim):
            vec[i] = vec[i] + scale * fp[i]

        for slot_name, delta in slots.items():
            idx = self.slot_indices.get(slot_name)
            if idx is None or idx < 0 or idx >= dim:
                continue
            vec[idx] = vec[idx] + float(delta)

        # Status-sensitive tweak for get_job_info
        if action.type == "get_job_info":
            status = str(action.params.get("status_hint") or action.params.get("status") or "")
            idx = self.slot_indices.get("job_status", 4)
            if status.lower() in ("completed", "done", "success") and idx < dim:
                vec[idx] += 0.15
            elif status.lower() in ("failed", "error") and idx < dim:
                vec[idx] -= 0.15

        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vec = [v / norm for v in vec]
        return LatentState(
            vector=vec,
            dim=dim,
            meta={
                "predictor": "tabular_transition_v1",
                "action_type": action.type,
                "scale": scale,
            },
        )
