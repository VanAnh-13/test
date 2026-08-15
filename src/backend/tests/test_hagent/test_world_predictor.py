"""
Comprehensive unit tests for World Model predictors (REFAC-014).
"""

from __future__ import annotations

import pytest

from hagent.world.predictor.factory import (
    create_outcome_ensemble,
    create_outcome_head,
    create_predictor,
)
from hagent.world.predictor.tabular_transition_v1 import (
    TabularTransitionV1Predictor,
    _action_fingerprint,
)
from hagent.world.schema import AutoMLAction, LatentState


def test_create_predictor_factory() -> None:
    """create_predictor trả về instance đúng theo config backend."""
    p_tab = create_predictor({"backend": "tabular_transition_v1"})
    assert isinstance(p_tab, TabularTransitionV1Predictor)

    p_jepa = create_predictor({"backend": "neural_jepa_v1"})
    assert p_jepa is not None

    p_ens = create_predictor({"backend": "dynamics_ensemble"})
    assert p_ens is not None

    with pytest.raises(ValueError, match="Unsupported world_model.predictor.backend"):
        create_predictor({"backend": "non_existent_backend"})


def test_create_outcome_head_and_ensemble_factories() -> None:
    """Outcome head & ensemble factories khởi tạo chính xác."""
    assert create_outcome_head({"enabled": False}) is None
    head = create_outcome_head({"backend": "outcome_head_v1"})
    assert head is not None

    assert create_outcome_ensemble({"enabled": False}) is None
    ens = create_outcome_ensemble({"enabled": True})
    assert ens is not None


def test_action_fingerprint_deterministic() -> None:
    """_action_fingerprint tạo vector nhiễu xác định theo action type và params."""
    action1 = AutoMLAction(
        type="start_training", params={"dataset_id": "d1", "model": "rf"}
    )
    action2 = AutoMLAction(
        type="start_training", params={"dataset_id": "d1", "model": "rf"}
    )
    fp1 = _action_fingerprint(action1, dim=8)
    fp2 = _action_fingerprint(action2, dim=8)
    assert fp1 == fp2
    assert len(fp1) == 8


def test_tabular_transition_predictor_transitions() -> None:
    """TabularTransitionV1Predictor dịch chuyển latent vector theo hành động."""
    predictor = TabularTransitionV1Predictor()
    z0 = LatentState(vector=[0.0] * 8, dim=8)

    # 1. Action start_training -> dịch chuyển train_signal và job_pending
    action_train = AutoMLAction(type="start_training", params={"dataset_id": "ds_1"})
    z_train = predictor.predict(z0, action_train)
    assert z_train.dim == 8
    assert z_train.vector != z0.vector
    # Normalized L2 norm
    norm = sum(x * x for x in z_train.vector) ** 0.5
    assert abs(norm - 1.0) < 1e-5

    # 2. Action get_job_info with completed status
    action_job_done = AutoMLAction(
        type="get_job_info", params={"job_id": "j1", "status_hint": "completed"}
    )
    z_done = predictor.predict(z0, action_job_done)
    assert z_done.vector != z0.vector

    # 3. Action get_job_info with failed status
    action_job_failed = AutoMLAction(
        type="get_job_info", params={"job_id": "j1", "status_hint": "failed"}
    )
    z_failed = predictor.predict(z0, action_job_failed)
    assert z_failed.vector != z_done.vector


def test_tabular_transition_unknown_action_and_dimension_padding() -> None:
    """Xử lý hành động lạ và vector kích thước thiếu/thừa một cách an toàn."""
    predictor = TabularTransitionV1Predictor()
    # Short vector padding
    z_short = LatentState(vector=[0.5, 0.5], dim=6)
    action_unknown = AutoMLAction(type="unknown_custom_tool", params={})
    z_pred = predictor.predict(z_short, action_unknown)
    assert len(z_pred.vector) == 6
    assert z_pred.dim == 6
