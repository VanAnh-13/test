"""
Unit tests for adaptive update frequency in World Model (REFAC-013).
"""

from __future__ import annotations

from hagent.world.schema import SurpriseResult, WorldState
from hagent.world.service import WorldModelService


def test_world_state_has_adaptive_fields() -> None:
    """WorldState schema có các trường update_frequency và surprise_momentum."""
    ws = WorldState(user_id="user_test")
    assert ws.update_frequency == 1.0
    assert ws.surprise_momentum == 0.0

    d = ws.to_dict()
    assert d["update_frequency"] == 1.0
    assert d["surprise_momentum"] == 0.0


def test_adaptive_update_high_surprise_updates_immediately() -> None:
    """Khi surprise cao, WorldModelService duy trì update mỗi step (interval=1)."""
    service = WorldModelService(
        encoder=None,
        predictor=None,
        planner=None,
        adaptive_config={
            "enabled": True,
            "min_update_interval": 1,
            "max_update_interval": 5,
            "surprise_decay": 0.8,
        },
    )

    user_id = "user_high_surprise"
    high_surprise = SurpriseResult(
        value=0.85, level="high", predicted_dim=1, actual_dim=1
    )

    # 3 bước liên tiếp surprise cao đều trigger update
    for _ in range(3):
        assert service.should_update_adaptive(user_id, high_surprise) is True

    state = service.get_adaptive_state(user_id)
    assert state["current_interval"] == 1
    assert state["momentum"] > 0.40


def test_adaptive_update_low_surprise_decays_frequency() -> None:
    """Khi surprise liên tục thấp, khoảng cách giữa các lần update giãn ra (interval tăng)."""
    service = WorldModelService(
        encoder=None,
        predictor=None,
        planner=None,
        adaptive_config={
            "enabled": True,
            "min_update_interval": 1,
            "max_update_interval": 4,
            "surprise_decay": 0.5,
        },
    )

    user_id = "user_low_surprise"
    low_surprise = SurpriseResult(
        value=0.02, level="low", predicted_dim=1, actual_dim=1
    )

    # Step 0: Interval bắt đầu tại 1, nên update ngay
    assert service.should_update_adaptive(user_id, low_surprise) is True

    # Cung cấp low surprise để tăng interval
    service.record_step_surprise(user_id, low_surprise)
    service.record_step_surprise(user_id, low_surprise)
    state = service.get_adaptive_state(user_id)
    assert state["current_interval"] > 1


def test_adaptive_update_surprise_spike_resets_interval() -> None:
    """Đang ở trạng thái ổn định (interval cao) gặp surprise spike sẽ reset ngay về interval=1."""
    service = WorldModelService(
        encoder=None,
        predictor=None,
        planner=None,
        adaptive_config={
            "enabled": True,
            "min_update_interval": 1,
            "max_update_interval": 5,
            "surprise_decay": 0.8,
        },
    )

    user_id = "user_spike"
    low_surprise = SurpriseResult(
        value=0.01, level="low", predicted_dim=1, actual_dim=1
    )

    # Giả lập 5 bước low surprise
    for _ in range(5):
        service.record_step_surprise(user_id, low_surprise)

    state_before = service.get_adaptive_state(user_id)
    assert state_before["current_interval"] >= 3

    # Đột ngột gặp high surprise spike
    spike_surprise = SurpriseResult(
        value=1.50, level="high", predicted_dim=1, actual_dim=1
    )
    should_update = service.should_update_adaptive(user_id, spike_surprise)
    assert should_update is True

    state_after = service.get_adaptive_state(user_id)
    assert state_after["current_interval"] == 1


def test_disabled_adaptive_update_always_updates() -> None:
    """Khi adaptive_update.enabled=False, luôn trả về True."""
    service = WorldModelService(
        encoder=None,
        predictor=None,
        planner=None,
        adaptive_config={"enabled": False},
    )

    user_id = "user_disabled"
    low_surprise = SurpriseResult(
        value=0.01, level="low", predicted_dim=1, actual_dim=1
    )

    for _ in range(5):
        assert service.should_update_adaptive(user_id, low_surprise) is True
