"""Cấu hình runtime dùng chung cho các điểm vào của campaign graph."""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)

_DEFAULT_MAX_MONITOR_TICKS = 50


def max_monitor_ticks() -> int:
    """Trả về giới hạn theo dõi campaign đã chặn biên, có giá trị dự phòng cũ."""
    try:
        from hagent.bridge.config import get_campaign_config

        return int(
            get_campaign_config().get(
                "max_monitor_ticks",
                _DEFAULT_MAX_MONITOR_TICKS,
            )
        )
    except Exception as exc:  # noqa: BLE001 - boundary cấu hình giữ giá trị dự phòng
        logger.debug("campaign monitor config unavailable: %s", exc)
        return _DEFAULT_MAX_MONITOR_TICKS
