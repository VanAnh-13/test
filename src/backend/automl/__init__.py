"""
Gói AutoML Core và Bộ tiền xử lý dữ liệu (CLEAN-001).
"""

from __future__ import annotations

from automl.preprocessing import (
    detect_column_types,
    preprocess_data_unified,
)

__all__ = [
    "detect_column_types",
    "preprocess_data_unified",
]
