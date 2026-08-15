"""
Bộ tiền xử lý dữ liệu cho bài toán Hồi quy / Regression (REFAC/CLEAN-001).

Tái sử dụng logic chung từ automl.preprocessing để loại bỏ trùng lặp mã nguồn.
"""

from __future__ import annotations

import pandas as pd

from automl.preprocessing import (
    categorical_transformer,
    convert_to_string,
    detect_column_types,
    numeric_transformer,
    preprocess_data_unified,
    text_transformer,
    to_1d_array,
)

__all__ = [
    "categorical_transformer",
    "convert_to_string",
    "detect_column_types",
    "numeric_transformer",
    "preprocess_data",
    "text_transformer",
    "to_1d_array",
]


def preprocess_data(list_feature: list, target: str, data: pd.DataFrame):
    """Tiền xử lý đặc trưng và nhãn target cho bài toán hồi quy."""
    return preprocess_data_unified(
        list_feature=list_feature,
        target=target,
        data=data,
        problem_type="regression",
    )
