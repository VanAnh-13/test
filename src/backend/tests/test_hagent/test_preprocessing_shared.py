"""
Kiểm thử đơn vị cho Bộ tiền xử lý dữ liệu dùng chung (CLEAN-001).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from automl.preprocessing import (
    detect_column_types,
    preprocess_data_unified,
    to_1d_array,
)
from automl.process_classification import (
    preprocess_data as preprocess_classification,
)
from automl.process_regression import preprocess_data as preprocess_regression


@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    """Tạo DataFrame thử nghiệm chứa đầy đủ cột số, phân loại, text và missing values."""
    return pd.DataFrame(
        {
            "age": [25, 30, np.nan, 45, 50],
            "income": [50000.0, 60000.0, 75000.0, np.nan, 120000.0],
            "gender": ["M", "F", "F", "M", "M"],
            "review": [
                "Good product and fast shipping",
                "Poor customer support",
                "Excellent quality, highly recommended",
                "Average experience",
                "Terrible service and slow delivery",
            ],
            "target_class": ["yes", "no", "yes", "no", "yes"],
            "target_reg": [10.5, 20.0, 30.2, 40.8, 50.1],
        }
    )


def test_detect_column_types(sample_dataset: pd.DataFrame) -> None:
    """Kiểm tra khả năng tự động phân loại cột số, categorical và text."""
    numeric, categorical, text = detect_column_types(
        sample_dataset, text_cardinality_threshold=3
    )
    assert "age" in numeric
    assert "income" in numeric
    assert "gender" in categorical
    assert "review" in text


def test_to_1d_array_flattening() -> None:
    """Kiểm tra helper làm phẳng mảng nhiều chiều."""
    data_2d = np.array([[1, 2], [3, 4]])
    flattened = to_1d_array(data_2d)
    assert flattened.shape == (4,)
    assert list(flattened) == [1, 2, 3, 4]


def test_preprocess_classification_flow(sample_dataset: pd.DataFrame) -> None:
    """Kiểm tra tiền xử lý cho bài toán Phân loại qua process_classification."""
    features = ["age", "income", "gender", "review"]
    target = "target_class"

    X, y, preprocessor, le_target = preprocess_classification(
        features, target, sample_dataset
    )

    assert X.shape[0] == len(sample_dataset)
    assert X.shape[1] > 0
    assert y.shape[0] == len(sample_dataset)
    assert le_target is not None
    assert len(le_target.classes_) == 2
    assert preprocessor is not None


def test_preprocess_regression_flow(sample_dataset: pd.DataFrame) -> None:
    """Kiểm tra tiền xử lý cho bài toán Hồi quy qua process_regression."""
    features = ["age", "income", "gender", "review"]
    target = "target_reg"

    X, y, preprocessor, le_target = preprocess_regression(
        features, target, sample_dataset
    )

    assert X.shape[0] == len(sample_dataset)
    assert X.shape[1] > 0
    assert y.shape[0] == len(sample_dataset)
    assert le_target is None  # Regression không dùng LabelEncoder cho target
    assert preprocessor is not None


def test_missing_feature_raises_key_error(sample_dataset: pd.DataFrame) -> None:
    """Kiểm tra bắt lỗi ngoại lệ khi truyền tên đặc trưng không tồn tại."""
    with pytest.raises(KeyError) as exc_info:
        preprocess_data_unified(
            ["non_existent_feature"], "target_class", sample_dataset
        )
    assert "Not found feature" in str(exc_info.value)
