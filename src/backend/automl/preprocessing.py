"""
Bộ tiền xử lý dữ liệu chung cho AutoML (CLEAN-001).

Cung cấp các pipeline tiền xử lý chuẩn hóa cho các kiểu dữ liệu số, phân loại và văn bản,
loại bỏ hoàn toàn sự trùng lặp giữa bài toán Classification và Regression.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
)


def detect_column_types(
    df: pd.DataFrame, text_cardinality_threshold: int = 50
) -> tuple[list[str], list[str], list[str]]:
    """Tự động phân loại các cột trong DataFrame thành numeric, categorical hoặc text."""
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    text_cols: list[str] = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        elif pd.api.types.is_object_dtype(df[col]) or isinstance(
            df[col].dtype, pd.CategoricalDtype
        ):
            if df[col].nunique() > text_cardinality_threshold:
                text_cols.append(col)
            else:
                categorical_cols.append(col)

    return numeric_cols, categorical_cols, text_cols


def to_1d_array(x: object) -> np.ndarray:
    """Chuyển đổi dữ liệu đầu vào thành mảng NumPy 1 chiều (flatten)."""
    if hasattr(x, "values"):
        return x.values.ravel()
    if isinstance(x, np.ndarray):
        return x.ravel()
    return np.array(x).ravel()


def convert_to_string(x: object) -> str:
    """Chuyển đổi giá trị thành chuỗi ký tự an toàn."""
    return str(x) if x is not None else ""


# Định nghĩa các pipeline biến đổi cho từng loại đặc trưng
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ]
)

text_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="")),
        ("reshape", FunctionTransformer(to_1d_array, validate=False)),
        ("tfidf", TfidfVectorizer(max_features=50, preprocessor=convert_to_string)),
    ]
)


def build_column_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    text_cols: list[str],
) -> ColumnTransformer | None:
    """Khởi tạo ColumnTransformer dựa trên danh sách các cột đã phân loại."""
    transformers = []

    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))

    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    if text_cols:
        for col in text_cols:
            transformers.append((f"text_{col}", text_transformer, [col]))

    if not transformers:
        return None

    return ColumnTransformer(
        transformers=transformers, remainder="passthrough", sparse_threshold=0.3
    )


def preprocess_data_unified(
    list_feature: list[str],
    target: str,
    data: pd.DataFrame,
    problem_type: Literal["classification", "regression"] = "classification",
) -> tuple[np.ndarray, np.ndarray, ColumnTransformer | None, LabelEncoder | None]:
    """Hàm tiền xử lý dữ liệu hợp nhất cho cả bài toán Phân loại (Classification) và Hồi quy (Regression)."""
    features = [f for f in list_feature if f != target]

    try:
        data_process = data[features].copy()
    except KeyError as ke:
        raise KeyError(f"Not found feature {ke!s}") from ke

    if target not in data.columns:
        raise KeyError(f"Target '{target}' not exist")

    le_target: LabelEncoder | None = None

    if problem_type == "classification":
        le_target = LabelEncoder()
        y_imputed_as_str = data[target].fillna("").astype(str)
        y_processed = le_target.fit_transform(y_imputed_as_str)
    else:
        y_series = pd.to_numeric(data[target], errors="coerce")
        valid_mask = y_series.notna()
        y_processed = y_series[valid_mask].values
        data_process = data_process.loc[valid_mask]

    numeric_cols, categorical_cols, text_cols = detect_column_types(data_process)
    preprocessor = build_column_preprocessor(numeric_cols, categorical_cols, text_cols)

    if preprocessor is None:
        X_processed = data_process.values
    else:
        X_processed = preprocessor.fit_transform(data_process)

    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    return X_processed, y_processed, preprocessor, le_target
