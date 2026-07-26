"""
Meta-features của dataset — đầu vào để outcome model TRANSFER xuyên dataset.

Một bộ key cố định (META_KEYS_V2) dùng thống nhất ở ba nơi:
  - meta_features_from_frame: tính từ DataFrame thật (OpenML CSV, upload user);
  - DatasetProfile.meta trong benchmark mô phỏng;
  - outcome_features (meta_profile="v2") khi featurize.
"""

from __future__ import annotations

from typing import Any, Dict

META_KEYS_V2 = [
    "n_rows",
    "n_cols",
    "n_classes",
    "class_imbalance",
    "frac_categorical",
    "missing_frac",
    "mean_abs_skew",
]


def meta_features_from_frame(df: Any, target: str = "target") -> Dict[str, float]:
    """
    Tính meta-features từ pandas DataFrame (cột target đã chuẩn hóa).

    Trả dict đủ META_KEYS_V2; cột target thiếu → các key phụ thuộc target = 0.
    """
    import pandas as pd  # import cục bộ: module này còn được dùng ở worker nhẹ

    n_rows = int(df.shape[0])
    feature_cols = [c for c in df.columns if c != target]
    n_cols = len(feature_cols)

    n_classes = 0
    class_imbalance = 0.0
    if target in df.columns and n_rows > 0:
        counts = df[target].value_counts(dropna=True)
        n_classes = int(counts.shape[0])
        if counts.sum() > 0:
            class_imbalance = float(counts.iloc[0] / counts.sum())

    numeric_cols = [
        c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])
    ]
    frac_categorical = (
        (n_cols - len(numeric_cols)) / n_cols if n_cols else 0.0
    )

    missing_frac = 0.0
    if n_rows and n_cols:
        missing_frac = float(df[feature_cols].isna().sum().sum() / (n_rows * n_cols))

    mean_abs_skew = 0.0
    if numeric_cols and n_rows > 2:
        skews = df[numeric_cols].skew(numeric_only=True).abs()
        skews = skews[skews.notna()]
        if len(skews):
            mean_abs_skew = float(skews.mean())

    return {
        "n_rows": float(n_rows),
        "n_cols": float(n_cols),
        "n_classes": float(n_classes),
        "class_imbalance": class_imbalance,
        "frac_categorical": float(frac_categorical),
        "missing_frac": missing_frac,
        "mean_abs_skew": mean_abs_skew,
    }
