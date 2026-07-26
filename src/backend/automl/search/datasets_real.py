"""
Nạp dataset THẬT cho benchmark HPO — hoàn toàn offline.

Nguồn:
  - sklearn bundled (đóng gói theo thư viện, không cần mạng): iris, wine,
    breast_cancer, digits;
  - CSV có sẵn trong repo: glass (assets/end_users), online_shoppers
    (assets/online_shoppers).

Cố ý KHÔNG phụ thuộc OpenML: API v1 của họ trả 504 (26/07/2026), và benchmark
phải tái lập được mà không cần mạng.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[2]

# name -> (kind, spec). kind: 'sklearn' | 'csv'
REGISTRY: Dict[str, Dict[str, Any]] = {
    "iris": {"kind": "sklearn", "loader": "load_iris"},
    "wine": {"kind": "sklearn", "loader": "load_wine"},
    "breast_cancer": {"kind": "sklearn", "loader": "load_breast_cancer"},
    "digits": {"kind": "sklearn", "loader": "load_digits"},
    "glass": {
        "kind": "csv",
        "path": "assets/end_users/glass.csv",
        "target": "Type",
    },
    "online_shoppers": {
        "kind": "csv",
        "path": "assets/online_shoppers/online_shoppers_intention.csv",
        "target": "Revenue",
    },
}


def _encode_frame(df: pd.DataFrame, target: str) -> tuple:
    """One-hot cột phi số, ép target về nhãn nguyên. Trả (X, y, frame_chuẩn)."""
    frame = df.dropna(subset=[target]).reset_index(drop=True)
    y_raw = frame[target]
    features = frame.drop(columns=[target])

    cat_cols = [
        c for c in features.columns if not pd.api.types.is_numeric_dtype(features[c])
    ]
    if cat_cols:
        features = pd.get_dummies(features, columns=cat_cols, drop_first=False)
    features = features.fillna(features.median(numeric_only=True))

    # bool -> int, và nhãn chuỗi -> mã nguyên
    if pd.api.types.is_bool_dtype(y_raw):
        y = y_raw.astype(int).to_numpy()
    elif pd.api.types.is_numeric_dtype(y_raw):
        y = y_raw.to_numpy()
    else:
        y = pd.Categorical(y_raw).codes

    X = features.astype(float).to_numpy()
    # frame chuẩn hóa tên cột target = 'target' để dùng meta_features_from_frame
    normalized = features.copy()
    normalized["target"] = y
    return X, y, normalized


def load_dataset(name: str) -> Dict[str, Any]:
    """Nạp một dataset thật; raise KeyError/FileNotFoundError nếu không có."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset {name!r}. Available: {', '.join(REGISTRY)}")
    spec = REGISTRY[name]

    if spec["kind"] == "sklearn":
        from sklearn import datasets as skd

        bunch = getattr(skd, spec["loader"])()
        X = np.asarray(bunch.data, dtype=float)
        y = np.asarray(bunch.target)
        frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        frame["target"] = y
        source = f"sklearn.datasets.{spec['loader']}"
    else:
        path = BACKEND_DIR / spec["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Dataset CSV missing: {path}")
        df = pd.read_csv(path)
        X, y, frame = _encode_frame(df, spec["target"])
        source = str(path.relative_to(BACKEND_DIR)).replace("\\", "/")

    from hagent.world.meta_features import meta_features_from_frame

    meta = meta_features_from_frame(frame, target="target")
    return {
        "name": name,
        "X": X,
        "y": y,
        "meta": meta,
        "source": source,
        "n_rows": int(X.shape[0]),
        "n_cols": int(X.shape[1]),
        "n_classes": int(len(np.unique(y))),
    }


def load_real_datasets(names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Nạp nhiều dataset; bỏ qua (kèm cảnh báo) bộ nào không nạp được."""
    out: List[Dict[str, Any]] = []
    for name in names or list(REGISTRY):
        try:
            out.append(load_dataset(name))
        except Exception as exc:
            logger.warning("Bỏ qua dataset %s: %s", name, exc)
    return out


def available_datasets() -> List[str]:
    return list(REGISTRY)
