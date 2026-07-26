"""
NL → GoalSpec (structured). Uses lightweight rules + optional LLM.

Primary path is rule-based so tests and offline use need no LLM.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, Optional

from hagent.world.schema import GoalSpec


def _detect_goal_type(text: str) -> str:
    lower = text.lower()
    train_kw = (
        "train",
        "huấn luyện",
        "huan luyen",
        "training",
        "xây model",
        "xay model",
        "fit model",
    )
    analyze_kw = (
        "phân tích",
        "phan tich",
        "analyze",
        "feature",
        "dataset",
        "dữ liệu",
        "du lieu",
        "thống kê",
    )
    eval_kw = (
        "đánh giá",
        "danh gia",
        "evaluate",
        "so sánh",
        "compare",
        "kết quả",
        "best model",
    )
    monitor_kw = ("status", "trạng thái", "job", "theo dõi", "monitor")
    list_kw = ("liệt kê", "liet ke", "list", "danh sách", "danh sach")

    if any(k in lower for k in train_kw):
        return "train"
    if any(k in lower for k in eval_kw):
        return "evaluate"
    if any(k in lower for k in monitor_kw):
        return "monitor"
    if any(k in lower for k in list_kw):
        return "list"
    if any(k in lower for k in analyze_kw):
        return "analyze"
    return "respond"


def _detect_problem_type(text: str) -> Optional[str]:
    lower = text.lower()
    if "regress" in lower or "hồi quy" in lower or "hoi quy" in lower:
        return "regression"
    if "classif" in lower or "phân loại" in lower or "phan loai" in lower:
        return "classification"
    return None


def _detect_metric(text: str) -> Optional[str]:
    lower = text.lower()
    for m in (
        "f1",
        "accuracy",
        "precision",
        "recall",
        "roc_auc",
        "auc",
        "mae",
        "mse",
        "rmse",
        "r2",
    ):
        if re.search(rf"\b{re.escape(m)}\b", lower):
            return m
    return None


# Words that must never be treated as a target column name
_TARGET_STOPWORDS = frozenset(
    {
        "column",
        "col",
        "cột",
        "cot",
        "là",
        "la",
        "is",
        "the",
        "a",
        "an",
        "with",
        "với",
        "voi",
        "and",
        "or",
        "of",
        "for",
        "to",
        "on",
        "dataset",
        "model",
        "metric",
    }
)


def _detect_target_column(text: str) -> Optional[str]:
    """
    Extract target/label column from natural language.

    Supports CI/E2E prompts like:
      - target column là 'Revenue'
      - target column is Revenue
      - target_column=Revenue
      - cột mục tiêu Revenue
    """
    patterns = [
        # target column là/is/=/: Name  (must come before bare "target X")
        r"target\s*[_ ]?\s*column\s*(?:là|is|=|:)?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?",
        r"target(?:_column)?\s*[:=]\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?",
        r"cột\s+mục\s+tiêu\s*(?:là|is|=|:)?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?",
        r"cot\s+muc\s+tieu\s*(?:là|is|=|:)?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?",
        r"nhãn\s*(?:là|is|=|:)?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?",
        r"label\s*(?:column)?\s*(?:là|is|=|:)?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?",
        # bare: target Revenue (reject stopwords like "column")
        r"\btarget\s+['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            name = m.group(1)
            if name.lower() not in _TARGET_STOPWORDS:
                return name
    return None


def _detect_dataset_id(text: str, known_ids: Optional[list[str]] = None) -> Optional[str]:
    """
    Extract dataset id from NL.

    Supports:
      - known ids present in text
      - dataset ID <id> / dataset_id=<id>
      - Mongo ObjectId (24 hex chars)
    """
    if known_ids:
        for did in known_ids:
            if did and did in text:
                return did

    patterns = [
        r"dataset(?:\s*id|_id)?\s*[:=]\s*['\"]?([A-Za-z0-9_\-]+)['\"]?",
        r"dataset\s+id\s+['\"]?([A-Za-z0-9_\-]+)['\"]?",
        r"dataset\s+ID\s+['\"]?([A-Za-z0-9_\-]+)['\"]?",
        r"id\s+dataset\s+['\"]?([A-Za-z0-9_\-]+)['\"]?",
        r"dataset_id\s+['\"]?([A-Za-z0-9_\-]+)['\"]?",
        r"trên\s+dataset\s+(?:id\s+)?['\"]?([A-Za-z0-9_\-]+)['\"]?",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1)

    # Mongo ObjectId (common HAutoML dataset ids)
    m = re.search(r"\b([a-fA-F0-9]{24})\b", text)
    if m:
        return m.group(1)
    return None


def _detect_models(text: str) -> list[str]:
    """Extract ML model names mentioned in the prompt (order-preserving, unique)."""
    # Prefer explicit CamelCase *Classifier/*Regressor and common short names
    found: list[str] = []
    patterns = [
        r"\b([A-Z][A-Za-z0-9]*(?:Classifier|Regressor|ClassifierCV))\b",
        r"\b(XGB(?:Classifier|Regressor)?)\b",
        r"\b(SVC|SVR|LGBMClassifier|LGBMRegressor|CatBoostClassifier|CatBoostRegressor)\b",
        r"\b(RandomForest(?:Classifier|Regressor)?)\b",
        r"\b(LogisticRegression|LinearRegression|Ridge|Lasso|ElasticNet)\b",
    ]
    for p in patterns:
        for m in re.finditer(p, text):
            name = m.group(1)
            # normalize bare RandomForest → leave as stated
            if name not in found:
                found.append(name)
    return found


# Cụm từ → thuật toán HPO (EN + VN). Cụm dài match trước; tránh từ đơn mơ hồ
# ("ga", "sh", "random" trần) để không bắt nhầm hội thoại thường.
_SEARCH_ALGO_PATTERNS: list[tuple[str, str]] = [
    (r"successive[\s_-]?halving|halving", "successive_halving"),
    (r"random[\s_-]?search|tìm\s*kiếm\s*ngẫu\s*nhiên", "random_search"),
    (r"bayes(?:ian)?(?:[\s_-]?(?:search|optimi[sz]ation))?", "bayesian_search"),
    (r"genetic(?:[\s_-]?algorithm)?|di\s*truyền|tiến\s*hóa", "genetic_algorithm"),
    (r"grid[\s_-]?search|vét\s*cạn|tìm\s*kiếm\s*lưới", "grid_search"),
]


def _detect_search_algorithm(text: str) -> Optional[str]:
    lowered = text.lower()
    for pattern, algo in _SEARCH_ALGO_PATTERNS:
        if re.search(pattern, lowered, re.IGNORECASE):
            return algo
    return None


def _detect_time_limit(text: str) -> Optional[int]:
    m = re.search(r"(\d+)\s*(?:giây|seconds?|s)\b", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*(?:phút|minutes?|m)\b", text, re.IGNORECASE)
    if m:
        return int(m.group(1)) * 60
    return None


def parse_goal(
    message: str,
    *,
    known_dataset_ids: Optional[list[str]] = None,
    default_user_constraints: Optional[Dict[str, Any]] = None,
) -> GoalSpec:
    """Parse natural language into GoalSpec (deterministic rules)."""
    text = (message or "").strip()
    goal_type = _detect_goal_type(text)
    constraints: Dict[str, Any] = dict(default_user_constraints or {})
    time_limit = _detect_time_limit(text)
    if time_limit is not None:
        constraints["time_limit"] = time_limit

    models = _detect_models(text)
    if models:
        constraints["models"] = models

    search_algo = _detect_search_algorithm(text)
    if search_algo and "search_algorithm" not in constraints:
        constraints["search_algorithm"] = search_algo

    goal: GoalSpec = {
        "goal_type": goal_type,
        "description": text[:500],
        "metric": _detect_metric(text),
        "problem_type": _detect_problem_type(text),
        "dataset_id": _detect_dataset_id(text, known_dataset_ids),
        "target_column": _detect_target_column(text),
        "constraints": constraints,
        "goal_id": str(uuid.uuid4()),  # type: ignore[typeddict-unknown-key]
    }
    # Clean Nones for cleaner state
    return {k: v for k, v in goal.items() if v is not None}  # type: ignore[return-value]


def is_simple_query(message: str, simple_keywords: Optional[list[str]] = None) -> bool:
    """True when planner should be skipped (greeting / chitchat)."""
    lower = (message or "").strip().lower()
    if not lower:
        return True
    keywords = simple_keywords or []
    if any(k in lower for k in keywords):
        return True
    # Very short non-task messages
    if len(lower) < 12 and not any(
        k in lower for k in ("train", "dataset", "job", "model", "data")
    ):
        return True
    return False
