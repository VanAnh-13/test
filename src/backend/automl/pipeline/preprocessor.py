"""AutomL Pipeline — Preprocessor & Configuration Loader."""

from __future__ import annotations

import logging
import os
from typing import Any

import yaml
from bson import ObjectId
from fastapi import HTTPException
from pymongo.asynchronous.database import AsyncDatabase
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from automl.pipeline.errors import UnknownModelError

logger = logging.getLogger(__name__)

# Registry model classes for safe instantiation
_MODEL_CLASSES: dict[str, Any] = {
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "KNeighborsClassifier": KNeighborsClassifier,
    "SVC": SVC,
    "LogisticRegression": LogisticRegression,
    "GaussianNB": GaussianNB,
}


async def get_dataset_and_user_info(
    data_id: str,
    user_id: str,
    db: AsyncDatabase,
) -> tuple[str | None, str | None]:
    """Lấy thông tin dataset và user từ MongoDB."""
    data_collection = db.tbl_Data
    user_collection = db.tbl_User

    dataset = await data_collection.find_one({"_id": ObjectId(data_id)})
    if not dataset:
        raise HTTPException(status_code=404, detail="Không tìm thấy bộ dữ liệu")
    data_name = dataset.get("dataName")

    user = await user_collection.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=400, detail="Không tìm thấy người dùng")
    user_name = user.get("username")

    return data_name, user_name


def choose_model_version(choose: str) -> list[int]:
    """Xác định danh sách ID các mô hình cần huấn luyện dựa trên lựa chọn."""
    if choose == "new model":
        return [0, 1, 2, 3]
    return [2]


def get_model(
    base_dir: str = "assets/system_models",
) -> tuple[dict[Any, dict[str, Any]], list[str]]:
    """Tải danh sách các mô hình classification và metric từ file YAML hệ thống."""
    file_path = os.path.join(base_dir, "classification.yml")
    with open(file_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    models: dict[Any, dict[str, Any]] = {}
    for key, model_info in data["Classification_models"].items():
        model_name = model_info["model"]
        model_cls = _MODEL_CLASSES.get(model_name)
        if model_cls is None:
            # Không dùng eval(): tránh thực thi mã tùy ý từ tên model trong YAML.
            # Model mới phải được đăng ký tường minh trong _MODEL_CLASSES.
            raise UnknownModelError(model_name)
        params = model_info.get("params")
        if params is None or params == []:
            params = [{}]
        models[key] = {
            "model": model_cls(),
            "params": params,
        }
    metric_list = data["metric_list"]
    return models, metric_list


def get_config(
    file: Any,
) -> tuple[
    str, list[str], str, list[str], str, dict[Any, dict[str, Any]], str, float | None
]:
    """Đọc cấu hình huấn luyện từ file YAML."""
    config = yaml.safe_load(file)
    choose = config["choose"]
    list_feature = config["list_feature"]
    target = config["target"]
    metric_sort = config["metric_sort"]
    metric_sort = metric_sort.strip().lower().replace(" ", "_")

    search_algorithm = config.get("search_algorithm", "grid_search")
    max_time = config.get("max_time", None)

    models, metric_list = get_model()
    return (
        choose,
        list_feature,
        target,
        metric_list,
        metric_sort,
        models,
        search_algorithm,
        max_time,
    )
