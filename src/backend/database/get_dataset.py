import logging
import math
import re
from io import BytesIO

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from bson.errors import InvalidId
from bson.objectid import ObjectId
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import PyMongoError

from automl.process_classification import preprocess_data as classification
from automl.process_regression import preprocess_data as regression
from automl.v2.minio import minIOStorage

logger = logging.getLogger(__name__)


def _validated_evaluation(evaluation: dict) -> dict:
    """Chỉ cho phép evaluation summary hữu hạn và storage reference ổn định."""
    if not isinstance(evaluation, dict):
        raise ValueError("Evaluation evidence phải là object")
    metric = evaluation.get("metric")
    if not isinstance(metric, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", metric):
        raise ValueError("Evaluation metric không hợp lệ")
    numeric_fields = (
        "metric_value",
        "baseline_value",
        "train_metric",
        "cv_mean",
        "cv_variance",
    )
    numbers = {}
    for field_name in numeric_fields:
        value = evaluation.get(field_name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"Evaluation {field_name} không phải số")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Evaluation {field_name} không hữu hạn")
        numbers[field_name] = number
    if numbers["cv_variance"] < 0:
        raise ValueError("Evaluation cv_variance không được âm")
    calibration_error = evaluation.get("calibration_error")
    if calibration_error is not None:
        if (
            not isinstance(calibration_error, (int, float))
            or isinstance(calibration_error, bool)
            or not math.isfinite(float(calibration_error))
            or float(calibration_error) < 0
        ):
            raise ValueError("Evaluation calibration_error không hợp lệ")
        calibration_error = float(calibration_error)
    input_features = evaluation.get("input_features")
    if (
        not isinstance(input_features, list)
        or not input_features
        or any(
            not isinstance(item, str) or not item or len(item) > 256
            for item in input_features
        )
    ):
        raise ValueError("Evaluation input_features không hợp lệ")
    model_version = evaluation.get("model_version")
    if not isinstance(model_version, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}",
        model_version,
    ):
        raise ValueError("Evaluation model_version không hợp lệ")
    model_storage = evaluation.get("model_storage")
    if not isinstance(model_storage, dict) or any(
        not isinstance(model_storage.get(key), str) or not model_storage[key]
        for key in ("bucket_name", "object_name")
    ):
        raise ValueError("Evaluation model_storage không hợp lệ")
    return {
        "metric": metric,
        **numbers,
        "calibration_error": calibration_error,
        "input_features": list(input_features),
        "model_storage": {
            "bucket_name": model_storage["bucket_name"],
            "object_name": model_storage["object_name"],
        },
        "model_version": model_version,
    }


class MongoDataLoader:
    def __init__(self, db: AsyncDatabase):
        self.__data_collection = db.tbl_Data

    async def _get_data_link_from_db(
        self, id_data: str
    ) -> tuple[str | None, str | None]:
        """Lấy data link từ MongoDB theo ID"""

        try:
            data = await self.__data_collection.find_one(
                {"_id": ObjectId(id_data)}, {"data_link": 1}
            )
            if data:
                data_link = data.get("data_link", {})
                return data_link.get("bucket_name"), data_link.get("object_name")
            return None, None
        except (InvalidId, TypeError, PyMongoError):
            logger.warning(
                "Không lấy được data_link cho dataset %s", id_data, exc_info=True
            )
            return None, None

    async def get_data_preview(
        self, id_data: str, num_rows: int = 50
    ) -> tuple[pd.DataFrame | None, list | None]:
        bucket_name, object_name = await self._get_data_link_from_db(id_data)
        if not (bucket_name and object_name):
            return None

        try:
            parquet_stream = minIOStorage.get_object(bucket_name, object_name)
            df_retrieved = pd.read_parquet(parquet_stream)

            total_rows = len(df_retrieved)
            df_preview = df_retrieved.head(num_rows)

            return df_preview, total_rows

        except Exception:
            # Best-effort preview: nhiều nguồn lỗi (storage, parquet, pandas) nên
            # ghi log đầy đủ và trả về giá trị rỗng thay vì làm hỏng request.
            logger.warning(
                "Không tạo được preview cho dataset %s", id_data, exc_info=True
            )
            return None, 0

    @classmethod
    def analyze_column_for_target(cls, series: pd.Series, threshold_unique=50) -> str:
        """
        Trả về: 'classification', 'regression', hoặc 'both' (nếu không chắc chắn)
        """
        try:
            # Xử lý dữ liệu null
            clean_series = series.dropna()
            if clean_series.empty:
                return "none"

            # Ngày tháng thường không làm Target trực tiếp (trừ Time Series Forecasting đặc thù)
            if pd.api.types.is_datetime64_any_dtype(
                clean_series
            ) or pd.api.types.is_timedelta64_dtype(clean_series):
                return "none"

            # Thử ép sang kiểu số
            series_numeric = pd.to_numeric(clean_series, errors="coerce").dropna()
            is_numeric_column = len(series_numeric) >= 0.5 * len(clean_series)

            if not is_numeric_column:
                # Dạng Text/Boolean -> Classification
                return "classification"
            else:
                clean_series = series_numeric

            # Binary (0/1, True/False) -> Classification
            if clean_series.nunique() <= 2:
                return "classification"

            # Số thực (Float) có phần thập phân -> Regression
            is_float = not np.all(np.isclose(clean_series % 1, 0))
            if is_float:
                return "regression"

            # Số nguyên (Integer) -> Vùng nhập nhằng (Gray Area)
            num_unique = clean_series.nunique()

            # Nếu unique quá lớn so với số dòng -> Regression
            if num_unique > 0.9 * len(clean_series):
                return "regression"

            # Nếu unique nhỏ -> Ưu tiên Classification, nhưng Regression vẫn khả thi
            if num_unique <= threshold_unique:
                return "both"

            return "regression"
        except Exception:
            # Heuristic thuần tính toán: nếu gặp dữ liệu bất thường thì coi như
            # không xác định được loại target; log để phục vụ debug.
            logger.warning("Không phân tích được cột target", exc_info=True)
            return "none"

    async def get_features_suggest_target(
        self, id_data: str, selected_problem_type: str, num_row: int = 1000
    ) -> dict | None:
        bucket_name, object_name = await self._get_data_link_from_db(id_data)
        if not (bucket_name and object_name):
            return None

        # Loại bỏ các cột là ID
        pattern = r"^(?i:id|stt|no|key|code|uuid|guid)$|(?i:.*_id)$|^ID_.*$"

        features = {}

        try:
            # Lấy data stream từ MinIO
            response = minIOStorage.get_object(bucket_name, object_name)
            file_buffer = BytesIO(response.read())

            # Đọc metadata & preview data
            parquet_file = pq.ParquetFile(file_buffer)
            schema_names = parquet_file.schema.names

            table = parquet_file.read_row_group(0)
            df_preview = table.to_pandas().head(num_row)

            for col_name in schema_names:
                if re.match(pattern, col_name):
                    features[col_name] = False
                    continue

                series: pd.Series = df_preview[col_name]

                if series.isnull().all() or series.nunique() <= 1:
                    features[col_name] = False
                    continue

                suggested_type = self.analyze_column_for_target(series)

                if selected_problem_type == "classification":
                    # classification & both
                    if suggested_type in ["classification", "both"]:
                        features[col_name] = True
                    else:
                        features[col_name] = False

                elif selected_problem_type == "regression":
                    # regression & both
                    if suggested_type in ["regression", "both"]:
                        features[col_name] = True
                    else:
                        features[col_name] = False

                else:
                    features[col_name] = False

            return features
        except Exception:
            logger.warning(
                "Không suy luận được features cho dataset %s", id_data, exc_info=True
            )
            return None

    async def get_processed_data(
        self, id_data: str, list_features: list, target: str, problem_type: str
    ) -> (
        tuple[pd.DataFrame, pd.DataFrame, object, object]
        | tuple[None, None, None, None]
    ):
        """Load dataset từ MinIO"""
        bucket_name, object_name = await self._get_data_link_from_db(id_data)
        if not (bucket_name and object_name):
            return None, None, None, None

        try:
            parquet_stream = minIOStorage.get_object(bucket_name, object_name)
            df_retrieved = pd.read_parquet(parquet_stream)

            X_processed, y_processed, preprocessor, le_target = None, None, None, None

            if problem_type == "classification":
                # Classification processs
                X_processed, y_processed, preprocessor, le_target = classification(
                    list_features, target, df_retrieved
                )
            else:
                # Regression process
                X_processed, y_processed, preprocessor, le_target = regression(
                    list_features, target, df_retrieved
                )

            return X_processed, y_processed, preprocessor, le_target

        except Exception:
            logger.warning(
                "Không xử lý được dataset %s từ MinIO", id_data, exc_info=True
            )
            return None, None, None, None


class MongoJob:
    def __init__(self, db: AsyncDatabase):
        self.__job_collection = db.tbl_Job

    async def update_failure(self, job_id: str, error_msg: str):
        update_data = {"$set": {"status": -1, "infor": error_msg}}
        await self.__job_collection.update_one({"job_id": job_id}, update_data)

    async def update_success(self, job_id: str, final_result: dict):
        owner_id = final_result.get("owner_id")
        if not isinstance(owner_id, str) or not owner_id:
            raise ValueError("Training result thiếu owner_id")
        evaluation = _validated_evaluation(final_result.get("evaluation"))
        update_data = {
            "$set": {
                "best_model_id": final_result["best_model_id"],
                "best_model": final_result["best_model"],
                "model": {
                    "bucket_name": final_result["model"].get("bucket_name", ""),
                    "object_name": final_result["model"].get("object_name", ""),
                },
                "best_params": final_result["best_params"],
                "best_score": final_result["best_score"],
                "orther_model_scores": final_result["model_scores"],
                "evaluation": evaluation,
                "status": 1,
                # Thông tin giới hạn thời gian
                "time_limit_reached": final_result.get("time_limit_reached", False),
                "completed_models": final_result.get("completed_models"),
                "total_models": final_result.get("total_models"),
            }
        }
        result = await self.__job_collection.update_one(
            {"job_id": job_id, "user.id": owner_id},
            update_data,
        )
        if result.matched_count != 1:
            raise ValueError("Không tìm thấy training job thuộc owner để lưu kết quả")
