# Thư viện chuẩn
import asyncio
import hashlib
import io
import logging
import math
import os
import pickle
import re
from typing import Annotated

import aiofiles
import joblib
import numpy as np
import pandas as pd
import yaml

# Thư viện bên thứ ba
from fastapi import (
    Depends,
    File,
    Header,
    HTTPException,
    Path,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRouter
from pymongo.asynchronous.database import AsyncDatabase

from automl.v2.minio import minIOStorage
from automl.v2.schemas import InputRequest
from automl.v2.service import (
    TrainingAccessDeniedError,
    TrainingDispatchStatus,
    TrainingIdempotencyConflictError,
    query_jobs,
    save_job,
    send_message,
    set_training_dispatch_status,
)
from config.providers import get_db

# Module nội bộ
from database.get_dataset import MongoDataLoader
from users.routers import get_current_user

exp = APIRouter(prefix="/v2/auto", tags=["API thí nghiệm"])
logger = logging.getLogger(__name__)

_SAFE_TRAINING_KEY = re.compile(r"[A-Za-z0-9._:-]{1,128}")
_SAFE_JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_SAFE_DISPATCH_STATUSES = frozenset({"needs_reconciliation", "pending", "sent"})
_PREDICTION_START_DELAY_SECONDS = 15

_CurrentUserDependency = Annotated[dict, Depends(get_current_user)]
_DatabaseDependency = Annotated[AsyncDatabase, Depends(get_db)]
_UploadedFileDependency = Annotated[UploadFile, File()]


def _log_boundary_error(
    event: str,
    exc: Exception,
    *,
    job_id: str | None = None,
) -> None:
    """Ghi loại lỗi tại boundary mà không phát tán message có thể chứa secret."""

    logger.error(
        "Lỗi tại biên API thí nghiệm",
        extra={
            "event": event,
            "error_type": type(exc).__name__,
            "job_id": job_id,
        },
    )


# API lấy ra danh sách đặc trưng của dataset
@exp.get("/features")
async def get_features_of_dataset(
    id_data: str,
    problem_type: str,
    request: Request,
    _current_user: _CurrentUserDependency,
):
    dataset = MongoDataLoader(request.app.state.db)
    try:
        features = await dataset.get_features_suggest_target(id_data, problem_type)

        return {"features": features}
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        _log_boundary_error("dataset_features_unavailable", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Không thể đọc đặc trưng của dataset.",
        ) from exc


@exp.get("/data")
async def get_data_of_dataset(
    id_data: str,
    request: Request,
    _current_user: _CurrentUserDependency,
):
    dataset = MongoDataLoader(request.app.state.db)
    try:
        data_preview, total_rows = await dataset.get_data_preview(id_data)

        if data_preview is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Không tìm thấy hoặc không thể đọc dataset.",
            )

        return {"rows": total_rows, "data": data_preview.to_dict(orient="records")}
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        _log_boundary_error("dataset_preview_unavailable", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Không thể đọc dữ liệu xem trước của dataset.",
        ) from exc


# API huấn luyện model: client -> Kafka -> server
@exp.post("/jobs/training")
async def distributed_training(
    input: InputRequest,
    idempotency_key: Annotated[
        str,
        Header(
            alias="Idempotency-Key",
            min_length=1,
            max_length=128,
            pattern=r"^[A-Za-z0-9._:-]+$",
        ),
    ],
    db: _DatabaseDependency,
    current_user: _CurrentUserDependency,
):
    """Gửi thông điệp vào Kafka để khởi tạo quá trình huấn luyện model."""

    owner_id = str(current_user.get("_id", ""))
    if not owner_id or str(input.id_user) != owner_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "TRAINING_OWNER_MISMATCH",
                "message": "Không được tạo training job cho người dùng khác.",
            },
        )

    try:
        submission = await save_job(
            input,
            db,
            owner_id=owner_id,
            idempotency_key=idempotency_key,
        )
    except TrainingAccessDeniedError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "TRAINING_DATASET_FORBIDDEN",
                "message": "Không được huấn luyện bằng dataset của người dùng khác.",
            },
        )
    except TrainingIdempotencyConflictError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "TRAINING_IDEMPOTENCY_CONFLICT",
                "message": (
                    "Idempotency-Key đã được dùng cho một training payload khác."
                ),
            },
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as exc:
        _log_boundary_error("training_store_unavailable", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "TRAINING_STORE_UNAVAILABLE",
                "message": "Chưa thể lưu training job.",
            },
        ) from exc

    if not submission.created:
        replay_status = (
            "success"
            if submission.dispatch_status is TrainingDispatchStatus.SENT
            else TrainingDispatchStatus.NEEDS_RECONCILIATION.value
        )
        return {
            "status": replay_status,
            "message": "Training job đã tồn tại cho Idempotency-Key này.",
            "job_id": submission.job_id,
            "replayed": True,
        }

    try:
        await send_message(
            os.getenv("KAFKA_TOPIC"),
            submission.job_id,
            submission.message,
        )
        await set_training_dispatch_status(
            db,
            owner_id=owner_id,
            job_id=submission.job_id,
            dispatch_status=TrainingDispatchStatus.SENT,
        )
    except Exception as exc:
        _log_boundary_error(
            "training_dispatch_uncertain",
            exc,
            job_id=submission.job_id,
        )
        try:
            await set_training_dispatch_status(
                db,
                owner_id=owner_id,
                job_id=submission.job_id,
                dispatch_status=TrainingDispatchStatus.NEEDS_RECONCILIATION,
            )
        except Exception as status_exc:
            _log_boundary_error(
                "training_dispatch_state_unavailable",
                status_exc,
                job_id=submission.job_id,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "TRAINING_DISPATCH_STATE_UNAVAILABLE",
                    "message": (
                        "Không thể xác nhận trạng thái publish của training job."
                    ),
                    "job_id": submission.job_id,
                },
            ) from status_exc
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "TRAINING_DISPATCH_UNCERTAIN",
                "message": "Training job cần được đối soát trước khi thử lại.",
                "job_id": submission.job_id,
            },
        ) from exc

    return {
        "status": "success",
        "message": "Đã khởi tạo training job.",
        "job_id": submission.job_id,
        "replayed": False,
    }


def _training_owner_id(current_user: dict) -> str:
    owner_id = str(current_user.get("_id", "")).strip()
    if not owner_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "TRAINING_AUTH_REQUIRED"},
        )
    return owner_id


def _validated_training_key(value: str) -> str:
    if not isinstance(value, str) or not _SAFE_TRAINING_KEY.fullmatch(value):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "INVALID_IDEMPOTENCY_KEY"},
        )
    return value


def _finite_result_number(
    value,
    *,
    field_name: str,
    minimum: float | None = None,
) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{field_name} không phải số")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{field_name} không hợp lệ")
    return result


def _completed_training_result(document: dict) -> dict:
    evidence = document.get("evaluation")
    if not isinstance(evidence, dict):
        raise TypeError("Thiếu evaluation evidence")
    metric = evidence.get("metric")
    model_version = evidence.get("model_version")
    features = evidence.get("input_features")
    if not isinstance(metric, str) or not metric.strip():
        raise ValueError("Metric không hợp lệ")
    if not isinstance(model_version, str) or not _SAFE_JOB_ID.fullmatch(model_version):
        raise ValueError("Model version không hợp lệ")
    if (
        not isinstance(features, list)
        or not features
        or any(not isinstance(item, str) or not item for item in features)
    ):
        raise ValueError("Input features không hợp lệ")
    calibration_error = evidence.get("calibration_error")
    if calibration_error is not None:
        calibration_error = _finite_result_number(
            calibration_error,
            field_name="calibration_error",
            minimum=0.0,
        )
    return {
        "status": "completed",
        "job_id": str(document["job_id"]),
        "metric": metric.strip().lower(),
        "metric_value": _finite_result_number(
            evidence.get("metric_value"),
            field_name="metric_value",
        ),
        "baseline_value": _finite_result_number(
            evidence.get("baseline_value"),
            field_name="baseline_value",
        ),
        "train_metric": _finite_result_number(
            evidence.get("train_metric"),
            field_name="train_metric",
        ),
        "cv_mean": _finite_result_number(
            evidence.get("cv_mean"),
            field_name="cv_mean",
        ),
        "cv_variance": _finite_result_number(
            evidence.get("cv_variance"),
            field_name="cv_variance",
            minimum=0.0,
        ),
        "calibration_error": calibration_error,
        "model_version": model_version,
        "input_schema": {"features": list(features)},
        "decision_threshold": None,
    }


@exp.get("/jobs/by-idempotency/{idempotency_key}")
async def reconcile_training_job(
    idempotency_key: Annotated[str, Path(min_length=1, max_length=128)],
    db: _DatabaseDependency,
    current_user: _CurrentUserDependency,
):
    """Đối soát training bằng key và owner đã xác thực."""
    owner_id = _training_owner_id(current_user)
    safe_key = _validated_training_key(idempotency_key)
    key_hash = hashlib.sha256(f"{owner_id}\0{safe_key}".encode()).hexdigest()
    document = await db.tbl_Job.find_one(
        {
            "_id": f"training-idempotency:{key_hash}",
            "user.id": owner_id,
        },
        {"_id": 0, "job_id": 1, "dispatch.status": 1},
    )
    if not document:
        return {"found": False}
    job_id = document.get("job_id")
    if not isinstance(job_id, str) or not _SAFE_JOB_ID.fullmatch(job_id):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "TRAINING_RECONCILIATION_UNAVAILABLE"},
        )
    dispatch = document.get("dispatch")
    dispatch_status = (
        str(dispatch.get("status", "pending"))
        if isinstance(dispatch, dict)
        else "pending"
    )
    if dispatch_status not in _SAFE_DISPATCH_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "TRAINING_RECONCILIATION_UNAVAILABLE"},
        )
    return {
        "found": True,
        "job_id": job_id,
        "dispatch_status": dispatch_status,
        "cost": 0.0,
    }


@exp.post("/jobs/results")
async def get_training_results(
    job_ids: list[str],
    db: _DatabaseDependency,
    current_user: _CurrentUserDependency,
):
    """Trả evidence typed của đúng một job thuộc owner đã xác thực."""
    owner_id = _training_owner_id(current_user)
    if (
        len(job_ids) != 1
        or not isinstance(job_ids[0], str)
        or not _SAFE_JOB_ID.fullmatch(job_ids[0])
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "INVALID_TRAINING_JOB_IDS"},
        )
    job_id = job_ids[0]
    document = await db.tbl_Job.find_one(
        {"job_id": job_id, "user.id": owner_id},
        {
            "_id": 0,
            "evaluation": 1,
            "job_id": 1,
            "status": 1,
        },
    )
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "TRAINING_JOB_NOT_FOUND"},
        )
    training_status = document.get("status")
    if training_status == 0:
        return {"status": "running", "job_id": job_id}
    if training_status == -1:
        return {
            "status": "failed",
            "job_id": job_id,
            "failure_code": "TRAINING_FAILED",
        }
    if training_status != 1:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "TRAINING_STATUS_UNAVAILABLE"},
        )
    try:
        return _completed_training_result(document)
    except (KeyError, TypeError, ValueError) as exc:
        _log_boundary_error(
            "training_evidence_unavailable",
            exc,
            job_id=job_id,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "TRAINING_EVIDENCE_UNAVAILABLE"},
        ) from exc


# API lấy danh sách job theo id_user => Thêm phân trang
@exp.get("/jobs/offset/{id_user}", response_model=dict)
async def get_jobs_offset(
    id_user: str,
    db: _DatabaseDependency,
    current_user: _CurrentUserDependency,
    page: int = Query(1, ge=1),
    limit: int = Query(5, ge=1),
):
    job_list_raw, total_pages, total_jobs = await query_jobs(id_user, page, limit, db)

    jobs_data = [{**job, "_id": str(job["_id"])} for job in job_list_raw]

    return {
        "data": jobs_data,
        "pagination": {
            "total_jobs": total_jobs,
            "total_pages": total_pages,
            "current_page": page,
            "next_page": page + 1 if page < total_pages else None,
            "prev_page": page - 1 if page > 1 else None,
        },
    }


# API lấy danh sách độ đo
async def read_yaml_async(file_path: str):
    async with aiofiles.open(file_path, mode="r", encoding="utf-8") as file:
        content = await file.read()

    metrics = yaml.safe_load(content)
    return metrics["metric_list"]


@exp.get("/metrics")
async def metrics(
    problem_type: str,
    _current_user: _CurrentUserDependency,
) -> dict:
    if not problem_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Không nhận diện được loại bài toán.",
        )

    base_dir = "assets/system_models"
    metrics = None
    if problem_type == "classification":
        file_path = os.path.join(base_dir, "classification.yml")
        metrics = await read_yaml_async(file_path=file_path)
    elif problem_type == "regression":
        file_path = os.path.join(base_dir, "regression.yml")
        metrics = await read_yaml_async(file_path=file_path)

    return {"metrics": metrics}


_prediction_tasks: dict[str, asyncio.Task] = {}


@exp.delete("/{job_id}/predictions")
async def cancel_prediction_task(
    job_id: str,
    _current_user: _CurrentUserDependency,
):
    """Hủy tác vụ dự đoán đang chạy của một job."""
    if job_id in _prediction_tasks:
        task = _prediction_tasks[job_id]
        task.cancel()
        return {"detail": f"Đã yêu cầu dừng tác vụ dự đoán của job {job_id}."}

    return {"detail": "Không tìm thấy tác vụ đang chạy cho job này."}


# Suy luận model tạm thời
async def inference_model_batch(
    job_id: str, user_id: str, df: pd.DataFrame, db: AsyncDatabase
):
    await asyncio.sleep(_PREDICTION_START_DELAY_SECONDS)
    job_collection = db.tbl_Job

    try:
        stored_model_data = await job_collection.find_one(
            {"job_id": job_id, "status": 1}
        )
        if not stored_model_data:
            return {"error": "Không tìm thấy job hoặc job chưa hoàn tất."}
    # Mongo driver có thể phát sinh nhiều loại lỗi tại boundary truy cập dữ liệu.
    except Exception as exc:  # noqa: BLE001
        _log_boundary_error("prediction_job_unavailable", exc, job_id=job_id)
        return {"error": "Không thể truy xuất thông tin model."}

    config = stored_model_data.get("config", {})
    list_feature = config.get("list_feature", [])
    target_name = config.get("target", "Target")

    model_url = stored_model_data.get("model")
    if not isinstance(model_url, dict):
        return {"error": "Thông tin lưu trữ model không hợp lệ."}
    bucket_name = model_url.get("bucket_name")
    model_path = model_url.get("object_name")
    if not isinstance(bucket_name, str) or not isinstance(model_path, str):
        return {"error": "Thông tin lưu trữ model không hợp lệ."}
    preprocessor_path = f"{user_id}/{job_id}/preprocessor.joblib"
    target_path = f"{user_id}/{job_id}/target.joblib"

    async def load_artifact(bucket, path, file_type):
        try:
            buffer = await asyncio.to_thread(minIOStorage.get_object, bucket, path)
            return await asyncio.to_thread(file_type.load, buffer)
        except Exception as exc:
            _log_boundary_error("prediction_artifact_unavailable", exc, job_id=job_id)
            raise ValueError("Không thể tải artifact của model.") from exc

    try:
        model, preprocessor, le_target = await asyncio.gather(
            load_artifact(bucket_name, model_path, pickle),
            load_artifact(bucket_name, preprocessor_path, joblib),
            load_artifact(bucket_name, target_path, joblib),
        )
    except ValueError as exc:
        _log_boundary_error("prediction_artifacts_incomplete", exc, job_id=job_id)
        return {"error": "Không thể tải đầy đủ artifact cần thiết của model."}

    missing_cols = set(list_feature) - set(df.columns)
    if missing_cols:
        missing_names = ", ".join(sorted(str(column) for column in missing_cols))
        return {"error": f"File tải lên thiếu các cột bắt buộc: {missing_names}"}

    data_to_predict = df[list_feature]

    try:
        X_new_transformed = await asyncio.to_thread(
            preprocessor.transform, data_to_predict
        )

        if isinstance(X_new_transformed, np.ndarray):
            X_new_transformed = np.nan_to_num(
                X_new_transformed, nan=0.0, posinf=0.0, neginf=0.0
            )
        elif hasattr(X_new_transformed, "toarray"):
            X_new_transformed = X_new_transformed.toarray()
            X_new_transformed = np.nan_to_num(
                X_new_transformed, nan=0.0, posinf=0.0, neginf=0.0
            )

        y_pred_raw = await asyncio.to_thread(model.predict, X_new_transformed)
        if hasattr(y_pred_raw, "ravel"):
            y_pred_raw = y_pred_raw.ravel()

        if le_target is not None:
            y_pred_final = await asyncio.to_thread(
                le_target.inverse_transform, y_pred_raw
            )
        else:
            y_pred_final = y_pred_raw

    # Pipeline của model bên thứ ba có thể ném nhiều loại lỗi khi transform/predict.
    except Exception as exc:  # noqa: BLE001
        _log_boundary_error("prediction_execution_failed", exc, job_id=job_id)
        return {"error": "Quá trình dự đoán thất bại."}

    return {"predictions": y_pred_final.tolist(), "target_name": target_name}


@exp.post("/{job_id}/predictions")
async def create_batch_prediction(
    job_id: str,
    file_data: _UploadedFileDependency,
    db: _DatabaseDependency,
    current_user: _CurrentUserDependency,
):
    user_id = str(current_user["_id"])
    filename = (file_data.filename or "").lower()

    try:
        contents = await file_data.read()
        file_stream = io.BytesIO(contents)

        if filename.endswith(".csv"):
            df = pd.read_csv(file_stream)
        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file_stream)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Chỉ hỗ trợ định dạng file .csv, .xls hoặc .xlsx.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        _log_boundary_error("prediction_input_unreadable", exc, job_id=job_id)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Không thể đọc file tải lên.",
        ) from exc

    prediction_task = asyncio.create_task(
        inference_model_batch(job_id, user_id, df, db)
    )

    _prediction_tasks[job_id] = prediction_task

    try:
        prediction_result = await prediction_task

        if "error" in prediction_result:
            raise HTTPException(status_code=422, detail=prediction_result["error"])

        predictions = prediction_result["predictions"]
        target_name = prediction_result.get("target_name", "Target")

        df[f"{target_name}_prediction"] = predictions
    except asyncio.CancelledError:
        del df
        raise HTTPException(status_code=499, detail="Quá trình dự đoán đã bị hủy.")
    except HTTPException:
        raise
    except Exception as exc:
        _log_boundary_error("prediction_response_failed", exc, job_id=job_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Không thể hoàn tất kết quả dự đoán.",
        ) from exc
    finally:
        _prediction_tasks.pop(job_id, None)

    output_stream = io.BytesIO()
    media_type = ""

    if filename.endswith(".csv"):
        df.to_csv(output_stream, index=False)
        media_type = "text/csv"
    elif filename.endswith((".xls", ".xlsx")):
        df.to_excel(output_stream, index=False)
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    output_stream.seek(0)

    return StreamingResponse(
        output_stream,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="predicted_{file_data.filename}"',
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )
