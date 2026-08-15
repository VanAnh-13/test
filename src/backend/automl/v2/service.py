# Standard Libraries
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

# Third-party Libraries
from bson.objectid import ObjectId
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import DuplicateKeyError

from automl.v2.schemas import InputRequest

# Local Modules
from infrastructure.messaging.kafka import get_producer


class TrainingAccessDeniedError(RuntimeError):
    """Caller không được dùng dataset đã chọn để tạo training job."""


class TrainingIdempotencyConflictError(RuntimeError):
    """Một idempotency key đã được dùng với payload khác."""


class TrainingDispatchStatus(StrEnum):
    """Trạng thái publish của một training job."""

    PENDING = "pending"
    SENT = "sent"
    NEEDS_RECONCILIATION = "needs_reconciliation"


@dataclass(frozen=True, slots=True)
class TrainingJobSubmission:
    """Kết quả tạo mới hoặc replay một training job."""

    job_id: str
    message: dict
    created: bool
    dispatch_status: TrainingDispatchStatus


async def send_message(topic: str, key: str, message: dict):
    try:
        producer = get_producer()
    except RuntimeError:
        raise ConnectionError("Kafka Producer not initialized in the lifespan.")

    await producer.send_and_wait(topic=topic, key=key.encode("utf-8"), value=message)


async def save_job(
    input: InputRequest,
    db: AsyncDatabase,
    *,
    owner_id: str,
    idempotency_key: str,
) -> TrainingJobSubmission:
    user_collection = db.tbl_User
    job_collection = db.tbl_Job
    data_collection = db.tbl_Data

    canonical_payload = json.dumps(
        {
            "config": input.config,
            "id_data": str(input.id_data),
            "id_user": owner_id,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    payload_fingerprint = hashlib.sha256(canonical_payload.encode()).hexdigest()
    key_hash = hashlib.sha256(f"{owner_id}\0{idempotency_key}".encode()).hexdigest()
    document_id = f"training-idempotency:{key_hash}"
    idempotency_filter = {"_id": document_id}

    existing_job = await job_collection.find_one(idempotency_filter)
    if existing_job:
        return _existing_submission(existing_job, payload_fingerprint)

    # Tìm tên người dùng
    user_doc = await user_collection.find_one(
        {"_id": ObjectId(owner_id)},
        {"username": 1},
    )

    if not user_doc:
        raise ValueError("User not found")
    user_name = user_doc.get("username")

    # Tìm tên dữ liệu
    data_doc = await data_collection.find_one(
        {"_id": ObjectId(input.id_data)},
        {"dataName": 1, "userId": 1, "activate": 1},
    )
    if not data_doc:
        raise ValueError("Data not found")
    dataset_owner = str(data_doc.get("userId", ""))
    is_public_dataset = dataset_owner == "0" and data_doc.get("activate") == 1
    if dataset_owner != owner_id and not is_public_dataset:
        raise TrainingAccessDeniedError
    data_name = data_doc.get("dataName")

    job_id = str(uuid4())
    created_at = datetime.now(UTC).timestamp()

    # Tạo một bản ghi job mới
    new_job = {
        "_id": document_id,
        "job_id": job_id,
        "config": input.config,
        "data": {"id": input.id_data, "name": data_name},
        "user": {"id": owner_id, "name": user_name},
        "status": 0,
        "activate": 0,
        "create_at": created_at,
        "idempotency": {
            "key_hash": key_hash,
            "payload_fingerprint": payload_fingerprint,
        },
        "dispatch": {
            "status": TrainingDispatchStatus.PENDING.value,
            "updated_at": created_at,
        },
    }

    msg_job = {"id_data": input.id_data, "config": input.config, "id_user": owner_id}

    try:
        await job_collection.insert_one(new_job)
    except DuplicateKeyError:
        existing_job = await job_collection.find_one(idempotency_filter)
        if not existing_job:
            raise
        return _existing_submission(existing_job, payload_fingerprint)

    return TrainingJobSubmission(
        job_id=job_id,
        message=msg_job,
        created=True,
        dispatch_status=TrainingDispatchStatus.PENDING,
    )


def _existing_submission(
    job: dict,
    expected_fingerprint: str,
) -> TrainingJobSubmission:
    stored_fingerprint = str(job.get("idempotency", {}).get("payload_fingerprint", ""))
    if not stored_fingerprint or stored_fingerprint != expected_fingerprint:
        raise TrainingIdempotencyConflictError
    raw_dispatch_status = str(
        job.get("dispatch", {}).get(
            "status",
            TrainingDispatchStatus.PENDING.value,
        )
    )
    try:
        dispatch_status = TrainingDispatchStatus(raw_dispatch_status)
    except ValueError as exc:
        raise RuntimeError("Training job có dispatch status không hợp lệ") from exc

    return TrainingJobSubmission(
        job_id=str(job["job_id"]),
        message={
            "id_data": str(job.get("data", {}).get("id", "")),
            "config": job.get("config", {}),
            "id_user": str(job.get("user", {}).get("id", "")),
        },
        created=False,
        dispatch_status=dispatch_status,
    )


async def set_training_dispatch_status(
    db: AsyncDatabase,
    *,
    owner_id: str,
    job_id: str,
    dispatch_status: TrainingDispatchStatus,
) -> None:
    """Cập nhật trạng thái publish bằng filter owner-scoped."""

    result = await db.tbl_Job.update_one(
        {"job_id": job_id, "user.id": owner_id},
        {
            "$set": {
                "dispatch.status": dispatch_status.value,
                "dispatch.updated_at": datetime.now(UTC).timestamp(),
            }
        },
    )
    if result.matched_count != 1:
        raise RuntimeError(
            "Không tìm thấy training job thuộc owner để cập nhật dispatch"
        )


async def query_jobs(
    id_user: str, page: int, limit: int, db: AsyncDatabase
) -> tuple[list[dict], int, int]:
    job_collection = db.tbl_Job

    filter_query = {"user.id": id_user}

    # Đảm bảo trang và limit hợp lệ
    page = max(1, page)
    limit = max(1, limit)
    offset = (page - 1) * limit

    # Tính tổng số trang
    total_jobs = await job_collection.count_documents(filter_query)
    total_pages = (total_jobs + limit - 1) // limit if total_jobs > 0 else 1

    # Nếu trang yêu cầu vượt quá tổng số trang, điều chỉnh offset về trang cuối cùng
    if page > total_pages > 0:
        page = total_pages
        offset = (page - 1) * limit

    # Projection để lấy các trường cần thiết, tránh tải dữ liệu lớn
    projection_fields = {
        # Các trường loại bỏ
        "model": 0,
        "config": 0,
        "activate": 0,
        "item": 0,
    }

    jobs_list_raw = (
        await job_collection.find(filter_query, projection=projection_fields)
        .sort("create_at", -1)
        .skip(offset)
        .limit(limit)
        .to_list(length=None)
    )

    for job in jobs_list_raw:
        for key, value in job.items():
            if isinstance(value, datetime):
                job[key] = value.timestamp()

    return jobs_list_raw, total_pages, total_jobs
