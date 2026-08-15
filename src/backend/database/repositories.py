"""
Lớp Repository trừu tượng hóa và chuẩn hóa các thao tác cơ sở dữ liệu MongoDB (CLEAN-003).

Cung cấp DatasetRepository, JobRepository và UserRepository với type safety,
tự động serialize ObjectId sang string và xử lý lỗi nhất quán.
"""

from __future__ import annotations

from typing import Any

from bson.objectid import ObjectId
from pymongo.asynchronous.database import AsyncDatabase


def serialize_doc(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    """Chuyển đổi ObjectId và định dạng tài liệu MongoDB sang Python dict an toàn."""
    if not doc:
        return None
    serialized = dict(doc)
    if "_id" in serialized:
        serialized["_id"] = str(serialized["_id"])
    return serialized


def serialize_docs(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Serialize danh sách các tài liệu MongoDB."""
    return [d for d in (serialize_doc(doc) for doc in docs) if d is not None]


class DatasetRepository:
    """Repository quản lý collection 'tbl_Data' (cài đặt MongoDB: tbl_Data)."""

    def __init__(self, db: AsyncDatabase) -> None:
        self.db = db
        # P2-FIX: dùng đúng tên collection thực tế là 'tbl_Data'
        self.collection = db["tbl_Data"]

    async def get_by_id(self, dataset_id: str) -> dict[str, Any] | None:
        """Tìm kiếm Dataset theo _id (hỗ trợ cả ObjectId và string)."""
        query: dict[str, Any] = {"_id": dataset_id}
        if ObjectId.is_valid(dataset_id):
            query = {"$or": [{"_id": ObjectId(dataset_id)}, {"_id": dataset_id}]}
        doc = await self.collection.find_one(query)
        return serialize_doc(doc)

    async def get_by_username(self, username: str) -> list[dict[str, Any]]:
        """Lấy tất cả Dataset thuộc về một người dùng."""
        cursor = self.collection.find({"username": username})
        docs = []
        async for doc in cursor:
            docs.append(doc)
        return serialize_docs(docs)

    async def get_by_name(
        self, name: str, username: str
    ) -> dict[str, Any] | None:
        """Tìm Dataset theo tên trong phạm vi sở hữu của một người dùng."""
        doc = await self.collection.find_one({"name": name, "username": username})
        return serialize_doc(doc)

    async def list_all(self) -> list[dict[str, Any]]:
        """Lấy toàn bộ Dataset trong hệ thống (không phân trang, dùng cho admin)."""
        cursor = self.collection.find()
        docs = []
        async for doc in cursor:
            docs.append(doc)
        return serialize_docs(docs)

    async def find_all(self, limit: int = 100, skip: int = 0) -> list[dict[str, Any]]:
        """Lấy danh sách tất cả Dataset có phân trang."""
        cursor = self.collection.find().skip(skip).limit(limit)
        docs = []
        async for doc in cursor:
            docs.append(doc)
        return serialize_docs(docs)

    async def create(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Tạo mới một Dataset và trả về bản ghi đã serialize."""
        to_insert = dict(doc)
        result = await self.collection.insert_one(to_insert)
        to_insert["_id"] = str(result.inserted_id)
        return to_insert

    async def update_by_id(
        self, dataset_id: str, updates: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Cập nhật thông tin Dataset theo _id."""
        query: dict[str, Any] = {"_id": dataset_id}
        if ObjectId.is_valid(dataset_id):
            query = {"$or": [{"_id": ObjectId(dataset_id)}, {"_id": dataset_id}]}
        await self.collection.update_one(query, {"$set": updates})
        return await self.get_by_id(dataset_id)

    async def delete_by_id(self, dataset_id: str) -> bool:
        """Xóa Dataset theo _id."""
        query: dict[str, Any] = {"_id": dataset_id}
        if ObjectId.is_valid(dataset_id):
            query = {"$or": [{"_id": ObjectId(dataset_id)}, {"_id": dataset_id}]}
        result = await self.collection.delete_one(query)
        return result.deleted_count > 0


class JobRepository:
    """Repository quản lý collection 'tbl_Job' (cài đặt MongoDB: tbl_Job)."""

    def __init__(self, db: AsyncDatabase) -> None:
        self.db = db
        # P2-FIX: dùng đúng tên collection thực tế là 'tbl_Job'
        self.collection = db["tbl_Job"]

    async def get_by_id(self, job_id: str) -> dict[str, Any] | None:
        """Tìm kiếm Job theo ID (hỗ trợ trường id, _id string hoặc ObjectId)."""
        query: dict[str, Any] = {"$or": [{"id": job_id}, {"_id": job_id}]}
        if ObjectId.is_valid(job_id):
            query["$or"].append({"_id": ObjectId(job_id)})
        doc = await self.collection.find_one(query)
        return serialize_doc(doc)

    async def get_by_username(self, username: str) -> list[dict[str, Any]]:
        """Lấy danh sách các Job theo username."""
        cursor = self.collection.find({"username": username})
        docs = []
        async for doc in cursor:
            docs.append(doc)
        return serialize_docs(docs)

    @staticmethod
    def _user_scoped_identity_query(
        username: str, identifier: str
    ) -> dict[str, Any]:
        """Tạo query khớp Job theo `job_id` hoặc `_id`, giới hạn theo người dùng.

        Thay cho việc bắt ngoại lệ khi ép `ObjectId`, ta kiểm tra hợp lệ trước
        bằng `ObjectId.is_valid` nên không cần `try/except` ở tầng API.
        """
        or_clauses: list[dict[str, Any]] = [{"job_id": identifier}]
        if ObjectId.is_valid(identifier):
            or_clauses.append({"_id": ObjectId(identifier)})
        return {"username": username, "$or": or_clauses}

    async def get_for_user(
        self, username: str, identifier: str
    ) -> dict[str, Any] | None:
        """Lấy Job theo `job_id` hoặc `_id` trong phạm vi sở hữu của người dùng."""
        doc = await self.collection.find_one(
            self._user_scoped_identity_query(username, identifier)
        )
        return serialize_doc(doc)

    async def delete_for_user(self, username: str, identifier: str) -> bool:
        """Xóa Job theo `job_id` hoặc `_id` trong phạm vi sở hữu của người dùng."""
        result = await self.collection.delete_one(
            self._user_scoped_identity_query(username, identifier)
        )
        return result.deleted_count > 0

    async def find_all(self, limit: int = 100, skip: int = 0) -> list[dict[str, Any]]:
        """Lấy tất cả các Job trong hệ thống."""
        cursor = self.collection.find().skip(skip).limit(limit)
        docs = []
        async for doc in cursor:
            docs.append(doc)
        return serialize_docs(docs)

    async def create(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Tạo mới một Job."""
        to_insert = dict(doc)
        result = await self.collection.insert_one(to_insert)
        to_insert["_id"] = str(result.inserted_id)
        return to_insert

    async def update_status(
        self, job_id: str, status: int | str, extra: dict[str, Any] | None = None
    ) -> bool:
        """Cập nhật trạng thái của Job."""
        query: dict[str, Any] = {"$or": [{"id": job_id}, {"_id": job_id}]}
        if ObjectId.is_valid(job_id):
            query["$or"].append({"_id": ObjectId(job_id)})
        payload: dict[str, Any] = {"status": status}
        if extra:
            payload.update(extra)
        res = await self.collection.update_one(query, {"$set": payload})
        return res.modified_count > 0

    async def delete_by_id(self, job_id: str) -> bool:
        """Xóa Job theo ID."""
        query: dict[str, Any] = {"$or": [{"id": job_id}, {"_id": job_id}]}
        if ObjectId.is_valid(job_id):
            query["$or"].append({"_id": ObjectId(job_id)})
        result = await self.collection.delete_one(query)
        return result.deleted_count > 0


class UserRepository:
    """Repository quản lý collection 'tbl_User' (cài đặt MongoDB: tbl_User)."""

    def __init__(self, db: AsyncDatabase) -> None:
        self.db = db
        # P2-FIX: dùng đúng tên collection thực tế là 'tbl_User'
        self.collection = db["tbl_User"]

    async def get_by_username(self, username: str) -> dict[str, Any] | None:
        """Tìm kiếm người dùng theo username."""
        doc = await self.collection.find_one({"username": username})
        return serialize_doc(doc)

    async def find_all(self) -> list[dict[str, Any]]:
        """Lấy danh sách tất cả người dùng."""
        cursor = self.collection.find()
        docs = []
        async for doc in cursor:
            docs.append(doc)
        return serialize_docs(docs)
