"""
Kiểm thử đơn vị cho Database Repositories (CLEAN-003).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from bson.objectid import ObjectId

from database.repositories import (
    DatasetRepository,
    JobRepository,
    UserRepository,
    serialize_doc,
    serialize_docs,
)


def test_serialize_doc_converts_objectid() -> None:
    """Kiểm tra hàm serialize_doc chuyển ObjectId thành string an toàn."""
    oid = ObjectId()
    doc = {"_id": oid, "name": "sample_dataset", "status": "ready"}
    serialized = serialize_doc(doc)
    assert serialized is not None
    assert serialized["_id"] == str(oid)
    assert serialized["name"] == "sample_dataset"


def test_serialize_doc_handles_none() -> None:
    """Kiểm tra hàm serialize_doc xử lý đầu vào None."""
    assert serialize_doc(None) is None


def test_serialize_docs_handles_list() -> None:
    """Kiểm tra hàm serialize_docs xử lý danh sách tài liệu."""
    oid1, oid2 = ObjectId(), ObjectId()
    docs = [{"_id": oid1, "x": 1}, {"_id": oid2, "x": 2}]
    res = serialize_docs(docs)
    assert len(res) == 2
    assert res[0]["_id"] == str(oid1)
    assert res[1]["_id"] == str(oid2)


@pytest.mark.asyncio
async def test_dataset_repository_crud() -> None:
    """Kiểm thử các thao tác CRUD cơ bản trên DatasetRepository."""
    mock_collection = MagicMock()
    mock_db = {"tbl_Data": mock_collection}

    repo = DatasetRepository(mock_db)  # type: ignore[arg-type]

    # Test create
    inserted_oid = ObjectId()
    mock_collection.insert_one = AsyncMock(
        return_value=MagicMock(inserted_id=inserted_oid)
    )
    new_dataset = await repo.create({"name": "test.csv", "username": "admin"})
    assert new_dataset["_id"] == str(inserted_oid)
    assert new_dataset["name"] == "test.csv"

    # Test get_by_id
    mock_collection.find_one = AsyncMock(
        return_value={"_id": inserted_oid, "name": "test.csv"}
    )
    found = await repo.get_by_id(str(inserted_oid))
    assert found is not None
    assert found["_id"] == str(inserted_oid)

    # Test delete_by_id
    mock_collection.delete_one = AsyncMock(return_value=MagicMock(deleted_count=1))
    deleted = await repo.delete_by_id(str(inserted_oid))
    assert deleted is True


@pytest.mark.asyncio
async def test_job_repository_crud() -> None:
    """Kiểm thử các thao tác CRUD cơ bản trên JobRepository."""
    mock_collection = MagicMock()
    mock_db = {"tbl_Job": mock_collection}

    repo = JobRepository(mock_db)  # type: ignore[arg-type]

    # Test create
    inserted_oid = ObjectId()
    mock_collection.insert_one = AsyncMock(
        return_value=MagicMock(inserted_id=inserted_oid)
    )
    new_job = await repo.create({"id": "job-123", "status": 0, "username": "admin"})
    assert new_job["_id"] == str(inserted_oid)
    assert new_job["id"] == "job-123"

    # Test update_status
    mock_collection.update_one = AsyncMock(return_value=MagicMock(modified_count=1))
    updated = await repo.update_status("job-123", 1, {"best_score": 0.95})
    assert updated is True

    # Test delete_by_id
    mock_collection.delete_one = AsyncMock(return_value=MagicMock(deleted_count=1))
    deleted = await repo.delete_by_id("job-123")
    assert deleted is True


@pytest.mark.asyncio
async def test_dataset_repository_get_by_name_scopes_by_user() -> None:
    """get_by_name phải giới hạn theo tên và username của người dùng."""
    mock_collection = MagicMock()
    mock_db = {"tbl_Data": mock_collection}
    repo = DatasetRepository(mock_db)  # type: ignore[arg-type]

    oid = ObjectId()
    mock_collection.find_one = AsyncMock(
        return_value={"_id": oid, "name": "iris.csv", "username": "alice"}
    )

    dataset = await repo.get_by_name("iris.csv", "alice")

    assert dataset is not None
    assert dataset["_id"] == str(oid)
    mock_collection.find_one.assert_awaited_once_with(
        {"name": "iris.csv", "username": "alice"}
    )


@pytest.mark.asyncio
async def test_dataset_repository_list_all_returns_all_serialized() -> None:
    """list_all trả về toàn bộ dataset đã serialize (dùng cho admin)."""
    mock_collection = MagicMock()
    mock_db = {"tbl_Data": mock_collection}
    repo = DatasetRepository(mock_db)  # type: ignore[arg-type]

    oid1, oid2 = ObjectId(), ObjectId()

    class _Cursor:
        def __aiter__(self):
            async def gen():
                yield {"_id": oid1, "name": "a"}
                yield {"_id": oid2, "name": "b"}

            return gen()

    mock_collection.find = MagicMock(return_value=_Cursor())

    datasets = await repo.list_all()

    assert [d["_id"] for d in datasets] == [str(oid1), str(oid2)]


@pytest.mark.asyncio
async def test_job_repository_get_for_user_scopes_and_matches_identity() -> None:
    """get_for_user giới hạn theo username và khớp job_id hoặc _id ObjectId."""
    mock_collection = MagicMock()
    mock_db = {"tbl_Job": mock_collection}
    repo = JobRepository(mock_db)  # type: ignore[arg-type]

    oid = ObjectId()
    mock_collection.find_one = AsyncMock(
        return_value={"_id": oid, "job_id": "job-1", "username": "alice"}
    )

    job = await repo.get_for_user("alice", str(oid))

    assert job is not None and job["_id"] == str(oid)
    query = mock_collection.find_one.await_args.args[0]
    assert query["username"] == "alice"
    assert {"job_id": str(oid)} in query["$or"]
    assert {"_id": oid} in query["$or"]


@pytest.mark.asyncio
async def test_job_repository_get_for_user_with_non_objectid_identifier() -> None:
    """Khi identifier không phải ObjectId, chỉ khớp theo job_id (không raise)."""
    mock_collection = MagicMock()
    mock_db = {"tbl_Job": mock_collection}
    repo = JobRepository(mock_db)  # type: ignore[arg-type]

    mock_collection.find_one = AsyncMock(return_value=None)

    job = await repo.get_for_user("alice", "not-an-objectid")

    assert job is None
    query = mock_collection.find_one.await_args.args[0]
    assert query["$or"] == [{"job_id": "not-an-objectid"}]


@pytest.mark.asyncio
async def test_job_repository_delete_for_user() -> None:
    """delete_for_user trả True khi xóa được và giới hạn theo username."""
    mock_collection = MagicMock()
    mock_db = {"tbl_Job": mock_collection}
    repo = JobRepository(mock_db)  # type: ignore[arg-type]

    mock_collection.delete_one = AsyncMock(return_value=MagicMock(deleted_count=1))

    deleted = await repo.delete_for_user("alice", "job-1")

    assert deleted is True
    query = mock_collection.delete_one.await_args.args[0]
    assert query["username"] == "alice"


@pytest.mark.asyncio
async def test_user_repository_get_by_username() -> None:
    """Kiểm thử tìm kiếm user theo username."""
    mock_collection = MagicMock()
    mock_db = {"tbl_User": mock_collection}

    repo = UserRepository(mock_db)  # type: ignore[arg-type]

    mock_collection.find_one = AsyncMock(
        return_value={"_id": ObjectId(), "username": "admin", "role": "admin"}
    )
    user = await repo.get_by_username("admin")
    assert user is not None
    assert user["username"] == "admin"
    assert isinstance(user["_id"], str)
