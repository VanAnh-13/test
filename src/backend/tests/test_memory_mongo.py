from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest
from pymongo import ASCENDING, DESCENDING, ReturnDocument


class _AsyncCursor:
    def __init__(self, documents):
        self.documents = list(documents)
        self.sort_args = None
        self.limit_value = None

    def sort(self, *args):
        self.sort_args = args
        return self

    def limit(self, value):
        self.limit_value = value
        return self

    def __aiter__(self):
        async def iterate():
            documents = self.documents
            if self.limit_value is not None:
                documents = documents[: self.limit_value]
            for document in documents:
                yield document

        return iterate()


def _fact_document(user_id="owner", key="fact-1", **overrides):
    document = {
        "user_id": user_id,
        "key": key,
        "content": "Iris has 150 rows",
        "category": "dataset",
        "confidence": 0.9,
        "source": "tool:get_dataset_info",
        "created_at": 10.0,
        "updated_at": 20.0,
        "access_count": 0,
    }
    document.update(overrides)
    return document


def _collection():
    collection = Mock()
    collection.create_index = AsyncMock(return_value="user_key_unique")
    collection.update_one = AsyncMock()
    collection.find_one_and_update = AsyncMock()
    collection.delete_one = AsyncMock()
    collection.delete_many = AsyncMock()
    return collection


@pytest.mark.asyncio
async def test_save_creates_compound_unique_index_and_upserts_complete_document():
    from hagent.agent.memory import Fact
    from hagent.agent.memory.mongo_store import MongoFactStore

    collection = _collection()
    store = MongoFactStore(collection=collection)
    fact = Fact(
        key="dataset-d1",
        content="Iris has 150 rows",
        category="dataset",
        confidence=0.8,
        source="tool:get_dataset_info",
        created_at=10.0,
    )

    await store.save("owner", fact)

    collection.create_index.assert_awaited_once_with(
        [("user_id", ASCENDING), ("key", ASCENDING)],
        unique=True,
        name="memory_user_key_unique",
    )
    collection.update_one.assert_awaited_once()
    query, update = collection.update_one.await_args.args
    assert query == {"user_id": "owner", "key": "dataset-d1"}
    assert collection.update_one.await_args.kwargs == {"upsert": True}
    assert update["$set"]["user_id"] == "owner"
    assert update["$set"]["key"] == "dataset-d1"
    assert update["$set"]["content"] == fact.content
    assert update["$set"]["category"] == "dataset"
    assert update["$set"]["confidence"] == 0.8
    assert update["$set"]["source"] == "tool:get_dataset_info"
    assert update["$set"]["access_count"] == 0
    assert isinstance(update["$set"]["updated_at"], float)
    assert update["$setOnInsert"] == {"created_at": 10.0}


@pytest.mark.asyncio
async def test_get_is_owner_scoped_and_increments_access_count():
    from hagent.agent.memory.mongo_store import MongoFactStore

    collection = _collection()
    collection.find_one_and_update.return_value = _fact_document(access_count=3)
    store = MongoFactStore(collection=collection)

    fact = await store.get("owner", "fact-1")

    assert fact is not None
    assert fact.key == "fact-1"
    assert fact.access_count == 3
    collection.find_one_and_update.assert_awaited_once_with(
        {"user_id": "owner", "key": "fact-1"},
        {"$inc": {"access_count": 1}},
        return_document=ReturnDocument.AFTER,
    )


@pytest.mark.asyncio
async def test_search_and_get_all_are_owner_scoped_sorted_and_bounded():
    from hagent.agent.memory.mongo_store import MongoFactStore

    collection = _collection()
    search_cursor = _AsyncCursor([_fact_document(key="a")])
    all_cursor = _AsyncCursor([_fact_document(key="b")])
    collection.find = Mock(side_effect=[search_cursor, all_cursor])
    store = MongoFactStore(collection=collection)

    search_results = await store.search(
        "owner",
        category="dataset",
        query="iris.csv",
        limit=7,
    )
    all_results = await store.get_all("owner")

    assert [fact.key for fact in search_results] == ["a"]
    assert [fact.key for fact in all_results] == ["b"]
    assert collection.find.call_args_list[0].args[0] == {
        "user_id": "owner",
        "category": "dataset",
        "content": {"$regex": "iris\\.csv", "$options": "i"},
    }
    assert collection.find.call_args_list[1].args[0] == {"user_id": "owner"}
    assert search_cursor.sort_args == ("updated_at", DESCENDING)
    assert search_cursor.limit_value == 7
    assert all_cursor.sort_args == ("updated_at", DESCENDING)


@pytest.mark.asyncio
async def test_delete_and_clear_cannot_cross_user_boundary():
    from hagent.agent.memory.mongo_store import MongoFactStore

    collection = _collection()
    collection.delete_one.return_value = Mock(deleted_count=1)
    collection.delete_many.return_value = Mock(deleted_count=4)
    store = MongoFactStore(collection=collection)

    assert await store.delete("owner", "fact-1") is True
    assert await store.clear("owner") == 4

    collection.delete_one.assert_awaited_once_with(
        {"user_id": "owner", "key": "fact-1"}
    )
    collection.delete_many.assert_awaited_once_with({"user_id": "owner"})


@pytest.mark.asyncio
async def test_configured_mongo_runtime_failure_is_not_hidden():
    from hagent.agent.memory.mongo_store import MongoFactStore

    collection = _collection()
    collection.create_index.side_effect = RuntimeError("mongo unavailable")
    store = MongoFactStore(collection=collection)

    with pytest.raises(RuntimeError, match="mongo unavailable"):
        await store.get_all("owner")


def test_auto_backend_uses_local_without_mongodb_env(monkeypatch, tmp_path):
    from hagent.agent import memory
    from hagent.bridge import config

    monkeypatch.delenv("MONGODB_CONNECT", raising=False)
    monkeypatch.setattr(
        config,
        "load_config",
        lambda: {
            "memory": {
                "backend": "auto",
                "storage_dir": str(tmp_path / "facts"),
            }
        },
    )

    store = memory.create_fact_store()

    assert isinstance(store, memory.LocalFactStore)


def test_auto_backend_uses_mongo_when_env_is_configured(monkeypatch):
    from hagent.agent import memory
    from hagent.bridge import config

    sentinel = object()
    captured = {}

    def mongo_store(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setenv("MONGODB_CONNECT", "mongo.example:27017")
    monkeypatch.setenv("MONGODB_DB_NAME", "memory-test")
    monkeypatch.setattr(
        config,
        "load_config",
        lambda: {
            "memory": {"backend": "auto", "collection": "facts"},
            "mongodb": {"connect": "ignored:27017", "db_name": "ignored"},
        },
    )
    monkeypatch.setattr("hagent.agent.memory.mongo_store.MongoFactStore", mongo_store)

    store = memory.create_fact_store()

    assert store is sentinel
    assert captured == {
        "connect": "mongo.example:27017",
        "db_name": "memory-test",
        "collection_name": "facts",
    }


def test_configured_mongo_constructor_failure_does_not_fallback(monkeypatch):
    from hagent.agent import memory
    from hagent.bridge import config

    monkeypatch.setenv("MONGODB_CONNECT", "mongo.example:27017")
    monkeypatch.setattr(
        config,
        "load_config",
        lambda: {"memory": {"backend": "auto"}},
    )

    def fail_closed(**kwargs):
        raise RuntimeError("cannot configure mongo")

    monkeypatch.setattr("hagent.agent.memory.mongo_store.MongoFactStore", fail_closed)

    with pytest.raises(RuntimeError, match="cannot configure mongo"):
        memory.create_fact_store()


def test_yaml_defaults_memory_backend_to_auto():
    from hagent.bridge.config import load_config

    memory_config = load_config()["memory"]
    assert memory_config["backend"] == "auto"
    assert memory_config["collection"] == "memory_facts"


@pytest.mark.asyncio
async def test_search_zero_limit_returns_no_documents():
    from hagent.agent.memory.mongo_store import MongoFactStore

    collection = _collection()
    collection.find = Mock()
    store = MongoFactStore(collection=collection)

    assert await store.search("owner", limit=0) == []
    collection.find.assert_not_called()


def test_auto_backend_fails_closed_when_mongodb_env_is_empty(monkeypatch):
    from hagent.agent import memory
    from hagent.bridge import config

    monkeypatch.setenv("MONGODB_CONNECT", "")
    monkeypatch.setattr(
        config,
        "load_config",
        lambda: {
            "memory": {"backend": "auto"},
            "mongodb": {"connect": "localhost:27017"},
        },
    )

    with pytest.raises(RuntimeError, match="requires a non-empty MONGODB_CONNECT"):
        memory.create_fact_store()
