"""
Unit tests for Episodic & Semantic Memory Eviction and Lazy Loading (REFAC-017).
"""

from __future__ import annotations

import pytest

from hagent.agent.memory import create_episodic_memory, create_semantic_memory
from hagent.agent.memory.episodic import EpisodicMemory
from hagent.agent.memory.semantic import SemanticMemory


@pytest.mark.asyncio
async def test_episodic_memory_lru_eviction() -> None:
    """EpisodicMemory giới hạn số lượng bản ghi tối đa max_entries theo chính sách LRU."""
    # Khởi tạo memory với dung lượng tối đa 3 entries
    memory = EpisodicMemory(max_entries=3)
    user_id = "user_test_lru"

    # 1. Lưu 3 bản ghi ban đầu
    await memory.store(user_id, "Event 1", record_id="r1")
    await memory.store(user_id, "Event 2", record_id="r2")
    await memory.store(user_id, "Event 3", record_id="r3")

    assert memory.count(user_id) == 3

    # 2. Truy cập r1 để cập nhật last_accessed_at (làm r1 mới hơn r2)
    fetched_r1 = await memory.get(user_id, "r1")
    assert fetched_r1 is not None

    # 3. Lưu thêm bản ghi thứ 4 (r4) -> r2 là bản ghi LRU cũ nhất bị evict
    await memory.store(user_id, "Event 4", record_id="r4")

    assert memory.count(user_id) == 3
    assert await memory.get(user_id, "r2") is None  # r2 đã bị xóa
    assert await memory.get(user_id, "r1") is not None  # r1 vẫn còn
    assert await memory.get(user_id, "r3") is not None  # r3 vẫn còn
    assert await memory.get(user_id, "r4") is not None  # r4 mới nhất


@pytest.mark.asyncio
async def test_semantic_memory_lazy_loading() -> None:
    """SemanticMemory không load embedding provider trong __init__ mà chỉ lazy load khi cần."""
    load_count = 0

    class MockEmbedder:
        def embed_query(self, text: str) -> list[float]:
            return [0.1, 0.2, 0.3]

        def embed(self, text: str) -> list[float]:
            return [0.1, 0.2, 0.3]

    def mock_factory() -> MockEmbedder:
        nonlocal load_count
        load_count += 1
        return MockEmbedder()

    # 1. Khởi tạo với lazy_load=True
    memory = SemanticMemory(
        embedder_factory=mock_factory,
        max_entries=5,
        lazy_load=True,
    )

    # Chưa load embedder
    assert load_count == 0
    assert memory.is_embedder_loaded() is False

    # 2. Lưu bản ghi không cung cấp vector -> kích hoạt lazy load
    await memory.store("user1", "Kiến thức máy học mới")
    assert load_count == 1
    assert memory.is_embedder_loaded() is True

    # 3. Recall dùng lại embedder đã nạp sẵn, không khởi tạo lại
    results = await memory.recall("user1", "máy học")
    assert load_count == 1
    assert len(results) == 1


@pytest.mark.asyncio
async def test_semantic_memory_importance_eviction() -> None:
    """SemanticMemory ưu tiên giữ lại bản ghi có importance cao hoặc access_count lớn khi vượt max_entries."""
    memory = SemanticMemory(max_entries=3)
    user_id = "user_sem_evict"

    # Lưu 3 bản ghi với importance khác nhau
    # s1: low importance (0.2)
    await memory.store(user_id, "Text 1", importance=0.2, record_id="s1")
    # s2: high importance (5.0)
    await memory.store(user_id, "Text 2", importance=5.0, record_id="s2")
    # s3: medium importance (1.0)
    await memory.store(user_id, "Text 3", importance=1.0, record_id="s3")

    assert memory.count(user_id) == 3

    # Lưu thêm bản ghi thứ 4 -> s1 có score thấp nhất sẽ bị giải phóng
    await memory.store(user_id, "Text 4", importance=1.5, record_id="s4")

    assert memory.count(user_id) == 3
    assert await memory.get(user_id, "s1") is None  # s1 bị loại bỏ
    assert await memory.get(user_id, "s2") is not None  # s2 (importance 5.0) giữ nguyên
    assert await memory.get(user_id, "s3") is not None  # s3 giữ nguyên
    assert await memory.get(user_id, "s4") is not None  # s4 giữ nguyên


def test_memory_factories_from_config() -> None:
    """create_episodic_memory và create_semantic_memory khởi tạo đúng thông số cấu hình."""
    ep_mem = create_episodic_memory(max_entries=25)
    assert ep_mem.max_entries == 25

    sem_mem = create_semantic_memory(max_entries=40, lazy_load=True)
    assert sem_mem.max_entries == 40
    assert sem_mem.lazy_load is True
