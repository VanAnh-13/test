"""
Semantic Memory with Importance-based Eviction and Lazy Embedding Loading (REFAC-017).

Lưu trữ kiến thức/facts ngữ nghĩa, hỗ trợ vector similarity search.
- Lazy-load embedding model: không import/khởi tạo mô hình nặng trong `__init__`.
- Importance-based eviction: giải phóng các bản ghi có độ quan trọng và tần suất truy cập thấp nhất khi đầy bộ nhớ.
"""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

DEFAULT_SEMANTIC_MAX_ENTRIES = 100


@dataclass
class SemanticRecord:
    """Một bản ghi trong Semantic Memory."""

    record_id: str
    user_id: str
    text: str
    embedding: list[float] | None = None
    importance: float = 1.0  # 0.1 -> 10.0
    access_count: int = 0
    meta: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SemanticRecord:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


class SemanticMemory:
    """Bộ nhớ Semantic với Lazy Embedding Loading và Importance/Frequency Eviction."""

    def __init__(
        self,
        storage_dir: str | Path | None = None,
        embedder_factory: Callable[[], Any] | None = None,
        max_entries: int = DEFAULT_SEMANTIC_MAX_ENTRIES,
        lazy_load: bool = True,
    ) -> None:
        self.max_entries = max(1, int(max_entries))
        self.lazy_load = lazy_load
        self._dir = Path(storage_dir) if storage_dir else None
        if self._dir:
            self._dir.mkdir(parents=True, exist_ok=True)

        self._embedder_factory = embedder_factory
        self._embedder: Any | None = None

        if not lazy_load and self._embedder_factory:
            self._embedder = self._embedder_factory()

        self._memory: dict[str, dict[str, SemanticRecord]] = {}

    def _get_embedder(self) -> Any | None:
        """Lazy initialization của embedding provider."""
        if self._embedder is None and self._embedder_factory:
            logger.debug("Lazy loading embedding model in SemanticMemory...")
            self._embedder = self._embedder_factory()
        return self._embedder

    def is_embedder_loaded(self) -> bool:
        """Kiểm tra xem embedding model đã được khởi tạo thực sự hay chưa."""
        return self._embedder is not None

    def _calculate_retention_score(self, record: SemanticRecord) -> float:
        """
        Tính điểm giữ lại: kết hợp độ quan trọng (importance) và tần suất truy cập (access_count).
        Score = importance * (1.0 + log(1 + access_count)).
        """
        return float(
            record.importance * (1.0 + math.log1p(max(0, record.access_count)))
        )

    def _evict_least_important_if_needed(
        self, user_id: str, records: dict[str, SemanticRecord]
    ) -> None:
        """Giải phóng các bản ghi có retention score thấp nhất khi vượt quá max_entries."""
        if len(records) <= self.max_entries:
            return
        sorted_records = sorted(
            records.values(),
            key=lambda r: (
                self._calculate_retention_score(r),
                r.last_accessed_at,
            ),
        )
        excess = len(records) - self.max_entries
        for victim in sorted_records[:excess]:
            records.pop(victim.record_id, None)
            logger.debug(
                "Importance-evicted semantic record %s for user %s (score=%.3f)",
                victim.record_id,
                user_id,
                self._calculate_retention_score(victim),
            )

    def _embed_text(self, text: str) -> list[float] | None:
        embedder = self._get_embedder()
        if embedder is None:
            return None
        if hasattr(embedder, "embed_query"):
            return list(embedder.embed_query(text))
        if hasattr(embedder, "embed"):
            return list(embedder.embed(text))
        if callable(embedder):
            return list(embedder(text))
        return None

    async def store(
        self,
        user_id: str,
        text: str,
        *,
        importance: float = 1.0,
        embedding: list[float] | None = None,
        meta: dict[str, Any] | None = None,
        record_id: str | None = None,
    ) -> SemanticRecord:
        """Lưu bản ghi ngữ nghĩa, tự động tính embedding và áp dụng importance eviction."""
        records = self._memory.setdefault(user_id, {})
        rid = record_id or str(uuid.uuid4())
        vec = embedding if embedding is not None else self._embed_text(text)
        now = time.time()
        record = SemanticRecord(
            record_id=rid,
            user_id=user_id,
            text=text,
            embedding=vec,
            importance=max(0.1, float(importance)),
            access_count=0,
            meta=dict(meta or {}),
            created_at=now,
            last_accessed_at=now,
        )
        records[rid] = record
        self._evict_least_important_if_needed(user_id, records)
        return record

    async def recall(
        self,
        user_id: str,
        query: str,
        *,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> list[tuple[SemanticRecord, float]]:
        """Truy xuất các bản ghi tương đồng ngữ nghĩa nhất với query."""
        records = self._memory.get(user_id, {})
        if not records:
            return []

        query_vec = self._embed_text(query)
        scored: list[tuple[SemanticRecord, float]] = []

        now = time.time()
        for r in records.values():
            if query_vec is not None and r.embedding is not None:
                sim = _cosine_similarity(query_vec, r.embedding)
            else:
                # Text fallback substring match
                sim = 1.0 if query.lower() in r.text.lower() else 0.0
            if sim >= min_similarity:
                scored.append((r, sim))

        scored.sort(key=lambda item: item[1], reverse=True)
        results = scored[:top_k]

        # Update access count & last_accessed_at for recalled records
        for r, _ in results:
            r.access_count += 1
            r.last_accessed_at = now

        return results

    async def get(self, user_id: str, record_id: str) -> SemanticRecord | None:
        records = self._memory.get(user_id, {})
        record = records.get(record_id)
        if record:
            record.access_count += 1
            record.last_accessed_at = time.time()
            return record
        return None

    async def delete(self, user_id: str, record_id: str) -> bool:
        records = self._memory.get(user_id, {})
        if record_id in records:
            del records[record_id]
            return True
        return False

    async def clear(self, user_id: str) -> int:
        records = self._memory.get(user_id, {})
        count = len(records)
        self._memory[user_id] = {}
        return count

    def count(self, user_id: str) -> int:
        return len(self._memory.get(user_id, {}))
