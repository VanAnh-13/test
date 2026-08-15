"""
Episodic Memory with LRU Eviction Policy (REFAC-017).

Lưu trữ các tương tác, chuỗi sự kiện hoặc hội thoại theo từng user.
Tự động áp dụng cơ chế LRU eviction khi số lượng bản ghi vượt quá `max_entries`.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

DEFAULT_EPISODIC_MAX_ENTRIES = 50


@dataclass
class EpisodicRecord:
    """Một bản ghi trong Episodic Memory."""

    record_id: str
    user_id: str
    content: str
    event_type: str = "general"
    meta: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    last_access_seq: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodicRecord:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


class EpisodicMemory:
    """Bộ nhớ Episodic theo user với chính sách giải phóng LRU (Least Recently Used)."""

    def __init__(
        self,
        storage_dir: str | Path | None = None,
        max_entries: int = DEFAULT_EPISODIC_MAX_ENTRIES,
    ) -> None:
        self.max_entries = max(1, int(max_entries))
        self._dir = Path(storage_dir) if storage_dir else None
        if self._dir:
            self._dir.mkdir(parents=True, exist_ok=True)
        # In-memory storage cache: user_id -> dict of record_id -> EpisodicRecord
        self._memory: dict[str, dict[str, EpisodicRecord]] = {}
        self._seq: int = 0

    def _user_file(self, user_id: str) -> Path | None:
        if not self._dir:
            return None
        safe_id = user_id.replace("/", "_").replace("\\", "_")
        return self._dir / f"episodic_{safe_id}.json"

    def _load(self, user_id: str) -> dict[str, EpisodicRecord]:
        if user_id in self._memory:
            return self._memory[user_id]
        records: dict[str, EpisodicRecord] = {}
        fpath = self._user_file(user_id)
        if fpath and fpath.exists():
            try:
                raw = json.loads(fpath.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    records = {
                        k: EpisodicRecord.from_dict(v)
                        for k, v in raw.items()
                        if isinstance(v, dict)
                    }
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load episodic memory from %s: %s", fpath, exc)
        self._memory[user_id] = records
        return records

    def _save(self, user_id: str, records: dict[str, EpisodicRecord]) -> None:
        self._memory[user_id] = records
        fpath = self._user_file(user_id)
        if fpath:
            try:
                payload = {k: v.to_dict() for k, v in records.items()}
                fpath.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to save episodic memory to %s: %s", fpath, exc)

    def _evict_lru_if_needed(
        self, user_id: str, records: dict[str, EpisodicRecord]
    ) -> None:
        """Loại bỏ bản ghi có access sequence cũ nhất khi vượt quá max_entries."""
        if len(records) <= self.max_entries:
            return
        sorted_records = sorted(
            records.values(),
            key=lambda r: (r.last_access_seq, r.last_accessed_at, r.timestamp),
        )
        excess = len(records) - self.max_entries
        for victim in sorted_records[:excess]:
            records.pop(victim.record_id, None)
            logger.debug(
                "LRU evicted episodic record %s for user %s",
                victim.record_id,
                user_id,
            )

    async def store(
        self,
        user_id: str,
        content: Any,
        *,
        event_type: str = "general",
        meta: dict[str, Any] | None = None,
        record_id: str | None = None,
    ) -> EpisodicRecord:
        """Lưu một sự kiện/nội dung mới vào bộ nhớ episodic và áp dụng LRU eviction."""
        records = self._load(user_id)
        rid = record_id or str(uuid.uuid4())
        text_content = (
            content
            if isinstance(content, str)
            else json.dumps(content, ensure_ascii=False)
        )
        now = time.time()
        self._seq += 1
        record = EpisodicRecord(
            record_id=rid,
            user_id=user_id,
            content=text_content,
            event_type=event_type,
            meta=dict(meta or {}),
            timestamp=now,
            last_accessed_at=now,
            last_access_seq=self._seq,
        )
        records[rid] = record
        self._evict_lru_if_needed(user_id, records)
        self._save(user_id, records)
        return record

    async def get(self, user_id: str, record_id: str) -> EpisodicRecord | None:
        """Truy xuất một bản ghi và cập nhật thời điểm truy cập (LRU touch)."""
        records = self._load(user_id)
        record = records.get(record_id)
        if record:
            self._seq += 1
            record.last_accessed_at = time.time()
            record.last_access_seq = self._seq
            self._save(user_id, records)
            return record
        return None

    async def get_recent(
        self,
        user_id: str,
        limit: int = 10,
        *,
        event_type: str | None = None,
    ) -> list[EpisodicRecord]:
        """Lấy danh sách các bản ghi mới nhất theo timestamp."""
        records = self._load(user_id)
        matched = [
            r
            for r in records.values()
            if event_type is None or r.event_type == event_type
        ]
        matched.sort(key=lambda r: r.timestamp, reverse=True)
        return matched[:limit]

    async def search(
        self, user_id: str, query: str, limit: int = 10
    ) -> list[EpisodicRecord]:
        """Tìm kiếm bản ghi episodic theo từ khóa."""
        records = self._load(user_id)
        q = query.lower()
        matched = [r for r in records.values() if q in r.content.lower()]
        matched.sort(key=lambda r: r.last_accessed_at, reverse=True)
        return matched[:limit]

    async def delete(self, user_id: str, record_id: str) -> bool:
        records = self._load(user_id)
        if record_id in records:
            del records[record_id]
            self._save(user_id, records)
            return True
        return False

    async def clear(self, user_id: str) -> int:
        records = self._load(user_id)
        count = len(records)
        self._save(user_id, {})
        return count

    def count(self, user_id: str) -> int:
        return len(self._load(user_id))
