"""
HAgent — Memory Storage (Phase 3).

Lưu trữ facts (kiến thức rút trích từ conversations).
Config-driven: backend type + TTL đọc từ YAML.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# ── Fact schema ──────────────────────────────────────────


@dataclass
class Fact:
    """Một fact rút trích từ conversation."""

    key: str  # Unique identifier
    content: str  # Nội dung fact
    category: str = "general"  # general | dataset | model | preference | workflow
    confidence: float = 1.0  # 0.0 → 1.0
    source: str = ""  # Conversation/tool mà fact được rút trích từ
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0  # Số lần fact được truy cập

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Fact:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ── Abstract Store ───────────────────────────────────────


class FactStore:
    """
    Abstract base cho fact storage.
    Cung cấp CRUD operations cho facts.
    """

    async def save(self, user_id: str, fact: Fact) -> None:
        raise NotImplementedError

    async def get(self, user_id: str, key: str) -> Fact | None:
        raise NotImplementedError

    async def search(
        self,
        user_id: str,
        *,
        category: str | None = None,
        query: str | None = None,
        limit: int = 20,
    ) -> list[Fact]:
        raise NotImplementedError

    async def get_all(self, user_id: str) -> list[Fact]:
        raise NotImplementedError

    async def delete(self, user_id: str, key: str) -> bool:
        raise NotImplementedError

    async def clear(self, user_id: str) -> int:
        raise NotImplementedError


# ── Local File Store ─────────────────────────────────────


class LocalFactStore(FactStore):
    """
    File-based fact store — lưu JSON trên disk.
    Phù hợp cho dev/test. Production dùng MongoDB store.
    """

    def __init__(self, storage_dir: str | Path):
        self._dir = Path(storage_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _user_file(self, user_id: str) -> Path:
        safe_id = user_id.replace("/", "_").replace("\\", "_")
        return self._dir / f"{safe_id}.json"

    def _load_facts(self, user_id: str) -> dict[str, dict]:
        fpath = self._user_file(user_id)
        if fpath.exists():
            try:
                return json.loads(fpath.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

    def _save_facts(self, user_id: str, facts: dict[str, dict]) -> None:
        fpath = self._user_file(user_id)
        fpath.write_text(
            json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    async def save(self, user_id: str, fact: Fact) -> None:
        facts = self._load_facts(user_id)
        fact.updated_at = time.time()
        facts[fact.key] = fact.to_dict()
        self._save_facts(user_id, facts)
        logger.debug("Saved fact '%s' for user '%s'", fact.key, user_id)

    async def get(self, user_id: str, key: str) -> Fact | None:
        facts = self._load_facts(user_id)
        data = facts.get(key)
        if data:
            fact = Fact.from_dict(data)
            fact.access_count += 1
            facts[key] = fact.to_dict()
            self._save_facts(user_id, facts)
            return fact
        return None

    async def search(
        self,
        user_id: str,
        *,
        category: str | None = None,
        query: str | None = None,
        limit: int = 20,
    ) -> list[Fact]:
        facts = self._load_facts(user_id)
        results = []
        for data in facts.values():
            fact = Fact.from_dict(data)
            if category and fact.category != category:
                continue
            if query and query.lower() not in fact.content.lower():
                continue
            results.append(fact)

        # Sort by recency
        results.sort(key=lambda f: f.updated_at, reverse=True)
        return results[:limit]

    async def get_all(self, user_id: str) -> list[Fact]:
        facts = self._load_facts(user_id)
        result = [Fact.from_dict(d) for d in facts.values()]
        result.sort(key=lambda f: f.updated_at, reverse=True)
        return result

    async def delete(self, user_id: str, key: str) -> bool:
        facts = self._load_facts(user_id)
        if key in facts:
            del facts[key]
            self._save_facts(user_id, facts)
            return True
        return False

    async def clear(self, user_id: str) -> int:
        facts = self._load_facts(user_id)
        count = len(facts)
        self._save_facts(user_id, {})
        return count


# ── Factory ──────────────────────────────────────────────


def create_fact_store() -> FactStore:
    """Create the configured owner-scoped fact store."""
    try:
        from hagent.bridge.config import load_config

        cfg = load_config()
        memory_cfg = cfg.get("memory", {}) or {}
    except Exception:  # noqa: BLE001
        memory_cfg = {}

    configured_backend = str(memory_cfg.get("backend", "auto")).lower()
    mongo_configured = "MONGODB_CONNECT" in os.environ
    mongo_env = os.getenv("MONGODB_CONNECT")
    if configured_backend == "auto":
        backend = "mongodb" if mongo_configured else "local"
    else:
        backend = configured_backend

    if backend == "local":
        storage_dir = memory_cfg.get("storage_dir", "./data/memory")
        return LocalFactStore(storage_dir)

    if backend == "mongodb":
        from hagent.agent.memory.mongo_store import MongoFactStore
        from hagent.bridge.config import get_mongodb_config

        mongo_cfg = get_mongodb_config()
        connect = mongo_env if mongo_configured else mongo_cfg.get("connect")
        if not connect or not str(connect).strip():
            raise RuntimeError(
                "MongoDB memory backend requires a non-empty MONGODB_CONNECT"
            )
        return MongoFactStore(
            connect=str(connect),
            db_name=str(mongo_cfg.get("db_name") or "hagent"),
            collection_name=str(memory_cfg.get("collection") or "memory_facts"),
        )

    raise ValueError(f"Unsupported memory backend: {configured_backend!r}")


# ── Episodic & Semantic Memory Factories ─────────────────


from hagent.agent.memory.episodic import (
    DEFAULT_EPISODIC_MAX_ENTRIES,
    EpisodicMemory,
    EpisodicRecord,
)
from hagent.agent.memory.semantic import (
    DEFAULT_SEMANTIC_MAX_ENTRIES,
    SemanticMemory,
    SemanticRecord,
)


def create_episodic_memory(
    storage_dir: str | Path | None = None,
    max_entries: int | None = None,
) -> EpisodicMemory:
    """Tạo EpisodicMemory từ cấu hình hagent.yaml hoặc tham số."""
    try:
        from hagent.bridge.config import load_config

        cfg = load_config()
        episodic_cfg = (cfg.get("memory", {}) or {}).get("episodic", {}) or {}
    except Exception:  # noqa: BLE001
        episodic_cfg = {}

    limit = (
        max_entries
        if max_entries is not None
        else int(episodic_cfg.get("max_entries", DEFAULT_EPISODIC_MAX_ENTRIES))
    )
    s_dir = storage_dir or episodic_cfg.get("storage_dir", "./data/memory")
    return EpisodicMemory(storage_dir=s_dir, max_entries=limit)


def create_semantic_memory(
    storage_dir: str | Path | None = None,
    max_entries: int | None = None,
    embedder_factory: Any = None,
    lazy_load: bool | None = None,
) -> SemanticMemory:
    """Tạo SemanticMemory từ cấu hình hagent.yaml hoặc tham số."""
    try:
        from hagent.bridge.config import load_config

        cfg = load_config()
        semantic_cfg = (cfg.get("memory", {}) or {}).get("semantic", {}) or {}
    except Exception:  # noqa: BLE001
        semantic_cfg = {}

    limit = (
        max_entries
        if max_entries is not None
        else int(semantic_cfg.get("max_entries", DEFAULT_SEMANTIC_MAX_ENTRIES))
    )
    lazy = (
        lazy_load
        if lazy_load is not None
        else bool(semantic_cfg.get("lazy_load", True))
    )
    s_dir = storage_dir or semantic_cfg.get("storage_dir", "./data/memory")
    return SemanticMemory(
        storage_dir=s_dir,
        embedder_factory=embedder_factory,
        max_entries=limit,
        lazy_load=lazy,
    )


__all__ = [
    "DEFAULT_EPISODIC_MAX_ENTRIES",
    "DEFAULT_SEMANTIC_MAX_ENTRIES",
    "EpisodicMemory",
    "EpisodicRecord",
    "Fact",
    "FactStore",
    "LocalFactStore",
    "SemanticMemory",
    "SemanticRecord",
    "create_episodic_memory",
    "create_fact_store",
    "create_semantic_memory",
]
