"""
HAgent Bridge — Quản lý Cuộc hội thoại (MongoDB)

Lưu trữ và truy xuất lịch sử hội thoại trong MongoDB.
Cấu hình tải từ hagent.yaml.
"""

import uuid
from datetime import UTC, datetime

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING

from .config import get_mongodb_config
from .models import Conversation, ConversationMessage

# Client ở tầng module (khởi tạo khi startup)
_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


def _normalize_mongodb_uri(configured: object) -> str:
    """Chuẩn hóa cấu hình cũ mà không ghi lại giá trị nhạy cảm vào lỗi."""
    if not isinstance(configured, str) or not configured.strip():
        raise ValueError("Cấu hình kết nối MongoDB không hợp lệ")
    value = configured.strip()
    if value.startswith(("mongodb://", "mongodb+srv://")):
        return value
    if "://" in value:
        raise ValueError("Cấu hình kết nối MongoDB không hợp lệ")
    return f"mongodb://{value}"


async def init_db():
    """Khởi tạo kết nối MongoDB và tạo các index cần thiết."""
    global _client, _db
    mongo_cfg = get_mongodb_config()

    _client = AsyncIOMotorClient(_normalize_mongodb_uri(mongo_cfg.get("connect")))
    _db = _client[mongo_cfg["db_name"]]

    # Tạo TTL index — tự động xóa cuộc hội thoại sau N giờ (từ YAML)
    ttl_seconds = mongo_cfg["conversation_ttl_hours"] * 3600
    await _db.conversations.create_index(
        [("updated_at", ASCENDING)],
        expireAfterSeconds=ttl_seconds,
    )

    # Index để truy vấn nhanh theo user
    await _db.conversations.create_index(
        [("user_id", ASCENDING), ("conversation_id", ASCENDING)],
        unique=True,
    )


async def close_db():
    """Đóng kết nối MongoDB."""
    global _client
    if _client:
        _client.close()
        _client = None


def _get_collection():
    """Lấy collection conversations."""
    if _db is None:
        raise RuntimeError("Database chưa được khởi tạo — gọi init_db() trước")
    return _db.conversations


# ─── Thao tác CRUD ──────────────────────────────────────


async def create_conversation(user_id: str) -> str:
    """Tạo cuộc hội thoại mới và trả về ID."""
    conversation_id = uuid.uuid4().hex
    now = datetime.now(UTC)

    doc = {
        "conversation_id": conversation_id,
        "user_id": user_id,
        "messages": [],
        "created_at": now,
        "updated_at": now,
        "provider": "",
        "model": "",
    }
    await _get_collection().insert_one(doc)
    return conversation_id


async def get_conversation(conversation_id: str, user_id: str) -> Conversation | None:
    """Truy xuất cuộc hội thoại theo ID, kiểm tra quyền sở hữu."""
    doc = await _get_collection().find_one({
        "conversation_id": conversation_id,
        "user_id": user_id,
    })
    if doc is None:
        return None

    return Conversation(
        conversation_id=doc["conversation_id"],
        user_id=doc["user_id"],
        messages=[ConversationMessage(**m) for m in doc.get("messages", [])],
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
        provider=doc.get("provider", ""),
        model=doc.get("model", ""),
    )


async def add_message(
    conversation_id: str,
    user_id: str,
    role: str,
    content: str,
    provider: str = "",
    model: str = "",
) -> None:
    """Thêm một tin nhắn vào cuộc hội thoại."""
    now = datetime.now(UTC)
    message = {
        "role": role,
        "content": content,
        "timestamp": now,
        "provider": provider,
        "model": model,
    }

    update_fields: dict = {
        "$push": {"messages": message},
        "$set": {"updated_at": now},
    }
    if provider:
        update_fields["$set"]["provider"] = provider
    if model:
        update_fields["$set"]["model"] = model

    result = await _get_collection().update_one(
        {"conversation_id": conversation_id, "user_id": user_id},
        update_fields,
    )

    # Nếu cuộc hội thoại chưa tồn tại, tạo mới
    if result.matched_count == 0:
        doc = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "messages": [message],
            "created_at": now,
            "updated_at": now,
            "provider": provider,
            "model": model,
        }
        await _get_collection().insert_one(doc)


async def add_assistant_message_once(
    conversation_id: str,
    user_id: str,
    content: str,
    message_id: str,
    provider: str = "",
    model: str = "",
) -> bool:
    """Atomically append one final assistant message for a stream turn."""
    if not message_id or not message_id.strip():
        raise ValueError("message_id is required")

    owner = str(user_id)
    now = datetime.now(UTC)
    message = {
        "message_id": message_id,
        "role": "assistant",
        "content": content,
        "timestamp": now,
        "provider": provider,
        "model": model,
    }
    update_fields: dict = {
        "$push": {"messages": message},
        "$set": {"updated_at": now},
    }
    if provider:
        update_fields["$set"]["provider"] = provider
    if model:
        update_fields["$set"]["model"] = model

    collection = _get_collection()
    owner_query = {
        "conversation_id": conversation_id,
        "user_id": owner,
    }
    result = await collection.update_one(
        {
            **owner_query,
            "messages.message_id": {"$ne": message_id},
        },
        update_fields,
    )
    if result.matched_count:
        return bool(result.modified_count)

    existing = await collection.find_one(owner_query, {"_id": 1})
    if existing is None:
        raise RuntimeError("Conversation missing during assistant persistence")
    return False

async def get_message_history(
    conversation_id: str, user_id: str, limit: int = 50
) -> list[ConversationMessage]:
    """Lấy danh sách tin nhắn gần nhất từ cuộc hội thoại."""
    conv = await get_conversation(conversation_id, user_id)
    if conv is None:
        return []
    return conv.messages[-limit:]


async def delete_conversation(conversation_id: str, user_id: str) -> bool:
    """Xóa cuộc hội thoại. Trả về True nếu xóa thành công."""
    result = await _get_collection().delete_one({
        "conversation_id": conversation_id,
        "user_id": user_id,
    })
    return result.deleted_count > 0


async def list_conversations(user_id: str, limit: int = 20) -> list[dict]:
    """Liệt kê các cuộc hội thoại gần nhất (chỉ tóm tắt)."""
    cursor = _get_collection().find(
        {"user_id": user_id},
        {
            "conversation_id": 1,
            "created_at": 1,
            "updated_at": 1,
            "provider": 1,
            "model": 1,
            "messages": {"$slice": -1},
        },
    ).sort("updated_at", -1).limit(limit)

    results = []
    async for doc in cursor:
        last_msg = doc.get("messages", [{}])
        results.append({
            "conversation_id": doc["conversation_id"],
            "updated_at": doc["updated_at"].isoformat(),
            "provider": doc.get("provider", ""),
            "preview": last_msg[0].get("content", "")[:100] if last_msg else "",
        })
    return results


def get_db_client():
    """Trả về global MongoDB client instance."""
    return _client
