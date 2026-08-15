"""Khám phá chat và quản lý lịch sử hội thoại của người dùng."""

# FastAPI yêu cầu Depends trong chữ ký endpoint.
# ruff: noqa: B008

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException

from hagent.bridge import conversation as conv_store
from hagent.bridge.auth import TokenPayload, get_current_user, get_optional_user
from hagent.bridge.config import get_llm_config, get_llm_models
from hagent.bridge.models import ProviderInfo, ProvidersResponse, SuggestionsResponse

router = APIRouter(tags=["conversations"])


@router.get("/api/v1/chat/suggestions", response_model=SuggestionsResponse)
async def get_suggestions():
    """Trả về gợi ý khởi đầu cho giao diện chat."""
    return SuggestionsResponse(
        suggestions=[
            "Hien thi danh sach dataset cua toi",
            "Huan luyen model phan loai moi",
            "Co nhung thuat toan ML nao kha dung?",
            "Kiem tra trang thai cac job training",
            "Giup toi chon model phu hop",
            "Chuyen sang provider AI khac",
        ]
    )


@router.get("/api/v1/chat/providers", response_model=ProvidersResponse)
async def list_providers(
    user: TokenPayload | None = Depends(get_optional_user),
):
    """Liệt kê alias model đã cấu hình, nhóm theo provider của toolkit."""
    del user
    registry = [model for model in get_llm_models() if model.get("name")]
    grouped: dict[str, list[str]] = {}
    for model in registry:
        provider_id = str(model.get("provider") or "unknown")
        grouped.setdefault(provider_id, []).append(str(model["name"]))

    default_model = str(get_llm_config().get("default_model") or "")
    if not default_model and registry:
        default_model = str(registry[0]["name"])
    default_entry = next(
        (model for model in registry if str(model["name"]) == default_model), None
    )
    default_provider = (
        str(default_entry.get("provider") or "unknown") if default_entry else ""
    )
    providers = [
        ProviderInfo(
            name=provider_id.replace("_", " ").title(),
            provider_id=provider_id,
            models=model_names,
            available=True,
            description="Configured in the toolkit model registry",
        )
        for provider_id, model_names in grouped.items()
    ]
    return ProvidersResponse(
        default_provider=default_provider,
        default_model=default_model,
        providers=providers,
    )


@router.get("/api/v1/chat/conversations")
async def list_user_conversations(
    user: TokenPayload = Depends(get_current_user),
):
    """Liệt kê các cuộc hội thoại gần nhất của người dùng."""
    conversations = await conv_store.list_conversations(user.user_id)
    return {"conversations": conversations}


@router.get("/api/v1/chat/conversation/{conversation_id}")
async def get_conversation_history(
    conversation_id: str,
    user: TokenPayload = Depends(get_current_user),
):
    """Truy xuất lịch sử của một cuộc hội thoại thuộc người dùng hiện tại."""
    conversation = await conv_store.get_conversation(conversation_id, user.user_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Khong tim thay cuoc hoi thoai")
    return {
        "conversation_id": conversation.conversation_id,
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
        "provider": conversation.provider,
        "model": conversation.model,
        "messages": [
            {
                "id": str(uuid.uuid4()),
                "role": message.role,
                "content": message.content,
                "timestamp": (
                    message.timestamp.isoformat() if message.timestamp else None
                ),
                "provider": message.provider,
                "model": message.model,
            }
            for message in conversation.messages
        ],
    }


@router.delete("/api/v1/chat/conversation/{conversation_id}")
async def clear_conversation(
    conversation_id: str,
    user: TokenPayload = Depends(get_current_user),
):
    """Xóa một cuộc hội thoại và toàn bộ lịch sử của nó."""
    deleted = await conv_store.delete_conversation(conversation_id, user.user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Khong tim thay cuoc hoi thoai")
    return {"status": "deleted", "conversation_id": conversation_id}
