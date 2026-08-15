"""Các endpoint chat đồng bộ, streaming và upload file."""

# FastAPI yêu cầu Depends, File và Form trong chữ ký endpoint.
# ruff: noqa: B008

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import StreamingResponse

from hagent.bridge import conversation as conv_store
from hagent.bridge.auth import TokenPayload, get_current_user
from hagent.bridge.config import get_hautoml_config
from hagent.bridge.models import ChatRequest, ChatResponse
from hagent.bridge.routes.route_support import (
    apply_tool_outputs_to_world_state,
    bridge_event_stream,
    call_hagent_gateway,
    extract_training_job_id,
    runtime_context,
    schedule_training_result_notification,
    to_chat_response,
    upload_hautoml_dataset,
    validate_model_name,
)
from hagent.world.state_store import WorldStateStore

logger = structlog.get_logger("hagent.bridge.routes.chat")
router = APIRouter(tags=["chat"])


@router.post("/api/v1/chat/", response_model=ChatResponse)
async def chat(
    request: Request,
    req: ChatRequest,
    user: TokenPayload = Depends(get_current_user),
):
    validate_model_name(req.model)
    conversation_id = req.conversation_id or uuid.uuid4().hex
    world_state_store: WorldStateStore = request.app.state.world_state_store
    await world_state_store.ensure(user.user_id)
    snapshot = await world_state_store.get(user.user_id)
    history = await conv_store.get_message_history(
        conversation_id, user.user_id, limit=20
    )
    await conv_store.add_message(
        conversation_id=conversation_id,
        user_id=user.user_id,
        role="user",
        content=req.message,
    )
    result = await call_hagent_gateway(
        message=req.message,
        user_token=user.raw_token,
        user_id=user.user_id,
        session_id=conversation_id,
        context_extra=runtime_context(req.context, snapshot, history),
        model_name=req.model,
    )
    await apply_tool_outputs_to_world_state(world_state_store, user.user_id, result)
    await conv_store.add_message(
        conversation_id=conversation_id,
        user_id=user.user_id,
        role="assistant",
        content=result["message"],
        provider=result.get("provider", ""),
        model=result.get("model", ""),
    )
    job_id = extract_training_job_id(result["message"])
    if job_id:
        schedule_training_result_notification(
            conversation_id=conversation_id,
            user_id=user.user_id,
            user_token=user.raw_token,
            job_id=job_id,
        )
    return to_chat_response(result, conversation_id)


@router.post("/api/v1/chat/stream")
async def chat_stream(
    request: Request,
    req: ChatRequest,
    user: TokenPayload = Depends(get_current_user),
):
    """Streaming SSE có phân quyền; Bridge sở hữu lịch sử và lưu trữ."""
    validate_model_name(req.model)
    conversation_id = req.conversation_id or uuid.uuid4().hex
    world_state_store: WorldStateStore = request.app.state.world_state_store
    await world_state_store.ensure(user.user_id)
    snapshot = await world_state_store.get(user.user_id)
    history = await conv_store.get_message_history(
        conversation_id, user.user_id, limit=20
    )
    await conv_store.add_message(
        conversation_id=conversation_id,
        user_id=user.user_id,
        role="user",
        content=req.message,
    )
    return StreamingResponse(
        bridge_event_stream(
            message=req.message,
            user=user,
            conversation_id=conversation_id,
            context_extra=runtime_context(req.context, snapshot, history),
            model_name=req.model,
            world_state_store=world_state_store,
            message_id=f"stream:{uuid.uuid4().hex}",
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Conversation-Id": conversation_id,
        },
    )


@router.post("/api/v1/chat/upload", response_model=ChatResponse)
async def chat_with_file(
    request: Request,
    message: str = Form(...),
    file: UploadFile = File(...),
    conversation_id: str | None = Form(None),
    model: str | None = Form(None),
    user: TokenPayload = Depends(get_current_user),
):
    """Tải file vào HAutoML rồi gửi ngữ cảnh file cho agent."""
    validate_model_name(model)
    hautoml_cfg = get_hautoml_config()
    conv_id = conversation_id or uuid.uuid4().hex
    world_state_store: WorldStateStore = request.app.state.world_state_store
    await world_state_store.ensure(user.user_id)
    snapshot = await world_state_store.get(user.user_id)

    file_content, filename = await upload_hautoml_dataset(
        file,
        base_url=hautoml_cfg["base_url"],
        user_id=user.user_id,
        raw_token=user.raw_token,
        on_http_failure=lambda status_code: logger.warning(
            "HAutoML upload returned HTTP %d",
            status_code,
        ),
    )

    file_info = (
        f"\n[File da upload vao he thong dataset: {filename} — "
        f"{len(file_content)} bytes]"
    )
    history = await conv_store.get_message_history(conv_id, user.user_id, limit=20)
    full_message = f"{message}{file_info}"
    await conv_store.add_message(conv_id, user.user_id, "user", full_message)
    result = await call_hagent_gateway(
        message=full_message,
        user_token=user.raw_token,
        user_id=user.user_id,
        session_id=conv_id,
        context_extra=runtime_context(None, snapshot, history),
        model_name=model,
    )
    await apply_tool_outputs_to_world_state(world_state_store, user.user_id, result)
    await conv_store.add_message(
        conv_id,
        user.user_id,
        "assistant",
        result["message"],
        result.get("provider", ""),
        result.get("model", ""),
    )
    job_id = extract_training_job_id(result["message"])
    if job_id:
        schedule_training_result_notification(
            conversation_id=conv_id,
            user_id=user.user_id,
            user_token=user.raw_token,
            job_id=job_id,
        )
    return to_chat_response(result, conv_id)
