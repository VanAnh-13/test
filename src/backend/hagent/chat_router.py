"""
HAgent — Chat Router tích hợp HAgent

Các endpoint chat mount trực tiếp vào FastAPI app chính (app.py).
Sử dụng LangGraph agent runtime — runtime LangGraph duy nhất.
Hỗ trợ cả synchronous và SSE streaming responses.
Lưu lịch sử chat vào MongoDB database AutoML.
"""

import logging
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from pymongo.asynchronous.database import AsyncDatabase

from database.database import get_db
from hagent import chat_store
from hagent.bridge.config import (
    get_error_messages,
    get_hautoml_config,
    get_suggestions,
)
from users.routers import get_current_user

logger = logging.getLogger("hagent.chat_router")

router = APIRouter(prefix="/api/v1/chat", tags=["HAgent Chat"])


# ─── Schemas ─────────────────────────────────────────────


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(..., description="Nội dung tin nhắn")
    conversation_id: str | None = Field(
        None, description="ID cuộc hội thoại (tạo mới nếu null)"
    )
    context: dict | None = Field(None, description="Ngữ cảnh bổ sung")
    model: str | None = Field(
        None, description="Tên model LLM (None = dùng default từ config)"
    )


class ChatResponse(BaseModel):
    message: str
    conversation_id: str
    sources: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    tool_outputs: list[dict] = Field(default_factory=list)
    provider: str = ""
    model: str = ""
    route: str = ""
    # World Model / planning surface (optional, deep integration)
    plan_status: str | None = None
    selected_plan: dict | None = None
    planning: dict | None = None
    surprise: dict | None = None
    cost_metrics: dict | None = None
    execution_events: list = Field(default_factory=list)
    execution_log: list = Field(default_factory=list)
    revision_count: int = 0
    world_model: dict | None = None
    campaign: dict | None = None
    campaign_status: str | None = None
    hierarchy: dict | None = None
    hierarchy_status: str | None = None
    evaluation: dict | None = None


class StreamChatRequest(BaseModel):
    message: str = Field(..., description="Nội dung tin nhắn")
    conversation_id: str | None = Field(None, description="ID cuộc hội thoại")
    model: str | None = Field(None, description="Tên model LLM")


class HealthResponse(BaseModel):
    agent_runtime: str
    hautoml_connected: bool
    available_models: list[dict]
    mode: str


# ─── World Model helper ──────────────────────────────────


def _wm_summary(world_model: dict | None) -> dict | None:
    """Compact snapshot for API/UI (avoid huge payloads)."""
    if not isinstance(world_model, dict):
        return None
    datasets = world_model.get("datasets") or {}
    jobs = world_model.get("jobs") or {}
    return {
        "user_id": world_model.get("user_id"),
        "phase": world_model.get("phase"),
        "active_dataset_id": world_model.get("active_dataset_id"),
        "active_job_id": world_model.get("active_job_id"),
        "n_datasets": len(datasets),
        "n_jobs": len(jobs),
        "dataset_ids": list(datasets.keys())[:20],
        "job_ids": list(jobs.keys())[:20],
        "last_surprise": world_model.get("last_surprise"),
    }


_CONTEXT_FIELDS = (
    "dataset_id",
    "dataset_name",
    "target_column",
    "problem_type",
    "metric",
    "models",
)


def _history_from_context(context: dict | None) -> list[dict[str, str]]:
    if not isinstance(context, dict) or not isinstance(context.get("history"), list):
        return []
    history = context["history"]
    normalized: list[dict[str, str]] = []
    for item in history[-20:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def _apply_request_context(
    world_model: dict | None,
    context: dict | None,
    user_id: str,
) -> dict:
    """Merge only public request fields; persisted server state remains authoritative."""
    merged = dict(world_model or {"datasets": {}, "jobs": {}})
    merged["user_id"] = user_id
    if not isinstance(context, dict):
        return merged

    public_context = {
        key: context[key]
        for key in _CONTEXT_FIELDS
        if key in context and context[key] is not None
    }
    if public_context:
        merged["request_context"] = public_context

    dataset_id = public_context.get("dataset_id")
    if dataset_id:
        datasets = dict(merged.get("datasets") or {})
        datasets.setdefault(
            dataset_id,
            {
                "id": dataset_id,
                "name": public_context.get("dataset_name") or dataset_id,
                "target": public_context.get("target_column"),
                "problem_type_inferred": public_context.get("problem_type"),
            },
        )
        merged["datasets"] = datasets
        merged["active_dataset_id"] = dataset_id
    return merged


async def _load_world_model(db: AsyncDatabase, user_id: str) -> dict | None:
    """Load World Model snapshot cho user (unified store API)."""
    try:
        from hagent.bridge.config import get_world_state_config
        from hagent.world.state_store import WorldStateStore

        ws_cfg = get_world_state_config()
        # AsyncDatabase exposes .client and .name
        client = getattr(db, "client", None)
        db_name = getattr(db, "name", None) or "hagent"
        if client is None:
            logger.debug("World Model: db has no client")
            return None
        store = WorldStateStore(
            client=client,
            db_name=db_name,
            collection_name=ws_cfg["collection_name"],
            ttl_seconds=ws_cfg["ttl_seconds"],
        )
        await store.ensure(str(user_id))
        return await store.get_snapshot(str(user_id))
    except Exception as exc:
        logger.debug("Không load được World Model: %s", exc)
        return None


# ─── Gọi HAgent Agent ────────────────────────────


def _validate_model_name(model_name: str | None) -> None:
    if model_name is None:
        return

    from hagent.agent.llm_config import require_model_config

    try:
        require_model_config(model_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _call_agent(
    message: str,
    *,
    user_token: str | None = None,
    user_id: str | None = None,
    history: list[dict[str, str]] | None = None,
    world_model: dict | None = None,
    mongo_client=None,
    db_name: str | None = None,
    model_name: str | None = None,
) -> dict:
    """Gọi LangGraph agent runtime — runtime LangGraph."""
    error_messages = get_error_messages()

    # Tên model sai (kể cả chuỗi rỗng) → 400, KHÔNG âm thầm dùng default.
    _validate_model_name(model_name)

    command = None
    try:
        from hagent.agent.runtime import (
            build_start_turn,
            collect_runtime_result,
            get_agent_runtime,
        )

        command, scope = build_start_turn(
            message,
            user_id=user_id,
            user_token=user_token,
            history=history,
            world_model=world_model,
            mongo_client=mongo_client,
            db_name=db_name,
            model_name=model_name,
        )
        result = await collect_runtime_result(
            get_agent_runtime(),
            command,
            scope=scope,
        )
        plan_status = result.get("plan_status")
        selected_plan = result.get("selected_plan")
        planning = result.get("planning")
        if planning is None and (plan_status is not None or selected_plan is not None):
            planning = {"status": plan_status, "selected_plan": selected_plan}
        return {
            "message": result.get("message", result.get("response", "")),
            "sources": result.get("sources", []),
            "suggestions": [],
            "tool_outputs": result.get("tool_outputs", []),
            "provider": result.get("provider", "hagent"),
            "model": result.get("model", ""),
            "route": result.get("route", "direct"),
            "plan_status": plan_status,
            "selected_plan": selected_plan,
            "planning": planning,
            "surprise": result.get("surprise"),
            "cost_metrics": result.get("cost_metrics"),
            "execution_events": result.get("execution_events") or [],
            "execution_log": result.get("execution_log") or [],
            "revision_count": result.get("revision_count") or 0,
            "world_model": result.get("world_model"),
            "campaign": result.get("campaign"),
            "campaign_status": result.get("campaign_status"),
            "hierarchy": result.get("hierarchy"),
            "hierarchy_status": result.get("hierarchy_status"),
            "evaluation": result.get("evaluation"),
        }

    except HTTPException:
        raise
    except TimeoutError as exc:
        logger.warning("Agent runtime timeout")
        raise HTTPException(status_code=504, detail="Agent runtime timed out") from exc
    except Exception as exc:
        raw_runtime_code = getattr(exc, "code", "")
        runtime_code = (
            raw_runtime_code
            if isinstance(raw_runtime_code, str)
            and 1 <= len(raw_runtime_code) <= 64
            and all(char.isalnum() or char in "_.-" for char in raw_runtime_code)
            else "RUNTIME_UNEXPECTED"
        )
        logger.error(
            "Agent runtime lỗi code=%s type=%s run_id=%s model=%s",
            runtime_code,
            type(exc).__name__,
            command.run_id if command is not None else "unavailable",
            model_name or "default",
        )
        if runtime_code in {"DEADLINE_EXCEEDED", "LEGACY_RUNTIME_TIMEOUT"}:
            raise HTTPException(
                status_code=504,
                detail="Agent runtime timed out",
            ) from exc
        # Phân loại lỗi từ config
        err_str = str(exc).lower()
        if "api key" in err_str or "authentication" in err_str or "401" in err_str:
            message_key = "llm_auth"
        elif "rate limit" in err_str or "429" in err_str:
            message_key = "llm_rate_limit"
        elif "timeout" in err_str:
            message_key = "timeout"
        else:
            message_key = "generic"

        msg = (
            error_messages.get(message_key)
            or error_messages.get("generic")
            or "HAgent đang gặp lỗi khi xử lý yêu cầu."
        )

        raise HTTPException(status_code=500, detail=msg) from exc


def _to_chat_response(
    result: dict,
    conversation_id: str,
) -> ChatResponse:
    return ChatResponse(
        message=result["message"],
        conversation_id=conversation_id,
        sources=result.get("sources", []),
        suggestions=result.get("suggestions", []),
        tool_outputs=result.get("tool_outputs", []),
        provider=result.get("provider", "hagent"),
        model=result.get("model", ""),
        route="direct" if result.get("route") is None else result.get("route"),
        plan_status=result.get("plan_status"),
        selected_plan=result.get("selected_plan"),
        planning=result.get("planning"),
        surprise=result.get("surprise"),
        cost_metrics=result.get("cost_metrics"),
        execution_events=result.get("execution_events") or [],
        execution_log=result.get("execution_log") or [],
        revision_count=result.get("revision_count") or 0,
        world_model=_wm_summary(result.get("world_model")),
        campaign=result.get("campaign"),
        campaign_status=result.get("campaign_status"),
        hierarchy=result.get("hierarchy"),
        hierarchy_status=result.get("hierarchy_status"),
        evaluation=result.get("evaluation"),
    )


# ─── Endpoints ───────────────────────────────────────────


@router.post("/agent-run", response_model=ChatResponse)
async def agent_run(
    req: ChatRequest,
    request: Request,
    db: AsyncDatabase = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """
    Internal agent-runtime invoke for HAgent Bridge.

    Runs LangGraph multi-agent only — does NOT write conversation history
    (Bridge owns conversation store). Use this from docker bridge service.
    """
    user_id = current_user["_id"]
    conversation_id = req.conversation_id or uuid.uuid4().hex
    auth_header = request.headers.get("Authorization", "")
    user_token = (
        auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
    )
    server_world_model = await _load_world_model(db, str(user_id))
    forwarded_world_model = (
        req.context.get("world_state") if isinstance(req.context, dict) else None
    )
    merged_world_model = dict(forwarded_world_model or {})
    merged_world_model.update(server_world_model or {})
    world_model = _apply_request_context(merged_world_model, req.context, str(user_id))

    client = getattr(db, "client", None)
    db_name = getattr(db, "name", None)
    result = await _call_agent(
        message=req.message,
        user_token=user_token,
        user_id=str(user_id),
        history=_history_from_context(req.context),
        world_model=world_model,
        mongo_client=client,
        db_name=str(db_name) if db_name else None,
        model_name=req.model,
    )
    # Persist agent world_model snapshot back if present
    if result.get("world_model") and client is not None:
        try:
            from hagent.bridge.config import get_world_state_config
            from hagent.world.state_store import WorldStateStore

            ws_cfg = get_world_state_config()
            store = WorldStateStore(
                client=client,
                db_name=str(db_name or "hagent"),
                collection_name=ws_cfg["collection_name"],
                ttl_seconds=ws_cfg["ttl_seconds"],
            )
            snap = result["world_model"]
            patch = {
                k: snap[k]
                for k in (
                    "datasets",
                    "jobs",
                    "plans",
                    "goals",
                    "phase",
                    "active_dataset_id",
                    "active_job_id",
                    "active_plan_id",
                    "active_goal",
                    "last_surprise",
                    "cost_metrics",
                )
                if k in snap
            }
            if result.get("surprise"):
                patch["last_surprise"] = result["surprise"]
            if patch:
                await store.upsert(str(user_id), patch)
        except Exception as exc:
            logger.debug("agent-run WM persist failed: %s", exc)

    return _to_chat_response(result, conversation_id)


@router.post("/agent-run/stream")
async def agent_run_stream(
    req: ChatRequest,
    request: Request,
    db: AsyncDatabase = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Private stateless SSE invoke; Bridge owns all conversation writes."""
    _validate_model_name(req.model)

    user_id = str(current_user["_id"])
    conversation_id = req.conversation_id or uuid.uuid4().hex
    auth_header = request.headers.get("Authorization", "")
    user_token = (
        auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
    )

    server_world_model = await _load_world_model(db, user_id)
    forwarded_world_model = (
        req.context.get("world_state") if isinstance(req.context, dict) else None
    )
    merged_world_model = dict(forwarded_world_model or {})
    merged_world_model.update(server_world_model or {})
    world_model = _apply_request_context(merged_world_model, req.context, user_id)

    client = getattr(db, "client", None)
    db_name = getattr(db, "name", None)
    from hagent.agent.streaming import sse_stream

    return StreamingResponse(
        sse_stream(
            req.message,
            user_id=user_id,
            user_token=user_token,
            history=_history_from_context(req.context),
            world_model=world_model,
            mongo_client=client,
            db_name=str(db_name) if db_name else None,
            model_name=req.model,
            conversation_id=conversation_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Conversation-Id": conversation_id,
        },
    )


@router.post("/", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    request: Request,
    db: AsyncDatabase = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Endpoint chat chính — gọi LangGraph agent, lưu hội thoại vào database."""
    user_id = current_user["_id"]
    conversation_id = req.conversation_id or uuid.uuid4().hex

    # Lấy token từ header
    auth_header = request.headers.get("Authorization", "")
    user_token = (
        auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
    )

    # Lưu tin nhắn người dùng
    await chat_store.add_message(db, conversation_id, user_id, "user", req.message)

    # Load World Model
    world_model = _apply_request_context(
        await _load_world_model(db, str(user_id)), req.context, str(user_id)
    )
    client = getattr(db, "client", None)
    db_name = getattr(db, "name", None)

    # Gọi HAgent Agent
    result = await _call_agent(
        message=req.message,
        user_token=user_token,
        user_id=str(user_id),
        world_model=world_model,
        mongo_client=client,
        db_name=str(db_name) if db_name else None,
        model_name=req.model,
    )

    # Lưu phản hồi trợ lý
    await chat_store.add_message(
        db, conversation_id, user_id, "assistant", result["message"]
    )

    return _to_chat_response(result, conversation_id)


@router.post("/stream")
async def chat_stream(
    req: StreamChatRequest,
    request: Request,
    db: AsyncDatabase = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """
    SSE streaming endpoint — trả về real-time token-by-token.

    Frames contain event, increasing id, and JSON data fields.
    Event names follow the typed agent stream contract.
    The stream ends with exactly one done or error event.
    There is no sentinel frame.
    """
    user_id = current_user["_id"]
    conversation_id = req.conversation_id or uuid.uuid4().hex

    auth_header = request.headers.get("Authorization", "")
    user_token = (
        auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
    )

    # Lưu tin nhắn người dùng
    await chat_store.add_message(db, conversation_id, user_id, "user", req.message)

    # Load World Model
    world_model = await _load_world_model(db, str(user_id))

    from hagent.agent.streaming import sse_stream

    async def _stream_wrapper():
        """Wrap SSE stream và lưu response cuối cùng vào DB."""
        full_response = ""
        async for chunk in sse_stream(
            req.message,
            user_id=str(user_id),
            user_token=user_token,
            world_model=world_model,
            model_name=req.model,
            conversation_id=conversation_id,
        ):
            yield chunk
            # Thu thập response để lưu DB
            if '"type": "done"' in chunk:
                import json

                try:
                    data_line = next(
                        line for line in chunk.splitlines() if line.startswith("data:")
                    )
                    data = json.loads(data_line.removeprefix("data:").strip())
                    response = data.get("response")
                    if isinstance(response, dict):
                        full_response = str(response.get("message") or "")
                except (json.JSONDecodeError, ValueError, StopIteration):
                    pass

        # Lưu phản hồi hoàn chỉnh
        if full_response:
            await chat_store.add_message(
                db,
                conversation_id,
                user_id,
                "assistant",
                full_response,
            )

    return StreamingResponse(
        _stream_wrapper(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Conversation-Id": conversation_id,
        },
    )


@router.post("/upload", response_model=ChatResponse)
async def chat_with_file(
    request: Request,
    message: str = Form(...),
    file: UploadFile = File(...),
    conversation_id: str | None = Form(None),
    model: str | None = Form(None),
    db: AsyncDatabase = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Chat kèm upload file."""
    _validate_model_name(model)
    user_id = current_user["_id"]
    conv_id = conversation_id or uuid.uuid4().hex
    hautoml_cfg = get_hautoml_config()

    auth_header = request.headers.get("Authorization", "")
    user_token = (
        auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
    )

    import httpx

    file_content = await file.read()
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{hautoml_cfg['base_url']}/upload_files",
                files={"files": (file.filename, file_content, file.content_type)},
            )
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="HAutoML upload timed out") from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502, detail="HAutoML upload unavailable"
        ) from exc

    if not 200 <= resp.status_code < 300:
        status_code = resp.status_code if 400 <= resp.status_code < 500 else 502
        raise HTTPException(
            status_code=status_code,
            detail=f"HAutoML upload returned HTTP {resp.status_code}",
        )
    file_info = f"\n[File đã upload: {file.filename} — {len(file_content)} bytes]"

    full_message = f"{message}{file_info}"
    await chat_store.add_message(db, conv_id, user_id, "user", full_message)

    world_model = await _load_world_model(db, str(user_id))

    result = await _call_agent(
        message=full_message,
        user_token=user_token,
        user_id=str(user_id),
        world_model=world_model,
        model_name=model,
    )

    await chat_store.add_message(db, conv_id, user_id, "assistant", result["message"])

    return _to_chat_response(result, conv_id)


@router.get("/health")
async def health_check():
    """Kiểm tra kết nối — HAgent agent + HAutoML backend."""
    import httpx

    hautoml_cfg = get_hautoml_config()

    hautoml_ok = False
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{hautoml_cfg['base_url']}/home")
            hautoml_ok = resp.status_code == 200
    except Exception:
        pass

    # Liệt kê LLM models từ config
    from hagent.agent.llm_config import list_available_models

    models = list_available_models()

    return {
        "agent_runtime": "hagent (LangGraph)",
        "hautoml_connected": hautoml_ok,
        "available_models": models,
        "mode": "multi-agent",
    }


@router.get("/suggestions")
async def get_chat_suggestions():
    """Gợi ý chat ban đầu — đọc từ config."""
    suggestions = get_suggestions()
    return {"suggestions": suggestions}


@router.get("/models")
async def list_llm_models():
    """Liệt kê các LLM models khả dụng."""
    from hagent.agent.llm_config import list_available_models

    return {"models": list_available_models()}


@router.delete("/conversation/{conversation_id}")
async def clear_conversation(
    conversation_id: str,
    db: AsyncDatabase = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Xóa cuộc hội thoại."""
    deleted = await chat_store.delete_conversation(
        db, conversation_id, current_user["_id"]
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Không tìm thấy cuộc hội thoại")
    return {"status": "deleted", "conversation_id": conversation_id}


@router.get("/conversations")
async def list_user_conversations(
    db: AsyncDatabase = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Liệt kê các cuộc hội thoại gần nhất của người dùng."""
    conversations = await chat_store.list_conversations(db, current_user["_id"])
    return {"conversations": conversations}


@router.get("/conversation/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: str,
    db: AsyncDatabase = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Lấy toàn bộ tin nhắn của một cuộc hội thoại."""
    messages = await chat_store.get_message_history(
        db, conversation_id, current_user["_id"], limit=200
    )
    return {
        "conversation_id": conversation_id,
        "messages": [
            {
                "role": m["role"],
                "content": m["content"],
                "timestamp": m.get("timestamp", ""),
            }
            for m in messages
        ],
    }
