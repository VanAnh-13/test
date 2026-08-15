# Các import bên dưới là API tương thích ngược, được client và test gọi trực tiếp.
# ruff: noqa: F401

from __future__ import annotations

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from hagent.bridge import conversation as conv_store
from hagent.bridge.config import (
    get_bridge_config,
    get_hautoml_config,
    get_llm_config,
    get_llm_models,
    get_mongodb_config,
    get_world_state_config,
)
from hagent.bridge.models import ChatRequest, ChatResponse
from hagent.bridge.routes import (
    route_support,
    agent_control,
    conversations,
    world_model,
)
from hagent.bridge.routes import chat as chat_routes
from hagent.core.errors import HAgentError
from hagent.observability.logging import correlation_id_middleware

lifespan = route_support.bridge_lifespan
logger = route_support.logger
_call_agent_runtime = route_support.compat_call_agent_runtime
_call_hagent_gateway = route_support.call_hagent_gateway
_apply_tool_outputs_to_world_state = route_support.apply_tool_outputs_to_world_state
_bridge_event_stream = route_support.bridge_event_stream
_extract_training_job_id = route_support.extract_training_job_id
_runtime_context = route_support.runtime_context
_schedule_training_result_notification = route_support.schedule_training_result_notification
_stream_agent_runtime_lines = route_support.stream_agent_runtime_lines
_to_chat_response = route_support.to_chat_response
_validate_model_name = route_support.validate_model_name
chat = route_support.compat_chat
chat_stream = route_support.compat_chat_stream
chat_with_file = route_support.compat_chat_with_file
list_providers = route_support.compat_list_providers

_bridge_cfg = get_bridge_config()
hagent_bridge = FastAPI(
    title="HAgent Bridge",
    description="Lớp trung gian giữa ChatWidget và HAgent runtime",
    version="2.0.0",
    lifespan=lifespan,
)
hagent_bridge.add_middleware(
    CORSMiddleware,
    allow_origins=_bridge_cfg.get("cors_origins", ["http://localhost:3000"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
hagent_bridge.middleware("http")(correlation_id_middleware)


@hagent_bridge.exception_handler(HAgentError)
async def handle_hagent_error(_request: Request, exc: HAgentError) -> JSONResponse:
    """Chuyển lỗi domain sang HTTP mà không làm lộ context hoặc exception gốc."""
    logger.error("Lỗi HAgent: %s", type(exc).__name__, exc_info=exc.cause)
    return JSONResponse(
        status_code=exc.http_status_code,
        content={"detail": exc.to_public_dict()},
    )


for route_module in (
    chat_routes,
    conversations,
    agent_control,
    world_model,
):
    hagent_bridge.include_router(route_module.router)

app = hagent_bridge


def main() -> None:
    import uvicorn

    cfg = get_bridge_config()
    uvicorn.run(
        "hagent.bridge.app:hagent_bridge",
        host=cfg["host"],
        port=cfg["port"],
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
