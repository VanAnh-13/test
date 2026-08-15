"""
HAgent Bridge — Pydantic schemas

Định nghĩa các schema cho request/response của API.
Tất cả giá trị mặc định được tải từ hagent.yaml thông qua config.py.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    """Request chat chuẩn dùng cho cả điểm vào đồng bộ và streaming."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(..., description="Nội dung tin nhắn")
    conversation_id: str | None = Field(
        None,
        description="ID cuộc hội thoại (tạo mới nếu null)",
    )
    context: dict | None = Field(None, description="Ngữ cảnh bổ sung")
    model: str | None = Field(None, description="Model cụ thể — xem hagent.yaml")


class ChatResponse(BaseModel):
    """Response chuẩn được trả về từ trình xử lý chat của Bridge và toolkit."""

    message: str
    conversation_id: str
    sources: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    provider: str = ""
    model: str = ""
    route: str = ""
    tool_outputs: list[dict] = Field(default_factory=list)
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


# ─── Trạng thái hệ thống ─────────────────────────────────


class HealthResponse(BaseModel):
    """Schema kiểm tra trạng thái; tất cả giá trị được lấy từ YAML."""

    hagent_url: str
    connected: bool
    hautoml_connected: bool
    mode: str  # "hagent" hoặc "direct"
    active_provider: str
    active_model: str
    available_providers: list[str]


# ─── Gợi ý ───────────────────────────────────────────────


class SuggestionsResponse(BaseModel):
    """Schema cho gợi ý chat ban đầu."""

    suggestions: list[str]


# ─── Nhà cung cấp model ──────────────────────────────────


class ProviderInfo(BaseModel):
    """Thông tin một nhà cung cấp model được lấy từ YAML."""

    name: str
    provider_id: str
    models: list[str]
    available: bool
    description: str = ""


class ProvidersResponse(BaseModel):
    """Danh sách tất cả nhà cung cấp model được lấy từ YAML."""

    default_provider: str
    default_model: str
    providers: list[ProviderInfo]


# ─── Hội thoại (MongoDB) ────────────────────────────────


class ConversationMessage(BaseModel):
    """Một tin nhắn trong cuộc hội thoại."""

    role: str
    content: str
    timestamp: datetime | None = None
    provider: str = ""
    message_id: str | None = None
    model: str = ""


class Conversation(BaseModel):
    """Một cuộc hội thoại đầy đủ."""

    conversation_id: str
    user_id: str
    messages: list[ConversationMessage] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    provider: str = ""
    model: str = ""
