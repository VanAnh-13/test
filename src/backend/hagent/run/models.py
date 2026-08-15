"""Các hợp đồng request công khai cho API durable run của Toolkit."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SAFE_RUNTIME_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"


class RunHistoryMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=32768)


class StartRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=32768)
    run_id: str | None = Field(default=None, pattern=SAFE_RUNTIME_ID_PATTERN)
    command_id: str | None = Field(default=None, pattern=SAFE_RUNTIME_ID_PATTERN)
    history: list[RunHistoryMessage] = Field(default_factory=list, max_length=20)
    model: str | None = Field(default=None, min_length=1, max_length=128)


class ResolveRunApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool
    command_id: str | None = Field(default=None, pattern=SAFE_RUNTIME_ID_PATTERN)
    response: dict[str, Any] = Field(default_factory=dict, max_length=16)


class CancelRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command_id: str | None = Field(default=None, pattern=SAFE_RUNTIME_ID_PATTERN)
