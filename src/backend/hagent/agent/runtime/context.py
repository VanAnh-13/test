"""Context tạm thời của một lượt chạy LangGraph.

Module này là ranh giới authority giữa transport và runtime graph. Dữ liệu ở
đây chỉ tồn tại trong lúc thực thi; checkpoint state không được tham chiếu hoặc
sao chép credential và service handle từ context này.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any

_MAX_PRINCIPAL_ID_LENGTH = 256
_MAX_TRACE_ID_LENGTH = 256
_SENSITIVE_STATE_KEYS = frozenset(
    {
        "credential",
        "token",
        "user_token",
        "_wm_service",
        "_world_store",
    }
)


@dataclass(frozen=True, slots=True, kw_only=True)
class GraphRequestContext:
    """Authority và tài nguyên tạm thời được inject ngoài dữ liệu của model."""

    principal_id: str
    credential: str | None = field(default=None, repr=False, compare=False)
    trace_id: str | None = None
    deadline: datetime | None = None
    services: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    capability_snapshot: Any | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.principal_id, str):
            raise TypeError("GraphRequestContext.principal_id must be a string")
        principal_id = self.principal_id.strip()
        if not principal_id:
            raise ValueError("GraphRequestContext.principal_id must not be empty")
        if len(principal_id) > _MAX_PRINCIPAL_ID_LENGTH:
            raise ValueError("GraphRequestContext.principal_id is too long")
        if "\x00" in principal_id:
            raise ValueError("GraphRequestContext.principal_id contains NUL")
        if self.credential is not None and (
            not isinstance(self.credential, str) or not self.credential
        ):
            raise ValueError(
                "GraphRequestContext.credential must be a non-empty string"
            )
        if self.trace_id is not None:
            if not isinstance(self.trace_id, str) or not self.trace_id.strip():
                raise ValueError("GraphRequestContext.trace_id must not be empty")
            if len(self.trace_id) > _MAX_TRACE_ID_LENGTH:
                raise ValueError("GraphRequestContext.trace_id is too long")
        if self.deadline is not None and self.deadline.tzinfo is None:
            raise ValueError("GraphRequestContext.deadline must be timezone-aware")
        if not isinstance(self.services, Mapping):
            raise TypeError("GraphRequestContext.services must be a mapping")

        object.__setattr__(self, "principal_id", principal_id)
        object.__setattr__(
            self,
            "services",
            MappingProxyType(dict(self.services)),
        )


def _sanitize_node_output(
    value: Any,
    *,
    context: GraphRequestContext,
) -> Any:
    """Loại authority tạm thời khỏi dữ liệu có thể được checkpoint."""
    if (
        context.credential is not None
        and isinstance(value, str)
        and context.credential in value
    ):
        return value.replace(context.credential, "[REDACTED]")
    if any(value is service for service in context.services.values()):
        return None
    if isinstance(value, Mapping):
        return {
            key: _sanitize_node_output(item, context=context)
            for key, item in value.items()
            if str(key).lower() not in _SENSITIVE_STATE_KEYS
        }
    if isinstance(value, list):
        return [_sanitize_node_output(item, context=context) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_node_output(item, context=context) for item in value)
    return value


def bind_request_context(
    node: Callable[[dict[str, Any]], Awaitable[Any]],
    *,
    include_credential: bool = True,
) -> Callable[[Mapping[str, Any], Any], Awaitable[Any]]:
    """Bọc legacy node bằng state view tạm thời lấy authority từ context."""

    async def contextual_node(state: Mapping[str, Any], runtime: Any) -> Any:
        context = getattr(runtime, "context", None)
        if not isinstance(context, GraphRequestContext):
            raise TypeError("LANGGRAPH_REQUEST_CONTEXT_REQUIRED")

        state_view = dict(state)
        for key in _SENSITIVE_STATE_KEYS:
            state_view.pop(key, None)
        state_view["user_id"] = context.principal_id
        if include_credential:
            state_view["user_token"] = context.credential

        wm_service = context.services.get("wm_service")
        if wm_service is not None:
            state_view["_wm_service"] = wm_service
        world_store = context.services.get("world_store")
        if world_store is not None:
            state_view["_world_store"] = world_store

        result = await node(state_view)
        return _sanitize_node_output(result, context=context)

    contextual_node.__name__ = getattr(node, "__name__", "contextual_node")
    contextual_node.__doc__ = getattr(node, "__doc__", None)
    return contextual_node
