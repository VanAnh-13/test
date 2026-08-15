"""Kiểu công khai của capability seam, độc lập với transport và LangGraph."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from hagent.agent.runtime import RequestScope

CapabilityEffect = Literal["read", "write"]
CapabilityErrorCode = Literal[
    "AUTH_SCOPE_REQUIRED",
    "CAPABILITY_NOT_FOUND",
    "INVALID_INPUT",
    "INVALID_OUTPUT",
    "PROVIDER_FAILURE",
    "RESOURCE_FORBIDDEN",
    "SCOPE_DENIED",
    "TIMEOUT",
]

_CAPABILITY_ID_PATTERN = re.compile(
    r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*@[1-9][0-9]*"
)
_PROVIDER_ID_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*")
_SCOPE_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:[.:][a-z][a-z0-9_-]*)+")
_JSON_SCHEMA_TYPES = frozenset(
    {"array", "boolean", "integer", "null", "number", "object", "string"}
)


def freeze_json(value: Any) -> Any:
    """Đóng băng một JSON-like value để snapshot không bị sửa ngầm."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): freeze_json(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(freeze_json(item) for item in value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(f"Unsupported JSON schema value: {type(value).__name__}")


def thaw_json(value: Any) -> Any:
    """Tạo bản JSON mutable dùng cho canonical hash và validation."""
    if isinstance(value, Mapping):
        return {str(key): thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def _validate_schema_definition(schema: Mapping[str, Any], *, field_name: str) -> None:
    schema_type = schema.get("type")
    if schema_type is None:
        return
    schema_types = (schema_type,) if isinstance(schema_type, str) else schema_type
    if not isinstance(schema_types, list | tuple) or not schema_types:
        raise ValueError(f"{field_name}.type must be a string or non-empty list")
    if any(item not in _JSON_SCHEMA_TYPES for item in schema_types):
        raise ValueError(f"{field_name}.type contains an unsupported JSON type")
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        raise TypeError(f"{field_name}.properties must be an object")
    required = schema.get("required", ())
    if not isinstance(required, list | tuple) or any(
        not isinstance(item, str) for item in required
    ):
        raise ValueError(f"{field_name}.required must be a string list")
    for property_name, property_schema in properties.items():
        if not isinstance(property_schema, Mapping):
            raise TypeError(
                f"{field_name}.properties.{property_name} must be an object"
            )
        _validate_schema_definition(
            property_schema,
            field_name=f"{field_name}.properties.{property_name}",
        )
    items = schema.get("items")
    if items is not None:
        if not isinstance(items, Mapping):
            raise ValueError(f"{field_name}.items must be an object")
        _validate_schema_definition(items, field_name=f"{field_name}.items")


@dataclass(frozen=True, slots=True, kw_only=True)
class CapabilityDescriptor:
    """Contract bất biến của một hành động do provider cung cấp."""

    id: str
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    effect: CapabilityEffect
    provider_id: str
    required_scopes: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not _CAPABILITY_ID_PATTERN.fullmatch(self.id):
            raise ValueError("CapabilityDescriptor.id must be a versioned capability ID")
        if self.effect not in ("read", "write"):
            raise ValueError("CapabilityDescriptor.effect must be read or write")
        if not isinstance(self.provider_id, str) or not _PROVIDER_ID_PATTERN.fullmatch(
            self.provider_id
        ):
            raise ValueError("CapabilityDescriptor.provider_id is invalid")
        if not isinstance(self.input_schema, Mapping) or not isinstance(
            self.output_schema,
            Mapping,
        ):
            raise TypeError("Capability schemas must be mappings")
        scopes = frozenset(self.required_scopes)
        if any(not isinstance(scope, str) or not _SCOPE_PATTERN.fullmatch(scope) for scope in scopes):
            raise ValueError("CapabilityDescriptor.required_scopes contains an invalid scope")
        _validate_schema_definition(self.input_schema, field_name="input_schema")
        _validate_schema_definition(self.output_schema, field_name="output_schema")
        object.__setattr__(self, "input_schema", freeze_json(self.input_schema))
        object.__setattr__(self, "output_schema", freeze_json(self.output_schema))
        object.__setattr__(self, "required_scopes", scopes)

    def canonical_dict(self) -> dict[str, Any]:
        """Trả contract chuẩn hóa để tạo snapshot hash ổn định."""
        return {
            "effect": self.effect,
            "id": self.id,
            "input_schema": thaw_json(self.input_schema),
            "output_schema": thaw_json(self.output_schema),
            "provider_id": self.provider_id,
            "required_scopes": sorted(self.required_scopes),
        }


class CapabilityAdapter(Protocol):
    async def invoke(
        self,
        capability_id: str,
        arguments: Mapping[str, Any],
        *,
        scope: RequestScope,
    ) -> Any: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class CapabilitySnapshot:
    """Danh mục provider đã đóng băng tại đầu một run."""

    descriptors: Mapping[str, CapabilityDescriptor]
    adapters: Mapping[str, CapabilityAdapter] = field(repr=False, compare=False)
    digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "descriptors", MappingProxyType(dict(self.descriptors)))
        object.__setattr__(self, "adapters", MappingProxyType(dict(self.adapters)))


@dataclass(frozen=True, slots=True, kw_only=True)
class CapabilityResult:
    capability_id: str
    provider_id: str
    output: Any
    cache_hit: bool = False


class CapabilityInvocationError(RuntimeError):
    """Lỗi an toàn để runtime phân nhánh mà không parse message tự do."""

    def __init__(
        self,
        code: CapabilityErrorCode,
        message: str,
        *,
        capability_id: str | None = None,
        provider_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.capability_id = capability_id
        self.provider_id = provider_id
