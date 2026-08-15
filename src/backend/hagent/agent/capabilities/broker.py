"""Boundary gọi capability với validation, policy, timeout và cache owner-scoped."""

from __future__ import annotations

import asyncio
import copy
import json
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from hagent.agent.capabilities.models import (
    CapabilityDescriptor,
    CapabilityInvocationError,
    CapabilityResult,
    CapabilitySnapshot,
)
from hagent.agent.runtime import RequestScope

_DEFAULT_TIMEOUT_SECONDS = 30.0
_DEFAULT_MAX_CACHE_ENTRIES = 256
_SENSITIVE_ARGUMENT_KEYS = frozenset(
    {
        "authorization",
        "credential",
        "principalid",
        "token",
        "userid",
        "usertoken",
    }
)


class _SchemaViolation(ValueError):
    pass


def _matches_type(value: Any, expected_type: str) -> bool:
    if expected_type == "null":
        return value is None
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "array":
        return isinstance(value, list | tuple)
    if expected_type == "object":
        return isinstance(value, Mapping)
    return False


def _validate_json(value: Any, schema: Mapping[str, Any], *, path: str) -> None:
    expected = schema.get("type")
    expected_types = (expected,) if isinstance(expected, str) else tuple(expected or ())
    if expected_types and not any(_matches_type(value, item) for item in expected_types):
        raise _SchemaViolation(f"{path} has invalid type")
    if "enum" in schema and value not in schema["enum"]:
        raise _SchemaViolation(f"{path} is not an allowed value")
    if isinstance(value, str) and len(value) < int(schema.get("minLength", 0)):
        raise _SchemaViolation(f"{path} is shorter than minLength")

    if isinstance(value, Mapping):
        required = tuple(schema.get("required", ()))
        missing = [key for key in required if key not in value]
        if missing:
            raise _SchemaViolation(f"{path} is missing required fields")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(properties)
            if extra:
                raise _SchemaViolation(f"{path} contains unsupported fields")
        for key, item in value.items():
            child_schema = properties.get(key)
            if child_schema is not None:
                _validate_json(item, child_schema, path=f"{path}.{key}")

    if isinstance(value, list | tuple) and isinstance(schema.get("items"), Mapping):
        for index, item in enumerate(value):
            _validate_json(item, schema["items"], path=f"{path}[{index}]")


def _granted_scopes(scope: RequestScope) -> frozenset[str]:
    raw_scopes = scope.services.get("scopes", ())
    if isinstance(raw_scopes, str) or not isinstance(raw_scopes, Sequence | set | frozenset):
        return frozenset()
    return frozenset(item for item in raw_scopes if isinstance(item, str))


def _contains_model_authority(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = "".join(character for character in str(key).lower() if character.isalnum())
            if normalized_key in _SENSITIVE_ARGUMENT_KEYS or _contains_model_authority(item):
                return True
    elif isinstance(value, list | tuple):
        return any(_contains_model_authority(item) for item in value)
    return False


def _contains_credential(value: Any, credential: str | None) -> bool:
    if not credential:
        return False
    if isinstance(value, str):
        return credential in value
    if isinstance(value, Mapping):
        return any(
            _contains_credential(key, credential) or _contains_credential(item, credential)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple):
        return any(_contains_credential(item, credential) for item in value)
    return False


class InvocationBroker:
    """Thực thi capability từ snapshot cố định, không đọc global registry."""

    def __init__(
        self,
        snapshot: CapabilitySnapshot,
        *,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        max_cache_entries: int = _DEFAULT_MAX_CACHE_ENTRIES,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_cache_entries < 0:
            raise ValueError("max_cache_entries must not be negative")
        self._snapshot = snapshot
        self._timeout_seconds = float(timeout_seconds)
        self._max_cache_entries = max_cache_entries
        self._cache: OrderedDict[tuple[str, str, str], Any] = OrderedDict()

    def _authorize(self, descriptor: CapabilityDescriptor, scope: RequestScope) -> None:
        if descriptor.required_scopes and (
            not isinstance(scope.credential, str) or not scope.credential
        ):
            raise CapabilityInvocationError(
                "AUTH_SCOPE_REQUIRED",
                "Authenticated request scope is required",
                capability_id=descriptor.id,
                provider_id=descriptor.provider_id,
            )
        if not descriptor.required_scopes.issubset(_granted_scopes(scope)):
            raise CapabilityInvocationError(
                "SCOPE_DENIED",
                "Request scope does not permit this capability",
                capability_id=descriptor.id,
                provider_id=descriptor.provider_id,
            )

    def _effective_timeout(self, scope: RequestScope) -> float:
        timeout = self._timeout_seconds
        if scope.deadline is not None:
            remaining = (
                scope.deadline - datetime.now(scope.deadline.tzinfo)
            ).total_seconds()
            timeout = min(timeout, remaining)
        if timeout <= 0:
            raise CapabilityInvocationError("TIMEOUT", "Capability deadline has elapsed")
        return timeout

    async def invoke(
        self,
        capability_id: str,
        arguments: Mapping[str, Any],
        *,
        scope: RequestScope,
    ) -> CapabilityResult:
        descriptor = self._snapshot.descriptors.get(capability_id)
        if descriptor is None:
            raise CapabilityInvocationError(
                "CAPABILITY_NOT_FOUND",
                "Capability is not available in this run snapshot",
                capability_id=capability_id,
            )
        self._authorize(descriptor, scope)
        if not isinstance(arguments, Mapping):
            raise CapabilityInvocationError(
                "INVALID_INPUT",
                "Capability input must be an object",
                capability_id=capability_id,
                provider_id=descriptor.provider_id,
            )
        normalized_arguments = dict(arguments)
        if _contains_model_authority(normalized_arguments):
            raise CapabilityInvocationError(
                "INVALID_INPUT",
                "Capability input must not contain request authority",
                capability_id=capability_id,
                provider_id=descriptor.provider_id,
            )
        try:
            _validate_json(normalized_arguments, descriptor.input_schema, path="input")
        except _SchemaViolation as exc:
            raise CapabilityInvocationError(
                "INVALID_INPUT",
                str(exc),
                capability_id=capability_id,
                provider_id=descriptor.provider_id,
            ) from exc

        try:
            canonical_arguments = json.dumps(
                normalized_arguments,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise CapabilityInvocationError(
                "INVALID_INPUT",
                "Capability input must be valid JSON",
                capability_id=capability_id,
                provider_id=descriptor.provider_id,
            ) from exc
        cache_key = (
            scope.principal_id,
            capability_id,
            canonical_arguments,
        )
        if descriptor.effect == "read" and cache_key in self._cache:
            output = self._cache.pop(cache_key)
            self._cache[cache_key] = output
            return CapabilityResult(
                capability_id=capability_id,
                provider_id=descriptor.provider_id,
                output=copy.deepcopy(output),
                cache_hit=True,
            )

        adapter = self._snapshot.adapters[descriptor.provider_id]
        try:
            output = await asyncio.wait_for(
                adapter.invoke(capability_id, normalized_arguments, scope=scope),
                timeout=self._effective_timeout(scope),
            )
        except TimeoutError as exc:
            raise CapabilityInvocationError(
                "TIMEOUT",
                "Capability invocation timed out",
                capability_id=capability_id,
                provider_id=descriptor.provider_id,
            ) from exc
        except CapabilityInvocationError:
            raise
        except Exception as exc:
            raise CapabilityInvocationError(
                "PROVIDER_FAILURE",
                "Capability provider invocation failed",
                capability_id=capability_id,
                provider_id=descriptor.provider_id,
            ) from exc

        try:
            if _contains_credential(output, scope.credential):
                raise _SchemaViolation("output contains request credential")
            json.dumps(output, allow_nan=False, ensure_ascii=False)
            _validate_json(output, descriptor.output_schema, path="output")
        except (TypeError, ValueError, _SchemaViolation) as exc:
            raise CapabilityInvocationError(
                "INVALID_OUTPUT",
                "Capability provider returned an invalid contract",
                capability_id=capability_id,
                provider_id=descriptor.provider_id,
            ) from exc

        if descriptor.effect == "read" and self._max_cache_entries:
            self._cache[cache_key] = copy.deepcopy(output)
            while len(self._cache) > self._max_cache_entries:
                self._cache.popitem(last=False)
        return CapabilityResult(
            capability_id=capability_id,
            provider_id=descriptor.provider_id,
            output=copy.deepcopy(output),
        )
