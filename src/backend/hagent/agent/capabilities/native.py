"""HAutoML native adapter cho dataset và training journey v1."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from hagent.agent.capabilities.models import (
    CapabilityDescriptor,
    CapabilityInvocationError,
)
from hagent.agent.journey.result_critic import TRAINING_RESULTS_CAPABILITY_ID
from hagent.agent.journey.training_operator import (
    TRAINING_LOOKUP_CAPABILITY_ID,
    TRAINING_START_CAPABILITY_ID,
)
from hagent.agent.runtime import RequestScope

DATASET_LIST_CAPABILITY_ID = "automl.dataset.list@1"
DATASET_INSPECT_CAPABILITY_ID = "automl.dataset.inspect@1"
NATIVE_PROVIDER_ID = "hagent-native"

NativeInvoker = Callable[[Mapping[str, Any]], Awaitable[Any]]


async def _invoke_list_tool(arguments: Mapping[str, Any]) -> Any:
    from hagent.agent.tools.automl_tools import list_datasets

    return await list_datasets.ainvoke(dict(arguments))


async def _invoke_inspect_tool(arguments: Mapping[str, Any]) -> Any:
    from hagent.agent.tools.automl_tools import get_dataset_info

    return await get_dataset_info.ainvoke(dict(arguments))


async def _invoke_training_start_tool(arguments: Mapping[str, Any]) -> Any:
    from hagent.agent.tools.automl_tools import start_training

    return await start_training.ainvoke(dict(arguments))


async def _invoke_training_lookup_tool(arguments: Mapping[str, Any]) -> Any:
    from hagent.agent.tools.automl_tools import lookup_training_job

    return await lookup_training_job.ainvoke(dict(arguments))


async def _invoke_training_results_tool(arguments: Mapping[str, Any]) -> Any:
    from hagent.agent.tools.automl_tools import get_training_results

    return await get_training_results.ainvoke(dict(arguments))


def _decode_tool_output(raw_output: Any, *, capability_id: str) -> Any:
    if isinstance(raw_output, str):
        try:
            output = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            raise CapabilityInvocationError(
                "PROVIDER_FAILURE",
                "Native tool returned non-JSON output",
                capability_id=capability_id,
                provider_id=NATIVE_PROVIDER_ID,
            ) from exc
    else:
        output = raw_output
    if isinstance(output, Mapping) and "error" in output:
        raise CapabilityInvocationError(
            "PROVIDER_FAILURE",
            "Native HAutoML API request failed",
            capability_id=capability_id,
            provider_id=NATIVE_PROVIDER_ID,
        )
    return output


def _dataset_items(output: Any) -> list[Mapping[str, Any]]:
    if isinstance(output, list):
        return [item for item in output if isinstance(item, Mapping)]
    if isinstance(output, Mapping):
        for key in ("datasets", "data", "items"):
            items = output.get(key)
            if isinstance(items, list):
                return [item for item in items if isinstance(item, Mapping)]
    return []


def native_dataset_descriptors() -> tuple[CapabilityDescriptor, CapabilityDescriptor]:
    """Khai báo contract cố định của hai native dataset reads."""
    required_scopes = frozenset({"automl.dataset.read"})
    return (
        CapabilityDescriptor(
            id=DATASET_LIST_CAPABILITY_ID,
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            output_schema={"type": "array", "items": {"type": "object"}},
            effect="read",
            required_scopes=required_scopes,
            provider_id=NATIVE_PROVIDER_ID,
        ),
        CapabilityDescriptor(
            id=DATASET_INSPECT_CAPABILITY_ID,
            input_schema={
                "type": "object",
                "required": ["dataset_id"],
                "properties": {"dataset_id": {"type": "string", "minLength": 1}},
                "additionalProperties": False,
            },
            output_schema={"type": "object"},
            effect="read",
            required_scopes=required_scopes,
            provider_id=NATIVE_PROVIDER_ID,
        ),
    )


def native_journey_descriptors() -> tuple[CapabilityDescriptor, ...]:
    """Khai báo native dataset và training capabilities cho Journey v1."""
    training_read_scope = frozenset({"automl.training.read"})
    training_write_scope = frozenset({"automl.training.write"})
    return (
        *native_dataset_descriptors(),
        CapabilityDescriptor(
            id=TRAINING_START_CAPABILITY_ID,
            input_schema={
                "type": "object",
                "required": [
                    "dataset_id",
                    "problem_type",
                    "target_column",
                    "metric",
                    "models",
                    "time_limit",
                    "list_feature",
                    "idempotency_key",
                ],
                "properties": {
                    "dataset_id": {"type": "string", "minLength": 1},
                    "problem_type": {"type": "string", "minLength": 1},
                    "target_column": {"type": "string", "minLength": 1},
                    "metric": {"type": "string", "minLength": 1},
                    "models": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "minItems": 1,
                    },
                    "time_limit": {"type": "integer", "minimum": 1},
                    "list_feature": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "minItems": 1,
                    },
                    "idempotency_key": {"type": "string", "minLength": 1},
                },
                "additionalProperties": False,
            },
            output_schema={"type": "object"},
            effect="write",
            required_scopes=training_write_scope,
            provider_id=NATIVE_PROVIDER_ID,
        ),
        CapabilityDescriptor(
            id=TRAINING_LOOKUP_CAPABILITY_ID,
            input_schema={
                "type": "object",
                "required": ["idempotency_key"],
                "properties": {
                    "idempotency_key": {"type": "string", "minLength": 1},
                },
                "additionalProperties": False,
            },
            output_schema={"type": "object"},
            effect="read",
            required_scopes=training_read_scope,
            provider_id=NATIVE_PROVIDER_ID,
        ),
        CapabilityDescriptor(
            id=TRAINING_RESULTS_CAPABILITY_ID,
            input_schema={
                "type": "object",
                "required": ["job_ids"],
                "properties": {
                    "job_ids": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "minItems": 1,
                        "maxItems": 1,
                    },
                },
                "additionalProperties": False,
            },
            output_schema={"type": "object"},
            effect="read",
            required_scopes=training_read_scope,
            provider_id=NATIVE_PROVIDER_ID,
        ),
    )


class HAutoMLNativeAdapter:
    """Chuyển native tools thành typed output và bổ sung owner guard cho inspect."""

    def __init__(
        self,
        *,
        list_invoker: NativeInvoker = _invoke_list_tool,
        inspect_invoker: NativeInvoker = _invoke_inspect_tool,
        training_start_invoker: NativeInvoker = _invoke_training_start_tool,
        training_lookup_invoker: NativeInvoker = _invoke_training_lookup_tool,
        training_results_invoker: NativeInvoker = _invoke_training_results_tool,
    ) -> None:
        self._list_invoker = list_invoker
        self._inspect_invoker = inspect_invoker
        self._training_start_invoker = training_start_invoker
        self._training_lookup_invoker = training_lookup_invoker
        self._training_results_invoker = training_results_invoker

    async def _list_owned(self, scope: RequestScope) -> Any:
        raw_output = await self._list_invoker(
            {"user_id": scope.principal_id, "token": scope.credential}
        )
        return _decode_tool_output(raw_output, capability_id=DATASET_LIST_CAPABILITY_ID)

    async def invoke(
        self,
        capability_id: str,
        arguments: Mapping[str, Any],
        *,
        scope: RequestScope,
    ) -> Any:
        if not scope.credential:
            raise CapabilityInvocationError(
                "AUTH_SCOPE_REQUIRED",
                "Authenticated request scope is required",
                capability_id=capability_id,
                provider_id=NATIVE_PROVIDER_ID,
            )
        if capability_id == DATASET_LIST_CAPABILITY_ID:
            return await self._list_owned(scope)
        if capability_id == TRAINING_START_CAPABILITY_ID:
            trusted_arguments = dict(arguments)
            trusted_arguments["user_id"] = scope.principal_id
            trusted_arguments["token"] = scope.credential
            raw_output = await self._training_start_invoker(trusted_arguments)
            return _decode_tool_output(raw_output, capability_id=capability_id)
        if capability_id == TRAINING_LOOKUP_CAPABILITY_ID:
            raw_output = await self._training_lookup_invoker(
                {
                    "idempotency_key": arguments.get("idempotency_key"),
                    "token": scope.credential,
                }
            )
            return _decode_tool_output(raw_output, capability_id=capability_id)
        if capability_id == TRAINING_RESULTS_CAPABILITY_ID:
            raw_output = await self._training_results_invoker(
                {
                    "job_ids": arguments.get("job_ids"),
                    "token": scope.credential,
                }
            )
            return _decode_tool_output(raw_output, capability_id=capability_id)
        if capability_id != DATASET_INSPECT_CAPABILITY_ID:
            raise CapabilityInvocationError(
                "CAPABILITY_NOT_FOUND",
                "Native adapter does not expose this capability",
                capability_id=capability_id,
                provider_id=NATIVE_PROVIDER_ID,
            )

        dataset_id = str(arguments.get("dataset_id", ""))
        owned_output = await self._list_owned(scope)
        owned_ids = {
            str(item.get("_id") or item.get("id"))
            for item in _dataset_items(owned_output)
            if item.get("_id") is not None or item.get("id") is not None
        }
        if dataset_id not in owned_ids:
            raise CapabilityInvocationError(
                "RESOURCE_FORBIDDEN",
                "Dataset is not available in the request owner scope",
                capability_id=capability_id,
                provider_id=NATIVE_PROVIDER_ID,
            )
        raw_output = await self._inspect_invoker(
            {"dataset_id": dataset_id, "token": scope.credential}
        )
        return _decode_tool_output(raw_output, capability_id=capability_id)
