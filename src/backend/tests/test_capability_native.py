from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError

import pytest

from hagent.agent.runtime import RequestScope


def _descriptor(*, capability_id: str = "test.echo@1", provider_id: str = "fake"):
    from hagent.agent.capabilities.models import CapabilityDescriptor

    return CapabilityDescriptor(
        id=capability_id,
        input_schema={
            "type": "object",
            "required": ["value"],
            "properties": {"value": {"type": "string", "minLength": 1}},
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "required": ["value"],
            "properties": {"value": {"type": "string"}},
            "additionalProperties": False,
        },
        effect="read",
        required_scopes=frozenset({"dataset:read"}),
        provider_id=provider_id,
    )


class _FakeAdapter:
    def __init__(self, output=None, *, delay: float = 0):
        self.output = output if output is not None else {"value": "ok"}
        self.delay = delay
        self.calls = []

    async def invoke(self, capability_id, arguments, *, scope):
        self.calls.append((capability_id, dict(arguments), scope))
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.output


def _scope(
    owner: str = "owner-1",
    *,
    credential: str | None = "request-secret",
    scopes=("dataset:read",),
):
    return RequestScope(
        principal_id=owner,
        credential=credential,
        services={"scopes": scopes},
    )


def test_descriptor_is_immutable_and_rejects_invalid_contract():
    from hagent.agent.capabilities.models import CapabilityDescriptor

    source_schema = {"type": "object", "properties": {}}
    descriptor = CapabilityDescriptor(
        id="automl.dataset.list@1",
        input_schema=source_schema,
        output_schema={"type": "array", "items": {"type": "object"}},
        effect="read",
        required_scopes=frozenset({"dataset:read"}),
        provider_id="hagent-native",
    )
    source_schema["type"] = "string"

    assert descriptor.input_schema["type"] == "object"
    with pytest.raises(FrozenInstanceError):
        descriptor.effect = "write"
    with pytest.raises(TypeError):
        descriptor.input_schema["type"] = "string"
    with pytest.raises(ValueError, match="versioned"):
        CapabilityDescriptor(
            id="automl.dataset.list",
            input_schema={},
            output_schema={},
            effect="read",
            provider_id="native",
        )
    with pytest.raises(ValueError, match="effect"):
        CapabilityDescriptor(
            id="automl.dataset.list@1",
            input_schema={},
            output_schema={},
            effect="unsafe",
            provider_id="native",
        )


def test_catalog_snapshot_is_frozen_hashed_and_provider_toggle_is_next_run_only():
    from hagent.agent.capabilities.catalog import CapabilityCatalog

    adapter = _FakeAdapter()
    catalog = CapabilityCatalog()
    catalog.register_provider("fake", [_descriptor()], adapter)

    first = catalog.snapshot()
    same = catalog.snapshot()
    catalog.set_provider_enabled("fake", False)
    disabled = catalog.snapshot()
    catalog.set_provider_enabled("fake", True)
    enabled_again = catalog.snapshot()

    assert first.digest == same.digest == enabled_again.digest
    assert tuple(first.descriptors) == ("test.echo@1",)
    assert tuple(disabled.descriptors) == ()
    assert tuple(first.descriptors) == ("test.echo@1",)
    with pytest.raises(TypeError):
        first.descriptors["other@1"] = _descriptor(capability_id="other@1")


def test_catalog_rejects_duplicate_or_mismatched_provider_contract():
    from hagent.agent.capabilities.catalog import CapabilityCatalog

    catalog = CapabilityCatalog()
    catalog.register_provider("fake", [_descriptor()], _FakeAdapter())

    with pytest.raises(ValueError, match="Duplicate capability"):
        catalog.register_provider(
            "other", [_descriptor(provider_id="other")], _FakeAdapter()
        )
    with pytest.raises(ValueError, match="provider"):
        CapabilityCatalog().register_provider(
            "other",
            [_descriptor(provider_id="fake")],
            _FakeAdapter(),
        )


@pytest.mark.asyncio
async def test_broker_injects_scope_and_caches_reads_per_owner_without_credential_key():
    from hagent.agent.capabilities.broker import InvocationBroker
    from hagent.agent.capabilities.catalog import CapabilityCatalog

    adapter = _FakeAdapter()
    catalog = CapabilityCatalog()
    catalog.register_provider("fake", [_descriptor()], adapter)
    broker = InvocationBroker(catalog.snapshot(), timeout_seconds=1)

    first = await broker.invoke("test.echo@1", {"value": "hello"}, scope=_scope())
    cached = await broker.invoke(
        "test.echo@1",
        {"value": "hello"},
        scope=_scope(credential="rotated-secret"),
    )
    other_owner = await broker.invoke(
        "test.echo@1",
        {"value": "hello"},
        scope=_scope("owner-2"),
    )

    assert first.output == cached.output == other_owner.output == {"value": "ok"}
    assert not first.cache_hit
    assert cached.cache_hit
    assert not other_owner.cache_hit
    assert len(adapter.calls) == 2
    assert adapter.calls[0][2].credential == "request-secret"
    assert "request-secret" not in repr(first)
    assert "rotated-secret" not in repr(cached)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "code"),
    [
        (_scope(credential=None), "AUTH_SCOPE_REQUIRED"),
        (_scope(scopes=()), "SCOPE_DENIED"),
    ],
)
async def test_broker_fails_closed_for_missing_auth_or_scope(scope, code):
    from hagent.agent.capabilities.broker import InvocationBroker
    from hagent.agent.capabilities.catalog import CapabilityCatalog
    from hagent.agent.capabilities.models import CapabilityInvocationError

    adapter = _FakeAdapter()
    catalog = CapabilityCatalog()
    catalog.register_provider("fake", [_descriptor()], adapter)

    with pytest.raises(CapabilityInvocationError) as exc_info:
        await InvocationBroker(catalog.snapshot()).invoke(
            "test.echo@1",
            {"value": "hello"},
            scope=scope,
        )

    assert exc_info.value.code == code
    assert not adapter.calls


@pytest.mark.asyncio
async def test_broker_validates_input_and_output_contracts():
    from hagent.agent.capabilities.broker import InvocationBroker
    from hagent.agent.capabilities.catalog import CapabilityCatalog
    from hagent.agent.capabilities.models import CapabilityInvocationError

    adapter = _FakeAdapter(output={"wrong": True})
    catalog = CapabilityCatalog()
    catalog.register_provider("fake", [_descriptor()], adapter)
    broker = InvocationBroker(catalog.snapshot())

    with pytest.raises(CapabilityInvocationError) as input_error:
        await broker.invoke("test.echo@1", {"value": ""}, scope=_scope())
    assert input_error.value.code == "INVALID_INPUT"
    assert not adapter.calls

    with pytest.raises(CapabilityInvocationError) as output_error:
        await broker.invoke("test.echo@1", {"value": "valid"}, scope=_scope())
    assert output_error.value.code == "INVALID_OUTPUT"
    assert len(adapter.calls) == 1


@pytest.mark.asyncio
async def test_broker_rejects_model_supplied_authority_before_adapter():
    from hagent.agent.capabilities.broker import InvocationBroker
    from hagent.agent.capabilities.catalog import CapabilityCatalog
    from hagent.agent.capabilities.models import (
        CapabilityDescriptor,
        CapabilityInvocationError,
    )

    descriptor = CapabilityDescriptor(
        id="test.authority@1",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        effect="read",
        required_scopes=frozenset({"dataset:read"}),
        provider_id="fake",
    )
    adapter = _FakeAdapter()
    catalog = CapabilityCatalog()
    catalog.register_provider("fake", [descriptor], adapter)

    with pytest.raises(CapabilityInvocationError) as exc_info:
        await InvocationBroker(catalog.snapshot()).invoke(
            "test.authority@1",
            {"nested": {"user_token": "model-secret"}},
            scope=_scope(),
        )

    assert exc_info.value.code == "INVALID_INPUT"
    assert "model-secret" not in str(exc_info.value)
    assert not adapter.calls


@pytest.mark.asyncio
async def test_broker_rejects_reflected_credential_and_does_not_cache_it():
    from hagent.agent.capabilities.broker import InvocationBroker
    from hagent.agent.capabilities.catalog import CapabilityCatalog
    from hagent.agent.capabilities.models import CapabilityInvocationError

    adapter = _FakeAdapter(output={"value": "Bearer request-secret"})
    catalog = CapabilityCatalog()
    catalog.register_provider("fake", [_descriptor()], adapter)
    broker = InvocationBroker(catalog.snapshot())

    with pytest.raises(CapabilityInvocationError) as exc_info:
        await broker.invoke("test.echo@1", {"value": "hello"}, scope=_scope())
    adapter.output = {"value": "safe"}
    result = await broker.invoke("test.echo@1", {"value": "hello"}, scope=_scope())

    assert exc_info.value.code == "INVALID_OUTPUT"
    assert "request-secret" not in str(exc_info.value)
    assert result.output == {"value": "safe"}
    assert not result.cache_hit
    assert len(adapter.calls) == 2


@pytest.mark.asyncio
async def test_broker_turns_timeout_into_typed_error():
    from hagent.agent.capabilities.broker import InvocationBroker
    from hagent.agent.capabilities.catalog import CapabilityCatalog
    from hagent.agent.capabilities.models import CapabilityInvocationError

    catalog = CapabilityCatalog()
    catalog.register_provider("fake", [_descriptor()], _FakeAdapter(delay=0.05))

    with pytest.raises(CapabilityInvocationError) as exc_info:
        await InvocationBroker(catalog.snapshot(), timeout_seconds=0.001).invoke(
            "test.echo@1",
            {"value": "hello"},
            scope=_scope(),
        )

    assert exc_info.value.code == "TIMEOUT"


@pytest.mark.asyncio
async def test_native_adapter_lists_and_inspects_only_owned_dataset():
    from hagent.agent.capabilities.native import (
        DATASET_INSPECT_CAPABILITY_ID,
        DATASET_LIST_CAPABILITY_ID,
        HAutoMLNativeAdapter,
    )

    calls = []

    async def list_invoker(arguments):
        calls.append(("list", dict(arguments)))
        return [{"_id": "owned", "dataName": "train.csv"}]

    async def inspect_invoker(arguments):
        calls.append(("inspect", dict(arguments)))
        return {"_id": arguments["dataset_id"], "features": ["x", "target"]}

    adapter = HAutoMLNativeAdapter(
        list_invoker=list_invoker,
        inspect_invoker=inspect_invoker,
    )
    scope = _scope(scopes=("automl.dataset.read",))

    listed = await adapter.invoke(DATASET_LIST_CAPABILITY_ID, {}, scope=scope)
    inspected = await adapter.invoke(
        DATASET_INSPECT_CAPABILITY_ID,
        {"dataset_id": "owned"},
        scope=scope,
    )

    assert listed[0]["_id"] == "owned"
    assert inspected["_id"] == "owned"
    assert calls == [
        ("list", {"user_id": "owner-1", "token": "request-secret"}),
        ("list", {"user_id": "owner-1", "token": "request-secret"}),
        ("inspect", {"dataset_id": "owned", "token": "request-secret"}),
    ]


@pytest.mark.asyncio
async def test_native_adapter_denies_unowned_dataset_before_detail_call():
    from hagent.agent.capabilities.models import CapabilityInvocationError
    from hagent.agent.capabilities.native import (
        DATASET_INSPECT_CAPABILITY_ID,
        HAutoMLNativeAdapter,
    )

    inspect_called = False

    async def list_invoker(_arguments):
        return [{"_id": "owned"}]

    async def inspect_invoker(_arguments):
        nonlocal inspect_called
        inspect_called = True
        return {}

    adapter = HAutoMLNativeAdapter(
        list_invoker=list_invoker,
        inspect_invoker=inspect_invoker,
    )

    with pytest.raises(CapabilityInvocationError) as exc_info:
        await adapter.invoke(
            DATASET_INSPECT_CAPABILITY_ID,
            {"dataset_id": "other-owner-dataset"},
            scope=_scope(scopes=("automl.dataset.read",)),
        )

    assert exc_info.value.code == "RESOURCE_FORBIDDEN"
    assert not inspect_called


def test_native_descriptors_expose_exactly_two_read_capabilities():
    from hagent.agent.capabilities.native import native_dataset_descriptors

    descriptors = native_dataset_descriptors()

    assert {descriptor.id for descriptor in descriptors} == {
        "automl.dataset.list@1",
        "automl.dataset.inspect@1",
    }
    assert {descriptor.effect for descriptor in descriptors} == {"read"}
    assert all(
        descriptor.required_scopes == frozenset({"automl.dataset.read"})
        for descriptor in descriptors
    )
