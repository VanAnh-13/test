"""Registry provider và snapshot capability theo từng run."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass

from hagent.agent.capabilities.models import (
    CapabilityAdapter,
    CapabilityDescriptor,
    CapabilitySnapshot,
)


@dataclass(slots=True)
class _Provider:
    adapter: CapabilityAdapter
    descriptors: tuple[CapabilityDescriptor, ...]
    enabled: bool


class CapabilityCatalog:
    """Sở hữu provider mutable; caller chỉ nhận snapshot bất biến."""

    def __init__(self) -> None:
        self._providers: dict[str, _Provider] = {}
        self._capability_ids: set[str] = set()

    def register_provider(
        self,
        provider_id: str,
        descriptors: Iterable[CapabilityDescriptor],
        adapter: CapabilityAdapter,
        *,
        enabled: bool = True,
    ) -> None:
        if provider_id in self._providers:
            raise ValueError(f"Provider is already registered: {provider_id}")
        frozen_descriptors = tuple(descriptors)
        if not frozen_descriptors:
            raise ValueError("Provider must expose at least one capability")
        for descriptor in frozen_descriptors:
            if descriptor.provider_id != provider_id:
                raise ValueError("Capability provider does not match registration provider")
            if descriptor.id in self._capability_ids:
                raise ValueError(f"Duplicate capability ID: {descriptor.id}")
        self._providers[provider_id] = _Provider(
            adapter=adapter,
            descriptors=frozen_descriptors,
            enabled=bool(enabled),
        )
        self._capability_ids.update(descriptor.id for descriptor in frozen_descriptors)

    def set_provider_enabled(self, provider_id: str, enabled: bool) -> None:
        try:
            provider = self._providers[provider_id]
        except KeyError as exc:
            raise KeyError(f"Unknown capability provider: {provider_id}") from exc
        provider.enabled = bool(enabled)

    def snapshot(self) -> CapabilitySnapshot:
        descriptors: dict[str, CapabilityDescriptor] = {}
        adapters: dict[str, CapabilityAdapter] = {}
        for provider_id in sorted(self._providers):
            provider = self._providers[provider_id]
            if not provider.enabled:
                continue
            adapters[provider_id] = provider.adapter
            for descriptor in sorted(provider.descriptors, key=lambda item: item.id):
                descriptors[descriptor.id] = descriptor

        canonical = [descriptors[item].canonical_dict() for item in sorted(descriptors)]
        digest = hashlib.sha256(
            json.dumps(
                canonical,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        return CapabilitySnapshot(
            descriptors=descriptors,
            adapters=adapters,
            digest=digest,
        )

