"""Contract test cho Mongo ledger trong package runtime."""

from __future__ import annotations

from pathlib import Path


def test_runtime_package_owns_store_without_root_compatibility_module():
    from hagent.agent.runtime import (
        MongoRuntimeEventStore,
        RuntimeLedgerSensitiveData,
        RuntimeLedgerUnavailable,
    )

    assert MongoRuntimeEventStore.__module__ == "hagent.agent.runtime.store"
    assert RuntimeLedgerSensitiveData.__module__ == "hagent.agent.runtime.store"
    assert RuntimeLedgerUnavailable.__module__ == "hagent.agent.runtime.store"
    agent_dir = Path(__file__).parents[1] / "hagent" / "agent"
    assert not (agent_dir / "runtime_store.py").exists()
