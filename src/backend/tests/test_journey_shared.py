"""Test hồi quy cho serialization và adapter quyền hạn dùng chung của Journey."""

from datetime import UTC, datetime

from hagent.agent.journey.canonical import canonical_mapping_hash
from hagent.agent.journey.request_scope import request_scope_from_context
from hagent.agent.runtime.context import GraphRequestContext


def test_canonical_mapping_hash_is_order_independent_and_unicode_stable():
    expected = "c44a01bbb478d30c24977dde4448ff12e45ed078a9b36786b078a45579f259f8"

    assert canonical_mapping_hash({"b": "Tiếng Việt", "a": 1}) == expected
    assert canonical_mapping_hash({"a": 1, "b": "Tiếng Việt"}) == expected


def test_request_scope_preserves_transient_graph_authority():
    service = object()
    deadline = datetime(2026, 8, 14, 16, 0, tzinfo=UTC)
    context = GraphRequestContext(
        principal_id="tenant-user",
        credential="secret-credential",
        trace_id="trace-123",
        deadline=deadline,
        services={"service": service},
    )

    scope = request_scope_from_context(context)

    assert scope.principal_id == context.principal_id
    assert scope.credential == context.credential
    assert scope.trace_id == context.trace_id
    assert scope.deadline is deadline
    assert scope.services is context.services
    assert scope.services["service"] is service
