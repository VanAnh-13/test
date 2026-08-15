"""Adapter chuyển quyền hạn request cho các lời gọi capability của Journey."""

from __future__ import annotations

from hagent.agent.runtime import RequestScope
from hagent.agent.runtime.context import GraphRequestContext


def request_scope_from_context(context: GraphRequestContext) -> RequestScope:
    """Chuyển quyền hạn tạm thời của graph sang hợp đồng runtime capability."""
    return RequestScope(
        principal_id=context.principal_id,
        credential=context.credential,
        trace_id=context.trace_id,
        deadline=context.deadline,
        services=context.services,
    )
