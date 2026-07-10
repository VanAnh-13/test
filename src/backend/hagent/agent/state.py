"""
AutoML Agent State — Tham chiếu từ DeerFlow's ThreadState.

Defines the state schema for the LangGraph StateGraph, carrying
messages, AutoML-specific context, World Model snapshot, and memory.

Reference: deerflow/agents/thread_state.py
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

try:
    # Python 3.11+
    from typing import NotRequired
except ImportError:  # Python 3.10 (toolkit Docker image)
    from typing_extensions import NotRequired  # type: ignore[assignment]

try:
    from langgraph.graph.message import add_messages
except ImportError:  # pragma: no cover — unit tests without langgraph
    def add_messages(left, right):  # type: ignore[misc]
        """Minimal list merge reducer fallback."""
        if left is None:
            left = []
        if right is None:
            return list(left)
        if not isinstance(right, list):
            right = [right]
        return list(left) + list(right)


# ── Sub-state fragments (kiểu DeerFlow) ─────────────────


class DatasetContext(TypedDict, total=False):
    """Ngữ cảnh dataset đang được thao tác."""
    id: str
    name: str
    n_rows: int
    n_cols: int
    features: list[str]
    target: str | None
    problem_type: str | None  # classification | regression


class JobContext(TypedDict, total=False):
    """Ngữ cảnh training job."""
    id: str
    dataset_id: str
    status: str  # pending | running | completed | failed
    models: list[str]
    best_model: str | None
    best_score: float | None
    metrics: dict[str, float]


class EvaluationResult(TypedDict, total=False):
    """Kết quả đánh giá và so sánh models."""
    job_ids: list[str]
    comparison_table: list[dict[str, Any]]
    best_job_id: str | None
    recommendation: str | None


# ── Main state ───────────────────────────────────────────


class AutoMLState(TypedDict):
    """
    State cho LangGraph StateGraph — Tham chiếu DeerFlow's ThreadState.

    messages: LangGraph message list with auto-merge reducer.
    Các trường còn lại là AutoML-specific context được cập nhật
    bởi các sub-agent nodes.
    """

    # ── Core LangGraph messages ──────────────────────────
    messages: Annotated[list, add_messages]

    # ── Routing / orchestration ──────────────────────────
    next_agent: NotRequired[str | None]
    """Agent node tiếp theo do coordinator quyết định."""

    current_phase: NotRequired[str | None]
    """Phase hiện tại: analyze | select | train | evaluate | respond."""

    # ── AutoML-specific context ──────────────────────────
    dataset_context: NotRequired[DatasetContext | None]
    """Dataset đang được thao tác."""

    job_context: NotRequired[JobContext | None]
    """Training job đang chạy hoặc vừa hoàn thành."""

    active_jobs: NotRequired[list[JobContext] | None]
    """Nhiều jobs chạy song song (khi so sánh models)."""

    evaluation: NotRequired[EvaluationResult | None]
    """Kết quả đánh giá cuối cùng."""

    # ── World Model snapshot ─────────────────────────────
    world_model: NotRequired[dict | None]
    """Snapshot từ WorldStateStore — tất cả datasets + jobs đã biết."""

    # ── Phase-4 planning / LeWM latent ───────────────────
    user_requirements: NotRequired[dict | None]
    """Structured requirements / GoalSpec."""

    goal: NotRequired[dict | None]
    """Active GoalSpec for latent planning."""

    plans: NotRequired[list | None]
    """Candidate PlanResult dicts from CEM-lite."""

    selected_plan: NotRequired[dict | None]
    """Best plan chosen for execution."""

    plan_verification: NotRequired[dict | None]
    """Hard/soft verification report."""

    revision_count: NotRequired[int | None]
    """Number of plan revisions so far."""

    latent: NotRequired[dict | None]
    """Last LatentState dict z_t."""

    surprise: NotRequired[dict | None]
    """Last SurpriseResult dict."""

    cost_metrics: NotRequired[dict | None]
    """n_llm_calls / plans_generated / elapsed etc."""

    # ── Plan execution loop ──────────────────────────────
    plan_step_index: NotRequired[int | None]
    """Index of next step in selected_plan.steps."""

    plan_status: NotRequired[str | None]
    """ready | executing | need_revise | done | failed | aborted."""

    last_step_error: NotRequired[str | None]
    """Last executor error for reviser."""

    execution_log: NotRequired[list | None]
    """Per-step execution records."""

    execution_events: NotRequired[list | None]
    """Structured events for SSE (plan/step/surprise/revise)."""

    # ── Phase 6 multi-candidate campaign ─────────────────
    campaign: NotRequired[dict | None]
    """Campaign dict (variants, jobs, comparison)."""

    campaign_status: NotRequired[str | None]
    """building | submitting | monitoring | comparing | done | failed."""

    campaign_tick: NotRequired[int | None]
    """Monitor loop counter (caps infinite polling)."""

    # ── Phase 7 hierarchy ────────────────────────────────
    hierarchy: NotRequired[dict | None]
    """GoalHierarchy dict (subgoals + index)."""

    hierarchy_status: NotRequired[str | None]
    """running | done | failed — required for hierarchy_route (LangGraph drops unknown keys)."""

    hierarchy_train_active: NotRequired[bool | None]
    """True while train-leaf campaign is still monitoring (alias of _hierarchy_train_active)."""

    # ── Memory context (DeerFlow memory module) ──────────
    memory_context: NotRequired[str | None]
    """Long-term memory đã inject vào prompt."""

    # ── User context ─────────────────────────────────────
    user_id: NotRequired[str | None]
    user_token: NotRequired[str | None]
