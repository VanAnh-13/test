"""
Lập kế hoạch phân cấp cho HAgent.

- Phân rã mục tiêu cấp cao thành các mục tiêu con có thứ tự theo mẫu cấu hình.
- Bỏ qua thông minh các mục tiêu con đã được World Model đáp ứng.
- Bộ điều khiển hierarchy chỉ chuyển tiếp khi một leaf hoàn tất.

Cách làm gọn nhưng thích ứng: mỗi leaf có chân trời ngắn và dựa trên dữ kiện World Model.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from hagent.world.query import features_of, get_dataset, past_best_jobs


@dataclass
class SubGoal:
    goal_type: str
    description: str
    params: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending | active | done | skipped | failed
    subgoal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    skip_reason: str | None = None
    result_summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubGoal:
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class GoalHierarchy:
    root_goal: dict[str, Any]
    subgoals: list[SubGoal]
    current_index: int = 0
    hierarchy_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "hierarchy_id": self.hierarchy_id,
            "root_goal": self.root_goal,
            "subgoals": [s.to_dict() for s in self.subgoals],
            "current_index": self.current_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoalHierarchy:
        subs = [
            SubGoal.from_dict(s) if isinstance(s, dict) else s
            for s in (data.get("subgoals") or [])
        ]
        return cls(
            root_goal=dict(data.get("root_goal") or {}),
            subgoals=subs,
            current_index=int(data.get("current_index") or 0),
            hierarchy_id=str(data.get("hierarchy_id") or uuid.uuid4()),
        )

    def current(self) -> SubGoal | None:
        if 0 <= self.current_index < len(self.subgoals):
            return self.subgoals[self.current_index]
        return None

    def advance(
        self, *, status: str = "done", summary: str | None = None
    ) -> SubGoal | None:
        if self.current_index < len(self.subgoals):
            cur = self.subgoals[self.current_index]
            if cur.status not in ("skipped", "failed"):
                cur.status = status
            if summary:
                cur.result_summary = summary
            self.current_index += 1
            # Activate next pending
            nxt = self.current()
            if nxt and nxt.status == "pending":
                nxt.status = "active"
        return self.current()

    def is_complete(self) -> bool:
        return self.current_index >= len(self.subgoals)

    def progress(self) -> dict[str, int]:
        counts = {"done": 0, "skipped": 0, "failed": 0, "pending": 0, "active": 0}
        for s in self.subgoals:
            counts[s.status] = counts.get(s.status, 0) + 1
        counts["total"] = len(self.subgoals)
        counts["index"] = self.current_index
        return counts


def _hierarchy_config() -> dict:
    try:
        from hagent.bridge.config import get_hierarchy_config

        return get_hierarchy_config()
    except Exception:  # noqa: BLE001
        return {"enabled": True, "smart_skip": True}


def _hierarchy_templates() -> dict[str, list[dict[str, Any]]]:
    cfg = _hierarchy_config()
    tpl = cfg.get("templates") or {}
    if isinstance(tpl, dict) and tpl:
        return tpl
    return {
        "train": [
            {"goal_type": "analyze", "description": "Inspect dataset features"},
            {"goal_type": "select", "description": "Select models/metrics"},
            {"goal_type": "train", "description": "Run training campaign/jobs"},
            {"goal_type": "evaluate", "description": "Compare results"},
        ],
        "evaluate": [
            {"goal_type": "monitor", "description": "List/check jobs"},
            {"goal_type": "evaluate", "description": "Compare best models"},
        ],
    }


def decompose_goal(goal: dict[str, Any]) -> GoalHierarchy:
    """Mở rộng mục tiêu gốc thành các mục tiêu con; loại đơn giản chỉ có một node."""
    gtype = str(goal.get("goal_type") or "respond").lower()
    templates = _hierarchy_templates()
    steps = templates.get(gtype)

    if not steps or gtype in ("list", "respond", "analyze", "monitor", "select"):
        return GoalHierarchy(
            root_goal=dict(goal),
            subgoals=[
                SubGoal(
                    goal_type=gtype,
                    description=str(goal.get("description") or gtype),
                    params={
                        k: v
                        for k, v in goal.items()
                        if k not in ("goal_type", "description")
                    },
                    status="active",
                )
            ],
            current_index=0,
        )

    subgoals: list[SubGoal] = []
    for i, step in enumerate(steps):
        params = {
            k: v
            for k, v in goal.items()
            if k not in ("goal_type", "description", "goal_id")
        }
        stype = str(step.get("goal_type"))
        if stype != "train":
            params = {
                k: params[k]
                for k in ("dataset_id", "problem_type", "target_column", "metric")
                if k in params
            }
        subgoals.append(
            SubGoal(
                goal_type=stype,
                description=str(step.get("description") or stype),
                params=params,
                status="active" if i == 0 else "pending",
            )
        )

    return GoalHierarchy(root_goal=dict(goal), subgoals=subgoals, current_index=0)


def subgoal_as_goal(hierarchy: GoalHierarchy) -> dict[str, Any]:
    """Hiện thực hóa mục tiêu con hiện tại thành dict có dạng GoalSpec."""
    cur = hierarchy.current()
    if not cur:
        return dict(hierarchy.root_goal)
    g = dict(hierarchy.root_goal)
    g["goal_type"] = cur.goal_type
    g["description"] = cur.description
    g.update(cur.params or {})
    g["hierarchy_id"] = hierarchy.hierarchy_id
    g["subgoal_id"] = cur.subgoal_id
    g["subgoal_index"] = hierarchy.current_index
    return g


# ── Smart skip (World Model grounded) ────────────────────


def _dataset_id(goal: dict, world_model: dict | None) -> str | None:
    wm = world_model or {}
    return (
        goal.get("dataset_id")
        or wm.get("active_dataset_id")
        or (next(iter((wm.get("datasets") or {}).keys()), None))
    )


def should_skip_subgoal(
    subgoal: SubGoal,
    *,
    root_goal: dict,
    world_model: dict | None,
    evaluation: dict | None = None,
    campaign_status: str | None = None,
) -> tuple[bool, str]:
    """
    Trả về cặp (có bỏ qua hay không, lý do) theo World Model và campaign trước.

    Cấu hình: hierarchy.smart_skip, mặc định True.
    """
    cfg = _hierarchy_config()
    if not cfg.get("smart_skip", True):
        return False, ""

    gtype = str(subgoal.goal_type or "").lower()
    wm = world_model or {}
    goal = {**root_goal, **(subgoal.params or {})}
    ds_id = _dataset_id(goal, wm)

    if gtype == "analyze":
        if ds_id:
            feats = features_of(wm, ds_id)
            ds = get_dataset(wm, ds_id) or {}
            if feats and (ds.get("n_rows") or ds.get("n_cols") or len(feats) > 0):
                # Coi mục tiêu là đã biết nếu goal hoặc tập dữ liệu có giá trị này.
                target = goal.get("target_column") or ds.get("target")
                if target and target in feats:
                    return True, f"features+target known for {ds_id}"
                if feats and not goal.get("target_column"):
                    return True, f"features already in WM for {ds_id}"
        return False, ""

    if gtype == "select":
        # Bỏ qua nếu đã chọn problem_type và metric, đồng thời đã có model tốt nhất trước đó.
        if goal.get("problem_type") and (
            goal.get("metric") or goal.get("constraints", {}).get("models")
        ):
            past = past_best_jobs(wm, problem_type=goal.get("problem_type"), top_k=1)
            if past or goal.get("metric"):
                return True, "problem_type/metric (and optional warm models) ready"
        return False, ""

    if gtype == "train":
        # Không tự động bỏ qua huấn luyện trừ khi campaign đã hoàn tất trong phiên này.
        if campaign_status == "done" and evaluation and evaluation.get("best_job_id"):
            return True, "campaign already completed in this session"
        return False, ""

    if gtype == "evaluate":
        if evaluation and (
            evaluation.get("comparison_table") or evaluation.get("best_job_id")
        ):
            return True, "evaluation already available from campaign"
        past = past_best_jobs(wm, problem_type=goal.get("problem_type"), top_k=1)
        jobs = wm.get("jobs") or {}
        completed = [
            j
            for j in jobs.values()
            if str(j.get("status", "")).lower() in ("completed", "done", "success")
        ]
        if not completed and not past:
            # Nếu không có gì để đánh giá thì bỏ qua thay vì báo lỗi.
            return True, "no completed jobs to evaluate"
        return False, ""

    if gtype == "monitor":
        jobs = wm.get("jobs") or {}
        if jobs:
            return True, f"{len(jobs)} jobs already in WM"
        return False, ""

    return False, ""


def apply_smart_skips(
    hierarchy: GoalHierarchy,
    *,
    world_model: dict | None,
    evaluation: dict | None = None,
    campaign_status: str | None = None,
) -> list[dict[str, Any]]:
    """
    Từ current_index trở đi, đánh dấu các mục tiêu con có thể bỏ qua rồi chuyển tiếp.

    Trả về danh sách sự kiện bỏ qua cho log và SSE.
    """
    events: list[dict[str, Any]] = []
    # Duyệt từ vị trí hiện tại đến leaf hoạt động không thể bỏ qua hoặc đến cuối.
    guard = 0
    while hierarchy.current() and guard < len(hierarchy.subgoals) + 2:
        guard += 1
        cur = hierarchy.current()
        if not cur:
            break
        if cur.status in ("done", "skipped", "failed"):
            hierarchy.current_index += 1
            continue
        skip, reason = should_skip_subgoal(
            cur,
            root_goal=hierarchy.root_goal,
            world_model=world_model,
            evaluation=evaluation,
            campaign_status=campaign_status,
        )
        if skip:
            cur.status = "skipped"
            cur.skip_reason = reason
            cur.result_summary = f"skipped: {reason}"
            events.append(
                {
                    "type": "subgoal_skipped",
                    "goal_type": cur.goal_type,
                    "subgoal_id": cur.subgoal_id,
                    "reason": reason,
                    "index": hierarchy.current_index,
                }
            )
            hierarchy.current_index += 1
            nxt = hierarchy.current()
            if nxt and nxt.status == "pending":
                nxt.status = "active"
            continue
        # Ensure active
        if cur.status == "pending":
            cur.status = "active"
        break
    return events


def ensure_hierarchy(
    state: dict,
    *,
    goal: dict | None = None,
) -> GoalHierarchy:
    """Tải hierarchy từ state hoặc tạo từ goal rồi áp dụng bỏ qua thông minh một lần."""
    raw = state.get("hierarchy")
    if isinstance(raw, dict) and raw.get("subgoals"):
        return GoalHierarchy.from_dict(raw)
    g = goal or state.get("goal") or state.get("user_requirements") or {}
    hier = decompose_goal(dict(g))
    apply_smart_skips(
        hier,
        world_model=state.get("world_model"),
        evaluation=state.get("evaluation"),
        campaign_status=state.get("campaign_status"),
    )
    return hier
