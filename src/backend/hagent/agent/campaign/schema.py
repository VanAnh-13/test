"""Schema campaign huấn luyện nhiều ứng viên của giai đoạn 6."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CampaignVariant:
    """Một ứng viên cấu hình huấn luyện."""

    variant_id: str
    label: str
    params: dict[str, Any]  # kwargs truyền vào start_training
    source: str = "default"  # giá trị: default | warm_start | diversified
    job_id: str | None = None
    status: str = (
        "pending"  # giá trị: pending | submitted | running | completed | failed
    )
    metrics: dict[str, float] = field(default_factory=dict)
    best_model: str | None = None
    best_score: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_job_entry(self) -> dict[str, Any]:
        """Trả về bản ghi World Model đầy đủ của job huấn luyện này."""
        return {
            "id": self.job_id,
            "status": self.status,
            "best_model": self.best_model,
            "best_score": self.best_score,
            "metrics": self.metrics,
            "config": self.params,
            "dataset_id": self.params.get("dataset_id"),
        }

    def to_submission_job_entry(self) -> dict[str, Any]:
        """Trả về bản ghi rút gọn được phát ngay sau khi gửi job."""
        full_entry = self.to_job_entry()
        return {
            key: full_entry[key] for key in ("id", "status", "config", "dataset_id")
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignVariant:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Campaign:
    """Campaign nhiều job cho một mục tiêu của người dùng."""

    campaign_id: str
    goal: dict[str, Any]
    variants: list[CampaignVariant]
    status: str = "building"  # giá trị: building | submitting | monitoring | comparing | done | failed
    best_variant_id: str | None = None
    comparison: list[dict[str, Any]] = field(default_factory=list)
    warm_start_used: list[str] = field(default_factory=list)
    max_concurrent: int = 2
    # Số vòng mở rộng theo outcome surprise đã dùng — PHẢI là field schema:
    # Bảo toàn trường này khi campaign đi qua to_dict/from_dict giữa các tick của graph.
    extension_rounds: int = 0
    # Ngân sách job của campaign (goal constraints max_jobs, fallback
    # n_job_candidates×2) — vòng mở rộng không được vượt; spent tăng mỗi
    # lần submit thành công
    total_budget: int | None = None
    spent_budget: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "goal": self.goal,
            "variants": [v.to_dict() for v in self.variants],
            "status": self.status,
            "best_variant_id": self.best_variant_id,
            "comparison": list(self.comparison),
            "warm_start_used": list(self.warm_start_used),
            "max_concurrent": self.max_concurrent,
            "extension_rounds": self.extension_rounds,
            "total_budget": self.total_budget,
            "spent_budget": self.spent_budget,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Campaign:
        variants = [
            CampaignVariant.from_dict(v) if isinstance(v, dict) else v
            for v in (data.get("variants") or [])
        ]
        return cls(
            campaign_id=str(data.get("campaign_id") or uuid.uuid4()),
            goal=dict(data.get("goal") or {}),
            variants=variants,
            status=str(data.get("status") or "building"),
            best_variant_id=data.get("best_variant_id"),
            comparison=list(data.get("comparison") or []),
            warm_start_used=list(data.get("warm_start_used") or []),
            max_concurrent=int(data.get("max_concurrent") or 2),
            extension_rounds=int(data.get("extension_rounds") or 0),
            total_budget=(
                int(data["total_budget"])
                if data.get("total_budget") is not None
                else None
            ),
            spent_budget=int(data.get("spent_budget") or 0),
        )

    def pending_submit(self) -> list[CampaignVariant]:
        return [v for v in self.variants if v.status == "pending"]

    def in_flight(self) -> list[CampaignVariant]:
        return [
            v
            for v in self.variants
            if v.status in ("submitted", "running") and v.job_id
        ]

    def unfinished(self) -> list[CampaignVariant]:
        return [v for v in self.variants if v.status not in ("completed", "failed")]
