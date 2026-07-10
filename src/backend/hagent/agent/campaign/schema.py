"""Phase 6 — multi-candidate training campaign schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class CampaignVariant:
    """One training configuration candidate."""

    variant_id: str
    label: str
    params: Dict[str, Any]  # start_training kwargs
    source: str = "default"  # default | warm_start | diversified
    job_id: Optional[str] = None
    status: str = "pending"  # pending | submitted | running | completed | failed
    metrics: Dict[str, float] = field(default_factory=dict)
    best_model: Optional[str] = None
    best_score: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CampaignVariant":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Campaign:
    """Multi-job campaign for one user goal."""

    campaign_id: str
    goal: Dict[str, Any]
    variants: List[CampaignVariant]
    status: str = "building"  # building | submitting | monitoring | comparing | done | failed
    best_variant_id: Optional[str] = None
    comparison: List[Dict[str, Any]] = field(default_factory=list)
    warm_start_used: List[str] = field(default_factory=list)
    max_concurrent: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "goal": self.goal,
            "variants": [v.to_dict() for v in self.variants],
            "status": self.status,
            "best_variant_id": self.best_variant_id,
            "comparison": list(self.comparison),
            "warm_start_used": list(self.warm_start_used),
            "max_concurrent": self.max_concurrent,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Campaign":
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
        )

    def pending_submit(self) -> List[CampaignVariant]:
        return [v for v in self.variants if v.status == "pending"]

    def in_flight(self) -> List[CampaignVariant]:
        return [
            v
            for v in self.variants
            if v.status in ("submitted", "running") and v.job_id
        ]

    def unfinished(self) -> List[CampaignVariant]:
        return [
            v
            for v in self.variants
            if v.status not in ("completed", "failed")
        ]
