"""Compare campaign job results and pick best variant."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from hagent.agent.campaign.schema import Campaign, CampaignVariant


def _variant_score(v: CampaignVariant, metric: str | None = None) -> float:
    if v.best_score is not None:
        try:
            return float(v.best_score)
        except (TypeError, ValueError):
            pass
    if metric and v.metrics and metric in v.metrics:
        try:
            return float(v.metrics[metric])
        except (TypeError, ValueError):
            pass
    if v.metrics:
        try:
            return float(max(v.metrics.values()))
        except (TypeError, ValueError):
            pass
    return float("-inf")


def compare_campaign(
    campaign: Campaign,
    *,
    metric: str | None = None,
    higher_is_better: bool = True,
) -> Tuple[Optional[CampaignVariant], List[Dict[str, Any]]]:
    """
    Rank completed variants. Returns (best, comparison_table).
    Failed variants rank last.
    """
    metric = metric or (campaign.goal or {}).get("metric")
    rows: List[Dict[str, Any]] = []
    completed: List[CampaignVariant] = []

    for v in campaign.variants:
        score = _variant_score(v, metric)
        rows.append(
            {
                "variant_id": v.variant_id,
                "label": v.label,
                "source": v.source,
                "job_id": v.job_id,
                "status": v.status,
                "best_model": v.best_model,
                "best_score": v.best_score if v.best_score is not None else (
                    score if score != float("-inf") else None
                ),
                "search_algorithm": v.params.get("search_algorithm"),
                "time_limit": v.params.get("time_limit"),
                "error": v.error,
            }
        )
        if v.status == "completed":
            completed.append(v)

    if not completed:
        # Prefer any non-failed with score
        candidates = [v for v in campaign.variants if v.status != "failed"]
        if not candidates:
            return None, rows
        completed = candidates

    reverse = higher_is_better
    # Lower-is-better metrics common in regression
    if metric and str(metric).lower() in ("mae", "mse", "rmse", "rmsle", "loss"):
        reverse = False

    completed_sorted = sorted(
        completed,
        key=lambda v: _variant_score(v, metric),
        reverse=reverse,
    )
    best = completed_sorted[0] if completed_sorted else None

    # Sort comparison table similarly
    def row_score(r: dict) -> float:
        s = r.get("best_score")
        if s is None:
            return float("-inf") if reverse else float("inf")
        try:
            return float(s)
        except (TypeError, ValueError):
            return float("-inf") if reverse else float("inf")

    rows.sort(key=row_score, reverse=reverse)
    return best, rows


def best_config_payload(best: CampaignVariant, goal: dict) -> Dict[str, Any]:
    """Serializable config for memory warm-start write-back."""
    return {
        "problem_type": best.params.get("problem_type") or goal.get("problem_type"),
        "metric": best.params.get("metric") or goal.get("metric"),
        "search_algorithm": best.params.get("search_algorithm"),
        "models": best.params.get("models"),
        "time_limit": best.params.get("time_limit"),
        "best_model": best.best_model,
        "best_score": best.best_score,
        "dataset_id": best.params.get("dataset_id") or goal.get("dataset_id"),
        "source_variant": best.label,
        "job_id": best.job_id,
    }
