"""
Build N training campaign variants from goal + warm-start + diversification.

Diversification axes come from config (search algorithms, time budgets) — not hard-coded
call-site magic beyond YAML defaults.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from hagent.agent.campaign.schema import Campaign, CampaignVariant
from hagent.agent.campaign.warm_start import collect_warm_start_configs


def _campaign_config() -> dict:
    try:
        from hagent.bridge.config import get_campaign_config

        return get_campaign_config()
    except Exception:
        return {
            "enabled": True,
            "n_job_candidates": 3,
            "max_concurrent_jobs": 2,
            "warm_start_top_k": 3,
            "search_algorithms": [
                "grid_search",
                "bayesian_search",
                "genetic_algorithm",
            ],
            "time_limit_options": [180, 300, 600],
            "poll_only_status": True,
        }


def _base_train_params(goal: dict, user_id: str | None) -> Dict[str, Any]:
    constraints = goal.get("constraints") if isinstance(goal.get("constraints"), dict) else {}
    params: Dict[str, Any] = {
        "dataset_id": goal.get("dataset_id"),
        "problem_type": goal.get("problem_type") or "classification",
        "target_column": goal.get("target_column"),
    }
    if user_id:
        params["user_id"] = user_id
    if goal.get("metric"):
        params["metric"] = goal["metric"]
    if constraints.get("time_limit") is not None:
        params["time_limit"] = constraints["time_limit"]
    if constraints.get("models"):
        params["models"] = constraints["models"]
    if constraints.get("search_algorithm"):
        params["search_algorithm"] = constraints["search_algorithm"]
    return {k: v for k, v in params.items() if v is not None}


async def build_campaign(
    goal: dict,
    *,
    user_id: str | None = None,
    world_model: dict | None = None,
    fact_store: Any | None = None,
    config: dict | None = None,
) -> Campaign:
    """Create campaign with diversified + warm-started training variants."""
    cfg = dict(_campaign_config())
    if config:
        cfg.update(config)

    n = max(1, int(cfg.get("n_job_candidates", 3)))
    max_conc = max(1, int(cfg.get("max_concurrent_jobs", 2)))
    top_k = int(cfg.get("warm_start_top_k", 3))
    algorithms: List[str] = list(
        cfg.get("search_algorithms")
        or ["grid_search", "bayesian_search", "genetic_algorithm"]
    )
    time_opts: List[int] = list(cfg.get("time_limit_options") or [180, 300, 600])

    base = _base_train_params(goal, user_id)
    problem_type = base.get("problem_type")

    warm = await collect_warm_start_configs(
        world_model=world_model,
        user_id=user_id,
        problem_type=str(problem_type) if problem_type else None,
        top_k=top_k,
        fact_store=fact_store,
    )
    warm_labels = [
        str(w.get("_source_job_id") or w.get("_memory_key") or "warm")
        for w in warm
    ]

    variants: List[CampaignVariant] = []

    # 1) Warm-start variants
    for i, w in enumerate(warm):
        if len(variants) >= n:
            break
        params = dict(base)
        for key in (
            "search_algorithm",
            "search_strategy",
            "models",
            "metric",
            "time_limit",
            "problem_type",
        ):
            if w.get(key) is not None:
                # normalize search_strategy → search_algorithm
                if key == "search_strategy":
                    params["search_algorithm"] = w[key]
                else:
                    params[key] = w[key]
        variants.append(
            CampaignVariant(
                variant_id=str(uuid.uuid4()),
                label=f"warm_start_{i + 1}",
                params=params,
                source="warm_start",
            )
        )

    # 2) Diversify remaining slots by search algorithm / time budget
    algo_idx = 0
    time_idx = 0
    while len(variants) < n:
        params = dict(base)
        if algorithms:
            params["search_algorithm"] = algorithms[algo_idx % len(algorithms)]
            algo_idx += 1
        if "time_limit" not in params and time_opts:
            params["time_limit"] = time_opts[time_idx % len(time_opts)]
            time_idx += 1
        # Avoid exact duplicate of previous
        sig = (
            params.get("search_algorithm"),
            tuple(params.get("models") or []),
            params.get("time_limit"),
        )
        if any(
            (
                v.params.get("search_algorithm"),
                tuple(v.params.get("models") or []),
                v.params.get("time_limit"),
            )
            == sig
            for v in variants
        ):
            # bump time
            if time_opts:
                params["time_limit"] = time_opts[time_idx % len(time_opts)]
                time_idx += 1
            else:
                break
        source = "default" if not variants else "diversified"
        variants.append(
            CampaignVariant(
                variant_id=str(uuid.uuid4()),
                label=f"{source}_{len(variants) + 1}",
                params=params,
                source=source,
            )
        )
        if algo_idx > len(algorithms) * 3:
            break

    if not variants:
        variants.append(
            CampaignVariant(
                variant_id=str(uuid.uuid4()),
                label="default_1",
                params=base,
                source="default",
            )
        )

    return Campaign(
        campaign_id=str(uuid.uuid4()),
        goal=dict(goal),
        variants=variants[:n],
        status="building",
        warm_start_used=warm_labels,
        max_concurrent=max_conc,
    )
