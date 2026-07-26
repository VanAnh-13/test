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
                "random_search",
                "successive_halving",
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


_LOWER_IS_BETTER_METRICS = {"mae", "mse", "rmse", "rmsle", "loss"}


def _resolve_outcome_model(outcome_model: Any) -> Any | None:
    """
    Sentinel "auto" → model mặc định từ config (checkpoint trên đĩa).
    None → TẮT hẳn (không fallback — benchmark cần để tránh nhiễm checkpoint rác).
    Object → dùng nếu is_ready.
    """
    if outcome_model is None:
        return None
    if isinstance(outcome_model, str):
        if outcome_model != "auto":
            return None
        try:
            from hagent.agent.campaign.wm_hooks import _default_outcome_model

            return _default_outcome_model()
        except Exception:
            return None
    return outcome_model if getattr(outcome_model, "is_ready", False) else None


def _campaign_planner(cfg: dict) -> Any | None:
    try:
        from hagent.bridge.config import get_world_model_config
        from hagent.world.planner.factory import create_campaign_planner

        planner_cfg = dict(
            (get_world_model_config() or {}).get("campaign_planner") or {}
        )
        # Không gian tìm kiếm của planner đồng bộ với campaign config;
        # campaign config có key (kể cả []) thì THẮNG yaml — caller (benchmark)
        # phải kiểm soát được không gian theo từng dataset/điều kiện
        for key in ("search_algorithms", "time_limit_options", "model_options"):
            if cfg.get(key) is not None:
                planner_cfg[key] = cfg.get(key)
        return create_campaign_planner(planner_cfg)
    except Exception:
        return None


def _variant_signature(params: Dict[str, Any]) -> tuple:
    return (
        params.get("search_algorithm"),
        tuple(params.get("models") or []),
        params.get("time_limit"),
    )


def propose_extension_variants(
    campaign: Any,
    goal: dict,
    *,
    user_id: str | None = None,
    dataset_meta: dict | None = None,
    outcome_model: Any = "auto",
    n_extra: int = 2,
    exploration_weight: float = 0.5,
    config: dict | None = None,
) -> List[CampaignVariant]:
    """
    Đề xuất variant MỞ RỘNG sau outcome surprise cao: model vừa bị bất ngờ
    → dự đoán quanh vùng đó không đáng tin → tăng mạnh exploration_weight
    (ưu tiên σ cao) thay vì khai thác tiếp μ.

    Fallback khi model/planner vắng: thử các thuật toán CHƯA có trong
    campaign (khám phá thô nhưng vẫn hợp lý).
    """
    cfg = dict(_campaign_config())
    if config:
        cfg.update(config)

    # Tôn trọng ngân sách campaign: cạn budget thì KHÔNG mở rộng
    total = campaign.total_budget or (len(campaign.variants) * 2)
    remaining = max(0, total - max(campaign.spent_budget, len(campaign.variants)))
    if remaining <= 0:
        return []
    n_extra = max(1, min(int(n_extra), remaining))

    base = _base_train_params(goal, user_id)
    metric = str(base.get("metric") or "").lower()
    higher = metric not in _LOWER_IS_BETTER_METRICS

    existing = {_variant_signature(v.params) for v in campaign.variants}
    proposals: List[Dict[str, Any]] = []

    model = _resolve_outcome_model(outcome_model)
    planner = _campaign_planner(cfg) if model is not None else None
    if planner is not None:
        # Boost exploration trên planner (và pool bên trong CemMpc nếu có)
        for target in (planner, getattr(planner, "_pool_planner", None)):
            if target is not None and hasattr(target, "exploration_weight"):
                target.exploration_weight = float(exploration_weight)
        if hasattr(planner, "plan_batch"):
            # cem_mpc_v1: Thompson anneal theo budget còn lại — đây là chỗ
            # MPC thật sự có đất trong prod (vòng thứ hai của campaign)
            proposals = planner.plan_batch(
                base_params=base,
                dataset_meta=dataset_meta,
                outcome_model=model,
                n=n_extra,
                remaining_budget=remaining,
                total_budget=total,
                higher_is_better=higher,
            )
        else:
            proposals = planner.plan_campaign_configs(
                base_params=base,
                dataset_meta=dataset_meta,
                outcome_model=model,
                n_return=n_extra * 3,  # dư để sống sót dedup
                higher_is_better=higher,
            )

    if not proposals:
        # Fallback: các thuật toán chưa thử, time budget lớn nhất
        tried = {v.params.get("search_algorithm") for v in campaign.variants}
        algorithms = [a for a in (cfg.get("search_algorithms") or []) if a not in tried]
        time_opts = list(cfg.get("time_limit_options") or [180, 300, 600])
        proposals = [
            {"search_algorithm": algo, "time_limit": max(time_opts)}
            for algo in algorithms
        ]

    out: List[CampaignVariant] = []
    for prop in proposals:
        if len(out) >= n_extra:
            break
        params = dict(base)
        params.update({k: v for k, v in prop.items() if v is not None})
        sig = _variant_signature(params)
        if sig in existing:
            continue
        existing.add(sig)
        out.append(
            CampaignVariant(
                variant_id=str(uuid.uuid4()),
                label=f"ext{campaign.extension_rounds + 1}_{len(out) + 1}",
                params=params,
                source="surprise_extension",
            )
        )
    return out


async def build_campaign(
    goal: dict,
    *,
    user_id: str | None = None,
    world_model: dict | None = None,
    fact_store: Any | None = None,
    config: dict | None = None,
    outcome_model: Any = "auto",
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
        or [
            "grid_search",
            "bayesian_search",
            "genetic_algorithm",
            "random_search",
            "successive_halving",
        ]
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

    # 0) User yêu cầu thuật toán đích danh (goal_parser → constraints) →
    #    GHIM variant đầu tiên đúng yêu cầu. Không ghim thì vòng diversify
    #    bên dưới ghi đè search_algorithm và yêu cầu của user vô tác dụng.
    requested_algo = base.get("search_algorithm")
    if requested_algo:
        variants.append(
            CampaignVariant(
                variant_id=str(uuid.uuid4()),
                label="requested_1",
                params=dict(base),
                source="requested",
            )
        )

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

    # Meta của dataset đích — cùng phân phối feature với lúc train outcome head
    dataset_meta: dict | None = None
    ds_id = base.get("dataset_id")
    if ds_id and isinstance(world_model, dict):
        ds_entry = (world_model.get("datasets") or {}).get(ds_id)
        if isinstance(ds_entry, dict):
            dataset_meta = dict(ds_entry)

    # 2a) World-model proposals cho các slot còn lại (CEM trên config space).
    #     Model chưa sẵn sàng → bỏ qua, rơi về round-robin như cũ.
    model = None
    if cfg.get("wm_variant_proposal", True) and len(variants) < n:
        model = _resolve_outcome_model(outcome_model)
        planner = _campaign_planner(cfg) if model is not None else None
        if planner is not None:
            metric = str(base.get("metric") or "").lower()
            proposals = planner.plan_campaign_configs(
                base_params=base,
                dataset_meta=dataset_meta,
                outcome_model=model,
                n_return=n - len(variants),
                higher_is_better=metric not in _LOWER_IS_BETTER_METRICS,
            )
            existing_sigs = {
                (
                    v.params.get("search_algorithm"),
                    tuple(v.params.get("models") or []),
                    v.params.get("time_limit"),
                )
                for v in variants
            }
            for prop in proposals:
                if len(variants) >= n:
                    break
                params = dict(base)
                params.update({k: v for k, v in prop.items() if v is not None})
                sig = (
                    params.get("search_algorithm"),
                    tuple(params.get("models") or []),
                    params.get("time_limit"),
                )
                if sig in existing_sigs:
                    continue
                existing_sigs.add(sig)
                variants.append(
                    CampaignVariant(
                        variant_id=str(uuid.uuid4()),
                        label=f"wm_cem_{len(variants) + 1}",
                        params=params,
                        source="wm_planner",
                    )
                )

    # 2b) Diversify remaining slots by search algorithm / time budget
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

    # 3) Xếp thứ tự submit theo mean dự đoán — variant hứa hẹn nhất chạy trước
    #    (quan trọng khi max_concurrent < n). Model chưa sẵn sàng → giữ nguyên.
    if cfg.get("wm_rank_variants", True):
        if model is None:
            model = _resolve_outcome_model(outcome_model)
        if model is not None:
            try:
                from hagent.world.predictor.outcome_head_v1 import (
                    rank_variants_by_outcome,
                )

                metric = str(base.get("metric") or "").lower()
                ranked = rank_variants_by_outcome(
                    variants[:n],
                    head=model,
                    dataset_meta=dataset_meta,
                    higher_is_better=metric not in _LOWER_IS_BETTER_METRICS,
                )
                variants = [v for v, _ in ranked]
            except Exception:
                pass

    constraints = goal.get("constraints") if isinstance(goal.get("constraints"), dict) else {}
    try:
        total_budget = int(constraints.get("max_jobs") or 0) or n * 2
    except (TypeError, ValueError):
        total_budget = n * 2

    return Campaign(
        campaign_id=str(uuid.uuid4()),
        goal=dict(goal),
        variants=variants[:n],
        status="building",
        warm_start_used=warm_labels,
        max_concurrent=max_conc,
        total_budget=total_budget,
    )
