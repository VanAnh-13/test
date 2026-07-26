"""
Benchmark sample-efficiency — campaign điều khiển bởi world model so với
baseline, trên môi trường AutoML mô phỏng.

Dùng ĐÚNG máy móc campaign thật (build_campaign + campaign_step); môi trường
giả lập cắm qua set_tool_invoker nên không cần backend/Docker. Response
surface của mỗi profile là hàm biết trước → đo được regret so với optimum.

Điều kiện:
  wm          — outcome head train online giữa các campaign; CEM proposal +
                ranking bật (qua tham số outcome_model của build_campaign).
  no_wm       — builder round-robin thuần (gate wm_* tắt).
  random      — mỗi job một config uniform-random từ không gian.
  fixed_<algo>— mọi job cùng một config (algo cố định, time lớn nhất).
"""

from __future__ import annotations

import asyncio
import logging
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from hagent.agent.campaign.builder import build_campaign
from hagent.agent.campaign.runner import campaign_step
from hagent.agent.campaign.schema import Campaign, CampaignVariant
from hagent.agent.execution.tool_runner import set_tool_invoker
from hagent.agent.eval.metrics import (
    best_so_far_curve,
    jobs_to_threshold,
    normalized_regret,
)
from hagent.world.predictor.outcome_head_v1 import train_outcome_head

logger = logging.getLogger(__name__)

_ALGOS = ["grid_search", "bayesian_search", "genetic_algorithm"]
_TIME_OPTIONS = [180, 300, 600]
_MAX_TICKS = 200


# ── Dataset profiles (response surface biết trước) ───────


@dataclass
class DatasetProfile:
    name: str
    base: float
    algo_bonus: Dict[str, float]
    time_coef: float
    noise: float
    meta: Dict[str, Any] = field(default_factory=lambda: {"n_rows": 1000, "n_cols": 10})

    @property
    def optimum(self) -> float:
        return self.base + max(self.algo_bonus.values()) + self.time_coef

    def expected_score(self, algo: str, time_limit: float) -> float:
        return (
            self.base
            + self.algo_bonus.get(algo, 0.0)
            + self.time_coef * math.log1p(max(0.0, time_limit)) / math.log1p(max(_TIME_OPTIONS))
        )

    def sample_score(self, algo: str, time_limit: float, rng: np.random.Generator) -> float:
        return float(self.expected_score(algo, time_limit) + rng.normal(0.0, self.noise))


PROFILES: Dict[str, DatasetProfile] = {
    # Tín hiệu mạnh, nhiễu nhỏ — nơi steering phải thắng rõ
    "synth_strong": DatasetProfile(
        name="synth_strong",
        base=0.60,
        algo_bonus={"grid_search": 0.0, "bayesian_search": 0.20, "genetic_algorithm": 0.08},
        time_coef=0.08,
        noise=0.01,
    ),
    # Nhiễu lớn hơn tín hiệu một phần — kiểm tra tính bền
    "synth_noisy": DatasetProfile(
        name="synth_noisy",
        base=0.65,
        algo_bonus={"grid_search": 0.0, "bayesian_search": 0.06, "genetic_algorithm": 0.03},
        time_coef=0.03,
        noise=0.04,
    ),
    # Không có tín hiệu — mọi condition phải xấp xỉ nhau (sanity/null case)
    "synth_flat": DatasetProfile(
        name="synth_flat",
        base=0.75,
        algo_bonus={"grid_search": 0.0, "bayesian_search": 0.0, "genetic_algorithm": 0.0},
        time_coef=0.0,
        noise=0.02,
    ),
}


# ── Simulated environment ────────────────────────────────


class SimulatedAutoMLEnv:
    """Tool invoker giả lập: start_training / get_job_info trên response surface."""

    def __init__(self, profile: DatasetProfile, seed: int = 0):
        self.profile = profile
        self.rng = np.random.default_rng(seed)
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.submission_log: List[Dict[str, Any]] = []

    @property
    def jobs_used(self) -> int:
        return len(self.submission_log)

    async def invoke(self, action_type: str, params: dict) -> dict:
        if action_type == "start_training":
            algo = str(params.get("search_algorithm") or "grid_search")
            t = float(params.get("time_limit") or _TIME_OPTIONS[0])
            job_id = f"sim_{len(self.jobs) + 1}_{uuid.uuid4().hex[:6]}"
            score = self.profile.sample_score(algo, t, self.rng)
            record = {
                "job_id": job_id,
                "search_algorithm": algo,
                "time_limit": t,
                "best_score": score,
            }
            self.jobs[job_id] = record
            self.submission_log.append(record)
            return {"job_id": job_id, "status": 0}

        if action_type == "get_job_info":
            job = self.jobs.get(str(params.get("job_id")))
            if not job:
                return {"error": "job not found"}
            return {
                "id": job["job_id"],
                "status": "completed",
                "best_model": "sim_model",
                "best_score": job["best_score"],
            }

        return {}


# ── Variant builders cho baseline conditions ─────────────


def _campaign_from_params(goal: dict, params_list: List[dict], *, source: str) -> Campaign:
    variants = [
        CampaignVariant(
            variant_id=str(uuid.uuid4()),
            label=f"{source}_{i + 1}",
            params=params,
            source=source,
        )
        for i, params in enumerate(params_list)
    ]
    return Campaign(
        campaign_id=str(uuid.uuid4()),
        goal=dict(goal),
        variants=variants,
        status="building",
        max_concurrent=2,
    )


def _base_params(goal: dict) -> dict:
    return {
        "dataset_id": goal.get("dataset_id"),
        "problem_type": goal.get("problem_type") or "classification",
        "metric": goal.get("metric") or "accuracy",
        "target_column": goal.get("target_column"),
    }


# ── Core: run one condition ──────────────────────────────


async def _run_condition_async(
    condition: str,
    profile: DatasetProfile,
    *,
    budget_jobs: int = 20,
    seed: int = 0,
    campaign_size: int = 3,
    min_train_samples: int = 6,
    head_config: dict | None = None,
    train_epochs: int = 60,
) -> Dict[str, Any]:
    goal = {
        "goal_type": "train",
        "dataset_id": profile.name,
        "problem_type": "classification",
        "metric": "accuracy",
        "target_column": "target",
    }
    env = SimulatedAutoMLEnv(profile, seed=seed)
    set_tool_invoker(env.invoke)
    rng = np.random.default_rng(seed + 1)
    head_cfg = dict(head_config or {"use_latent": False, "hidden_dim": 32})

    samples: List[Dict[str, Any]] = []
    outcome_model = None
    wm_trained_after: Optional[int] = None
    outcome_surprise_events: List[dict] = []

    try:
        while env.jobs_used < budget_jobs:
            n = min(campaign_size, budget_jobs - env.jobs_used)

            if condition == "wm":
                camp = await build_campaign(
                    goal,
                    user_id=f"bench_{profile.name}_{seed}",
                    config={
                        "n_job_candidates": n,
                        "warm_start_top_k": 0,
                        "wm_variant_proposal": True,
                        "wm_rank_variants": True,
                    },
                    outcome_model=outcome_model,
                )
            elif condition == "no_wm":
                camp = await build_campaign(
                    goal,
                    user_id=f"bench_{profile.name}_{seed}",
                    config={
                        "n_job_candidates": n,
                        "warm_start_top_k": 0,
                        "wm_variant_proposal": False,
                        "wm_rank_variants": False,
                    },
                )
            elif condition == "random":
                params_list = [
                    dict(
                        _base_params(goal),
                        search_algorithm=str(rng.choice(_ALGOS)),
                        time_limit=int(rng.choice(_TIME_OPTIONS)),
                    )
                    for _ in range(n)
                ]
                camp = _campaign_from_params(goal, params_list, source="random")
            elif condition.startswith("fixed_"):
                algo = condition.removeprefix("fixed_")
                if algo not in _ALGOS:
                    raise ValueError(f"Unknown fixed condition algorithm: {algo!r}")
                params_list = [
                    dict(
                        _base_params(goal),
                        search_algorithm=algo,
                        time_limit=max(_TIME_OPTIONS),
                    )
                    for _ in range(n)
                ]
                camp = _campaign_from_params(goal, params_list, source=condition)
            else:
                raise ValueError(f"Unknown benchmark condition: {condition!r}")

            events: List[dict] = []
            ticks = 0
            while camp.status not in ("done", "failed") and ticks < _MAX_TICKS:
                camp = await campaign_step(
                    camp,
                    user_id=f"bench_{profile.name}_{seed}",
                    user_token=None,
                    world_model={"datasets": {profile.name: dict(profile.meta)}},
                    surprise_events=events,
                )
                ticks += 1
            outcome_surprise_events.extend(
                e for e in events if e.get("type") == "campaign_outcome_surprise"
            )

            for v in camp.variants:
                if v.status == "completed" and v.best_score is not None:
                    samples.append(
                        {
                            "params": dict(v.params),
                            "dataset_meta": dict(profile.meta),
                            "best_score": float(v.best_score),
                        }
                    )

            if condition == "wm" and len(samples) >= min_train_samples:
                outcome_model = train_outcome_head(
                    samples, config=dict(head_cfg), epochs=train_epochs, seed=seed
                )
                if wm_trained_after is None:
                    wm_trained_after = env.jobs_used
    finally:
        set_tool_invoker(None)

    scores = [r["best_score"] for r in env.submission_log[:budget_jobs]]
    curve = best_so_far_curve(scores)
    final_best = curve[-1] if curve else None
    threshold = profile.base + 0.95 * (profile.optimum - profile.base)
    return {
        "condition": condition,
        "profile": profile.name,
        "seed": seed,
        "budget_jobs": budget_jobs,
        "jobs_used": env.jobs_used,
        "scores": scores,
        "curve": curve,
        "final_best": final_best,
        "optimum": profile.optimum,
        "regret": (profile.optimum - final_best) if final_best is not None else None,
        "normalized_regret": (
            normalized_regret(final_best, profile.optimum, baseline=profile.base)
            if final_best is not None
            else None
        ),
        "jobs_to_95pct": jobs_to_threshold(curve, threshold),
        "wm_trained_after_jobs": wm_trained_after,
        "n_outcome_surprise_events": len(outcome_surprise_events),
        "n_train_samples": len(samples),
    }


def run_condition(condition: str, profile: DatasetProfile | str, **kwargs) -> Dict[str, Any]:
    """Sync wrapper — mỗi lần chạy trên event loop mới (an toàn khi gọi lặp)."""
    prof = PROFILES[profile] if isinstance(profile, str) else profile
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            _run_condition_async(condition, prof, **kwargs)
        )
    finally:
        loop.close()


def run_benchmark_matrix(
    *,
    conditions: List[str],
    profiles: List[str],
    budget_jobs: int = 20,
    seeds: List[int] | None = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Chạy đủ ma trận conditions × profiles × seeds, trả list kết quả."""
    results = []
    for prof_name in profiles:
        for condition in conditions:
            for seed in seeds or [0]:
                logger.info(
                    "benchmark: %s × %s × seed=%d", condition, prof_name, seed
                )
                results.append(
                    run_condition(
                        condition,
                        prof_name,
                        budget_jobs=budget_jobs,
                        seed=seed,
                        **kwargs,
                    )
                )
    return results
