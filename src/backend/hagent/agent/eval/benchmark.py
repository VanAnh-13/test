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
    # Hiệu ứng model-subset: điểm cộng của model TỐT NHẤT trong subset đã chọn,
    # trừ dilution cho mỗi model thừa (chọn đúng model > quét cả catalog).
    # Rỗng = chiều models không tác động (tương thích profile cũ).
    model_effects: Dict[str, float] = field(default_factory=dict)
    model_dilution: float = 0.01

    def _model_term(self, models: Sequence[str] | None) -> float:
        if not self.model_effects:
            return 0.0
        chosen = list(models) if models else list(self.model_effects)
        effects = [self.model_effects.get(str(m), 0.0) for m in chosen]
        best = max(effects) if effects else 0.0
        return best - self.model_dilution * max(0, len(chosen) - 1)

    @property
    def optimum(self) -> float:
        best_model = max(self.model_effects.values()) if self.model_effects else 0.0
        return self.base + max(self.algo_bonus.values()) + self.time_coef + best_model

    def expected_score(
        self,
        algo: str,
        time_limit: float,
        models: Sequence[str] | None = None,
    ) -> float:
        return (
            self.base
            + self.algo_bonus.get(algo, 0.0)
            + self.time_coef * math.log1p(max(0.0, time_limit)) / math.log1p(max(_TIME_OPTIONS))
            + self._model_term(models)
        )

    def sample_score(
        self,
        algo: str,
        time_limit: float,
        rng: np.random.Generator,
        models: Sequence[str] | None = None,
    ) -> float:
        return float(
            self.expected_score(algo, time_limit, models) + rng.normal(0.0, self.noise)
        )


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
    # Model subset quyết định phần lớn tín hiệu — không gian hành động lớn
    # (3 algo × 3 time × 2^4 subset), steering phải thắng rõ round-robin
    "synth_models": DatasetProfile(
        name="synth_models",
        base=0.55,
        algo_bonus={"grid_search": 0.0, "bayesian_search": 0.05, "genetic_algorithm": 0.02},
        time_coef=0.04,
        noise=0.01,
        model_effects={
            "DecisionTreeClassifier": 0.00,
            "RandomForestClassifier": 0.15,
            "KNeighborsClassifier": 0.04,
            "SVC": 0.08,
        },
        model_dilution=0.02,
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
            models = params.get("models") or None
            job_id = f"sim_{len(self.jobs) + 1}_{uuid.uuid4().hex[:6]}"
            score = self.profile.sample_score(algo, t, self.rng, models)
            record = {
                "job_id": job_id,
                "search_algorithm": algo,
                "time_limit": t,
                "models": list(models) if models else None,
                "best_score": score,
                # Kỳ vọng không nhiễu của config này — metric steering-quality
                # phải tính trên giá trị này, không phải max của các draw nhiễu
                "expected_score": self.profile.expected_score(algo, t, models),
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


def make_transfer_profiles(k: int = 6, seed: int = 0) -> List[DatasetProfile]:
    """
    Họ profile cho thí nghiệm transfer: thuật toán tốt nhất PHỤ THUỘC meta
    theo luật cố định mà outcome head nhìn thấy được qua feature v2:
      - dataset lớn (log-rows chuẩn hóa > 0.55) → bayesian_search thắng;
      - dataset nhỏ → grid_search thắng;
      - frac_categorical > 0.5 → genetic_algorithm được cộng thêm.
    """
    rng = np.random.default_rng(seed)
    profiles: List[DatasetProfile] = []
    row_options = [200, 500, 1000, 5000, 20000, 100000]
    for i in range(k):
        n_rows = int(row_options[int(rng.integers(0, len(row_options)))])
        frac_cat = float(rng.uniform(0.0, 1.0))
        big = math.log1p(n_rows) / math.log(1e6) > 0.55
        bonus = {
            "grid_search": 0.0 if big else 0.12,
            "bayesian_search": 0.12 if big else 0.0,
            "genetic_algorithm": 0.06 if frac_cat > 0.5 else 0.0,
        }
        meta = {
            "n_rows": n_rows,
            "n_cols": int(rng.integers(5, 50)),
            "n_classes": int(rng.integers(2, 10)),
            "class_imbalance": float(rng.uniform(0.3, 0.9)),
            "frac_categorical": frac_cat,
            "missing_frac": float(rng.uniform(0.0, 0.2)),
            "mean_abs_skew": float(rng.uniform(0.0, 2.0)),
        }
        profiles.append(
            DatasetProfile(
                name=f"transfer_{i}",
                base=0.60,
                algo_bonus=bonus,
                time_coef=0.04,
                noise=0.01,
                meta=meta,
            )
        )
    return profiles


def generate_offline_samples(
    profile: DatasetProfile, m: int = 60, seed: int = 0
) -> List[Dict[str, Any]]:
    """Sinh sample (config → score) offline từ một profile — dữ liệu pretrain."""
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(m):
        algo = _ALGOS[int(rng.integers(0, len(_ALGOS)))]
        t = _TIME_OPTIONS[int(rng.integers(0, len(_TIME_OPTIONS)))]
        samples.append(
            {
                "params": {
                    "search_algorithm": algo,
                    "problem_type": "classification",
                    "metric": "accuracy",
                    "time_limit": t,
                },
                "dataset_meta": dict(profile.meta),
                "best_score": profile.sample_score(algo, t, rng),
            }
        )
    return samples


def run_transfer_loo(
    *,
    k: int = 6,
    heldout_index: int = 0,
    budget_jobs: int = 12,
    seed: int = 0,
    samples_per_profile: int = 60,
    profile_seed: int = 0,
) -> Dict[str, Any]:
    """
    Leave-one-dataset-out: pretrain outcome head trên k-1 profile, so
    wm-pretrained vs wm-scratch trên profile giữ lại.
    """
    profiles = make_transfer_profiles(k, seed=profile_seed)
    held = profiles[heldout_index % len(profiles)]
    pretrain: List[Dict[str, Any]] = []
    for i, p in enumerate(profiles):
        if p.name == held.name:
            continue
        pretrain.extend(generate_offline_samples(p, samples_per_profile, seed + i))

    pretrained = run_condition(
        "wm", held, budget_jobs=budget_jobs, seed=seed, initial_samples=pretrain
    )
    scratch = run_condition("wm", held, budget_jobs=budget_jobs, seed=seed)
    return {
        "heldout": held.name,
        "heldout_meta": dict(held.meta),
        "n_pretrain_samples": len(pretrain),
        "pretrained": pretrained,
        "scratch": scratch,
    }


def validate_condition(condition: str) -> None:
    """Raise ValueError nếu condition không hợp lệ — gọi TRƯỚC khi chạy ma trận."""
    if condition in ("wm", "wm_mpc", "no_wm", "random"):
        return
    if condition.startswith("fixed_") and condition.removeprefix("fixed_") in _ALGOS:
        return
    raise ValueError(
        f"Unknown benchmark condition: {condition!r}. "
        f"Valid: wm, wm_mpc, no_wm, random, fixed_<algo> with algo in {_ALGOS}"
    )


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
    initial_samples: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    validate_condition(condition)
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
    # Chiều model-subset chỉ bật khi profile có model_effects; [] tắt hẳn
    # (override yaml) để các profile cũ giữ nguyên không gian tìm kiếm
    model_opts: List[str] = sorted(profile.model_effects) if profile.model_effects else []
    head_cfg = dict(head_config or {"use_latent": False, "hidden_dim": 32})
    head_cfg.setdefault("model_vocab", model_opts)
    # user_id riêng theo condition — chặn nhiễm chéo qua warm-start memory
    uid = f"bench_{profile.name}_{condition}_{seed}"
    wm_snapshot = {"datasets": {profile.name: dict(profile.meta)}}

    samples: List[Dict[str, Any]] = list(initial_samples or [])
    outcome_model = None
    wm_trained_after: Optional[int] = None
    outcome_surprise_events: List[dict] = []

    # Pretrained transfer: đủ sample từ dataset khác → có model từ job 0
    if condition in ("wm", "wm_mpc") and len(samples) >= min_train_samples:
        outcome_model = train_outcome_head(
            samples, config=dict(head_cfg), epochs=train_epochs, seed=seed
        )
        wm_trained_after = 0

    mpc_planner = None
    if condition == "wm_mpc":
        from hagent.world.planner.cem_mpc_v1 import CemMpcV1Planner

        mpc_planner = CemMpcV1Planner(
            {
                "seed": seed,
                "search_algorithms": _ALGOS,
                "time_limit_options": _TIME_OPTIONS,
                "model_options": model_opts,
            }
        )

    try:
        while env.jobs_used < budget_jobs:
            n = min(campaign_size, budget_jobs - env.jobs_used)

            if condition == "wm":
                camp = await build_campaign(
                    goal,
                    user_id=uid,
                    world_model=wm_snapshot,
                    config={
                        "n_job_candidates": n,
                        "warm_start_top_k": 0,
                        "wm_variant_proposal": True,
                        "wm_rank_variants": True,
                        "model_options": model_opts,
                    },
                    # None (chưa train) = TẮT fallback checkpoint đĩa — không
                    # để model lạ trên máy lọt vào điều kiện thí nghiệm
                    outcome_model=outcome_model,
                )
            elif condition == "no_wm":
                camp = await build_campaign(
                    goal,
                    user_id=uid,
                    world_model=wm_snapshot,
                    config={
                        "n_job_candidates": n,
                        "warm_start_top_k": 0,
                        "wm_variant_proposal": False,
                        "wm_rank_variants": False,
                    },
                    outcome_model=None,
                )
            elif condition == "wm_mpc":
                props = mpc_planner.plan_batch(
                    base_params=_base_params(goal),
                    dataset_meta=dict(profile.meta),
                    outcome_model=outcome_model,
                    n=n,
                    remaining_budget=budget_jobs - env.jobs_used,
                    total_budget=budget_jobs,
                )
                params_list = [dict(_base_params(goal), **p) for p in props]
                camp = _campaign_from_params(goal, params_list, source="wm_mpc")
            elif condition == "random":
                params_list = []
                for _ in range(n):
                    p = dict(
                        _base_params(goal),
                        search_algorithm=str(rng.choice(_ALGOS)),
                        time_limit=int(rng.choice(_TIME_OPTIONS)),
                    )
                    if model_opts:
                        k = int(rng.integers(1, len(model_opts) + 1))
                        p["models"] = sorted(
                            str(m)
                            for m in rng.choice(model_opts, size=k, replace=False)
                        )
                    params_list.append(p)
                camp = _campaign_from_params(goal, params_list, source="random")
            elif condition.startswith("fixed_"):
                algo = condition.removeprefix("fixed_")
                params_list = [
                    dict(
                        _base_params(goal),
                        search_algorithm=algo,
                        time_limit=max(_TIME_OPTIONS),
                    )
                    for _ in range(n)
                ]
                camp = _campaign_from_params(goal, params_list, source=condition)
            else:  # đã validate ở đầu hàm — chỉ còn "random" rơi vào nhánh trên
                raise ValueError(f"Unknown benchmark condition: {condition!r}")

            events: List[dict] = []
            ticks = 0
            while camp.status not in ("done", "failed") and ticks < _MAX_TICKS:
                camp = await campaign_step(
                    camp,
                    user_id=uid,
                    user_token=None,
                    world_model=wm_snapshot,
                    surprise_events=events,
                    # Model train online đo surprise cho wm/wm_mpc; condition
                    # khác tắt hẳn (None) để không đụng checkpoint đĩa
                    outcome_model=(
                        outcome_model if condition in ("wm", "wm_mpc") else None
                    ),
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

            if condition in ("wm", "wm_mpc") and len(samples) >= min_train_samples:
                outcome_model = train_outcome_head(
                    samples, config=dict(head_cfg), epochs=train_epochs, seed=seed
                )
                if wm_trained_after is None:
                    wm_trained_after = env.jobs_used
    finally:
        set_tool_invoker(None)

    # Steering-quality metrics tính trên EXPECTED score của config đã submit —
    # so max của điểm nhiễu với optimum không nhiễu sẽ bị lệch bởi
    # max-order-statistic (~noise·E[max N] > cả khoảng cách giữa các policy
    # trên profile nhiễu). Điểm quan sát (nhiễu) vẫn báo cáo riêng.
    log = env.submission_log[:budget_jobs]
    scores = [r["expected_score"] for r in log]
    observed_scores = [r["best_score"] for r in log]
    curve = best_so_far_curve(scores)
    observed_curve = best_so_far_curve(observed_scores)
    final_best = curve[-1] if curve else None
    threshold = profile.base + 0.95 * (profile.optimum - profile.base)
    return {
        "condition": condition,
        "profile": profile.name,
        "seed": seed,
        "budget_jobs": budget_jobs,
        "jobs_used": env.jobs_used,
        "scores": scores,
        "observed_scores": observed_scores,
        "curve": curve,
        "observed_curve": observed_curve,
        "final_best": final_best,
        "observed_final_best": observed_curve[-1] if observed_curve else None,
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
    # Fail-fast: một condition sai chính tả không được phép đốt cả ma trận
    for condition in conditions:
        validate_condition(condition)
    for prof_name in profiles:
        if prof_name not in PROFILES:
            raise ValueError(
                f"Unknown profile {prof_name!r}. Available: {', '.join(PROFILES)}"
            )
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
