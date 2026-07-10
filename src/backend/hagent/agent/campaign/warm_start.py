"""
Warm-start configs from World Model history + memory facts.

No hard-coded model names beyond what's already in past job configs / config YAML.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from hagent.world.query import past_best_jobs

logger = logging.getLogger(__name__)


def _score_job(job: dict) -> float:
    s = job.get("best_score")
    if s is not None:
        try:
            return float(s)
        except (TypeError, ValueError):
            pass
    metrics = job.get("metrics") or {}
    if isinstance(metrics, dict) and metrics:
        try:
            return float(max(metrics.values()))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def configs_from_world_model(
    world_model: dict | None,
    *,
    problem_type: str | None,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Extract training configs from past successful jobs."""
    jobs = past_best_jobs(
        world_model or {},
        problem_type=problem_type,
        top_k=top_k,
    )
    configs: List[Dict[str, Any]] = []
    for j in jobs:
        cfg = dict(j.get("config") or {})
        if not cfg and j.get("best_model"):
            cfg = {"models": [j["best_model"]]}
        if j.get("best_model") and "models" not in cfg:
            cfg["models"] = [j["best_model"]]
        if problem_type and "problem_type" not in cfg:
            cfg["problem_type"] = problem_type
        cfg["_source_job_id"] = j.get("id")
        cfg["_source_score"] = _score_job(j)
        configs.append(cfg)
    return configs


async def configs_from_memory(
    user_id: str | None,
    *,
    problem_type: str | None,
    fact_store: Any | None = None,
) -> List[Dict[str, Any]]:
    """Load warm-start facts previously written by campaigns."""
    if not user_id:
        return []
    try:
        if fact_store is None:
            from hagent.agent.memory import create_fact_store

            fact_store = create_fact_store()
        facts = await fact_store.search(user_id, category="model", limit=20)
        out: List[Dict[str, Any]] = []
        for fact in facts:
            # Keys written by campaigns: warm_start_{problem_type}
            if not str(fact.key).startswith("warm_start"):
                # Also accept content that embeds warm-start JSON
                if "warm_start" not in (fact.content or "").lower() and "best_score" not in (
                    fact.content or ""
                ):
                    continue
            try:
                data = json.loads(fact.content)
                if not isinstance(data, dict):
                    continue
                if problem_type and data.get("problem_type"):
                    if str(data["problem_type"]).lower() != str(problem_type).lower():
                        continue
                data["_memory_key"] = fact.key
                out.append(data)
            except (json.JSONDecodeError, TypeError):
                continue
        return out
    except Exception as exc:
        logger.debug("Memory warm-start skipped: %s", exc)
        return []


def merge_warm_starts(
    *,
    from_wm: List[Dict[str, Any]],
    from_memory: List[Dict[str, Any]],
    max_items: int = 5,
) -> List[Dict[str, Any]]:
    """Dedupe warm-start configs (by search_algorithm + models signature)."""
    seen = set()
    merged: List[Dict[str, Any]] = []
    for cfg in list(from_memory) + list(from_wm):
        models = cfg.get("models") or cfg.get("model")
        if isinstance(models, str):
            models = [models]
        key = (
            str(cfg.get("search_algorithm") or cfg.get("search_strategy") or ""),
            tuple(models or ()),
            str(cfg.get("metric") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(cfg)
        if len(merged) >= max_items:
            break
    return merged


async def collect_warm_start_configs(
    *,
    world_model: dict | None,
    user_id: str | None,
    problem_type: str | None,
    top_k: int = 3,
    fact_store: Any | None = None,
) -> List[Dict[str, Any]]:
    wm_cfgs = configs_from_world_model(
        world_model, problem_type=problem_type, top_k=top_k
    )
    mem_cfgs = await configs_from_memory(
        user_id, problem_type=problem_type, fact_store=fact_store
    )
    return merge_warm_starts(from_wm=wm_cfgs, from_memory=mem_cfgs, max_items=top_k + 2)
