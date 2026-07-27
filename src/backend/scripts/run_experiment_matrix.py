#!/usr/bin/env python3
"""
Runner ma trận thí nghiệm agent — conditions × datasets × models × seeds.

Thiết kế:
  - LLM THẬT (per-request model, T12) + job sklearn THẬT chạy in-process:
    RealJobEnv chặn invoke_tool, start_training chạy đúng
    SearchStrategyFactory trên dataset thật từ automl/search/datasets_real
    → không docker, không mock score, vô hiệu phản biện "toàn mock".
  - Điều kiện A/B/C/C_mpc render thành yaml tạm từ hagent.yaml gốc,
    HAGENT_CONFIG trỏ vào + load_config.cache_clear() mỗi ô.
  - JSONL mỗi ô một dòng, RESUMABLE (ô đã có bị bỏ qua).

Usage:
  cd src/backend
  python scripts/run_experiment_matrix.py --dry-run
  python scripts/run_experiment_matrix.py                 # chạy theo config
  python scripts/run_experiment_matrix.py --only A:iris:openai-gpt4o-mini:0
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import math
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

# ── Điều kiện → patch yaml ───────────────────────────────

_MISSING = "./data/world_model/__disabled__"
PROTOCOL_VERSION = "paired-meta-advice-v1"
PROTOCOL_METRIC = "accuracy"
PROTOCOL_CONDITIONS = ("A", "B", "C")
PROTOCOL_SEEDS = (0, 1, 2)
ADVICE_ALGORITHMS = (
    "grid_search",
    "bayesian_search",
    "genetic_algorithm",
    "random_search",
    "successive_halving",
)
META_FEATURE_KEYS = (
    "n_rows",
    "n_cols",
    "n_classes",
    "class_imbalance",
    "frac_categorical",
    "missing_frac",
    "mean_abs_skew",
)


class ProtocolError(RuntimeError):
    """Fail-closed protocol violation."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_experiment_design(cfg: dict) -> Dict[str, Any]:
    dimensions: Dict[str, list] = {}
    for name in ("conditions", "datasets", "models", "seeds"):
        values = cfg.get(name)
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise ProtocolError(f"{name} must be a non-empty unique list")
        dimensions[name] = list(values)
    if tuple(dimensions["conditions"]) != PROTOCOL_CONDITIONS:
        raise ProtocolError("conditions must be exactly A, B, C")
    if tuple(dimensions["seeds"]) != PROTOCOL_SEEDS:
        raise ProtocolError("seeds must be exactly 0, 1, 2")
    cell_count = (
        len(dimensions["conditions"])
        * len(dimensions["datasets"])
        * len(dimensions["models"])
        * len(dimensions["seeds"])
    )
    if cell_count != 54:
        raise ProtocolError(f"main matrix must contain 54 cells, got {cell_count}")
    prompt = cfg.get("prompt")
    job = cfg.get("job")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ProtocolError("prompt must be a non-empty string")
    if not isinstance(job, dict):
        raise ProtocolError("job must be an object")
    return {
        "protocol_version": PROTOCOL_VERSION,
        **dimensions,
        "metric": PROTOCOL_METRIC,
        "search_algorithms": list(ADVICE_ALGORITHMS),
        "job": job,
        "prompt": prompt,
    }


def design_sha256(cfg: dict) -> str:
    return hashlib.sha256(
        _canonical_json(build_experiment_design(cfg)).encode("utf-8")
    ).hexdigest()


def dataset_sha256(dataset: Dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in ("X", "y"):
        if name not in dataset:
            raise ProtocolError(f"dataset missing {name}")
        array = np.ascontiguousarray(np.asarray(dataset[name], dtype="<f8"))
        digest.update(name.encode("ascii"))
        digest.update(_canonical_json(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def anonymized_advice_payload(
    dataset: Dict[str, Any],
    *,
    metric: str = PROTOCOL_METRIC,
    algorithms: tuple[str, ...] = ADVICE_ALGORITHMS,
) -> Dict[str, Any]:
    if metric != PROTOCOL_METRIC:
        raise ProtocolError(f"metric must be {PROTOCOL_METRIC}")
    if tuple(algorithms) != ADVICE_ALGORITHMS:
        raise ProtocolError("search algorithm pool differs from frozen protocol")
    meta = dataset.get("meta")
    if not isinstance(meta, dict):
        raise ProtocolError("dataset meta-features are missing")
    safe_meta: Dict[str, float] = {}
    for key in META_FEATURE_KEYS:
        value = meta.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise ProtocolError(f"meta-feature {key} is not numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ProtocolError(f"meta-feature {key} is not finite")
        safe_meta[key] = number
    return {
        "meta_features": safe_meta,
        "metric": metric,
        "search_algorithms": list(algorithms),
    }



CONDITION_PATCHES: Dict[str, Dict[str, Any]] = {
    "A": {
        "campaign": {
            "wm_variant_proposal": False,
            "wm_rank_variants": False,
            "surprise_extension": {"enabled": False},
        },
        # Trỏ checkpoint vào path không tồn tại — model không thể bị nạp lén
        "world_model": {
            "outcome_head": {"checkpoint_path": _MISSING + ".npz"},
            "outcome_ensemble": {"checkpoint_dir": _MISSING},
        },
    },
    "B": {
        "campaign": {
            "wm_variant_proposal": True,
            "wm_rank_variants": True,
            "surprise_extension": {"enabled": False},
        },
    },
    "C": {
        "campaign": {
            "wm_variant_proposal": True,
            "wm_rank_variants": True,
            "surprise_extension": {
                "enabled": True,
                "max_rounds": 1,
                "n_extra": 2,
                "exploration_weight": 0.5,
            },
        },
    },
    "C_mpc": {
        "campaign": {
            "wm_variant_proposal": True,
            "wm_rank_variants": True,
            "surprise_extension": {
                "enabled": True,
                "max_rounds": 1,
                "n_extra": 2,
                "exploration_weight": 0.5,
            },
        },
        "world_model": {"campaign_planner": {"backend": "cem_mpc_v1"}},
    },
}


def _deep_merge(base: dict, patch: dict) -> dict:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def render_condition_yaml(condition: str, scratch_dir: Path) -> Path:
    """hagent.yaml gốc + patch điều kiện → file tạm; trả path."""
    if condition not in CONDITION_PATCHES:
        raise ValueError(
            f"Condition {condition!r} không hợp lệ. Có: {list(CONDITION_PATCHES)}"
        )
    base_path = BACKEND / "hagent" / "hagent.yaml"
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    patch = CONDITION_PATCHES[condition]

    merged = dict(base)
    if "campaign" in patch:
        agent = dict(merged.get("agent") or {})
        agent["campaign"] = _deep_merge(dict(agent.get("campaign") or {}), patch["campaign"])
        merged["agent"] = agent
    if "world_model" in patch:
        merged["world_model"] = _deep_merge(
            dict(merged.get("world_model") or {}), patch["world_model"]
        )

    out = scratch_dir / f"hagent_{condition}.yaml"
    out.write_text(yaml.safe_dump(merged, allow_unicode=True), encoding="utf-8")
    return out


def apply_condition(condition: str, scratch_dir: Path) -> None:
    """Đặt HAGENT_CONFIG + xóa cache config — PHẢI gọi trước mỗi ô."""
    path = render_condition_yaml(condition, scratch_dir)
    os.environ["HAGENT_CONFIG"] = str(path)
    from hagent.bridge import config as bridge_config

    bridge_config.load_config.cache_clear()
    # Memo outcome model phụ thuộc config path → xóa để nạp đúng điều kiện
    from hagent.agent.campaign import wm_hooks

    wm_hooks._outcome_model_cache["fingerprint"] = None


# ── RealJobEnv: job sklearn thật in-process ──────────────


class RealJobEnv:
    """Tool invoker: start_training chạy THẬT search strategy trên dataset thật."""

    def __init__(self, dataset: Dict[str, Any], *, job_cfg: dict, seed: int):
        self.dataset = dataset
        self.job_cfg = dict(job_cfg or {})
        self.seed = seed
        self.jobs: Dict[str, Dict[str, Any]] = {}

    async def invoke(self, action_type: str, params: dict) -> dict:
        d = self.dataset
        if action_type in ("list_datasets",):
            return {
                "datasets": [
                    {"id": d["name"], "name": d["name"], "n_rows": d["n_rows"],
                     "n_cols": d["n_cols"]}
                ]
            }
        if action_type in ("get_dataset_info", "get_features", "preview_data"):
            return {
                "id": d["name"],
                "name": d["name"],
                "n_rows": d["n_rows"],
                "n_cols": d["n_cols"],
                "features": [f"f{i}" for i in range(d["n_cols"])] + ["target"],
                "target": "target",
            }
        if action_type == "get_available_models":
            return {"models": ["RandomForestClassifier"]}
        if action_type == "start_training":
            job_id = f"real_{len(self.jobs) + 1}_{uuid.uuid4().hex[:6]}"
            record = await asyncio.to_thread(self._run_real_job, dict(params))
            record["job_id"] = job_id
            self.jobs[job_id] = record
            return {"job_id": job_id, "status": 0}
        if action_type == "get_job_info":
            job = self.jobs.get(str(params.get("job_id")))
            if not job:
                return {"error": "job not found"}
            return {
                "id": job["job_id"],
                "status": "completed",
                "best_model": job["best_model"],
                "best_score": job["best_score"],
                "metrics": {"accuracy": job["best_score"]},
            }
        if action_type == "list_jobs":
            return {"jobs": list(self.jobs.values())}
        return {}

    def _run_real_job(self, params: dict) -> dict:
        """Job sklearn thật: strategy.search trên dataset thật (blocking)."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold

        from automl.search.factory.search_strategy_factory import (
            SearchStrategyFactory,
        )

        algo = str(params.get("search_algorithm") or "grid_search")
        cfg = dict(
            cv=StratifiedKFold(
                n_splits=int(self.job_cfg.get("cv", 3)), shuffle=True,
                random_state=self.seed,
            ),
            scoring={"accuracy": "accuracy"},
            metric_sort="accuracy",
            n_jobs=-1,
            random_state=self.seed,
            save_log=False,
            verbose=0,
            max_time=float(params.get("time_limit") or self.job_cfg.get("time_limit") or 60),
            # BO mặc định infer_dimensions=True sẽ tự nới grid thành không
            # gian liên tục — mọi thuật toán PHẢI cùng không gian 18 điểm
            infer_dimensions=False,
        )
        strategy = SearchStrategyFactory.create_strategy(algo, cfg)
        grid = dict(self.job_cfg.get("param_grid") or {})
        t0 = time.perf_counter()
        best_params, best_score, _, _, time_limited = strategy.search(
            RandomForestClassifier(random_state=self.seed),
            grid,
            self.dataset["X"],
            self.dataset["y"],
        )
        return {
            "search_algorithm": algo,
            "best_params": best_params,
            "best_score": float(best_score),
            "best_model": "RandomForestClassifier",
            "seconds": round(time.perf_counter() - t0, 2),
            "time_limited": bool(time_limited),
        }


# ── Cell execution ───────────────────────────────────────


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(BACKEND), text=True
        ).strip()
    except Exception:
        return "unknown"


def _checkpoint_sha() -> Optional[str]:
    try:
        from hagent.bridge.config import get_world_model_config

        path = (get_world_model_config().get("outcome_head") or {}).get("checkpoint_path")
        p = Path(path) if path else None
        if p and not p.is_absolute():
            p = BACKEND / p
        if p and p.is_file():
            return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    except Exception:
        pass
    return None


def cell_key(condition: str, dataset: str, model: str, seed: int) -> str:
    return f"{condition}:{dataset}:{model}:{seed}"


def run_cell(
    condition: str,
    dataset_name: str,
    model: str,
    seed: int,
    *,
    cfg: dict,
    scratch_dir: Path,
) -> Dict[str, Any]:
    from automl.search.datasets_real import load_dataset
    from hagent.agent.execution.tool_runner import set_tool_invoker
    from hagent.agent.graph import run_agent

    apply_condition(condition, scratch_dir)
    dataset = load_dataset(dataset_name)
    env = RealJobEnv(dataset, job_cfg=cfg.get("job") or {}, seed=seed)
    set_tool_invoker(env.invoke)

    prompt = str(cfg.get("prompt") or "Train a model on {dataset}, target {target}.")
    message = prompt.format(dataset=dataset_name, target="target")

    t0 = time.perf_counter()
    error = None
    result: Dict[str, Any] = {}
    try:
        result = asyncio.new_event_loop().run_until_complete(
            run_agent(
                message,
                user_id=f"mx_{condition}_{dataset_name}_{seed}",
                world_model={
                    "user_id": f"mx_{condition}_{dataset_name}_{seed}",
                    "datasets": {
                        dataset_name: {
                            "id": dataset_name,
                            "name": dataset_name,
                            **{k: v for k, v in dataset["meta"].items()},
                            "features": [f"f{i}" for i in range(dataset["n_cols"])]
                            + ["target"],
                            "target": "target",
                        }
                    },
                },
                model_name=model,
            )
        )
    except Exception as exc:  # ghi lỗi vào row, không sập cả ma trận
        error = f"{type(exc).__name__}: {exc}"
    finally:
        set_tool_invoker(None)
    elapsed = time.perf_counter() - t0

    campaign = result.get("campaign") or {}
    variants = campaign.get("variants") or []
    events = result.get("execution_events") or []
    ev_types = [e.get("type") for e in events if isinstance(e, dict)]
    real_scores = [j["best_score"] for j in env.jobs.values()]

    return {
        "key": cell_key(condition, dataset_name, model, seed),
        "condition": condition,
        "dataset": dataset_name,
        "model": model,
        "seed": seed,
        "error": error,
        "route": result.get("route"),
        "response_chars": len(result.get("response") or ""),
        "campaign_status": result.get("campaign_status"),
        "n_variants": len(variants),
        "variant_sources": sorted({v.get("source") for v in variants if isinstance(v, dict)}),
        "extension_rounds": campaign.get("extension_rounds"),
        "best_real_score": max(real_scores) if real_scores else None,
        "n_real_jobs": len(env.jobs),
        "job_seconds_total": round(
            sum(j.get("seconds") or 0 for j in env.jobs.values()), 2
        ),
        "n_outcome_surprise": ev_types.count("campaign_outcome_surprise"),
        "n_extended": ev_types.count("campaign_extended"),
        "cost_metrics": result.get("cost_metrics"),
        "checkpoint_sha": _checkpoint_sha(),
        "git_sha": _git_sha(),
        "wall_seconds": round(elapsed, 2),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Ma trận thí nghiệm agent")
    parser.add_argument(
        "--config", default="benchmarks/agent_matrix_config.yaml"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only", help="Chạy đúng một ô: COND:DATASET:MODEL:SEED"
    )
    args = parser.parse_args()

    cfg_path = BACKEND / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    cells = [
        (c, d, m, s)
        for c in cfg["conditions"]
        for d in cfg["datasets"]
        for m in cfg["models"]
        for s in cfg["seeds"]
    ]
    if args.only:
        c, d, m, s = args.only.split(":")
        cells = [(c, d, m, int(s))]

    out_path = BACKEND / str(cfg.get("output") or "benchmarks/agent_matrix_results.jsonl")
    done: set = set()
    if out_path.is_file():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if not row.get("error"):
                    done.add(row["key"])
            except Exception:
                continue

    todo = [c for c in cells if cell_key(*c) not in done]
    print(
        f"Ma trận: {len(cells)} ô | đã xong {len(cells) - len(todo)} | còn {len(todo)}"
    )
    if args.dry_run:
        for c in todo:
            print("  ", cell_key(*c))
        return 0

    # Cảnh báo checkpoint cho điều kiện B/C
    needs_ckpt = [c for c in todo if c[0] != "A"]
    if needs_ckpt:
        head = BACKEND / "data" / "world_model" / "outcome_head_v2.npz"
        if not head.is_file():
            print(
                f"CẢNH BÁO: điều kiện B/C cần checkpoint {head} — chạy "
                f"scripts/train_outcome_model.py trước, nếu không WM sẽ bất hoạt."
            )

    scratch = Path(tempfile.mkdtemp(prefix="hagent_matrix_"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as fh:
        for i, (c, d, m, s) in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {cell_key(c, d, m, s)} ...", flush=True)
            row = run_cell(c, d, m, s, cfg=cfg, scratch_dir=scratch)
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            fh.flush()
            status = row["error"] or (
                f"route={row['route']} jobs={row['n_real_jobs']} "
                f"best={row['best_real_score']} ext={row['n_extended']} "
                f"{row['wall_seconds']}s"
            )
            print(f"    -> {status}", flush=True)

    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
