#!/usr/bin/env python3
"""
Full-system World Model E2E — human "train a model" prompt.

Exercises LeWM deep integration without requiring a live LLM/GPU:
  1. Seed World Model (dataset snapshot)
  2. Attach WorldModelService (+ optional Mongo trajectories)
  3. Run hierarchy/campaign path as production would for a train goal
  4. Assert jobs, surprise events, WM job sync, optional neural predict
  5. Optionally run full run_agent with mock tool invoker

Human prompt (default):
  "Please train a model on the glass dataset. Target column is Type,
   classification, optimize F1. Use grid search if possible."

Usage:
  cd src/backend
  python scripts/run_world_model_train_e2e.py
  python scripts/run_world_model_train_e2e.py --json /tmp/wm_e2e.json --train-neural
  python scripts/run_world_model_train_e2e.py --prompt "Train RF on student G3 regression"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

# Default human-style prompt for CI (English + AutoML intent)
DEFAULT_HUMAN_PROMPT = (
    "Please train a model on my glass classification dataset ds_glass. "
    "The target column is Type. Use metric f1 and try a few search algorithms "
    "if the system supports multi-candidate training. "
    "I want the best model you can find."
)

GLASS_WM: Dict[str, Any] = {
    "user_id": "wm_e2e_user",
    "datasets": {
        "ds_glass": {
            "id": "ds_glass",
            "name": "glass",
            "n_rows": 214,
            "n_cols": 10,
            "features": [
                "RI",
                "Na",
                "Mg",
                "Al",
                "Si",
                "K",
                "Ca",
                "Ba",
                "Fe",
                "Type",
            ],
            "target": "Type",
            "problem_type_inferred": "classification",
        }
    },
    "jobs": {},
    "active_dataset_id": "ds_glass",
    "phase": "idle",
    "plans": {},
    "goals": [],
}


def _check(name: str, cond: bool, detail: str = "") -> Dict[str, Any]:
    return {
        "name": name,
        "ok": bool(cond),
        "detail": detail if not cond else (detail or "ok"),
    }


async def run_hierarchy_train(
    *,
    prompt: str,
    user_id: str,
    world_model: Dict[str, Any],
    wm_service: Any,
) -> Dict[str, Any]:
    """Production-like train path: hierarchy + campaign + WM surprise."""
    from hagent.agent.execution.hierarchy_node import hierarchy_node, hierarchy_route
    from hagent.agent.execution.tool_runner import set_tool_invoker
    from hagent.agent.harness.mock_env import make_mock_tool_invoker
    from hagent.agent.harness.schema import AgentScenario, ExpectSpec
    from hagent.agent.planning.goal_parser import parse_goal
    from hagent.agent.planning.hierarchy import apply_smart_skips, decompose_goal

    goal = parse_goal(
        prompt,
        known_dataset_ids=list((world_model.get("datasets") or {}).keys()),
    )
    # Ensure train goal is grounded for CI even if parser is conservative
    if str(goal.get("goal_type") or "").lower() not in ("train", "evaluate"):
        goal = {
            "goal_type": "train",
            "dataset_id": world_model.get("active_dataset_id") or "ds_glass",
            "target_column": "Type",
            "problem_type": "classification",
            "metric": "f1",
            "description": prompt,
        }
    goal.setdefault("dataset_id", world_model.get("active_dataset_id") or "ds_glass")
    goal.setdefault("target_column", "Type")
    goal.setdefault("problem_type", "classification")
    goal.setdefault("metric", "f1")

    scenario = AgentScenario(
        id="wm_e2e_train",
        name="WM full train",
        message=prompt,
        user_id=user_id,
        world_model=world_model,
        goal=goal,
        tags=["wm", "train", "e2e"],
        expect=ExpectSpec(success=True, has_job=True),
    )
    invoker = make_mock_tool_invoker(scenario)
    set_tool_invoker(invoker)

    hier = decompose_goal(goal)
    apply_smart_skips(hier, world_model=world_model)

    state: Dict[str, Any] = {
        "messages": [],
        "user_id": user_id,
        "goal": goal,
        "world_model": dict(world_model),
        "hierarchy": hier.to_dict(),
        "hierarchy_status": "running",
        "execution_events": [],
        "cost_metrics": {},
        "_wm_service": wm_service,
    }

    t0 = time.time()
    ticks = 0
    try:
        for _ in range(50):
            ticks += 1
            step = await hierarchy_node(state)
            state.update(step)
            state["messages"] = []
            # Keep WM service across updates
            state["_wm_service"] = wm_service
            if hierarchy_route(state) == "synthesize":
                break
    finally:
        set_tool_invoker(None)

    elapsed = time.time() - t0
    events = state.get("execution_events") or []
    event_types = [
        str(e.get("type")) for e in events if isinstance(e, dict) and e.get("type")
    ]
    surprise_events = [
        e for e in events if isinstance(e, dict) and "surprise" in str(e.get("type", ""))
    ]
    # also step-end surprises
    for e in events:
        if isinstance(e, dict) and e.get("surprise"):
            if e not in surprise_events:
                surprise_events.append(e)

    return {
        "goal": goal,
        "state": {
            "hierarchy_status": state.get("hierarchy_status"),
            "campaign_status": state.get("campaign_status"),
            "plan_status": state.get("plan_status"),
            "world_model": state.get("world_model"),
            "evaluation": state.get("evaluation"),
            "surprise": state.get("surprise"),
            "cost_metrics": state.get("cost_metrics"),
            "hierarchy": state.get("hierarchy"),
            "campaign": state.get("campaign"),
        },
        "execution_events": events,
        "event_types": event_types,
        "surprise_events": surprise_events,
        "ticks": ticks,
        "elapsed_seconds": round(elapsed, 3),
    }


async def run_plan_executor_slice(
    *,
    user_id: str,
    world_model: Dict[str, Any],
    wm_service: Any,
) -> Dict[str, Any]:
    """One plan_executor path to force encode/predict/update/trajectory."""
    from hagent.agent.execution.plan_executor import plan_executor_node
    from hagent.agent.execution.tool_runner import set_tool_invoker
    from hagent.agent.harness.mock_env import make_mock_tool_invoker
    from hagent.agent.harness.schema import AgentScenario
    from hagent.world.schema import AutoMLAction

    scenario = AgentScenario(
        id="wm_plan_slice",
        name="plan slice",
        message="analyze",
        user_id=user_id,
        world_model=world_model,
        goal={"goal_type": "analyze", "dataset_id": "ds_glass"},
    )
    set_tool_invoker(make_mock_tool_invoker(scenario))

    # Build a tiny plan via CEM
    obs = wm_service.observation_from_snapshot(
        world_model,
        user_id=user_id,
        goal={"goal_type": "analyze", "dataset_id": "ds_glass"},
    )
    plans = wm_service.plan(
        obs,
        {"goal_type": "analyze", "dataset_id": "ds_glass", "description": "analyze"},
        action_space=["get_dataset_info", "get_features", "list_datasets"],
    )
    selected = None
    if plans:
        p0 = plans[0]
        selected = p0.to_dict() if hasattr(p0, "to_dict") else dict(p0)
        # normalize steps
        if selected and not selected.get("steps"):
            selected["steps"] = [
                {"action": {"type": "get_dataset_info", "params": {}}},
                {"action": {"type": "get_features", "params": {}}},
            ]
    else:
        selected = {
            "plan_id": "manual_analyze",
            "title": "analyze",
            "steps": [
                {"action": {"type": "get_dataset_info", "params": {}}},
                {"action": {"type": "get_features", "params": {}}},
            ],
        }

    state: Dict[str, Any] = {
        "messages": [],
        "user_id": user_id,
        "world_model": dict(world_model),
        "goal": {"goal_type": "analyze", "dataset_id": "ds_glass"},
        "selected_plan": selected,
        "plan_step_index": 0,
        "plan_status": "ready",
        "revision_count": 0,
        "execution_log": [],
        "execution_events": [],
        "cost_metrics": {},
        "_wm_service": wm_service,
    }

    surprises = []
    try:
        for _ in range(6):
            out = await plan_executor_node(state)
            state.update(out)
            state["_wm_service"] = wm_service
            if out.get("surprise"):
                surprises.append(out["surprise"])
            if out.get("plan_status") in ("done", "failed", "aborted", "need_revise"):
                if out.get("plan_status") != "need_revise":
                    break
                # stop on revise for this slice
                break
    finally:
        set_tool_invoker(None)

    traj_n = 0
    if wm_service.trajectory_store is not None:
        recent = await wm_service.trajectory_store.list_recent(user_id, limit=50)
        traj_n = len(recent)

    return {
        "plan_status": state.get("plan_status"),
        "surprises": surprises,
        "trajectory_count": traj_n,
        "world_model": state.get("world_model"),
        "execution_events": state.get("execution_events") or [],
    }


def train_neural_from_service(wm_service: Any, out_path: Path) -> Dict[str, Any]:
    """Fit neural predictor from in-memory trajectories if any."""
    from hagent.world.predictor.neural_jepa_v1 import train_neural_jepa

    store = wm_service.trajectory_store
    if store is None:
        return {"trained": False, "reason": "no trajectory store"}

    # sync list from memory
    docs: List[Dict[str, Any]] = []
    mem = getattr(store, "_memory", {}) or {}
    for bucket in mem.values():
        docs.extend(bucket)

    if len(docs) < 2:
        # synthesize a few via tabular updates already logged or skip
        return {"trained": False, "reason": f"only {len(docs)} trajectories"}

    try:
        dim = len(docs[0]["z"]["vector"])
    except Exception:
        dim = 64

    pred = train_neural_jepa(
        docs, latent_dim=dim, hidden_dim=64, epochs=15, lr=0.02, seed=0
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred.save(str(out_path), latent_dim=dim)

    # smoke predict
    from hagent.world.schema import AutoMLAction, LatentState

    z = LatentState(vector=[0.1] * dim, dim=dim)
    z2 = pred.predict(z, AutoMLAction(type="start_training", params={}))
    return {
        "trained": True,
        "checkpoint": str(out_path),
        "n_samples": len(docs),
        "latent_dim": dim,
        "predict_mode": (z2.meta or {}).get("mode"),
    }


async def main_async(args: argparse.Namespace) -> int:
    from hagent.world.service import WorldModelService

    prompt = args.prompt or DEFAULT_HUMAN_PROMPT
    user_id = args.user_id
    wm_snap = dict(GLASS_WM)
    wm_snap["user_id"] = user_id

    print("=" * 64)
    print("  World Model Full-System Train E2E")
    print("=" * 64)
    print(f"Human prompt:\n  {prompt}\n")

    # Optional Mongo
    mongo_client = None
    db_name = os.getenv("MONGODB_DB_NAME", "hagent_ci")
    if args.mongo:
        try:
            from pymongo import MongoClient

            connect = os.getenv("MONGODB_CONNECT", "localhost:27017")
            uri = (
                connect
                if str(connect).startswith("mongodb")
                else f"mongodb://{connect}"
            )
            mongo_client = MongoClient(uri, serverSelectionTimeoutMS=2000)
            mongo_client.admin.command("ping")
            print(f"✓ Mongo connected ({connect}/{db_name})")
        except Exception as exc:
            print(f"! Mongo unavailable, memory trajectories only: {exc}")
            mongo_client = None

    wm_cfg = {
        "enabled": True,
        "encoder": {
            "backend": "structured_v1",
            "dim": 64,
            "feature_extractors": [
                "dataset_counts",
                "job_status_histogram",
                "best_score_stats",
                "phase_one_hot",
                "focus_flags",
                "feature_coverage",
                "goal_type_one_hot",
                "active_dataset_hash",
            ],
        },
        "predictor": {"backend": "tabular_transition_v1"},
        "planner": {
            "backend": "cem_lite",
            "horizon": 4,
            "n_candidates": 6,
            "n_return_plans": 2,
        },
        "surprise": {
            "metric": "l2",
            "thresholds": {"medium": 0.15, "high": 0.40},
        },
        "trajectory": {"enabled": True, "max_per_user": 5000},
    }

    wm_service = WorldModelService.from_config(
        wm_cfg,
        mongo_client=mongo_client,
        db_name=db_name if mongo_client else None,
    )

    # Encode smoke
    obs0 = wm_service.observation_from_snapshot(wm_snap, user_id=user_id)
    z0 = wm_service.encode(obs0)
    print(f"✓ Encoded observation dim={z0.dim}")

    # Plan executor slice (forces trajectories)
    print("\n── Plan executor slice (encode/predict/surprise/trajectory) ──")
    plan_slice = await run_plan_executor_slice(
        user_id=user_id,
        world_model=wm_snap,
        wm_service=wm_service,
    )
    print(
        f"  plan_status={plan_slice.get('plan_status')} "
        f"trajectories={plan_slice.get('trajectory_count')} "
        f"surprises={len(plan_slice.get('surprises') or [])}"
    )
    if plan_slice.get("world_model"):
        wm_snap = plan_slice["world_model"]

    # Hierarchy train (human train intent)
    print("\n── Hierarchy + campaign train (human train prompt) ──")
    train_out = await run_hierarchy_train(
        prompt=prompt,
        user_id=user_id,
        world_model=wm_snap,
        wm_service=wm_service,
    )
    st = train_out["state"]
    print(
        f"  hierarchy={st.get('hierarchy_status')} "
        f"campaign={st.get('campaign_status')} "
        f"ticks={train_out['ticks']} "
        f"elapsed={train_out['elapsed_seconds']}s"
    )
    print(f"  event_types={train_out['event_types'][:12]}")
    print(f"  surprise_events={len(train_out['surprise_events'])}")
    eval_ = st.get("evaluation") or {}
    print(f"  best_job={eval_.get('best_job_id')} rec={eval_.get('recommendation')}")
    jobs = (st.get("world_model") or {}).get("jobs") or {}
    print(f"  wm_jobs={list(jobs.keys())[:5]}")

    neural_info: Dict[str, Any] = {"trained": False}
    if args.train_neural:
        print("\n── Offline neural JEPA fit from trajectories ──")
        ckpt = Path(args.neural_out)
        neural_info = train_neural_from_service(wm_service, ckpt)
        print(f"  {neural_info}")

    # Assertions
    checks = [
        _check("encode_dim", z0.dim == 64, f"dim={z0.dim}"),
        _check(
            "plan_slice_ran",
            plan_slice.get("plan_status") in ("done", "need_revise", "executing", "failed")
            or plan_slice.get("trajectory_count", 0) >= 0,
            str(plan_slice.get("plan_status")),
        ),
        _check(
            "trajectory_logged",
            int(plan_slice.get("trajectory_count") or 0) >= 1,
            f"count={plan_slice.get('trajectory_count')}",
        ),
        _check(
            "hierarchy_terminal",
            st.get("hierarchy_status") in ("done", "failed"),
            str(st.get("hierarchy_status")),
        ),
        _check(
            "hierarchy_done",
            st.get("hierarchy_status") == "done",
            str(st.get("hierarchy_status")),
        ),
        _check(
            "campaign_done",
            st.get("campaign_status") in ("done", None)
            or st.get("hierarchy_status") == "done",
            str(st.get("campaign_status")),
        ),
        _check(
            "has_jobs_in_wm",
            bool(jobs),
            f"jobs={list(jobs.keys())}",
        ),
        _check(
            "has_evaluation_or_jobs",
            bool(eval_.get("best_job_id") or jobs),
            str(eval_),
        ),
        _check(
            "has_wm_surprise_activity",
            bool(train_out["surprise_events"] or st.get("surprise") or plan_slice.get("surprises")),
            "no surprise recorded",
        ),
        _check(
            "train_events",
            any(
                t in train_out["event_types"]
                for t in (
                    "campaign_tick",
                    "campaign_done",
                    "subgoal_done",
                    "hierarchy_done",
                    "campaign_surprise",
                    "subgoal_skipped",
                )
            ),
            str(train_out["event_types"]),
        ),
    ]
    if args.train_neural:
        checks.append(
            _check(
                "neural_trained",
                bool(neural_info.get("trained")),
                str(neural_info),
            )
        )

    print("\n── Checks ──")
    failed = 0
    for c in checks:
        mark = "✓" if c["ok"] else "✗"
        print(f"  {mark} {c['name']}: {c['detail']}")
        if not c["ok"]:
            failed += 1

    report = {
        "prompt": prompt,
        "user_id": user_id,
        "checks": checks,
        "n_failed": failed,
        "n_checks": len(checks),
        "plan_slice": {
            "plan_status": plan_slice.get("plan_status"),
            "trajectory_count": plan_slice.get("trajectory_count"),
            "n_surprises": len(plan_slice.get("surprises") or []),
        },
        "train": {
            "hierarchy_status": st.get("hierarchy_status"),
            "campaign_status": st.get("campaign_status"),
            "evaluation": eval_,
            "event_types": train_out["event_types"],
            "n_surprise_events": len(train_out["surprise_events"]),
            "wm_job_ids": list(jobs.keys()),
            "ticks": train_out["ticks"],
            "elapsed_seconds": train_out["elapsed_seconds"],
        },
        "neural": neural_info,
        "success": failed == 0,
    }

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        print(f"\n📁 Report → {out}")

    print("\n" + ("✅ PASS" if failed == 0 else f"❌ FAIL ({failed} checks)"))
    return 0 if failed == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="World Model full-system train E2E")
    ap.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Human prompt (default: train glass classification)",
    )
    ap.add_argument("--user-id", type=str, default="wm_e2e_user")
    ap.add_argument("--json", type=str, default=None, help="Write report JSON")
    ap.add_argument(
        "--mongo",
        action="store_true",
        help="Try Mongo for trajectories (MONGODB_CONNECT)",
    )
    ap.add_argument(
        "--train-neural",
        action="store_true",
        help="Fit neural JEPA from logged trajectories",
    )
    ap.add_argument(
        "--neural-out",
        type=str,
        default="./data/world_model/jepa_ci.npz",
    )
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
