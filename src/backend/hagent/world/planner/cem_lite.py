"""
CEM-lite latent planner (LeWM-style, structured AutoML domain).

1. Expand goal-type skeletons from config into candidate action sequences
2. Roll out predictor for horizon H
3. Cost = w_latent * ||ẑ_H - z_g|| + w_constraint * violations + w_step * steps
4. Return top-k plans

Skeletons / weights / horizon come from config — not hard-coded at graph layer.
"""

from __future__ import annotations

import itertools
import uuid
from typing import Any, Dict, List, Optional, Sequence

from hagent.world.schema import (
    AutoMLAction,
    GoalSpec,
    LatentState,
    PlanResult,
    PlanStep,
)
from hagent.world.surprise import latent_distance


# Fallback skeletons when config omits goal_skeletons
_FALLBACK_SKELETONS: Dict[str, List[List[str]]] = {
    "train": [
        ["get_dataset_info", "get_features", "get_available_models", "start_training", "get_job_info"],
        ["list_datasets", "get_dataset_info", "start_training", "get_job_info"],
        ["get_dataset_info", "start_training", "list_jobs", "get_job_info"],
    ],
    "analyze": [
        ["list_datasets", "get_dataset_info", "get_features"],
        ["get_dataset_info", "preview_data", "get_features"],
    ],
    "evaluate": [
        ["list_jobs", "get_job_info"],
        ["get_job_info", "list_jobs"],
    ],
    "monitor": [
        ["list_jobs", "get_job_info"],
        ["get_job_info"],
    ],
    "list": [
        ["list_datasets"],
        ["list_jobs"],
    ],
    "respond": [
        ["get_world_state"],
    ],
}

# Map action type → preferred specialist agent (config can override)
_FALLBACK_ACTION_AGENTS: Dict[str, str] = {
    "list_datasets": "data_analyst",
    "get_dataset_info": "data_analyst",
    "get_features": "data_analyst",
    "preview_data": "data_analyst",
    "get_available_models": "model_selector",
    "get_metrics": "model_selector",
    "start_training": "training_monitor",
    "get_job_info": "training_monitor",
    "list_jobs": "training_monitor",
    "check_system_health": "training_monitor",
    "get_world_state": "data_analyst",
}


def _params_for_action(
    action_type: str,
    goal: GoalSpec,
    context: dict | None,
) -> Dict[str, Any]:
    """Fill action params from goal + observation context (no LLM)."""
    ctx = context or {}
    params: Dict[str, Any] = {}
    dataset_id = goal.get("dataset_id") or ctx.get("dataset_id")
    user_id = ctx.get("user_id")
    if action_type in (
        "get_dataset_info",
        "get_features",
        "preview_data",
        "start_training",
    ):
        if dataset_id:
            params["dataset_id"] = dataset_id
    if action_type == "start_training":
        if user_id:
            params["user_id"] = user_id
        if goal.get("problem_type"):
            params["problem_type"] = goal["problem_type"]
        if goal.get("target_column"):
            params["target_column"] = goal["target_column"]
        if goal.get("metric"):
            params["metric"] = goal["metric"]
        constraints = goal.get("constraints") or {}
        if isinstance(constraints, dict):
            if constraints.get("time_limit") is not None:
                params["time_limit"] = constraints["time_limit"]
            if constraints.get("models"):
                params["models"] = constraints["models"]
            if constraints.get("search_algorithm"):
                params["search_algorithm"] = constraints["search_algorithm"]
    if action_type in ("list_datasets", "list_jobs", "start_training") and user_id:
        params["user_id"] = user_id
    if action_type == "get_available_models" and goal.get("problem_type"):
        params["problem_type"] = goal["problem_type"]
    if action_type == "get_metrics" and goal.get("problem_type"):
        params["problem_type"] = goal["problem_type"]
    if action_type == "get_job_info":
        job_id = ctx.get("job_id") or goal.get("constraints", {}).get("job_id")
        if job_id:
            params["job_id"] = job_id
    return params


class CEMLitePlanner:
    def __init__(self, predictor: Any, config: dict | None = None):
        self.predictor = predictor
        self.config = dict(config or {})
        self.horizon = int(self.config.get("horizon", 4))
        self.n_candidates = int(self.config.get("n_candidates", 8))
        self.n_return = int(self.config.get("n_return_plans", 2))
        weights = self.config.get("cost_weights") or {}
        self.w_latent = float(weights.get("latent_goal", 1.0))
        self.w_constraint = float(weights.get("constraint_violation", 5.0))
        self.w_step = float(weights.get("step_penalty", 0.05))
        self.skeletons: Dict[str, List[List[str]]] = dict(
            self.config.get("goal_skeletons") or _FALLBACK_SKELETONS
        )
        self.action_agents: Dict[str, str] = dict(
            self.config.get("action_agents") or _FALLBACK_ACTION_AGENTS
        )
        self.distance_metric = str(self.config.get("distance_metric") or "l2")

    def _candidate_sequences(
        self,
        goal: GoalSpec,
        action_space: Sequence[str],
    ) -> List[List[str]]:
        gtype = str(goal.get("goal_type") or "respond").lower()
        space = set(action_space)
        skeletons = list(self.skeletons.get(gtype) or self.skeletons.get("respond") or [])

        # Also add truncated / shuffled variants up to n_candidates
        seqs: List[List[str]] = []
        for sk in skeletons:
            filtered = [a for a in sk if a in space]
            if filtered:
                seqs.append(filtered[: self.horizon])

        # Pad with single-step explores from action space if few skeletons
        if len(seqs) < self.n_candidates:
            for a in action_space:
                if len(seqs) >= self.n_candidates:
                    break
                seqs.append([a])

        # Combinations of first tools for diversity
        if len(seqs) < self.n_candidates and len(action_space) >= 2:
            for combo in itertools.islice(
                itertools.permutations(list(action_space)[:5], min(2, self.horizon)),
                self.n_candidates,
            ):
                seq = list(combo)
                if seq not in seqs:
                    seqs.append(seq)
                if len(seqs) >= self.n_candidates:
                    break

        return seqs[: self.n_candidates]

    def _constraint_penalty(
        self,
        actions: List[AutoMLAction],
        goal: GoalSpec,
        context: dict | None,
    ) -> float:
        penalty = 0.0
        gtype = str(goal.get("goal_type") or "")
        types = [a.type for a in actions]
        if gtype == "train":
            if "start_training" not in types:
                penalty += 1.0
            if not goal.get("dataset_id") and not (context or {}).get("dataset_id"):
                if "list_datasets" not in types and "get_dataset_info" not in types:
                    penalty += 0.5
            if goal.get("target_column") is None and "start_training" in types:
                # May still be valid if user already configured — light penalty
                penalty += 0.1
        if gtype == "analyze" and not any(
            t in types for t in ("get_dataset_info", "get_features", "list_datasets")
        ):
            penalty += 0.8
        for a in actions:
            if a.type == "start_training" and not a.params.get("dataset_id"):
                penalty += 0.5
            if a.type == "get_job_info" and not a.params.get("job_id"):
                penalty += 0.2
        return penalty

    def _rollout_cost(
        self,
        z0: LatentState,
        z_goal: LatentState,
        actions: List[AutoMLAction],
        goal: GoalSpec,
        context: dict | None,
    ) -> float:
        z = z0
        for action in actions:
            z = self.predictor.predict(z, action)
        dist = latent_distance(z, z_goal, metric=self.distance_metric)
        pen = self._constraint_penalty(actions, goal, context)
        step_cost = self.w_step * len(actions)
        return self.w_latent * dist + self.w_constraint * pen + step_cost

    def plan(
        self,
        z0: LatentState,
        z_goal: LatentState,
        *,
        goal: GoalSpec,
        action_space: List[str],
        observation_context: dict | None = None,
    ) -> List[PlanResult]:
        if not action_space:
            return []

        sequences = self._candidate_sequences(goal, action_space)
        scored: List[PlanResult] = []

        for seq in sequences:
            actions = [
                AutoMLAction(
                    type=a_type,
                    params=_params_for_action(a_type, goal, observation_context),
                )
                for a_type in seq
            ]
            cost = self._rollout_cost(z0, z_goal, actions, goal, observation_context)
            steps = [
                PlanStep(
                    action=act,
                    agent=self.action_agents.get(act.type),
                )
                for act in actions
            ]
            scored.append(
                PlanResult(
                    plan_id=str(uuid.uuid4()),
                    steps=steps,
                    cost=cost,
                    score_estimate=1.0 / (1.0 + cost),
                    title=f"{goal.get('goal_type', 'plan')}:{'→'.join(seq[:4])}",
                    meta={"action_types": seq},
                )
            )

        scored.sort(key=lambda p: p.cost)
        return scored[: max(1, self.n_return)]
