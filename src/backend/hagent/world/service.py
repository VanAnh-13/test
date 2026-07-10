"""
WorldModelService — single facade agents use (LeWM-style pipeline).

  encode → predict → plan → step(update with surprise)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from hagent.world.encoder.factory import create_encoder
from hagent.world.planner.factory import create_planner
from hagent.world.predictor.factory import create_predictor
from hagent.world.schema import (
    AutoMLAction,
    AutoMLObservation,
    GoalSpec,
    LatentState,
    PlanResult,
    SurpriseResult,
    WorldState,
)
from hagent.world.surprise import compute_surprise
from hagent.world.trajectory_store import TrajectoryStore

logger = logging.getLogger(__name__)


class WorldModelService:
    """
    Facade: Encoder + Predictor + Planner + Surprise + optional trajectory log.

    Construct via WorldModelService.from_config(...).
    """

    def __init__(
        self,
        *,
        encoder: Any,
        predictor: Any,
        planner: Any,
        surprise_config: dict | None = None,
        trajectory_store: TrajectoryStore | None = None,
        action_space: Sequence[str] | None = None,
        enabled: bool = True,
    ):
        self.encoder = encoder
        self.predictor = predictor
        self.planner = planner
        self.surprise_config = dict(surprise_config or {})
        self.trajectory_store = trajectory_store
        self.action_space: List[str] = list(action_space or [])
        self.enabled = enabled

    @classmethod
    def from_config(
        cls,
        config: dict | None = None,
        *,
        trajectory_store: TrajectoryStore | None = None,
        action_space: Sequence[str] | None = None,
    ) -> "WorldModelService":
        """
        Build service from world_model config section.

        If config is None, load from hagent.yaml via bridge.config.
        """
        if config is None:
            try:
                from hagent.bridge.config import get_world_model_config

                config = get_world_model_config()
            except Exception:
                config = {}

        cfg = dict(config or {})
        enabled = bool(cfg.get("enabled", True))
        encoder = create_encoder(cfg.get("encoder") or {})
        predictor = create_predictor(cfg.get("predictor") or {})
        planner = create_planner(predictor, cfg.get("planner") or {})
        surprise_cfg = cfg.get("surprise") or {}

        traj_cfg = cfg.get("trajectory") or {}
        if trajectory_store is None and traj_cfg.get("enabled", True):
            trajectory_store = TrajectoryStore(
                collection=None,
                max_per_user=int(traj_cfg.get("max_per_user", 5000)),
                enabled=bool(traj_cfg.get("enabled", True)),
            )

        space = list(action_space or cfg.get("action_space") or [])
        if not space:
            # Default closed action space from known tools (overridable in YAML)
            space = list(
                cfg.get("default_action_space")
                or [
                    "list_datasets",
                    "get_dataset_info",
                    "get_features",
                    "preview_data",
                    "get_available_models",
                    "get_metrics",
                    "start_training",
                    "get_job_info",
                    "list_jobs",
                    "check_system_health",
                    "get_world_state",
                    "cancel_job",
                    "predict_batch",
                ]
            )

        return cls(
            encoder=encoder,
            predictor=predictor,
            planner=planner,
            surprise_config=surprise_cfg,
            trajectory_store=trajectory_store,
            action_space=space,
            enabled=enabled,
        )

    # ── Core LeWM operations ─────────────────────────────

    def encode(self, observation: AutoMLObservation) -> LatentState:
        return self.encoder.encode(observation)

    def encode_goal(
        self, goal: GoalSpec, observation: AutoMLObservation
    ) -> LatentState:
        return self.encoder.encode_goal(goal, observation)

    def predict(self, z: LatentState, action: AutoMLAction) -> LatentState:
        return self.predictor.predict(z, action)

    def plan(
        self,
        observation: AutoMLObservation,
        goal: GoalSpec,
        *,
        action_space: Sequence[str] | None = None,
    ) -> List[PlanResult]:
        z0 = self.encode(observation)
        z_goal = self.encode_goal(goal, observation)
        space = list(action_space or self.action_space)
        ctx = {
            "user_id": observation.user_id,
            "dataset_id": (observation.focus or {}).get("dataset_id")
            or goal.get("dataset_id"),
            "job_id": (observation.focus or {}).get("job_id"),
        }
        return self.planner.plan(
            z0,
            z_goal,
            goal=goal,
            action_space=space,
            observation_context=ctx,
        )

    def measure_surprise(
        self, predicted: LatentState, actual: LatentState
    ) -> SurpriseResult:
        return compute_surprise(predicted, actual, self.surprise_config)

    async def update(
        self,
        observation: AutoMLObservation,
        action: AutoMLAction,
        next_observation: AutoMLObservation,
    ) -> Tuple[LatentState, LatentState, LatentState, SurpriseResult]:
        """
        After env step: encode o, predict ẑ', encode o', surprise, log trajectory.
        """
        z = self.encode(observation)
        z_hat = self.predict(z, action)
        z_next = self.encode(next_observation)
        surprise = self.measure_surprise(z_hat, z_next)

        if self.trajectory_store is not None:
            await self.trajectory_store.append(
                user_id=observation.user_id,
                observation=observation,
                action=action,
                next_observation=next_observation,
                z=z,
                z_hat=z_hat,
                z_next=z_next,
                surprise=surprise,
            )
        return z, z_hat, z_next, surprise

    async def step(
        self,
        observation: AutoMLObservation,
        action: AutoMLAction,
        *,
        env: Callable[[AutoMLAction], Any],
        apply_result: Callable[
            [AutoMLObservation, AutoMLAction, Any], AutoMLObservation
        ],
    ) -> Tuple[AutoMLObservation, SurpriseResult]:
        """
        Full env step: action → env → next obs → surprise update.

        env: sync or async callable(action) -> tool payload
        apply_result: merge payload into next observation
        """
        import inspect

        result = env(action)
        if inspect.isawaitable(result):
            result = await result
        next_obs = apply_result(observation, action, result)
        _, _, _, surprise = await self.update(observation, action, next_obs)
        return next_obs, surprise

    # ── Helpers ──────────────────────────────────────────

    def observation_from_state(
        self, state: WorldState, *, goal: GoalSpec | None = None
    ) -> AutoMLObservation:
        return state.to_observation(goal=goal)

    def observation_from_snapshot(
        self,
        snapshot: dict,
        *,
        user_id: str | None = None,
        goal: GoalSpec | None = None,
    ) -> AutoMLObservation:
        uid = user_id or snapshot.get("user_id") or ""
        focus = {
            "dataset_id": snapshot.get("active_dataset_id"),
            "job_id": snapshot.get("active_job_id"),
            "plan_id": snapshot.get("active_plan_id"),
        }
        # Drop empty focus keys
        focus = {k: v for k, v in focus.items() if v}
        return AutoMLObservation(
            user_id=str(uid),
            datasets=dict(snapshot.get("datasets") or {}),
            jobs=dict(snapshot.get("jobs") or {}),
            focus=focus,  # type: ignore[arg-type]
            phase=str(snapshot.get("phase") or "idle"),
            goal=goal or snapshot.get("active_goal"),
        )
