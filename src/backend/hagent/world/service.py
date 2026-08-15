"""
WorldModelService là facade duy nhất các agent sử dụng theo pipeline kiểu LeWM.

  encode → predict → plan → step(update with surprise)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import structlog

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

logger = structlog.get_logger(__name__)


class WorldModelService:
    """
    Facade gồm Encoder, Predictor, Planner, Surprise và log trajectory tùy chọn.

    Khởi tạo qua WorldModelService.from_config(...).
    """

    def __init__(
        self,
        *,
        encoder: Any,
        predictor: Any,
        planner: Any,
        surprise_config: dict | None = None,
        adaptive_config: dict | None = None,
        trajectory_store: TrajectoryStore | None = None,
        action_space: Sequence[str] | None = None,
        enabled: bool = True,
    ):
        self.encoder = encoder
        self.predictor = predictor
        self.planner = planner
        self.surprise_config = dict(surprise_config or {})
        self.adaptive_config = dict(adaptive_config or {})
        self.trajectory_store = trajectory_store
        self.action_space: list[str] = list(action_space or [])
        self.enabled = enabled
        self._user_adaptive_state: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_config(
        cls,
        config: dict | None = None,
        *,
        trajectory_store: TrajectoryStore | None = None,
        action_space: Sequence[str] | None = None,
        mongo_client: Any | None = None,
        db_name: str | None = None,
    ) -> WorldModelService:
        """
        Tạo dịch vụ từ phần cấu hình world_model.

        Nếu config là None, tải từ hagent.yaml qua bridge.config.
        Khi có mongo_client, các trajectory được lưu bền vững vào Mongo.
        """
        if config is None:
            try:
                from hagent.bridge.config import get_world_model_config

                config = get_world_model_config()
            except (ImportError, KeyError, ValueError, RuntimeError, OSError):
                config = {}

        cfg = dict(config or {})
        enabled = bool(cfg.get("enabled", True))
        encoder = create_encoder(cfg.get("encoder") or {})
        predictor = create_predictor(cfg.get("predictor") or {})
        planner = create_planner(predictor, cfg.get("planner") or {})
        surprise_cfg = cfg.get("surprise") or {}
        adaptive_cfg = cfg.get("adaptive_update") or {}

        traj_cfg = cfg.get("trajectory") or {}
        if trajectory_store is None and traj_cfg.get("enabled", True):
            from hagent.world.trajectory_store import create_trajectory_store

            trajectory_store = create_trajectory_store(
                mongo_client,
                db_name=db_name,
                collection_name=traj_cfg.get("collection"),
                max_per_user=int(traj_cfg.get("max_per_user", 5000)),
                enabled=bool(traj_cfg.get("enabled", True)),
            )

        space = list(action_space or cfg.get("action_space") or [])
        if not space:
            # Không gian action đóng mặc định từ công cụ đã biết và có thể ghi đè bằng YAML.
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
            adaptive_config=adaptive_cfg,
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
    ) -> list[PlanResult]:
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
    ) -> tuple[LatentState, LatentState, LatentState, SurpriseResult]:
        """
        Sau bước môi trường: mã hóa o, dự đoán ẑ', mã hóa o', tính surprise và ghi trajectory.
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
    ) -> tuple[AutoMLObservation, SurpriseResult]:
        """
        Bước môi trường đầy đủ: action → môi trường → observation kế tiếp → cập nhật surprise.

        env là callable đồng bộ hoặc bất đồng bộ nhận action và trả payload công cụ.
        apply_result gộp payload vào observation kế tiếp.
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

    # ── Adaptive Update Scheduling ───────────────────────

    def should_update_adaptive(
        self,
        user_id: str,
        current_surprise: float | SurpriseResult | None = None,
        *,
        force: bool = False,
    ) -> bool:
        """
        Xác định có nên cập nhật trạng thái và động lực học World Model ở bước này hay không.

        - Nếu surprise hoặc momentum cao, cập nhật ở mọi bước với interval tối thiểu bằng 1.
        - Nếu surprise liên tục thấp, giảm tần suất đến interval tối đa.
        - Nếu force là True, trả về True ngay và đặt lại bộ đếm.
        """
        if force or not self.adaptive_config.get("enabled", True):
            if user_id in self._user_adaptive_state:
                self._user_adaptive_state[user_id]["steps_since_update"] = 0
            return True

        min_interval = int(self.adaptive_config.get("min_update_interval", 1))
        max_interval = int(self.adaptive_config.get("max_update_interval", 5))
        decay = float(self.adaptive_config.get("surprise_decay", 0.8))

        state = self._user_adaptive_state.setdefault(
            user_id,
            {
                "steps_since_update": 0,
                "momentum": 0.0,
                "current_interval": min_interval,
            },
        )

        s_val = 0.0
        is_high = False
        if current_surprise is not None:
            if isinstance(current_surprise, SurpriseResult):
                s_val = float(current_surprise.value)
                is_high = current_surprise.level == "high"
            else:
                s_val = float(current_surprise)
                is_high = s_val >= 0.40

            state["momentum"] = decay * state["momentum"] + (1.0 - decay) * s_val

        # Kích hoạt ngay khi surprise hoặc momentum tăng cao.
        if is_high or state["momentum"] >= 0.30:
            state["current_interval"] = min_interval
            state["steps_since_update"] = 0
            return True

        state["steps_since_update"] += 1
        should_update = state["steps_since_update"] >= state["current_interval"]
        if should_update:
            state["steps_since_update"] = 0

        # Điều chỉnh interval cho các bước sau nếu surprise vẫn thấp.
        if current_surprise is not None and s_val < 0.15 and state["momentum"] < 0.20:
            state["current_interval"] = min(max_interval, state["current_interval"] + 1)

        return should_update

    def record_step_surprise(
        self,
        user_id: str,
        surprise: SurpriseResult | float,
    ) -> tuple[float, float]:
        """
        Ghi nhận surprise đã quan sát và trả về surprise_momentum cùng update_frequency mới.
        """
        min_interval = int(self.adaptive_config.get("min_update_interval", 1))
        max_interval = int(self.adaptive_config.get("max_update_interval", 5))
        decay = float(self.adaptive_config.get("surprise_decay", 0.8))

        state = self._user_adaptive_state.setdefault(
            user_id,
            {
                "steps_since_update": 0,
                "momentum": 0.0,
                "current_interval": min_interval,
            },
        )

        if isinstance(surprise, SurpriseResult):
            s_val = float(surprise.value)
            is_high = surprise.level == "high"
        else:
            s_val = float(surprise)
            is_high = s_val >= 0.40

        state["momentum"] = decay * state["momentum"] + (1.0 - decay) * s_val

        if is_high or state["momentum"] >= 0.30:
            state["current_interval"] = min_interval
        elif s_val < 0.15 and state["momentum"] < 0.20:
            state["current_interval"] = min(max_interval, state["current_interval"] + 1)

        update_freq = 1.0 / max(state["current_interval"], 1)
        return float(state["momentum"]), float(update_freq)

    def get_adaptive_state(self, user_id: str) -> dict[str, Any]:
        return dict(
            self._user_adaptive_state.get(
                user_id,
                {
                    "steps_since_update": 0,
                    "momentum": 0.0,
                    "current_interval": int(
                        self.adaptive_config.get("min_update_interval", 1)
                    ),
                },
            )
        )

    def reset_adaptive_state(self, user_id: str | None = None) -> None:
        if user_id is not None:
            self._user_adaptive_state.pop(user_id, None)
        else:
            self._user_adaptive_state.clear()
