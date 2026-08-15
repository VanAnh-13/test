"""
Các schema Pydantic cho từng phần cấu hình HAgent.

Mỗi phần trong hagent.yaml có một BaseModel tương ứng.
Việc kiểm tra xảy ra khi tải qua loader.py thay vì lúc chạy, nên lỗi cấu hình
được phát hiện sớm khi khởi động.

Thiết kế:
- Dùng `model_config = ConfigDict(extra="allow")` để tương thích
  ngược với các khóa YAML có thể xuất hiện trong hagent.yaml gốc hoặc
  phần mở rộng trong tương lai chưa được schema hóa.
- Tất cả trường đều có giá trị mặc định để việc tải từng phần vẫn hợp lệ.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ── Hạ tầng và giá trị mặc định ──────────────────────────────────────────────


class BridgeConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    host: str = "0.0.0.0"
    port: int = 9900
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])


class HAutoMLConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    base_url: str = "http://localhost:8585"


class MongoDBConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    connect: str = "localhost:27017"
    db_name: str = "hagent"
    conversation_ttl_hours: int = 24


class AuthConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    secret_key: str = ""
    algorithm: str = "HS256"


class MemoryExtractionConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    from_tools: bool = True
    from_responses: bool = True


class MemoryConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    backend: str = "auto"
    collection: str = "memory_facts"
    storage_dir: str = "./data/memory"
    max_facts: int = 15
    extraction: MemoryExtractionConfig = Field(default_factory=MemoryExtractionConfig)


class MiddlewaresConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    pipeline: list[str] = Field(
        default_factory=lambda: [
            "timing",
            "input_sanitizer",
            "world_model",
            "memory",
        ]
    )


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    level: str = "info"
    file: str = "./logs/hagent.log"


class ErrorMessagesConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    model_memory: str = ""
    model_endpoint: str = ""
    timeout: str = ""
    invalid_response: str = ""
    generic: str = ""
    llm_auth: str = ""
    llm_rate_limit: str = ""


# ── LLM ──────────────────────────────────────────────────────────────────────


class ModelPricingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    input_per_1m: float = 0.0
    output_per_1m: float = 0.0


class UsageTrackingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    pricing: dict[str, ModelPricingConfig] = Field(default_factory=dict)


class LLMModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    provider: str
    model: str
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = 4096
    base_url: str = ""
    default: bool = False


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    default_model: str = "openai-gpt4o-mini"
    usage_tracking: UsageTrackingConfig = Field(default_factory=UsageTrackingConfig)
    models: list[LLMModelConfig] = Field(default_factory=list)


# ── World Model ───────────────────────────────────────────────────────────────


class WorldStateConfig(BaseModel):
    """Phần world_state cấp cao nhất cũ vẫn được hỗ trợ để tương thích."""

    model_config = ConfigDict(extra="allow")

    collection_name: str = "world_states"
    ttl_seconds: int = 86400
    snapshot_size_limit: int = 16384


class EncoderConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    backend: str = "structured_v1"
    dim: int = 64
    feature_extractors: list[str] = Field(default_factory=list)
    phases: list[str] = Field(default_factory=list)
    goal_types: list[str] = Field(default_factory=list)
    job_statuses: list[str] = Field(default_factory=list)


class PredictorConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    backend: str = "tabular_transition_v1"
    default_scale: float = 0.03
    hidden_dim: int = 128
    checkpoint_path: str = "./data/world_model/jepa_v1.npz"
    fallback: str = "tabular_transition_v1"
    k: int = 5
    checkpoint_dir: str = "./data/world_model/dynamics_ensemble"


class OutcomeHeadConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    backend: str = "outcome_head_v1"
    hidden_dim: int = 64
    use_latent: bool = False
    latent_dim: int = 64
    checkpoint_path: str = "./data/world_model/outcome_head_v2.npz"
    time_limit_norm: int = 600
    search_algorithms: list[str] = Field(default_factory=list)
    problem_types: list[str] = Field(
        default_factory=lambda: ["classification", "regression"]
    )
    model_vocab: list[str] = Field(default_factory=list)


class OutcomeEnsembleConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    k: int = 5
    hidden_dim: int = 64
    use_latent: bool = False
    latent_dim: int = 64
    model_vocab: list[str] = Field(default_factory=list)
    checkpoint_dir: str = "./data/world_model/outcome_ensemble_v2"
    time_limit_norm: int = 600
    search_algorithms: list[str] = Field(default_factory=list)


class CampaignPlannerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    backend: str = "cem_config_v1"
    n_candidates: int = 32
    n_iterations: int = 8
    elite_fraction: float = 0.25
    smoothing: float = 0.25
    exploration_weight: float = 0.1
    seed: int = 0
    model_options: list[str] = Field(default_factory=list)
    min_models: int = 1


class CostWeightsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    latent_goal: float = 1.0
    constraint_violation: float = 5.0
    step_penalty: float = 0.05


class PlannerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    backend: str = "cem_lite"
    horizon: int = 4
    n_candidates: int = 8
    n_return_plans: int = 2
    elite_fraction: float = 0.25
    distance_metric: str = "l2"
    cost_weights: CostWeightsConfig = Field(default_factory=CostWeightsConfig)


class SurpriseThresholdsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    medium: float = 0.15
    high: float = 0.40


class SurpriseConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    metric: str = "l2"
    thresholds: SurpriseThresholdsConfig = Field(
        default_factory=SurpriseThresholdsConfig
    )
    outcome_enabled: bool = True
    outcome_thresholds: dict[str, float] = Field(
        default_factory=lambda: {"medium": 1.5, "high": 3.0}
    )
    normalized_thresholds: dict[str, float] = Field(
        default_factory=lambda: {"medium": 1.5, "high": 3.0}
    )
    sigma_floor: float = 0.001


class TrajectoryConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    collection: str = "world_trajectories"
    max_per_user: int = 5000


class WorldModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    predictor: PredictorConfig = Field(default_factory=PredictorConfig)
    outcome_head: OutcomeHeadConfig = Field(default_factory=OutcomeHeadConfig)
    outcome_ensemble: OutcomeEnsembleConfig = Field(
        default_factory=OutcomeEnsembleConfig
    )
    campaign_planner: CampaignPlannerConfig = Field(
        default_factory=CampaignPlannerConfig
    )
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    surprise: SurpriseConfig = Field(default_factory=SurpriseConfig)
    trajectory: TrajectoryConfig = Field(default_factory=TrajectoryConfig)
    state: WorldStateConfig = Field(default_factory=WorldStateConfig)
    default_action_space: list[str] = Field(default_factory=list)


# ── Agent và điều phối ───────────────────────────────────────────────────────


class PlanningConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    ground_on_world_model: bool = True
    execute_plans: bool = True
    max_revisions: int = 2
    revise_on_high_surprise: bool = True
    surprise_ignore_actions: list[str] = Field(default_factory=list)
    skip_planner_for_simple_queries: bool = True
    simple_query_keywords: list[str] = Field(default_factory=list)


class SurpriseExtensionConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    max_rounds: int = 1
    n_extra: int = 2
    exploration_weight: float = 0.5


class CampaignConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    n_job_candidates: int = 3
    max_concurrent_jobs: int = 2
    warm_start_top_k: int = 3
    prefer_for_goal_types: list[str] = Field(default_factory=lambda: ["train"])
    search_algorithms: list[str] = Field(
        default_factory=lambda: [
            "grid_search",
            "bayesian_search",
            "genetic_algorithm",
        ]
    )
    time_limit_options: list[int] = Field(default_factory=lambda: [180, 300, 600])
    max_monitor_ticks: int = 50
    wm_variant_proposal: bool = True
    wm_rank_variants: bool = True
    surprise_extension: SurpriseExtensionConfig = Field(
        default_factory=SurpriseExtensionConfig
    )


class HierarchyConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    live_controller: bool = True
    smart_skip: bool = True
    abort_on_leaf_fail: bool = False
    templates: dict[str, Any] = Field(default_factory=dict)


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    default_modes: list[str] = Field(
        default_factory=lambda: [
            "single_shot",
            "plan_executor",
            "campaign",
            "hierarchical",
        ]
    )
    default_tags: list[str] = Field(default_factory=lambda: ["tabular"])


class HarnessConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    default_layers: list[str] = Field(default_factory=lambda: ["offline", "graph"])
    default_offline_modes: list[str] = Field(
        default_factory=lambda: [
            "single_shot",
            "plan_executor",
            "campaign",
            "hierarchical",
        ]
    )
    smoke_tags: list[str] = Field(default_factory=lambda: ["smoke"])


class SubagentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    module: str
    node_function: str
    prompt_file: str = ""
    tools: list[str] = Field(default_factory=list)


class RoutingKeywordsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    keywords: list[str] = Field(default_factory=list)


class CacheConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    ttl_seconds: int = 300
    max_entries: int = 100


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    system_prompt_path: str = "./prompts/coordinator.md"
    soul_path: str = "./SOUL.md"
    planning: PlanningConfig = Field(default_factory=PlanningConfig)
    campaign: CampaignConfig = Field(default_factory=CampaignConfig)
    hierarchy: HierarchyConfig = Field(default_factory=HierarchyConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    harness: HarnessConfig = Field(default_factory=HarnessConfig)
    subagents: dict[str, SubagentConfig] = Field(default_factory=dict)
    routing: dict[str, RoutingKeywordsConfig] = Field(default_factory=dict)
    suggestions: list[str] = Field(default_factory=list)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    max_iterations: int = 10
    timeout_seconds: int = 120


# ── Cấu hình gốc có kiểu ─────────────────────────────────────────────────────


class HAgentConfig(BaseModel):
    """Đối tượng cấu hình gốc kết hợp tất cả các phần."""

    model_config = ConfigDict(extra="allow")

    bridge: BridgeConfig = Field(default_factory=BridgeConfig)
    hautoml: HAutoMLConfig = Field(default_factory=HAutoMLConfig)
    mongodb: MongoDBConfig = Field(default_factory=MongoDBConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    middlewares: MiddlewaresConfig = Field(default_factory=MiddlewaresConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    error_messages: ErrorMessagesConfig = Field(default_factory=ErrorMessagesConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    world_state: WorldStateConfig = Field(default_factory=WorldStateConfig)
    world_model: WorldModelConfig = Field(default_factory=WorldModelConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
