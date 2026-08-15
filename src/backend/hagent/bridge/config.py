"""
HAgent Bridge — Cấu hình hệ thống

Backward-compatible facade trên hagent.config.loader.

Toàn bộ các hàm public (`get_bridge_config`, `get_hautoml_config`, ...)
giữ nguyên signature và hành vi. Consumer code không cần thay đổi.

Implementation detail:
    `load_config()` nay delegate sang `hagent.config.loader.load_raw_config()`
    thay vì tải trực tiếp hagent.yaml — cho phép sử dụng đồng thời các file
    YAML dạng module trong config/ và config/hagent.yaml (module ưu tiên hơn).
"""

import os
from functools import lru_cache
from pathlib import Path

from hagent.config.loader import load_raw_config as _load_raw_config

# ── Đường dẫn ────────────────────────────────────────────

# Tìm file hagent.yaml — ưu tiên biến môi trường, sau đó tìm tự động.
_PACKAGE_DIR = Path(__file__).parent.parent
_DEFAULT_CONFIG_PATHS = [
    _PACKAGE_DIR / "config" / "hagent.yaml",  # vị trí chuẩn trong package
    _PACKAGE_DIR / "hagent.yaml",  # vị trí cũ hagent/hagent.yaml
    Path(__file__).parent.parent.parent / "hagent.yaml",  # backend/hagent.yaml
    Path.home() / ".hagent" / "hagent.yaml",  # ~/.hagent/hagent.yaml
]


def _find_config_path() -> Path:
    """Tìm file cấu hình hagent.yaml theo thứ tự ưu tiên."""
    # Ưu tiên biến môi trường
    env_path = os.getenv("HAGENT_CONFIG")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"HAGENT_CONFIG trỏ tới file không tồn tại: {env_path}")

    # Tìm tự động
    for p in _DEFAULT_CONFIG_PATHS:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Không tìm thấy hagent.yaml. "
        "Đặt biến HAGENT_CONFIG hoặc đặt file tại: "
        + ", ".join(str(p) for p in _DEFAULT_CONFIG_PATHS)
    )


# ── Tải cấu hình ────────────────────────────────────────


@lru_cache
def load_config() -> dict:
    """
    Tải và cache cấu hình.

    Chuyển tiếp sang ``hagent.config.loader.load_raw_config()`` để tải các file
    YAML dạng module và/hoặc config/hagent.yaml, sau đó phân giải biến môi
    trường. Chữ ký được giữ nguyên để tương thích ngược.
    """
    return _load_raw_config()


# ── Các hàm truy xuất cấu hình ──────────────────────────


def get_bridge_config() -> dict:
    """Lấy cấu hình Bridge service."""
    cfg = load_config()
    bridge = cfg.get("bridge", {})
    # Cho phép ghi đè qua biến môi trường
    bridge["host"] = os.getenv("BRIDGE_HOST", bridge.get("host", "0.0.0.0"))
    bridge["port"] = int(os.getenv("BRIDGE_PORT", bridge.get("port", 9900)))
    bridge["cors_origins"] = bridge.get("cors_origins", ["http://localhost:3000"])
    return bridge


def get_hautoml_config() -> dict:
    """Lấy cấu hình HAutoML backend."""
    cfg = load_config()
    h = cfg.get("hautoml", {})
    h["base_url"] = os.getenv(
        "HAUTOML_BASE_URL", h.get("base_url", "http://localhost:8080")
    )
    return h


def get_mongodb_config() -> dict:
    """Lấy cấu hình MongoDB."""
    cfg = load_config()
    m = cfg.get("mongodb", {})
    m["connect"] = os.getenv("MONGODB_CONNECT", m.get("connect", "localhost:27017"))
    m["db_name"] = os.getenv("MONGODB_DB_NAME", m.get("db_name", "hagent"))
    m["conversation_ttl_hours"] = int(
        os.getenv("CONVERSATION_TTL_HOURS", m.get("conversation_ttl_hours", 24))
    )
    return m


def get_auth_config() -> dict:
    """Lấy cấu hình JWT authentication."""
    cfg = load_config()
    a = cfg.get("auth", {})
    a["secret_key"] = os.getenv("SECRET_KEY", a.get("secret_key", ""))
    a["algorithm"] = os.getenv("ALGORITHM", a.get("algorithm", "HS256"))
    return a


def get_world_state_config() -> dict:
    """Lấy cấu hình World State (legacy section + world_model.state)."""
    cfg = load_config()
    ws = cfg.get("world_state", {}) or {}
    # Prefer nested world_model.state when present
    wm_state = (cfg.get("world_model") or {}).get("state") or {}
    merged = {**ws, **wm_state}
    merged["collection_name"] = os.getenv(
        "WORLD_STATE_COLLECTION",
        merged.get("collection_name", "world_states"),
    )
    merged["ttl_seconds"] = int(
        os.getenv(
            "WORLD_STATE_TTL_SECONDS",
            merged.get("ttl_seconds", 86400),
        )
    )
    merged["snapshot_size_limit"] = int(
        os.getenv(
            "WORLD_STATE_SNAPSHOT_SIZE_LIMIT",
            merged.get("snapshot_size_limit", 16384),
        )
    )
    return merged


def get_world_model_config() -> dict:
    """
    Lấy cấu hình LeWM-style World Model (encoder/predictor/planner/surprise).

    Section: world_model in hagent.yaml
    """
    cfg = load_config()
    wm = dict(cfg.get("world_model") or {})
    wm.setdefault("enabled", True)
    wm.setdefault("encoder", {"backend": "structured_v1", "dim": 64})
    wm.setdefault("predictor", {"backend": "tabular_transition_v1"})
    wm.setdefault(
        "planner",
        {
            "backend": "cem_lite",
            "horizon": 4,
            "n_candidates": 8,
            "n_return_plans": 2,
        },
    )
    wm.setdefault(
        "surprise",
        {"metric": "l2", "thresholds": {"medium": 0.15, "high": 0.40}},
    )
    wm.setdefault(
        "trajectory",
        {"enabled": True, "collection": "world_trajectories", "max_per_user": 5000},
    )
    return wm


def get_planning_config() -> dict:
    """Lấy agent.planning config."""
    agent = get_agent_config()
    planning = dict(agent.get("planning") or {})
    planning.setdefault("enabled", True)
    planning.setdefault("ground_on_world_model", True)
    planning.setdefault("execute_plans", True)
    planning.setdefault("max_revisions", 2)
    planning.setdefault("skip_planner_for_simple_queries", True)
    return planning


def get_campaign_config() -> dict:
    """Phase 6 multi-candidate job campaign config."""
    agent = get_agent_config()
    camp = dict(agent.get("campaign") or {})
    camp.setdefault("enabled", True)
    camp.setdefault("n_job_candidates", 3)
    camp.setdefault("max_concurrent_jobs", 2)
    camp.setdefault("warm_start_top_k", 3)
    camp.setdefault(
        "search_algorithms",
        ["grid_search", "bayesian_search", "genetic_algorithm"],
    )
    camp.setdefault("time_limit_options", [180, 300, 600])
    camp.setdefault("prefer_for_goal_types", ["train"])
    camp.setdefault("max_monitor_ticks", 50)
    return camp


def get_hierarchy_config() -> dict:
    """Hierarchical goal decomposition + live controller config."""
    agent = get_agent_config()
    h = dict(agent.get("hierarchy") or {})
    h.setdefault("enabled", True)
    h.setdefault("live_controller", True)
    h.setdefault("smart_skip", True)
    h.setdefault("abort_on_leaf_fail", False)
    h.setdefault(
        "templates",
        {
            "train": [
                {"goal_type": "analyze", "description": "Inspect dataset features"},
                {"goal_type": "select", "description": "Select models/metrics"},
                {"goal_type": "train", "description": "Run training campaign/jobs"},
                {"goal_type": "evaluate", "description": "Compare results"},
            ],
            "evaluate": [
                {"goal_type": "monitor", "description": "List/check jobs"},
                {"goal_type": "evaluate", "description": "Compare best models"},
            ],
        },
    )
    return h


def get_eval_config() -> dict:
    """Phase 7 offline eval harness defaults."""
    agent = get_agent_config()
    e = dict(agent.get("eval") or {})
    e.setdefault(
        "default_modes",
        ["single_shot", "plan_executor", "campaign", "hierarchical"],
    )
    e.setdefault("default_tags", ["tabular"])
    return e


# ── HAgent config accessors ────────────────────


def get_llm_config() -> dict:
    """Lấy cấu hình LLM providers."""
    cfg = load_config()
    llm = cfg.get("llm", {}) or {}
    llm["default_model"] = os.getenv("LLM_DEFAULT_MODEL", llm.get("default_model", ""))
    return llm


def get_llm_models() -> list[dict]:
    """Lấy danh sách model configs đã resolve env vars."""
    llm = get_llm_config()
    return llm.get("models", [])


def get_agent_config() -> dict:
    """Lấy cấu hình agent orchestration."""
    cfg = load_config()
    agent = cfg.get("agent", {}) or {}
    agent["max_iterations"] = int(
        os.getenv("AGENT_MAX_ITERATIONS", agent.get("max_iterations", 10))
    )
    agent["timeout_seconds"] = int(
        os.getenv("AGENT_TIMEOUT_SECONDS", agent.get("timeout_seconds", 120))
    )
    return agent


def get_subagents_config() -> dict:
    """
    Lấy cấu hình sub-agents registry từ agent.subagents.

    Returns:
        Dict[agent_name, agent_config] từ YAML.
    """
    agent = get_agent_config()
    return agent.get("subagents", {}) or {}


def get_routing_config() -> dict[str, list[str]]:
    """
    Lấy routing keywords cho từng sub-agent.

    Returns:
        Dict[agent_name, list[keyword]], ví dụ:
        {"data_analyst": ["dataset", "data", ...], ...}
    """
    agent = get_agent_config()
    routing_raw = agent.get("routing", {}) or {}
    result = {}
    for agent_name, conf in routing_raw.items():
        if isinstance(conf, dict):
            result[agent_name] = conf.get("keywords", [])
        elif isinstance(conf, list):
            result[agent_name] = conf
    return result


def get_suggestions() -> list[str]:
    """Lấy danh sách gợi ý chat mặc định."""
    agent = get_agent_config()
    return agent.get("suggestions", [])


def get_cache_config() -> dict:
    """Lấy cấu hình cache cho tool results."""
    agent = get_agent_config()
    cache = agent.get("cache", {}) or {}
    cache["enabled"] = os.getenv(
        "AGENT_CACHE_ENABLED", str(cache.get("enabled", True))
    ).lower() in ("true", "1", "yes")
    cache["ttl_seconds"] = int(
        os.getenv("AGENT_CACHE_TTL", cache.get("ttl_seconds", 300))
    )
    cache["max_entries"] = int(
        os.getenv("AGENT_CACHE_MAX_ENTRIES", cache.get("max_entries", 100))
    )
    return cache


def get_error_messages() -> dict[str, str]:
    """Lấy cấu hình thông báo lỗi (section error_messages top-level)."""
    cfg = load_config()
    return dict(cfg.get("error_messages", {}) or {})


def load_prompt_file(relative_path: str | None = None) -> str:
    """
    Đọc nội dung file prompt (.md) — tương đối so với thư mục hagent/.

    Args:
        relative_path: Đường dẫn tương đối từ thư mục chứa hagent.yaml.
                       Nếu None, lấy từ agent.system_prompt_path trong config.

    Returns:
        Nội dung file prompt dạng string.
    """
    if not relative_path:
        agent = get_agent_config()
        relative_path = agent.get("system_prompt_path", "./prompts/coordinator.md")

    # Ưu tiên prompt cạnh file cấu hình ngoài; cấu hình đóng gói dùng hagent/prompts.
    config_dir = _find_config_path().parent
    prompt_candidates = (config_dir / relative_path, _PACKAGE_DIR / relative_path)
    prompt_path = next(
        (candidate for candidate in prompt_candidates if candidate.exists()),
        prompt_candidates[0],
    )

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file prompt tại {prompt_path}. "
            f"Kiểm tra agent.system_prompt_path trong hagent.yaml."
        )

    return prompt_path.read_text(encoding="utf-8")
