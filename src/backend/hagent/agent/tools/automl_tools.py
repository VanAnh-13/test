"""
DeerFlow-AutoML Tools — HAutoML API wrappers as LangChain Tools.

Replaces the old CLI-based hautoml_tools.py with proper async LangChain
tools that call the HAutoML REST API directly.

"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ── Config-driven helpers ────────────────────────────────


def _get_base_url() -> str:
    """Lấy HAutoML base URL từ config hautoml.base_url."""
    from hagent.bridge.config import get_hautoml_config
    return get_hautoml_config()["base_url"].rstrip("/")


def _get_timeout() -> int:
    """Lấy timeout từ agent config."""
    from hagent.bridge.config import get_agent_config
    return get_agent_config().get("timeout_seconds", 120)


def _auth_headers(token: str | None = None) -> dict[str, str]:
    """Build auth headers — token từ tham số hoặc env var."""
    import os
    tok = token or os.getenv("USER_TOKEN", "")
    if tok:
        return {"Authorization": f"Bearer {tok}"}
    return {}


# ── Tool result cache ────────────────────────────────────

_cache: dict[str, tuple[float, Any]] = {}


def _get_cache_config() -> dict:
    from hagent.bridge.config import get_cache_config
    return get_cache_config()


def _cache_key(path: str, params: dict | None) -> str:
    return f"{path}:{json.dumps(params or {}, sort_keys=True)}"


def _get_cached(key: str) -> Any | None:
    """Trả về cached result nếu còn hiệu lực."""
    cfg = _get_cache_config()
    if not cfg.get("enabled", False):
        return None
    entry = _cache.get(key)
    if entry and (time.time() - entry[0]) < cfg.get("ttl_seconds", 300):
        logger.debug("Cache hit: %s", key)
        return entry[1]
    return None


def _set_cache(key: str, value: Any) -> None:
    """Lưu kết quả vào cache."""
    cfg = _get_cache_config()
    if not cfg.get("enabled", False):
        return
    # Evict nếu quá max_entries
    max_entries = cfg.get("max_entries", 100)
    if len(_cache) >= max_entries:
        # Xóa entry cũ nhất
        oldest_key = min(_cache, key=lambda k: _cache[k][0])
        del _cache[oldest_key]
    _cache[key] = (time.time(), value)


# ── HTTP helpers ─────────────────────────────────────────


async def _api_get(
    path: str,
    *,
    params: dict | None = None,
    token: str | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """GET request tới HAutoML API."""
    if use_cache:
        key = _cache_key(path, params)
        cached = _get_cached(key)
        if cached is not None:
            return cached

    url = f"{_get_base_url()}{path}"
    async with httpx.AsyncClient(timeout=_get_timeout()) as client:
        resp = await client.get(url, params=params, headers=_auth_headers(token))
        resp.raise_for_status()
        result = resp.json()

    if use_cache:
        _set_cache(_cache_key(path, params), result)
    return result


async def _api_post(
    path: str,
    *,
    params: dict | None = None,
    data: dict | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """POST request tới HAutoML API (không cache mutations)."""
    url = f"{_get_base_url()}{path}"
    async with httpx.AsyncClient(timeout=_get_timeout()) as client:
        resp = await client.post(url, params=params, json=data, headers=_auth_headers(token))
        resp.raise_for_status()
        return resp.json()


def _result(data: Any) -> str:
    """Serialize kết quả tool thành JSON string."""
    return json.dumps(data, ensure_ascii=False, default=str)


def _error(exc: Exception) -> str:
    """Serialize lỗi thành JSON string."""
    return json.dumps({"error": str(exc)}, ensure_ascii=False)


# ── Dataset Tools ────────────────────────────────────────


@tool
async def list_datasets(user_id: str, token: str | None = None) -> str:
    """Liệt kê tất cả datasets của người dùng.

    Args:
        user_id: ID của người dùng.
        token: JWT token (tùy chọn, sẽ lấy từ env nếu không truyền).

    Returns:
        Danh sách datasets dạng JSON string.
    """
    try:
        result = await _api_post(
            "/get-list-data-by-userid",
            params={"id": user_id},
            token=token,
        )
        return _result(result)
    except Exception as exc:
        return _error(exc)


@tool
async def get_dataset_info(dataset_id: str, token: str | None = None) -> str:
    """Lấy thông tin chi tiết về một dataset.

    Args:
        dataset_id: ID của dataset cần xem.
        token: JWT token (tùy chọn).

    Returns:
        Thông tin dataset dạng JSON string (features, target, statistics).
    """
    try:
        result = await _api_get(
            "/get-data-info",
            params={"id": dataset_id},
            token=token,
        )
        return _result(result)
    except Exception as exc:
        return _error(exc)


@tool
async def get_available_models(problem_type: str) -> str:
    """Lấy danh sách thuật toán ML khả dụng cho loại bài toán.

    Args:
        problem_type: Loại bài toán — 'classification' hoặc 'regression'.

    Returns:
        Danh sách models, metrics, và hyperparameters dạng JSON string.
    """
    try:
        result = await _api_get(f"/api/v1/available-models/{problem_type}")
        return _result(result)
    except Exception as exc:
        return _error(exc)


async def _fetch_feature_list(
    dataset_id: str,
    *,
    problem_type: str | None = None,
    token: str | None = None,
) -> list[str]:
    """Best-effort feature list for start_training (v2 then legacy)."""
    ptype = problem_type or "classification"
    # v2 features API (preferred)
    try:
        result = await _api_get(
            "/v2/auto/features",
            params={"id_data": dataset_id, "problem_type": ptype},
            token=token,
            use_cache=True,
        )
        if isinstance(result, dict):
            feats = result.get("features") or result.get("list_feature") or []
            # features may be list[str] or list[dict]
            names: list[str] = []
            for f in feats:
                if isinstance(f, str):
                    names.append(f)
                elif isinstance(f, dict):
                    name = f.get("name") or f.get("feature") or f.get("column")
                    if name:
                        names.append(str(name))
            if names:
                return names
    except Exception as exc:
        logger.debug("v2 features fetch failed: %s", exc)

    # Legacy / dataset info
    try:
        result = await _api_get(
            "/get-data-info",
            params={"id": dataset_id},
            token=token,
            use_cache=True,
        )
        if isinstance(result, dict):
            feats = (
                result.get("features")
                or result.get("columns")
                or result.get("list_feature")
                or []
            )
            return [str(x) for x in feats if x]
    except Exception as exc:
        logger.debug("legacy features fetch failed: %s", exc)
    return []


@tool
async def start_training(
    user_id: str,
    dataset_id: str,
    problem_type: str,
    target_column: str,
    models: list[str] | None = None,
    metric: str | None = None,
    time_limit: int = 300,
    search_algorithm: str | None = None,
    list_feature: list[str] | None = None,
    token: str | None = None,
) -> str:
    """Khởi tạo một job training AutoML (distributed v2 API).

    Calls ``POST /v2/auto/jobs/training`` (Kafka-backed pipeline).
    Auto-fetches feature list when not provided.

    Args:
        user_id: ID người dùng.
        dataset_id: ID dataset để train.
        problem_type: 'classification' hoặc 'regression'.
        target_column: Tên cột mục tiêu (target/label).
        models: Danh sách tên model muốn thử (None = backend default).
        metric: Metric đánh giá (None = accuracy/rmse default).
        time_limit: Giới hạn thời gian (giây), mặc định 300s.
        search_algorithm: grid_search | bayesian_search | genetic_algorithm | ...
        list_feature: Feature columns (optional; auto-resolved if missing).
        token: JWT token (tùy chọn).

    Returns:
        Thông tin job đã tạo (bao gồm job_id) dạng JSON string.
    """
    try:
        # Resolve features (required by InputRequest.config.list_feature)
        features = list(list_feature or [])
        if not features:
            features = await _fetch_feature_list(
                dataset_id, problem_type=problem_type, token=token
            )
        # Never train on the target column itself
        features = [f for f in features if str(f) != str(target_column)]
        if not features:
            return _error(
                ValueError(
                    "start_training requires list_feature; "
                    f"could not resolve features for dataset_id={dataset_id}"
                )
            )

        search = search_algorithm or "grid_search"
        metric_sort = metric or (
            "accuracy" if str(problem_type).lower() == "classification" else "rmse"
        )

        body: dict[str, Any] = {
            "id_data": dataset_id,
            "id_user": user_id,
            "config": {
                "choose": search,
                "metric_sort": metric_sort,
                "list_feature": features,
                "target": target_column,
                "problem_type": problem_type,
                "search_algorithm": search,
                "max_time": int(time_limit or 300),
            },
        }
        if models:
            body["config"]["models"] = models

        # Primary: distributed training API (creates job in Mongo + Kafka)
        try:
            result = await _api_post(
                "/v2/auto/jobs/training",
                data=body,
                token=token,
            )
            return _result(result)
        except Exception as v2_exc:
            logger.warning(
                "v2 /jobs/training failed (%s); trying legacy train endpoint",
                v2_exc,
            )

        # Legacy fallback (expects different shape — best effort for older deploys)
        legacy = {
            "data": [],
            "config": {
                "choose": search,
                "metric_sort": metric_sort,
                "list_feature": features,
                "target": target_column,
                "problem_type": problem_type,
                "search_algorithm": search,
                "max_time": int(time_limit or 300),
                "models": models or [],
            },
        }
        result = await _api_post(
            "/train-from-requestbody-json/",
            params={"userId": user_id, "id_data": dataset_id},
            data=legacy,
            token=token,
        )
        return _result(result)
    except Exception as exc:
        return _error(exc)


@tool
async def get_features(
    dataset_id: str,
    problem_type: str | None = None,
    token: str | None = None,
) -> str:
    """Lấy danh sách features của dataset.

    Args:
        dataset_id: ID dataset.
        problem_type: Optional problem type hint.
        token: JWT token (tùy chọn).
    """
    try:
        ptype = problem_type or "classification"
        # Prefer v2 auto features API
        try:
            result = await _api_get(
                "/v2/auto/features",
                params={"id_data": dataset_id, "problem_type": ptype},
                token=token,
            )
            if isinstance(result, dict) and (
                result.get("features") or result.get("list_feature")
            ):
                feats = result.get("features") or result.get("list_feature") or []
                names: list[str] = []
                for f in feats:
                    if isinstance(f, str):
                        names.append(f)
                    elif isinstance(f, dict):
                        name = f.get("name") or f.get("feature") or f.get("column")
                        if name:
                            names.append(str(name))
                return _result(
                    {
                        "dataset_id": dataset_id,
                        "features": names or feats,
                        "problem_type": ptype,
                        **{k: v for k, v in result.items() if k != "features"},
                    }
                )
        except Exception as exc:
            logger.debug("v2 get_features failed: %s", exc)

        # Fallback: dataset info
        result = await _api_get(
            "/get-data-info",
            params={"id": dataset_id},
            token=token,
        )
        if isinstance(result, dict):
            features = (
                result.get("features")
                or result.get("columns")
                or result.get("list_feature")
                or []
            )
            return _result(
                {
                    "dataset_id": dataset_id,
                    "features": features,
                    "problem_type": problem_type or result.get("problem_type"),
                    "target": result.get("target"),
                    **{k: v for k, v in result.items() if k not in ("features",)},
                }
            )
        return _result(result)
    except Exception as exc:
        return _error(exc)


@tool
async def get_metrics(problem_type: str) -> str:
    """Lấy danh sách metric khả dụng theo problem type.

    Args:
        problem_type: classification | regression
    """
    try:
        # Reuse available-models payload which often includes metrics
        result = await _api_get(f"/api/v1/available-models/{problem_type}")
        if isinstance(result, dict) and "metrics" in result:
            return _result(result)
        return _result(
            {
                "problem_type": problem_type,
                "metrics": result if isinstance(result, list) else result,
            }
        )
    except Exception as exc:
        return _error(exc)


@tool
async def preview_data(
    dataset_id: str,
    n_rows: int = 5,
    token: str | None = None,
) -> str:
    """Xem trước vài dòng dữ liệu dataset.

    Args:
        dataset_id: ID dataset.
        n_rows: Số dòng preview (mặc định 5).
        token: JWT token (tùy chọn).
    """
    try:
        result = await _api_get(
            "/get-data-info",
            params={"id": dataset_id, "preview_rows": n_rows},
            token=token,
        )
        if isinstance(result, dict):
            result = {**result, "dataset_id": dataset_id, "id": result.get("id", dataset_id)}
        return _result(result)
    except Exception as exc:
        return _error(exc)


@tool
async def get_world_state(user_id: str, token: str | None = None) -> str:
    """Lấy snapshot World Model của user (datasets/jobs/plans đã biết).

    Args:
        user_id: ID người dùng.
        token: JWT token (tùy chọn, hiện dùng cho auth đồng bộ).
    """
    try:
        # Prefer in-process snapshot if store injected later; else return hint
        # Chat/middleware normally injects world_model into state — this tool
        # is a fallback for agents that need an explicit refresh signal.
        _ = token  # reserved for authenticated store access
        return _result(
            {
                "user_id": user_id,
                "note": (
                    "World state is injected via middleware. "
                    "Call list_datasets/list_jobs to refresh."
                ),
                "datasets": {},
                "jobs": {},
            }
        )
    except Exception as exc:
        return _error(exc)


@tool
async def get_job_info(job_id: str, token: str | None = None) -> str:
    """Lấy trạng thái và kết quả của một training job.

    Args:
        job_id: ID của job cần kiểm tra.
        token: JWT token (tùy chọn).

    Returns:
        Thông tin job (status, best_model, best_score, metrics) dạng JSON string.
    """
    try:
        result = await _api_post(
            "/get-job-info",
            params={"id": job_id},
            token=token,
        )
        return _result(result)
    except Exception as exc:
        return _error(exc)


@tool
async def list_jobs(user_id: str, token: str | None = None) -> str:
    """Liệt kê tất cả training jobs của người dùng.

    Args:
        user_id: ID người dùng.
        token: JWT token (tùy chọn).

    Returns:
        Danh sách jobs dạng JSON string.
    """
    try:
        result = await _api_post(
            "/get-list-job-by-userId",
            params={"user_id": user_id},
            token=token,
        )
        return _result(result)
    except Exception as exc:
        return _error(exc)


@tool
async def check_system_health() -> str:
    """Kiểm tra trạng thái hệ thống HAutoML.

    Returns:
        Thông tin health check dạng JSON string.
    """
    try:
        result = await _api_get("/home", use_cache=False)
        return _result({"status": "ok", **result})
    except Exception as exc:
        return _result({"status": "error", "error": str(exc)})


@tool
async def cancel_job(job_id: str, token: str | None = None) -> str:
    """Hủy / force-stop job training hoặc prediction nếu API hỗ trợ.

    Args:
        job_id: ID job.
        token: JWT token (tùy chọn).
    """
    try:
        # Prefer v2 experiment cancel endpoints when available
        try:
            result = await _api_post(
                f"/v2/auto/{job_id}/cancel",
                token=token,
            )
            return _result(result)
        except Exception:
            result = await _api_post(
                "/cancel-job",
                params={"id": job_id},
                token=token,
            )
            return _result(result)
    except Exception as exc:
        return _error(exc)


@tool
async def predict_batch(
    job_id: str,
    file_path: str | None = None,
    token: str | None = None,
) -> str:
    """Chạy batch prediction trên job đã train (API v2).

    Args:
        job_id: ID job model đã train.
        file_path: Đường dẫn file CSV/XLSX trên server (nếu API nhận path).
                   Nếu None, trả hướng dẫn upload qua /v2/auto/{job_id}/predictions.
        token: JWT token (tùy chọn).
    """
    try:
        if not file_path:
            return _result(
                {
                    "status": "need_upload",
                    "message": (
                        f"Upload file qua POST /v2/auto/{job_id}/predictions "
                        "với multipart file_data."
                    ),
                    "job_id": job_id,
                    "endpoint": f"/v2/auto/{job_id}/predictions",
                }
            )
        # Path-based convenience for agent/tooling environments
        import os

        if not os.path.exists(file_path):
            return _error(FileNotFoundError(f"File không tồn tại: {file_path}"))

        url = f"{_get_base_url()}/v2/auto/{job_id}/predictions"
        filename = os.path.basename(file_path)
        content_type = (
            "text/csv"
            if filename.endswith(".csv")
            else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        async with httpx.AsyncClient(timeout=_get_timeout()) as client:
            with open(file_path, "rb") as f:
                resp = await client.post(
                    url,
                    headers=_auth_headers(token),
                    files={"file_data": (filename, f, content_type)},
                )
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")
            if "application/json" in ctype:
                return _result(resp.json())
            return _result(
                {
                    "status": "success",
                    "job_id": job_id,
                    "content_type": ctype,
                    "size_bytes": len(resp.content),
                    "note": "Binary prediction file returned by API",
                }
            )
    except Exception as exc:
        return _error(exc)


# ── Tool registry ────────────────────────────────────────

ALL_TOOLS = [
    list_datasets,
    get_dataset_info,
    get_features,
    preview_data,
    get_available_models,
    get_metrics,
    start_training,
    get_job_info,
    list_jobs,
    check_system_health,
    get_world_state,
    cancel_job,
    predict_batch,
]

DATASET_TOOLS = [list_datasets, get_dataset_info, get_features, preview_data]
TRAINING_TOOLS = [start_training, get_job_info, list_jobs, cancel_job]
MODEL_TOOLS = [get_available_models, get_metrics]
SYSTEM_TOOLS = [check_system_health, get_world_state]
PREDICT_TOOLS = [predict_batch]
