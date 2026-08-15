"""Gọi công cụ đã đăng ký bằng loại action và các tham số.

Module này tách bộ thực thi plan khỏi phần nối dây ToolNode của LangChain.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Hook tùy chọn dành cho kiểm thử
_tool_invoker: Callable[[str, dict[str, Any]], Any] | None = None

_USER_ID_SCOPED_ACTIONS = frozenset(
    {
        "get_world_state",
        "list_datasets",
        "list_jobs",
        "start_training",
    }
)
_CREDENTIAL_SCOPED_ACTIONS = frozenset(
    {
        "cancel_job",
        "get_dataset_info",
        "get_features",
        "get_job_info",
        "get_world_state",
        "list_datasets",
        "list_jobs",
        "predict_batch",
        "preview_data",
        "start_training",
    }
)


def _auth_scope_error() -> dict[str, Any]:
    return {
        "error": {
            "code": "AUTH_SCOPE_REQUIRED",
            "message": "Thiếu credential xác thực của request",
        }
    }


def _tool_accepts_parameter(tool: Any, parameter: str) -> bool:
    args = getattr(tool, "args", None)
    return isinstance(args, dict) and parameter in args


def set_tool_invoker(fn: Callable[[str, dict[str, Any]], Any] | None) -> None:
    """Inject mock invoker đồng bộ hoặc bất đồng bộ dành cho kiểm thử."""
    global _tool_invoker
    _tool_invoker = fn


def _parse_tool_output(raw: Any) -> Any:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if hasattr(raw, "content"):
        raw = raw.content
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}
    return {"raw": raw}


def _normalize_tool_output(raw: Any) -> dict[str, Any]:
    payload = _parse_tool_output(raw)
    error = payload.get("error") if isinstance(payload, dict) else None
    if error is None:
        return payload

    raw_code = error.get("code") if isinstance(error, dict) else None
    code = (
        raw_code
        if isinstance(raw_code, str)
        and 1 <= len(raw_code) <= 64
        and all(char.isalnum() or char in "_.-" for char in raw_code)
        else "TOOL_REPORTED_ERROR"
    )
    return {
        "error": {
            "code": code,
            "message": "Tool trả về lỗi khi thực thi",
        }
    }


async def invoke_tool(action_type: str, params: dict[str, Any]) -> dict[str, Any]:
    """Chạy công cụ theo tên và trả về payload dict đã phân tích.

    Hàm không phát sinh ngoại lệ ra ngoài; lỗi được chuyển thành payload ``error``.
    """
    try:
        if _tool_invoker is not None:
            result = _tool_invoker(action_type, params)
            if inspect.isawaitable(result):
                result = await result
            return _normalize_tool_output(result)

        from hagent.agent.orchestration import registry as registry_module

        tmap = registry_module.get_tool_map()
        tool = tmap.get(action_type)
        if tool is None:
            return {"error": f"Unknown tool: {action_type}"}

        if _tool_accepts_parameter(tool, "token") and not (
            isinstance(params.get("token"), str) and params["token"].strip()
        ):
            return _auth_scope_error()

        # Tool LangChain nhận dict input qua ainvoke.
        if hasattr(tool, "ainvoke"):
            result = await tool.ainvoke(params)
        elif hasattr(tool, "invoke"):
            result = tool.invoke(params)
        elif callable(tool):
            result = tool(**params)
            if inspect.isawaitable(result):
                result = await result
        else:
            return {"error": f"Tool {action_type} is not invokable"}

        return _normalize_tool_output(result)
    except Exception as exc:  # noqa: BLE001 - boundary công cụ phải chuẩn hóa mọi lỗi
        logger.error(
            "Gọi tool thất bại action=%s type=%s",
            action_type,
            type(exc).__name__,
        )
        return {
            "error": {
                "code": "TOOL_INVOCATION_FAILED",
                "message": "Không thể thực thi tool",
            }
        }


def enrich_params(
    action_type: str,
    params: dict[str, Any],
    *,
    user_id: str | None,
    user_token: str | None,
    goal: dict[str, Any] | None,
    world_model: dict[str, Any] | None,
    action_id: str | None = None,
) -> dict[str, Any]:
    """Bổ sung user_id, token và dataset_id từ context state đáng tin cậy."""
    out = dict(params or {})
    goal = goal or {}
    wm = world_model or {}

    if action_type in _USER_ID_SCOPED_ACTIONS:
        out.pop("user_id", None)
        if user_id:
            out["user_id"] = user_id

    out.pop("token", None)
    if action_type in _CREDENTIAL_SCOPED_ACTIONS and user_token:
        out["token"] = user_token

    ds = out.get("dataset_id") or goal.get("dataset_id") or wm.get("active_dataset_id")
    if not ds:
        focus = wm.get("focus") or {}
        if isinstance(focus, dict):
            ds = focus.get("dataset_id")
    if (
        ds
        and "dataset_id" not in out
        and action_type
        in (
            "get_dataset_info",
            "get_features",
            "preview_data",
            "start_training",
        )
    ):
        out["dataset_id"] = ds

    if action_type == "start_training":
        out.pop("idempotency_key", None)
        normalized_action_id = str(action_id or "").strip()
        if user_id and 1 <= len(normalized_action_id) <= 512:
            digest = hashlib.sha256(
                f"{user_id}\0{normalized_action_id}".encode()
            ).hexdigest()
            out["idempotency_key"] = f"hagent-{digest}"
        if "problem_type" not in out and goal.get("problem_type"):
            out["problem_type"] = goal["problem_type"]
        if "target_column" not in out and goal.get("target_column"):
            out["target_column"] = goal["target_column"]
        if "metric" not in out and goal.get("metric"):
            out["metric"] = goal["metric"]
        constraints = goal.get("constraints") or {}
        if isinstance(constraints, dict):
            if "time_limit" not in out and constraints.get("time_limit") is not None:
                out["time_limit"] = constraints["time_limit"]
            if "search_algorithm" not in out and constraints.get("search_algorithm"):
                out["search_algorithm"] = constraints["search_algorithm"]
            if "models" not in out and constraints.get("models"):
                out["models"] = constraints["models"]
            if "list_feature" not in out and constraints.get("list_feature"):
                out["list_feature"] = constraints["list_feature"]
        # Lấy features từ world model nếu analyze leaf đã tải trước đó.
        if "list_feature" not in out and ds:
            wm_ds = (wm.get("datasets") or {}).get(str(ds)) or {}
            feats = wm_ds.get("features") or wm_ds.get("list_feature")
            if feats:
                out["list_feature"] = list(feats)
        if "user_id" not in out and user_id:
            out["user_id"] = user_id

    if action_type == "get_job_info" and "job_id" not in out:
        jid = wm.get("active_job_id")
        if jid:
            out["job_id"] = jid

    if action_type in ("get_available_models", "get_metrics"):
        if "problem_type" not in out and goal.get("problem_type"):
            out["problem_type"] = goal["problem_type"]
        out.setdefault("problem_type", "classification")

    return out
