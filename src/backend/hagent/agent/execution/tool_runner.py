"""
Invoke registered tools by action type + params.

Decouples plan executor from LangChain ToolNode wiring.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Optional inject for tests
_tool_invoker: Callable[[str, Dict[str, Any]], Any] | None = None


def set_tool_invoker(fn: Callable[[str, Dict[str, Any]], Any] | None) -> None:
    """Test hook — inject mock invoker (sync or async)."""
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


async def invoke_tool(action_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run tool by name. Returns parsed dict payload.
    Never raises — errors become {"error": "..."}.
    """
    try:
        if _tool_invoker is not None:
            result = _tool_invoker(action_type, params)
            if inspect.isawaitable(result):
                result = await result
            return _parse_tool_output(result)

        from hagent.agent.registry import get_tool_map

        tmap = get_tool_map()
        tool = tmap.get(action_type)
        if tool is None:
            return {"error": f"Unknown tool: {action_type}"}

        # LangChain tools: ainvoke with dict input
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

        return _parse_tool_output(result)
    except Exception as exc:
        logger.exception("Tool invoke failed: %s", action_type)
        return {"error": str(exc)}


def enrich_params(
    action_type: str,
    params: Dict[str, Any],
    *,
    user_id: str | None,
    user_token: str | None,
    goal: Dict[str, Any] | None,
    world_model: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Fill user_id/token/dataset_id from state context."""
    out = dict(params or {})
    goal = goal or {}
    wm = world_model or {}

    if user_id and "user_id" not in out:
        if action_type in (
            "list_datasets",
            "list_jobs",
            "start_training",
            "get_world_state",
        ):
            out["user_id"] = user_id

    if user_token and "token" not in out:
        out["token"] = user_token

    ds = out.get("dataset_id") or goal.get("dataset_id") or wm.get("active_dataset_id")
    if not ds:
        focus = wm.get("focus") or {}
        if isinstance(focus, dict):
            ds = focus.get("dataset_id")
    if ds and "dataset_id" not in out and action_type in (
        "get_dataset_info",
        "get_features",
        "preview_data",
        "start_training",
    ):
        out["dataset_id"] = ds

    if action_type == "start_training":
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
        # Pull features from world model when analyze leaf already loaded them
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
