"""Interface cho tool adapter và hạ tầng hỗ trợ gọi tool của HAgent."""

from hagent.agent.tools.cache import ToolCache, get_tool_cache, reset_cache

__all__ = ("ToolCache", "get_tool_cache", "reset_cache")
