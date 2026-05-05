from __future__ import annotations

from .base import Tool

_tool_registry: dict[str, Tool] = {}


def register_tool(tool: Tool) -> None:
    _tool_registry[tool.name] = tool


def get_tool(name: str) -> Tool | None:
    return _tool_registry.get(name)


def list_tools() -> list[Tool]:
    return list(_tool_registry.values())


def clear_tools() -> None:
    _tool_registry.clear()
