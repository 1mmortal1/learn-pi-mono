from __future__ import annotations

from .base import Tool


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}

        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def clear(self) -> None:
        self._tools.clear()


_global_tool_registry = ToolRegistry()


def register_tool(tool: Tool) -> None:
    _global_tool_registry.register(tool)


def get_tool(name: str) -> Tool | None:
    return _global_tool_registry.get(name)


def list_tools() -> list[Tool]:
    return _global_tool_registry.list()


def clear_tools() -> None:
    _global_tool_registry.clear()
