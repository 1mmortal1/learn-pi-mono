from __future__ import annotations

from typing import Any, Protocol


class Tool(Protocol):
    name: str
    description: str
    parameters: dict[str, Any]

    async def execute(self, arguments: dict[str, Any]) -> str: ...
