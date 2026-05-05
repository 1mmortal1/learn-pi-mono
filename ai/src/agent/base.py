from __future__ import annotations

from typing import Any, Protocol


class Tool(Protocol):
    name: str
    description: str

    async def execute(self, arguments: dict[str, Any]) -> str:
        ...
