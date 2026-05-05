from __future__ import annotations

from typing import Any

from ..base import Tool


class EchoTool:
    name = "echo"
    description = "Return the same text passed in the arguments."

    async def execute(self, arguments: dict[str, Any]) -> str:
        text = arguments.get("text", "")
        return str(text)
