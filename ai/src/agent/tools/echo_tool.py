from __future__ import annotations

from typing import Any

from ..base import Tool


class EchoTool:
    name = "echo"
    description = "Return the same text passed in the arguments."
    parameters = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to return.",
            },
        },
        "required": ["text"],
        "additionalProperties": False,
    }

    async def execute(self, arguments: dict[str, Any]) -> str:
        text = arguments["text"]
        return str(text)
