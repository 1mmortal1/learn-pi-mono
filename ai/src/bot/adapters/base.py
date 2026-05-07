from __future__ import annotations

from typing import Protocol

from ..runtime import BotRuntime


class BotAdapter(Protocol):
    name: str

    async def run(self, runtime: BotRuntime) -> None: ...
