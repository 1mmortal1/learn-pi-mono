from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Generic, TypeVar

TEvent = TypeVar("TEvent")
TResult = TypeVar("TResult")


class EventStream(Generic[TEvent, TResult]):
    """A small stream object that supports async iteration and a final result."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[TEvent | None] = asyncio.Queue()
        self._result_future: asyncio.Future[TResult] = asyncio.get_running_loop().create_future()

    async def push(self, event: TEvent) -> None:
        await self._queue.put(event)

    async def end(self, result: TResult) -> None:
        if not self._result_future.done():
            self._result_future.set_result(result)
        await self._queue.put(None)

    async def result(self) -> TResult:
        return await self._result_future

    def __aiter__(self) -> AsyncIterator[TEvent]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[TEvent]:
        while True:
            item = await self._queue.get()
            if item is None:
                break
            yield item
