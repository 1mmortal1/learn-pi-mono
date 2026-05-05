from __future__ import annotations

import asyncio
import time

from ..models import ModelSpec
from ..types import AssistantContentPart, AssistantEvent, AssistantMessage, Context, DoneEvent, StartEvent, TextDeltaEvent, TextPart
from ..utils.event_stream import EventStream


class DummyProvider:
    api = "dummy"

    async def complete(self, model: ModelSpec, context: Context) -> AssistantMessage:
        return await self.stream(model, context).result()

    def stream_simple(self, model: ModelSpec, context: Context) -> EventStream[AssistantEvent, AssistantMessage]:
        return self.stream(model, context)

    def stream(self, model: ModelSpec, context: Context) -> EventStream[AssistantEvent, AssistantMessage]:
        response_stream = EventStream[AssistantEvent, AssistantMessage]()
        reply_chunks = self._build_reply_chunks(model, context)
        text_part = TextPart(text="")
        parts: list[AssistantContentPart] = [text_part]
        final_message = self._build_message(
            model,
            [TextPart(text="".join(reply_chunks))],
        )

        async def produce() -> None:
            start_message = self._build_message(model, parts)
            await response_stream.push(StartEvent(message=start_message))

            for chunk in reply_chunks:
                await asyncio.sleep(0.2)
                text_part.text += chunk
                partial_message = self._build_message(model, parts)
                await response_stream.push(
                    TextDeltaEvent(
                        content_index=0,
                        delta=chunk,
                        message=partial_message,
                    )
                )

            await response_stream.push(DoneEvent(message=final_message))
            await response_stream.end(final_message)

        asyncio.create_task(produce())
        return response_stream

    def _build_reply_chunks(self, model: ModelSpec, context: Context) -> list[str]:
        user_text = ""
        if context.messages:
            last_message = context.messages[-1]
            if hasattr(last_message, "content") and isinstance(last_message.content, list):
                user_text = "".join(part.text for part in last_message.content if hasattr(part, "text"))
        return [
            "Dummy reply ",
            f"from {model.provider}/{model.id}: ",
            f"I received '{user_text}'",
        ]

    def _build_message(self, model: ModelSpec, content: list[AssistantContentPart]) -> AssistantMessage:
        return AssistantMessage(
            content=[part.model_copy(deep=True) for part in content],
            model=model.id,
            provider=model.provider,
            api=model.api,
            timestamp=int(time.time() * 1000),
        )
