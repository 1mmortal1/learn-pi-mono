from __future__ import annotations

import asyncio
import os
import time

from openai import AsyncOpenAI

from ..errors import LLMConfigurationError
from ..models import ModelSpec
from ..types import (
    AssistantContentPart,
    AssistantEvent,
    AssistantMessage,
    Context,
    DoneEvent,
    ErrorEvent,
    StartEvent,
    TextDeltaEvent,
    TextPart,
)
from ..utils.event_stream import EventStream
from .transform_messages import build_openai_input


class OpenAIResponsesProvider:
    api = "openai-responses"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._base_url = base_url

    async def complete(self, model: ModelSpec, context: Context) -> AssistantMessage:
        return await self.stream(model, context).result()

    def stream_simple(self, model: ModelSpec, context: Context) -> EventStream[AssistantEvent, AssistantMessage]:
        return self.stream(model, context)

    def stream(self, model: ModelSpec, context: Context) -> EventStream[AssistantEvent, AssistantMessage]:
        response_stream = EventStream[AssistantEvent, AssistantMessage]()

        async def produce() -> None:
            text_part = TextPart(text="")
            parts: list[AssistantContentPart] = [text_part]

            try:
                client = self._build_client(model)
                input_items = build_openai_input(context)

                start_message = self._build_message(model, parts)
                await response_stream.push(StartEvent(message=start_message))

                # 接 OpenAI 的真实流式调用

                stream = await client.responses.create(
                    model=model.id,
                    input=input_items,
                    stream=True,
                )

                async for event in stream:
                    if event.type == "response.output_text.delta":
                        delta = event.delta or ""
                        text_part.text += delta

                        partial_message = self._build_message(model, parts)
                        await response_stream.push(
                            TextDeltaEvent(
                                content_index=0,
                                delta=delta,
                                message=partial_message,
                            )
                        )

                    elif event.type == "response.completed":
                        final_message = self._build_message(model, parts)
                        await response_stream.push(DoneEvent(message=final_message))
                        await response_stream.end(final_message)
                        return

                final_message = self._build_message(model, parts)
                await response_stream.push(DoneEvent(message=final_message))
                await response_stream.end(final_message)

            except Exception as exc:
                error_message = self._build_message(
                    model,
                    parts,
                    stop_reason="error",
                    error_message=str(exc),
                )
                await response_stream.push(ErrorEvent(message=error_message))
                await response_stream.end(error_message)

        asyncio.create_task(produce())
        return response_stream

    def _build_client(self, model: ModelSpec) -> AsyncOpenAI:
        if not self._api_key:
            raise LLMConfigurationError("OPENAI_API_KEY is required.")
        return AsyncOpenAI(
            api_key=self._api_key,
            base_url=model.base_url or self._base_url,
        )


    def _build_message(
        self,
        model: ModelSpec,
        content: list[AssistantContentPart],
        stop_reason: str = "stop",
        error_message: str | None = None,
    ) -> AssistantMessage:
        return AssistantMessage(
            content=[part.model_copy(deep=True) for part in content],
            model=model.id,
            provider=model.provider,
            api=model.api,
            stop_reason=stop_reason,
            error_message=error_message,
            timestamp=int(time.time() * 1000),
        )
