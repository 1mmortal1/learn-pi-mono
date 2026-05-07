from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

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
    ToolCallEvent,
    ToolCallPart,
)
from ..utils.event_stream import EventStream
from .transform_messages import build_openai_input


class OpenAIChatCompletionsProvider:
    api = "openai-chat-completions"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url

    async def complete(self, model: ModelSpec, context: Context) -> AssistantMessage:
        return await self.stream(model, context).result()

    def stream_simple(
        self, model: ModelSpec, context: Context
    ) -> EventStream[AssistantEvent, AssistantMessage]:
        return self.stream(model, context)

    def stream(
        self, model: ModelSpec, context: Context
    ) -> EventStream[AssistantEvent, AssistantMessage]:
        response_stream = EventStream[AssistantEvent, AssistantMessage]()

        async def produce() -> None:
            text_part = TextPart(text="")
            parts: list[AssistantContentPart] = [text_part]
            tool_call_deltas: dict[int, dict[str, str]] = {}

            try:
                client = self._build_client(model)
                messages = build_openai_input(context)

                start_message = self._build_message(model, parts)
                await response_stream.push(StartEvent(message=start_message))

                create_kwargs = self._build_create_kwargs(model, context, messages)
                stream = await client.chat.completions.create(**create_kwargs)

                async for chunk in stream:
                    if not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta
                    text_delta = getattr(delta, "content", None) or ""
                    if text_delta:
                        text_part.text += text_delta
                        partial_message = self._build_message(model, parts)
                        await response_stream.push(
                            TextDeltaEvent(
                                content_index=0,
                                delta=text_delta,
                                message=partial_message,
                            )
                        )

                    for tool_call_delta in getattr(delta, "tool_calls", None) or []:
                        self._collect_tool_call_delta(
                            tool_call_deltas,
                            tool_call_delta,
                        )

                for index in sorted(tool_call_deltas):
                    tool_call = self._build_tool_call_part(
                        index,
                        tool_call_deltas[index],
                    )
                    parts.append(tool_call)
                    partial_message = self._build_message(model, parts)
                    await response_stream.push(
                        ToolCallEvent(
                            content_index=len(parts) - 1,
                            tool_call=tool_call,
                            message=partial_message,
                        )
                    )

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
        api_key = self._resolve_api_key(model)
        if not api_key:
            env_name = model.compat.get("api_key_env", "OPENAI_API_KEY")
            raise LLMConfigurationError(f"{env_name} is required.")

        return AsyncOpenAI(
            api_key=api_key,
            base_url=model.base_url or self._base_url,
        )

    def _resolve_api_key(self, model: ModelSpec) -> str | None:
        env_name = model.compat.get("api_key_env")
        if isinstance(env_name, str):
            return self._api_key or os.getenv(env_name)
        return self._api_key or os.getenv("OPENAI_API_KEY")

    def _build_create_kwargs(
        self,
        model: ModelSpec,
        context: Context,
        messages: list[dict[str, Any]],
    ) -> dict[str, object]:
        create_kwargs: dict[str, object] = {
            "model": model.id,
            "messages": messages,
            "stream": True,
        }

        if model.max_output_tokens is not None:
            create_kwargs["max_tokens"] = model.max_output_tokens

        if model.supports_tools:
            tools = self._build_openai_tools_from_context(context)
            if tools:
                create_kwargs["tools"] = tools

        extra_body = model.compat.get("extra_body")
        if isinstance(extra_body, dict):
            create_kwargs["extra_body"] = extra_body

        return create_kwargs

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

    def _build_openai_tools_from_context(
        self,
        context: Context,
    ) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in context.tools
        ]

    def _collect_tool_call_delta(
        self,
        tool_call_deltas: dict[int, dict[str, str]],
        tool_call_delta: Any,
    ) -> None:
        index = getattr(tool_call_delta, "index", 0)
        entry = tool_call_deltas.setdefault(
            index,
            {
                "id": "",
                "name": "",
                "arguments": "",
            },
        )

        tool_call_id = getattr(tool_call_delta, "id", None)
        if tool_call_id:
            entry["id"] = tool_call_id

        function = getattr(tool_call_delta, "function", None)
        if function is None:
            return

        name_delta = getattr(function, "name", None)
        if name_delta:
            entry["name"] += name_delta

        arguments_delta = getattr(function, "arguments", None)
        if arguments_delta:
            entry["arguments"] += arguments_delta

    def _build_tool_call_part(
        self,
        index: int,
        data: dict[str, str],
    ) -> ToolCallPart:
        raw_arguments = data.get("arguments", "")
        try:
            arguments = json.loads(raw_arguments) if raw_arguments else {}
        except json.JSONDecodeError:
            arguments = {}

        return ToolCallPart(
            id=data.get("id") or f"tool_call_{index}",
            name=data.get("name", ""),
            arguments=arguments,
        )
