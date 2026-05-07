from __future__ import annotations

import json
import time
from typing import Any

from ..types import (
    AssistantContentPart,
    AssistantMessage,
    Context,
    Message,
    TextPart,
    ToolCallPart,
    ToolResultMessage,
    UserMessage,
)


def build_openai_input(context: Context) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    if context.system_prompt:
        items.append(
            {
                "role": "system",
                "content": context.system_prompt,
            }
        )

    for message in normalize_tool_result_pairs(context.messages):
        items.append(map_message_to_openai(message))

    return items


def normalize_tool_result_pairs(messages: list[Message]) -> list[Message]:
    result: list[Message] = []
    pending_tool_calls: list[ToolCallPart] = []

    for message in messages:
        if isinstance(message, ToolResultMessage):
            if any(
                tool_call.id == message.tool_call_id for tool_call in pending_tool_calls
            ):
                result.append(message)
                pending_tool_calls = [
                    tool_call
                    for tool_call in pending_tool_calls
                    if tool_call.id != message.tool_call_id
                ]
            else:
                result.append(build_orphan_tool_result_note(message))
            continue

        if pending_tool_calls:
            result.extend(
                build_synthetic_tool_result(tool_call)
                for tool_call in pending_tool_calls
            )
            pending_tool_calls = []

        result.append(message)

        if isinstance(message, AssistantMessage):
            pending_tool_calls = collect_tool_calls(message.content)

    if pending_tool_calls:
        result.extend(
            build_synthetic_tool_result(tool_call) for tool_call in pending_tool_calls
        )

    return result


def build_synthetic_tool_result(tool_call: ToolCallPart) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=[TextPart(text="No result provided")],
        is_error=True,
        timestamp=int(time.time() * 1000),
    )


def build_orphan_tool_result_note(message: ToolResultMessage) -> UserMessage:
    text = flatten_text_parts(message.content)
    return UserMessage(
        content=[
            TextPart(
                text=(
                    f"[orphaned tool result:{message.tool_name}] "
                    f"{text or 'No content'}"
                )
            )
        ],
        timestamp=message.timestamp,
    )


def map_message_to_openai(message: Message) -> dict[str, Any]:
    if isinstance(message, UserMessage):
        return {
            "role": "user",
            "content": flatten_text_parts(message.content),
        }

    if isinstance(message, AssistantMessage):
        item: dict[str, Any] = {
            "role": "assistant",
            "content": flatten_assistant_text(message.content),
        }
        tool_calls = collect_tool_calls(message.content)
        if tool_calls:
            item["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json_dumps(tool_call.arguments),
                    },
                }
                for tool_call in tool_calls
            ]
        return item

    if isinstance(message, ToolResultMessage):
        tool_text = flatten_text_parts(message.content)
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": tool_text,
        }

    raise ValueError(f"Unsupported message type: {type(message)!r}")


def flatten_text_parts(parts: list[TextPart]) -> str:
    return "".join(part.text for part in parts)


def flatten_assistant_text(parts: list[AssistantContentPart]) -> str:
    chunks: list[str] = []

    for part in parts:
        if part.type == "text":
            chunks.append(part.text)
        elif part.type == "thinking":
            continue
        elif part.type == "tool_call":
            continue

    return "".join(chunks)


def collect_tool_calls(parts: list[AssistantContentPart]) -> list[ToolCallPart]:
    tool_calls: list[ToolCallPart] = []

    for part in parts:
        if part.type == "tool_call":
            tool_calls.append(part)

    return tool_calls


def json_dumps(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False)
