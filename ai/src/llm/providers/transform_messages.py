from __future__ import annotations

from ..types import AssistantContentPart, AssistantMessage, Context, Message, TextPart, ToolResultMessage, UserMessage


def build_openai_input(context: Context) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []

    if context.system_prompt:
        items.append(
            {
                "role": "system",
                "content": context.system_prompt,
            }
        )

    for message in context.messages:
        items.append(map_message_to_openai(message))

    return items


def map_message_to_openai(message: Message) -> dict[str, str]:
    if isinstance(message, UserMessage):
        return {
            "role": "user",
            "content": flatten_text_parts(message.content),
        }

    if isinstance(message, AssistantMessage):
        return {
            "role": "assistant",
            "content": flatten_assistant_text(message.content),
        }

    if isinstance(message, ToolResultMessage):
        tool_text = flatten_text_parts(message.content)
        return {
            "role": "user",
            "content": f"[tool:{message.tool_name}] {tool_text}",
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
