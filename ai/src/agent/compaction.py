from __future__ import annotations

import json
import math
import time

from pydantic import BaseModel

from ..llm.models import ModelSpec
from ..llm.stream import complete_simple
from ..llm.types import (
    AssistantMessage,
    Context,
    Message,
    TextPart,
    ToolResultMessage,
    UserMessage,
)

SUMMARY_PREFIX = "[context summary]\n"

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Your task is to read a "
    "conversation between a user and an AI coding assistant, then produce a "
    "structured summary following the exact format specified.\n\n"
    "Do NOT continue the conversation. Do NOT respond to any questions in the "
    "conversation. ONLY output the structured summary."
)

SUMMARIZATION_PROMPT = """The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish?]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

UPDATE_SUMMARIZATION_PROMPT = """The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section based on what was accomplished
- UPDATE Next Steps based on the current state
- PRESERVE exact file paths, function names, and error messages

Use the same structured format as the previous summary."""


class CompactionSettings(BaseModel):
    enabled: bool = True
    reserve_tokens: int = 16_384
    keep_recent_tokens: int = 20_000


async def compact_context_in_place(
    model: ModelSpec,
    context: Context,
    settings: CompactionSettings,
) -> None:
    if not settings.enabled or model.context_window is None:
        return

    tokens = estimate_context_tokens(context.messages)
    if tokens <= model.context_window - settings.reserve_tokens:
        return

    cut_index = find_cut_index(context.messages, settings.keep_recent_tokens)
    if cut_index <= 0:
        return

    old_messages = context.messages[:cut_index]
    recent_messages = context.messages[cut_index:]
    previous_summary = find_previous_summary(old_messages)
    messages_to_summarize = [
        message for message in old_messages if not is_summary_message(message)
    ]

    if not messages_to_summarize:
        return

    summary = await generate_summary(
        model,
        messages_to_summarize,
        previous_summary=previous_summary,
    )
    summary_message = build_summary_message(summary)

    context.messages[:] = [summary_message, *recent_messages]


def estimate_context_tokens(messages: list[Message]) -> int:
    return sum(estimate_tokens(message) for message in messages)


def estimate_tokens(message: Message) -> int:
    chars = 0

    if isinstance(message, UserMessage):
        chars += sum(len(part.text) for part in message.content)

    elif isinstance(message, AssistantMessage):
        for part in message.content:
            if part.type == "text":
                chars += len(part.text)
            elif part.type == "thinking":
                chars += len(part.thinking)
            elif part.type == "tool_call":
                chars += len(part.name) + len(json.dumps(part.arguments))

    elif isinstance(message, ToolResultMessage):
        chars += sum(len(part.text) for part in message.content)

    return math.ceil(chars / 4)


def find_cut_index(messages: list[Message], keep_recent_tokens: int) -> int:
    total = 0

    for index in range(len(messages) - 1, -1, -1):
        total += estimate_tokens(messages[index])
        if total < keep_recent_tokens:
            continue

        for cut in range(index, len(messages)):
            if isinstance(messages[cut], UserMessage) and not is_summary_message(
                messages[cut]
            ):
                return normalize_cut_index_for_tool_pairs(messages, cut)

        return normalize_cut_index_for_tool_pairs(messages, index)

    return 0


def normalize_cut_index_for_tool_pairs(
    messages: list[Message],
    cut_index: int,
) -> int:
    while cut_index < len(messages) and isinstance(
        messages[cut_index], ToolResultMessage
    ):
        owner_index = find_tool_call_owner_index(messages, cut_index)
        if owner_index is not None:
            return owner_index
        cut_index += 1

    return cut_index


def find_tool_call_owner_index(
    messages: list[Message],
    tool_result_index: int,
) -> int | None:
    tool_result = messages[tool_result_index]
    if not isinstance(tool_result, ToolResultMessage):
        return None

    for index in range(tool_result_index - 1, -1, -1):
        message = messages[index]
        if isinstance(message, UserMessage):
            return None
        if not isinstance(message, AssistantMessage):
            continue
        for part in message.content:
            if part.type == "tool_call" and part.id == tool_result.tool_call_id:
                return index
        return None

    return None


async def generate_summary(
    model: ModelSpec,
    messages: list[Message],
    *,
    previous_summary: str | None = None,
) -> str:
    conversation_text = serialize_conversation(messages)
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n"

    if previous_summary:
        prompt_text += (
            f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
        )
        prompt_text += UPDATE_SUMMARIZATION_PROMPT
    else:
        prompt_text += SUMMARIZATION_PROMPT

    response = await complete_simple(
        model,
        Context(
            system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
            messages=[
                UserMessage(
                    content=[TextPart(text=prompt_text)],
                    timestamp=now_ms(),
                )
            ],
        ),
    )

    if response.stop_reason == "error":
        raise RuntimeError(response.error_message or "Compaction summarization failed.")

    return "\n".join(
        part.text for part in response.content if part.type == "text"
    ).strip()


def serialize_conversation(messages: list[Message]) -> str:
    parts: list[str] = []

    for message in messages:
        if isinstance(message, UserMessage):
            text = flatten_text_parts(message.content)
            if text:
                parts.append(f"[User]: {text}")

        elif isinstance(message, AssistantMessage):
            text_parts: list[str] = []
            thinking_parts: list[str] = []
            tool_calls: list[str] = []

            for part in message.content:
                if part.type == "text":
                    text_parts.append(part.text)
                elif part.type == "thinking":
                    thinking_parts.append(part.thinking)
                elif part.type == "tool_call":
                    args = ", ".join(
                        f"{key}={json.dumps(value)}"
                        for key, value in part.arguments.items()
                    )
                    tool_calls.append(f"{part.name}({args})")

            if thinking_parts:
                parts.append(f"[Assistant thinking]: {'\n'.join(thinking_parts)}")
            if text_parts:
                parts.append(f"[Assistant]: {'\n'.join(text_parts)}")
            if tool_calls:
                parts.append(f"[Assistant tool calls]: {'; '.join(tool_calls)}")

        elif isinstance(message, ToolResultMessage):
            text = truncate_for_summary(flatten_text_parts(message.content))
            if text:
                parts.append(f"[Tool result {message.tool_name}]: {text}")

    return "\n\n".join(parts)


def flatten_text_parts(parts: list[TextPart]) -> str:
    return "".join(part.text for part in parts)


def truncate_for_summary(text: str, max_chars: int = 2_000) -> str:
    if len(text) <= max_chars:
        return text
    truncated_chars = len(text) - max_chars
    return f"{text[:max_chars]}\n\n[... {truncated_chars} more characters truncated]"


def build_summary_message(summary: str) -> UserMessage:
    return UserMessage(
        content=[TextPart(text=f"{SUMMARY_PREFIX}{summary}")],
        timestamp=now_ms(),
    )


def find_previous_summary(messages: list[Message]) -> str | None:
    for message in messages:
        if is_summary_message(message):
            return flatten_text_parts(message.content)[len(SUMMARY_PREFIX) :]
    return None


def is_summary_message(message: Message) -> bool:
    if not isinstance(message, UserMessage):
        return False
    return flatten_text_parts(message.content).startswith(SUMMARY_PREFIX)


def now_ms() -> int:
    return int(time.time() * 1000)
