from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

ApiName = str
ProviderName = str


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: list[TextPart] = Field(default_factory=list)
    timestamp: int


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: list[AssistantContentPart] = Field(default_factory=list)
    model: str
    provider: str
    api: str
    stop_reason: str = "stop"
    error_message: str | None = None
    timestamp: int


class ToolResultMessage(BaseModel):
    role: Literal["toolResult"] = "toolResult"
    tool_call_id: str
    tool_name: str
    content: list[TextPart] = Field(default_factory=list)
    is_error: bool = False
    timestamp: int


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ThinkingPart(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str


class ToolCallPart(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


AssistantContentPart = TextPart | ThinkingPart | ToolCallPart


Message = UserMessage | AssistantMessage | ToolResultMessage


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class Context(BaseModel):
    system_prompt: str | None = None
    messages: list[Message] = Field(default_factory=list)
    tools: list[ToolSpec] = Field(default_factory=list)


class StartEvent(BaseModel):
    type: Literal["start"] = "start"
    message: AssistantMessage
    session_id: str = ""
    node_id: str | None = None
    run_id: str = ""
    turn_index: int = 0
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))


class TextDeltaEvent(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    content_index: int
    delta: str
    message: AssistantMessage
    session_id: str = ""
    node_id: str | None = None
    run_id: str = ""
    turn_index: int = 0
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))


class ThinkingDeltaEvent(BaseModel):
    type: Literal["thinking_delta"] = "thinking_delta"
    content_index: int
    delta: str
    message: AssistantMessage
    session_id: str = ""
    node_id: str | None = None
    run_id: str = ""
    turn_index: int = 0
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))


class ToolCallEvent(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    content_index: int
    tool_call: ToolCallPart
    message: AssistantMessage
    session_id: str = ""
    node_id: str | None = None
    run_id: str = ""
    turn_index: int = 0
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    message: AssistantMessage
    session_id: str = ""
    node_id: str | None = None
    run_id: str = ""
    turn_index: int = 0
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: AssistantMessage
    session_id: str = ""
    node_id: str | None = None
    run_id: str = ""
    turn_index: int = 0
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))


AssistantEvent = (
    StartEvent
    | TextDeltaEvent
    | DoneEvent
    | ErrorEvent
    | ThinkingDeltaEvent
    | ToolCallEvent
)
