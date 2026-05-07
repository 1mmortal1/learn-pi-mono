from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from ..llm.types import AssistantMessage, ToolCallPart


class ToolExecutionResult(BaseModel):
    tool_call_id: str
    tool_name: str
    output: str
    is_error: bool = False


class ToolContext(BaseModel):
    session_id: str | None = None


class AgentEventBase(BaseModel):
    session_id: str
    node_id: str | None = None
    run_id: str
    turn_index: int | None = None
    timestamp: int


class RunStartEvent(AgentEventBase):
    type: Literal["run_start"] = "run_start"


class RunEndEvent(AgentEventBase):
    type: Literal["run_end"] = "run_end"
    message: AssistantMessage


class TurnStartEvent(AgentEventBase):
    type: Literal["turn_start"] = "turn_start"


class TurnEndEvent(AgentEventBase):
    type: Literal["turn_end"] = "turn_end"


class AssistantMessageEvent(AgentEventBase):
    type: Literal["assistant_message"] = "assistant_message"
    message: AssistantMessage


class AssistantTextDeltaEvent(AgentEventBase):
    type: Literal["assistant_text_delta"] = "assistant_text_delta"
    content_index: int
    delta: str
    message: AssistantMessage


class ToolCallExecutionStartEvent(AgentEventBase):
    type: Literal["tool_call_execution_start"] = "tool_call_execution_start"
    tool_call: ToolCallPart


class ToolCallExecutionEndEvent(AgentEventBase):
    type: Literal["tool_call_execution_end"] = "tool_call_execution_end"
    tool_call: ToolCallPart
    result: ToolExecutionResult


class AbortedEvent(AgentEventBase):
    type: Literal["aborted"] = "aborted"
    reason: str | None = None


AgentEvent = (
    RunStartEvent
    | RunEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | AssistantMessageEvent
    | AssistantTextDeltaEvent
    | ToolCallExecutionStartEvent
    | ToolCallExecutionEndEvent
    | AbortedEvent
)
