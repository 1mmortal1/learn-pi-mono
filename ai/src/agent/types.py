from __future__ import annotations

from pydantic import BaseModel

from typing import Literal

from ..llm.types import AssistantMessage, ToolCallPart, ToolResultMessage



class ToolExecutionResult(BaseModel):
    tool_call_id: str
    tool_name: str
    output: str
    is_error: bool = False


class ToolContext(BaseModel):
    session_id: str | None = None

class RunStartEvent(BaseModel):
    type: Literal["run_start"] = "run_start"


class RunEndEvent(BaseModel):
    type: Literal["run_end"] = "run_end"
    message: AssistantMessage


class TurnStartEvent(BaseModel):
    type: Literal["turn_start"] = "turn_start"

class TurnEndEvent(BaseModel):
    type: Literal["turn_end"] = "turn_end"

class AssistantMessageEvent(BaseModel):
    type: Literal["assistant_message"] = "assistant_message"
    message: AssistantMessage


class ToolCallExecutionStartEvent(BaseModel):
    type: Literal["tool_call_execution_start"] = "tool_call_execution_start"
    tool_call: ToolCallPart


class ToolCallExecutionEndEvent(BaseModel):
    type: Literal["tool_call_execution_end"] = "tool_call_execution_end"
    tool_call: ToolCallPart
    result: ToolResultMessage


class AbortedEvent(BaseModel):
    type: Literal["aborted"] = "aborted"


AgentEvent = (
    RunStartEvent
    | RunEndEvent
    | TurnStartEvent
    | AssistantMessageEvent
    | ToolCallExecutionStartEvent
    | ToolCallExecutionEndEvent
    | AbortedEvent
)