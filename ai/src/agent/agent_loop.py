from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from .tool_registry import get_tool
from .types import AgentEvent, AssistantMessageEvent, ToolCallExecutionStartEvent, ToolExecutionResult, TurnStartEvent,ToolCallExecutionEndEvent,TurnEndEvent,AbortedEvent
from ..llm.models import ModelSpec
from ..llm.stream import complete
from ..llm.types import AssistantMessage, Context, TextPart, ToolCallPart, ToolResultMessage,UserMessage

CompleteFn = Callable[[ModelSpec, Context], Awaitable[AssistantMessage]]


async def run_agent_loop(
    model: ModelSpec,
    context: Context,
    *,
    complete_fn: CompleteFn | None = None,
    get_steering_messages: Callable[[], list[UserMessage]] | None = None,
    get_follow_up_messages: Callable[[], list[UserMessage]] | None = None,
    should_abort: Callable[[], bool] | None = None,
    emit: Callable[[AgentEvent], None] | None = None,
) -> AssistantMessage:
    llm_complete = complete_fn or complete
    steering_reader = get_steering_messages or (lambda: [])
    follow_up_reader = get_follow_up_messages or (lambda: [])
    abort_checker = should_abort or (lambda: False)
    emitter = emit or (lambda event: None)

    last_assistant_message: AssistantMessage | None = None
    pending_messages = steering_reader()

    while True:
        has_more_tool_calls = True

        while has_more_tool_calls or pending_messages:
            if abort_checker():
                emitter(AbortedEvent())
                raise RuntimeError("Agent run aborted.")
            
            emitter(TurnStartEvent())

            if pending_messages:
                context.messages.extend(pending_messages)
                pending_messages = []

            assistant_message = await llm_complete(model, context)
            context.messages.append(assistant_message)
            last_assistant_message = assistant_message
            emitter(AssistantMessageEvent(message=assistant_message))

            tool_calls = _collect_tool_calls(assistant_message)
            has_more_tool_calls = len(tool_calls) > 0

            if has_more_tool_calls:
                for tool_call in tool_calls:
                    if abort_checker():
                        emitter(AbortedEvent())
                        raise RuntimeError("Agent run aborted.")
                    
                    emitter(ToolCallExecutionStartEvent(tool_call=tool_call))
                    execution_result = await _execute_tool_call(tool_call)
                    tool_result_message = _build_tool_result_message(execution_result)
                    context.messages.append(tool_result_message)
                    emitter(
                        ToolCallExecutionEndEvent(
                            tool_call = tool_call, 
                            result=execution_result
                        )
                    )
                    
            emitter(TurnEndEvent())
            pending_messages = steering_reader()

        follow_up_messages = follow_up_reader()
        if follow_up_messages:
            pending_messages = follow_up_messages
            continue

        if last_assistant_message is None:
            raise RuntimeError("Agent loop ended without an assistant message.")

        return last_assistant_message



def _collect_tool_calls(message: AssistantMessage) -> list[ToolCallPart]:
    tool_calls: list[ToolCallPart] = []

    for part in message.content:
        if part.type == "tool_call":
            tool_calls.append(part)

    return tool_calls


async def _execute_tool_call(tool_call: ToolCallPart) -> ToolExecutionResult:
    tool = get_tool(tool_call.name)

    if tool is None:
        return ToolExecutionResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            output=f"Tool '{tool_call.name}' not found.",
            is_error=True,
        )

    try:
        output = await tool.execute(tool_call.arguments)
        return ToolExecutionResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            output=output,
            is_error=False,
        )
    except Exception as exc:
        return ToolExecutionResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            output=str(exc),
            is_error=True,
        )


def _build_tool_result_message(result: ToolExecutionResult) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=result.tool_call_id,
        tool_name=result.tool_name,
        content=[TextPart(text=result.output)],
        is_error=result.is_error,
        timestamp=int(time.time() * 1000),
    )
