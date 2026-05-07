from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from ..llm.models import ModelSpec
from ..llm.stream import complete, stream
from ..llm.types import (
    AssistantMessage,
    Context,
    TextDeltaEvent,
    TextPart,
    ToolCallPart,
    ToolResultMessage,
    ToolSpec,
    UserMessage,
)
from .base import Tool
from .tool_registry import get_tool
from .tool_validation import ToolValidationError, validate_tool_arguments
from .types import (
    AbortedEvent,
    AgentEvent,
    AssistantMessageEvent,
    AssistantTextDeltaEvent,
    ToolCallExecutionEndEvent,
    ToolCallExecutionStartEvent,
    ToolExecutionResult,
    TurnEndEvent,
    TurnStartEvent,
)

CompleteFn = Callable[[ModelSpec, Context], Awaitable[AssistantMessage]]
TransformContextFn = Callable[[Context], Awaitable[None]]


async def run_agent_loop(
    model: ModelSpec,
    context: Context,
    *,
    session_id: str = "",
    run_id: str = "",
    node_id: str | None = None,
    complete_fn: CompleteFn | None = None,
    transform_context: TransformContextFn | None = None,
    tools: list[Tool] | None = None,
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
    turn_index = 0

    while True:
        has_more_tool_calls = True

        while has_more_tool_calls or pending_messages:
            if abort_checker():
                emitter(
                    AbortedEvent(
                        session_id=session_id,
                        node_id=node_id,
                        run_id=run_id,
                        turn_index=turn_index,
                        timestamp=_now(),
                        reason="abort_requested",
                    )
                )
                raise RuntimeError("Agent run aborted.")

            turn_index += 1

            emitter(
                TurnStartEvent(
                    session_id=session_id,
                    node_id=node_id,
                    run_id=run_id,
                    turn_index=turn_index,
                    timestamp=_now(),
                )
            )

            if pending_messages:
                context.messages.extend(pending_messages)
                pending_messages = []

            if transform_context is not None:
                await transform_context(context)

            llm_context = _with_tool_specs(context, tools)
            assistant_message = await _complete_assistant_message(
                model,
                llm_context,
                complete_fn=llm_complete if complete_fn is not None else None,
                session_id=session_id,
                node_id=node_id,
                run_id=run_id,
                turn_index=turn_index,
                emit=emitter,
            )
            context.messages.append(assistant_message)
            last_assistant_message = assistant_message

            emitter(
                AssistantMessageEvent(
                    session_id=session_id,
                    node_id=node_id,
                    run_id=run_id,
                    turn_index=turn_index,
                    timestamp=_now(),
                    message=assistant_message,
                )
            )

            tool_calls = _collect_tool_calls(assistant_message)
            has_more_tool_calls = len(tool_calls) > 0

            if has_more_tool_calls:
                for tool_call in tool_calls:
                    if abort_checker():
                        emitter(
                            AbortedEvent(
                                session_id=session_id,
                                node_id=node_id,
                                run_id=run_id,
                                turn_index=turn_index,
                                timestamp=_now(),
                                reason="abort_requested",
                            )
                        )
                        raise RuntimeError("Agent run aborted.")

                    emitter(
                        ToolCallExecutionStartEvent(
                            session_id=session_id,
                            node_id=node_id,
                            run_id=run_id,
                            turn_index=turn_index,
                            timestamp=_now(),
                            tool_call=tool_call,
                        )
                    )

                    execution_result = await _execute_tool_call(tool_call, tools)
                    tool_result_message = _build_tool_result_message(execution_result)
                    context.messages.append(tool_result_message)

                    emitter(
                        ToolCallExecutionEndEvent(
                            session_id=session_id,
                            node_id=node_id,
                            run_id=run_id,
                            turn_index=turn_index,
                            timestamp=_now(),
                            tool_call=tool_call,
                            result=execution_result,
                        )
                    )

            emitter(
                TurnEndEvent(
                    session_id=session_id,
                    node_id=node_id,
                    run_id=run_id,
                    turn_index=turn_index,
                    timestamp=_now(),
                )
            )

            pending_messages = steering_reader()

        follow_up_messages = follow_up_reader()
        if follow_up_messages:
            pending_messages = follow_up_messages
            continue

        if last_assistant_message is None:
            raise RuntimeError("Agent loop ended without an assistant message.")

        return last_assistant_message


async def _complete_assistant_message(
    model: ModelSpec,
    context: Context,
    *,
    complete_fn: CompleteFn | None,
    session_id: str,
    node_id: str | None,
    run_id: str,
    turn_index: int,
    emit: Callable[[AgentEvent], None],
) -> AssistantMessage:
    if complete_fn is not None:
        return await complete_fn(model, context)

    response_stream = stream(model, context)

    async for event in response_stream:
        if isinstance(event, TextDeltaEvent):
            emit(
                AssistantTextDeltaEvent(
                    session_id=session_id,
                    node_id=node_id,
                    run_id=run_id,
                    turn_index=turn_index,
                    timestamp=_now(),
                    content_index=event.content_index,
                    delta=event.delta,
                    message=event.message,
                )
            )

    return await response_stream.result()


def _collect_tool_calls(message: AssistantMessage) -> list[ToolCallPart]:
    tool_calls: list[ToolCallPart] = []

    for part in message.content:
        if part.type == "tool_call":
            tool_calls.append(part)

    return tool_calls


async def _execute_tool_call(
    tool_call: ToolCallPart,
    tools: list[Tool] | None = None,
) -> ToolExecutionResult:
    tool = _find_tool(tool_call.name, tools)

    if tool is None:
        return ToolExecutionResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            output=f"Tool '{tool_call.name}' not found.",
            is_error=True,
        )

    try:
        arguments = validate_tool_arguments(tool, tool_call.arguments)
        output = await tool.execute(arguments)
        return ToolExecutionResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            output=output,
            is_error=False,
        )
    except ToolValidationError as exc:
        return ToolExecutionResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            output=str(exc),
            is_error=True,
        )
    except Exception as exc:
        return ToolExecutionResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            output=str(exc),
            is_error=True,
        )


def _find_tool(name: str, tools: list[Tool] | None = None) -> Tool | None:
    if tools is not None:
        for tool in tools:
            if tool.name == name:
                return tool
        return None

    return get_tool(name)


def _with_tool_specs(context: Context, tools: list[Tool] | None) -> Context:
    if not tools:
        return context

    llm_context = context.model_copy(deep=True)
    llm_context.tools = [
        ToolSpec(
            name=tool.name,
            description=tool.description,
            parameters=getattr(tool, "parameters", {}),
        )
        for tool in tools
    ]
    return llm_context


def _build_tool_result_message(result: ToolExecutionResult) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=result.tool_call_id,
        tool_name=result.tool_name,
        content=[TextPart(text=result.output)],
        is_error=result.is_error,
        timestamp=_now(),
    )


def _now() -> int:
    return int(time.time() * 1000)
