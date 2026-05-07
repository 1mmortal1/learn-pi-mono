from __future__ import annotations

import asyncio

from dotenv import load_dotenv

from ai.src.agent.agent import Agent
from ai.src.agent.tools.echo_tool import EchoTool
from ai.src.agent.types import (
    AssistantMessageEvent,
    AssistantTextDeltaEvent,
    RunEndEvent,
    ToolCallExecutionEndEvent,
    ToolCallExecutionStartEvent,
)
from ai.src.bot import BotMessageEvent, BotResponse, BotRuntime
from ai.src.bot.adapters import CliAdapter
from ai.src.llm.models import ModelSpec, register_model
from ai.src.llm.providers.register_builtins import register_builtins
from ai.src.session.types import SessionNode

SHOW_EVENTS = False


class ConsoleRenderer:
    def __init__(self) -> None:
        self._streaming = False
        self._streamed_since_last_result = False

    def handle_event(self, event) -> None:
        if isinstance(event, AssistantTextDeltaEvent):
            if not event.delta:
                return
            if not self._streaming:
                print("Mio: ", end="", flush=True)
                self._streaming = True
            self._streamed_since_last_result = True
            print(event.delta, end="", flush=True)
            return

        if isinstance(event, RunEndEvent) and self._streaming:
            print()
            self._streaming = False

    def consume_streamed_flag(self) -> bool:
        streamed = self._streamed_since_last_result
        self._streamed_since_last_result = False
        return streamed


def print_event(event) -> None:
    if not SHOW_EVENTS:
        return

    print(
        f"[event] type={event.type} "
        f"run_id={event.run_id} "
        f"turn_index={event.turn_index} "
        f"node_id={event.node_id}"
    )

    if isinstance(event, AssistantMessageEvent):
        print(f"  assistant_message={event.message}")

    if isinstance(event, ToolCallExecutionStartEvent):
        print(f"  tool_start name={event.tool_call.name}")

    if isinstance(event, ToolCallExecutionEndEvent):
        print(
            f"  tool_end name={event.tool_call.name} "
            f"output={event.result.output} "
            f"is_error={event.result.is_error}"
        )


def print_path(path: list[SessionNode]) -> None:
    if not path:
        print("session path: (empty)")
        return

    print("session path:")
    for node in path:
        print(
            f"  {node.id} "
            f"parent={node.parent_id or '-'} "
            f"children={len(node.children_ids)} "
            f"messages={len(node.messages)}"
        )


def print_help() -> None:
    print(
        "commands:\n"
        "  /help                         show commands\n"
        "  /quit                         exit\n"
        "  /tools                        list enabled tools\n"
        "  /node                         show current node id\n"
        "  /path                         show current session path\n"
        "  /tree                         show session tree\n"
        "  /checkout <node_id>           switch to a previous node\n"
        "  /fork <node_id> <message>     fork from a previous node\n"
    )


def print_assistant_message(result, *, already_streamed: bool = False) -> None:
    text = "".join(part.text for part in result.content if part.type == "text")
    if result.stop_reason == "error":
        print(f"Mio error: {result.error_message or text or 'unknown error'}")
        return
    if already_streamed:
        return
    print(f"Mio: {text}")


def print_bot_response(
    response: BotResponse, *, already_streamed: bool = False
) -> None:
    if response.stop_reason == "error":
        print(
            f"Mio error: {response.error_message or response.text or 'unknown error'}"
        )
        return
    if already_streamed:
        return
    print(f"Mio: {response.text}")


def print_tree(agent: Agent) -> None:
    nodes = agent.get_session_nodes()
    if not nodes:
        print("session tree: (empty)")
        return

    node_by_id = {node.id: node for node in nodes}
    roots = [node for node in nodes if node.parent_id is None]
    current_node_id = agent.get_current_node_id()

    print("session tree:")

    def print_node(node: SessionNode, depth: int) -> None:
        marker = "*" if node.id == current_node_id else " "
        indent = "  " * depth
        print(
            f"{indent}{marker} {node.id} "
            f"messages={len(node.messages)} "
            f"children={len(node.children_ids)}"
        )
        for child_id in node.children_ids:
            child = node_by_id.get(child_id)
            if child is not None:
                print_node(child, depth + 1)

    for root in roots:
        print_node(root, 0)


async def handle_command(agent: Agent, renderer: ConsoleRenderer, text: str) -> bool:
    command, _, rest = text.partition(" ")

    if command in {"/quit", "/exit"}:
        return False

    if command == "/help":
        print_help()
        return True

    if command == "/node":
        print(f"current node: {agent.get_current_node_id() or '(none)'}")
        return True

    if command == "/tools":
        tools = agent.list_tools()
        if not tools:
            print("tools: (none)")
            return True
        print("tools:")
        for tool in tools:
            required = tool.parameters.get("required", [])
            required_text = f" required={', '.join(required)}" if required else ""
            print(f"  {tool.name}: {tool.description}{required_text}")
        return True

    if command == "/path":
        print_path(agent.get_current_path())
        return True

    if command == "/tree":
        print_tree(agent)
        return True

    if command == "/checkout":
        node_id = rest.strip()
        if not node_id:
            print("usage: /checkout <node_id>")
            return True
        try:
            agent.checkout_node(node_id)
        except RuntimeError as exc:
            print(f"checkout failed: {exc}")
        else:
            print(f"checked out node: {node_id}")
        return True

    if command == "/fork":
        node_id, _, message = rest.strip().partition(" ")
        if not node_id or not message:
            print("usage: /fork <node_id> <message>")
            return True
        try:
            result = await agent.fork_from(node_id, message)
        except RuntimeError as exc:
            print(f"fork failed: {exc}")
        else:
            print_assistant_message(
                result,
                already_streamed=renderer.consume_streamed_flag(),
            )
        return True

    print(f"unknown command: {command}")
    print("type /help to see available commands")
    return True


async def main() -> None:
    load_dotenv()

    register_builtins()

    model = ModelSpec(
        id="glm-4.7",
        name="GLM-4.7",
        api="openai-chat-completions",
        provider="zai",
        base_url="https://api.z.ai/api/paas/v4/",
        supports_tools=True,
        context_window=203_000,
        max_output_tokens=1_024,
        compat={
            "api_key_env": "ZAI_API_KEY",
            "extra_body": {
                "thinking": {
                    "type": "disabled",
                },
            },
        },
    )
    register_model(model)

    renderer = ConsoleRenderer()
    runtime = BotRuntime(
        lambda _event: Agent(model=model, tools=[EchoTool()]),
        agent_listeners=[renderer.handle_event, print_event],
    )
    cli_agent = runtime.get_or_create_agent(
        BotMessageEvent(
            platform="cli",
            conversation_id="default",
            conversation_type="private",
            user_id="local",
            message_id="cli-0",
            text="",
        )
    )

    adapter = CliAdapter(
        command_handler=lambda text: handle_command(
            cli_agent,
            renderer,
            text,
        ),
        response_renderer=lambda response: print_bot_response(
            response,
            already_streamed=renderer.consume_streamed_flag(),
        ),
    )

    await adapter.run(runtime)


if __name__ == "__main__":
    asyncio.run(main())
