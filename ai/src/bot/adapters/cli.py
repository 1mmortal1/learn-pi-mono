from __future__ import annotations

from collections.abc import Awaitable, Callable

from ...agent.agent import Agent
from ..runtime import BotRuntime
from ..types import BotMessageEvent, BotResponse, BotSender

CommandHandler = Callable[[str], Awaitable[bool]]
ResponseRenderer = Callable[[BotResponse], None]


class CliAdapter:
    name = "cli"

    def __init__(
        self,
        *,
        conversation_id: str = "default",
        user_id: str = "local",
        command_handler: CommandHandler | None = None,
        response_renderer: ResponseRenderer | None = None,
    ) -> None:
        self._conversation_id = conversation_id
        self._user_id = user_id
        self._command_handler = command_handler
        self._response_renderer = response_renderer or default_response_renderer
        self._message_index = 0

    def get_agent(self, runtime: BotRuntime) -> Agent:
        return runtime.get_or_create_agent(self._build_event("", "cli-0"))

    async def run(self, runtime: BotRuntime) -> None:
        print("Mio CLI started. Type /help for commands, /quit to exit.")

        while True:
            try:
                text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return

            if not text:
                continue

            if text.startswith("/") and self._command_handler is not None:
                should_continue = await self._command_handler(text)
                if not should_continue:
                    return
                continue

            self._message_index += 1
            response = await runtime.handle_message(
                self._build_event(text, f"cli-{self._message_index}")
            )
            if response is not None:
                self._response_renderer(response)

    def _build_event(self, text: str, message_id: str) -> BotMessageEvent:
        return BotMessageEvent(
            platform=self.name,
            conversation_id=self._conversation_id,
            conversation_type="private",
            user_id=self._user_id,
            message_id=message_id,
            text=text,
            raw_text=text,
            sender=BotSender(id=self._user_id, name=self._user_id),
        )


def default_response_renderer(response: BotResponse) -> None:
    if response.stop_reason == "error":
        print(
            f"Mio error: {response.error_message or response.text or 'unknown error'}"
        )
        return
    print(f"Mio: {response.text}")
