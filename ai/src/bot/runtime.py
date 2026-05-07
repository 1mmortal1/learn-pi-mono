from __future__ import annotations

from collections.abc import Callable

from ..agent.agent import Agent
from ..agent.types import AgentEvent
from ..llm.types import AssistantMessage
from .message_format import format_event_for_agent
from .trigger import TriggerPolicy, prepare_event_for_agent
from .types import BotMessageEvent, BotResponse

AgentFactory = Callable[[BotMessageEvent], Agent]
AgentListener = Callable[[AgentEvent], None]


class BotRuntime:
    def __init__(
        self,
        agent_factory: AgentFactory,
        *,
        agent_listeners: list[AgentListener] | None = None,
        trigger_policy: TriggerPolicy | None = None,
    ) -> None:
        self._agent_factory = agent_factory
        self._agent_listeners = list(agent_listeners or [])
        self._trigger_policy = trigger_policy or TriggerPolicy()
        self._agents: dict[str, Agent] = {}

    async def handle_message(self, event: BotMessageEvent) -> BotResponse | None:
        prepared_event = prepare_event_for_agent(event, self._trigger_policy)
        if prepared_event is None:
            return None

        agent = self.get_or_create_agent(prepared_event)
        result = await agent.prompt(format_event_for_agent(prepared_event))
        return BotResponse(
            platform=prepared_event.platform,
            conversation_id=prepared_event.conversation_id,
            message_id=prepared_event.message_id,
            text=assistant_text(result),
            stop_reason=result.stop_reason,
            error_message=result.error_message,
        )

    def get_or_create_agent(self, event: BotMessageEvent) -> Agent:
        key = event.conversation_key()
        agent = self._agents.get(key)
        if agent is not None:
            return agent

        agent = self._agent_factory(event)
        for listener in self._agent_listeners:
            agent.subscribe(listener)
        self._agents[key] = agent
        return agent

    def get_agent(self, platform: str, conversation_id: str) -> Agent | None:
        return self._agents.get(self.session_key(platform, conversation_id))

    def list_agents(self) -> list[Agent]:
        return list(self._agents.values())

    @staticmethod
    def session_key(platform: str, conversation_id: str) -> str:
        return f"{platform}:{conversation_id}"


def assistant_text(message: AssistantMessage) -> str:
    return "".join(part.text for part in message.content if part.type == "text")
