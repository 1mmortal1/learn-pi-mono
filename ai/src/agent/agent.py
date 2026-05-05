from __future__ import annotations

import time
from collections.abc import Callable

from .agent_loop import run_agent_loop
from .types import AgentEvent, RunEndEvent, RunStartEvent
from ..llm.models import ModelSpec
from ..llm.types import AssistantMessage, Context, Message, TextPart, UserMessage
from ..session.session_manager import SessionManager
from ..session.types import Session, SessionNode

class Agent:
    def __init__(
        self,
        model: ModelSpec,
        *,
        system_prompt: str | None = None,
        messages: list[Message] | None = None,
    ) -> None:
        self._model = model
        self._system_prompt = system_prompt

        self._session_manager = SessionManager()
        self._session: Session = self._session_manager.create_session()

        if messages:
            self._session_manager.create_root_node(
                self._session,
                list(messages),
            )

        self._is_running = False
        self._abort_requested = False

        self._steering_queue: list[UserMessage] = []
        self._follow_up_queue: list[UserMessage] = []

        self._listeners: list[Callable[[AgentEvent], None]] = []

    @property
    def model(self) -> ModelSpec:
        return self._model
    
    @property
    def is_running(self) -> bool:
        return self._is_running
    
    def subscribe(self, listener: Callable[[AgentEvent], None]) -> None:
        self._listeners.append(listener)

    def abort(self) -> None:
        self._abort_requested = True

    def steer(self, text: str) -> None:
        self._steering_queue.append(self._build_user_message(text))

    def follow_up(self, text: str) -> None:
        self._follow_up_queue.append(self._build_user_message(text))


    async def prompt(self, text: str) -> AssistantMessage:
        if self._is_running:
            raise RuntimeError("Agent is already running.")

        base_messages = self._session_manager.get_current_messages(self._session)
        user_message = self._build_user_message(text)
        run_messages = [*base_messages, user_message]

        context = self._create_context(run_messages)

        self._is_running = True
        self._abort_requested = False
        self._emit(RunStartEvent())

        try:
            result = await run_agent_loop(self._model, context, 
                get_steering_messages=self._drain_steering_queue,
                get_follow_up_messages=self._drain_follow_up_queue,
                should_abort=self._should_abort,
                emit=self._emit,
            )
            self._save_context_to_session(context)
            self._emit(RunEndEvent(message=result))
            return result
        finally:
            self._is_running = False

    async def continue_run(self) -> AssistantMessage:
        if self._is_running:
            raise RuntimeError("Agent is already running.")

        run_messages = self._session_manager.get_current_messages(self._session)
        context = self._create_context(run_messages)


        self._is_running = True
        self._abort_requested = False
        self._emit(RunStartEvent())

        try:
            result = await run_agent_loop(self._model, context, 
                get_steering_messages=self._drain_steering_queue,
                get_follow_up_messages=self._drain_follow_up_queue,
                should_abort=self._should_abort,
                emit=self._emit,
            )
            self._save_context_to_session(context)
            self._emit(RunEndEvent(message=result))
            return result
        finally:
            self._is_running = False

    def get_messages(self) -> list[Message]:
        return self._session_manager.get_current_messages(self._session)
    
    def _create_context(self, messages: list[Message]) -> Context:
        return Context(
            system_prompt=self._system_prompt,
            messages=list(messages),
        )

    def _save_context_to_session(self, context: Context) -> None:
        current_node = self._session_manager.get_current_node(self._session)

        if current_node is None:
            self._session_manager.create_root_node(
                self._session,
                list(context.messages),
            )
            return

        current_messages = current_node.messages
        if current_messages == context.messages:
            return

        self._session_manager.create_child_node(
            self._session,
            list(context.messages),
        )
    

    def _build_user_message(self, text: str) -> UserMessage:
        return UserMessage(
            content=[TextPart(text=text)],
            timestamp=int(time.time() * 1000),
        )

    def _emit(self, event: AgentEvent) -> None:
        for listener in self._listeners:
            listener(event)

    def _drain_steering_queue(self) -> list[UserMessage]:
        messages = list(self._steering_queue)
        self._steering_queue.clear()
        return messages
    
    def _drain_follow_up_queue(self) -> list[UserMessage]:
        messages = list(self._follow_up_queue)
        self._follow_up_queue.clear()
        return messages
    
    def _should_abort(self) -> bool:
        return self._abort_requested

    def get_current_node_id(self) -> str | None:
        return self._session.current_node_id
    
    def checkout_node(self, node_id: str) -> None:
        if self._is_running:
            raise RuntimeError("Cannot checkout while agent is running.")

        self._session_manager.checkout_node(self._session, node_id)

    def get_current_path(self) -> list[SessionNode]:
        return self._session_manager.get_path_to_current(self._session)
    
    
    async def fork_from(self, node_id: str, text: str) -> AssistantMessage:
        if self._is_running:
            raise RuntimeError("Agent is already running.")

        node = self._session_manager.get_node(self._session, node_id)
        if node is None:
            raise RuntimeError(f"Node '{node_id}' not found.")

        user_message = self._build_user_message(text)
        run_messages = [*node.messages, user_message]
        context = self._create_context(run_messages)

        self._is_running = True
        self._abort_requested = False
        self._emit(RunStartEvent())

        try:
            result = await run_agent_loop(
                self._model,
                context,
                get_steering_messages=self._drain_steering_queue,
                get_follow_up_messages=self._drain_follow_up_queue,
                should_abort=self._should_abort,
                emit=self._emit,
            )

            self._session_manager.fork_from(
                self._session,
                node_id,
                list(context.messages),
            )

            self._emit(RunEndEvent(message=result))
            return result
        finally:
            self._is_running = False
