from __future__ import annotations

import time
import uuid
from collections.abc import Callable

from ..llm.models import ModelSpec
from ..llm.types import AssistantMessage, Context, Message, TextPart, UserMessage
from ..session.session_manager import SessionManager
from ..session.types import Session, SessionNode
from .agent_loop import run_agent_loop
from .base import Tool
from .compaction import CompactionSettings, compact_context_in_place
from .tool_registry import ToolRegistry
from .types import AgentEvent, RunEndEvent, RunStartEvent


class Agent:
    def __init__(
        self,
        model: ModelSpec,
        *,
        system_prompt: str | None = None,
        messages: list[Message] | None = None,
        tools: list[Tool] | None = None,
        compaction_settings: CompactionSettings | None = None,
    ) -> None:
        self._model = model
        self._system_prompt = system_prompt
        self._tool_registry = ToolRegistry(tools)
        self._compaction_settings = compaction_settings or CompactionSettings()

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
        base_messages = self._session_manager.get_current_messages(self._session)
        user_message = self._build_user_message(text)
        context = self._create_context([*base_messages, user_message])
        return await self._run_context(context, save_mode="append")

    async def continue_run(self) -> AssistantMessage:
        run_messages = self._session_manager.get_current_messages(self._session)
        context = self._create_context(run_messages)
        return await self._run_context(context, save_mode="append")

    async def fork_from(self, node_id: str, text: str) -> AssistantMessage:
        node = self._session_manager.get_node(self._session, node_id)
        if node is None:
            raise RuntimeError(f"Node '{node_id}' not found.")

        user_message = self._build_user_message(text)
        context = self._create_context([*node.messages, user_message])
        return await self._run_context(
            context,
            save_mode="fork",
            fork_parent_node_id=node_id,
        )

    def get_messages(self) -> list[Message]:
        return self._session_manager.get_current_messages(self._session)

    def get_current_node_id(self) -> str | None:
        return self._session.current_node_id

    def checkout_node(self, node_id: str) -> None:
        if self._is_running:
            raise RuntimeError("Cannot checkout while agent is running.")
        self._session_manager.checkout_node(self._session, node_id)

    def get_current_path(self) -> list[SessionNode]:
        return self._session_manager.get_path_to_current(self._session)

    def get_session_nodes(self) -> list[SessionNode]:
        return self._session_manager.list_nodes(self._session)

    def register_tool(self, tool: Tool) -> None:
        if self._is_running:
            raise RuntimeError("Cannot register tools while agent is running.")
        self._tool_registry.register(tool)

    def list_tools(self) -> list[Tool]:
        return self._tool_registry.list()

    async def _run_context(
        self,
        context: Context,
        *,
        save_mode: str,
        fork_parent_node_id: str | None = None,
    ) -> AssistantMessage:
        if self._is_running:
            raise RuntimeError("Agent is already running.")

        run_id = self._new_run_id()
        start_node_id = self.get_current_node_id()
        session_id = self._session.id

        self._is_running = True
        self._abort_requested = False

        self._emit(
            RunStartEvent(
                session_id=session_id,
                node_id=start_node_id,
                run_id=run_id,
                turn_index=None,
                timestamp=self._now(),
            )
        )

        try:
            result = await run_agent_loop(
                self._model,
                context,
                session_id=session_id,
                run_id=run_id,
                node_id=start_node_id,
                get_steering_messages=self._drain_steering_queue,
                get_follow_up_messages=self._drain_follow_up_queue,
                should_abort=self._should_abort,
                transform_context=self._compact_context,
                tools=self._tool_registry.list(),
                emit=self._emit,
            )

            if save_mode == "append":
                self._save_context_to_session(context)
            elif save_mode == "fork":
                if fork_parent_node_id is None:
                    raise RuntimeError("fork_parent_node_id is required for fork mode.")
                self._session_manager.fork_from(
                    self._session,
                    fork_parent_node_id,
                    list(context.messages),
                )
            else:
                raise RuntimeError(f"Unknown save mode: {save_mode}")

            self._emit(
                RunEndEvent(
                    session_id=session_id,
                    node_id=self.get_current_node_id(),
                    run_id=run_id,
                    turn_index=None,
                    timestamp=self._now(),
                    message=result,
                )
            )
            return result
        finally:
            self._is_running = False

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

        if current_node.messages == context.messages:
            return

        self._session_manager.create_child_node(
            self._session,
            list(context.messages),
        )

    def _build_user_message(self, text: str) -> UserMessage:
        return UserMessage(
            content=[TextPart(text=text)],
            timestamp=self._now(),
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

    async def _compact_context(self, context: Context) -> None:
        await compact_context_in_place(
            self._model,
            context,
            self._compaction_settings,
        )

    def _new_run_id(self) -> str:
        return str(uuid.uuid4())

    def _now(self) -> int:
        return int(time.time() * 1000)
