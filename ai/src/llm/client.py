from __future__ import annotations

from .event_stream import EventStream
from .models import ModelSpec
from .api_registry import get_api_adapter
from .types import AssistantEvent, AssistantMessage, Context

class LLMClient:
    """Very small facade that delegates work to an API adapter."""


    async def complete(self, model: ModelSpec, context: Context) -> AssistantMessage:
        adapter = get_api_adapter(model.api)
        if not adapter:
            raise ValueError(f"No adapter found for API '{model.api}'.")
        return await adapter.complete(model, context)


    def stream(self, model: ModelSpec, context: Context) -> EventStream[AssistantEvent, AssistantMessage]:
        adapter = get_api_adapter(model.api)
        if adapter is None:
            raise ValueError(f"No adapter found for API '{model.api}'.")
        return adapter.stream(model, context)
