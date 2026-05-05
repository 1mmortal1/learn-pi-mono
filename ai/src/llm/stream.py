from __future__ import annotations

from .api_registry import get_api_provider
from .models import ModelSpec
from .providers.register_builtins import register_builtins
from .types import AssistantEvent, AssistantMessage, Context
from .utils.event_stream import EventStream

register_builtins()


def _resolve_api_provider(api: str):
    provider = get_api_provider(api)
    if provider is None:
        raise ValueError(f"No API provider registered for api '{api}'.")
    return provider


def stream(model: ModelSpec, context: Context) -> EventStream[AssistantEvent, AssistantMessage]:
    provider = _resolve_api_provider(model.api)
    return provider.stream(model, context)


async def complete(model: ModelSpec, context: Context) -> AssistantMessage:
    response_stream = stream(model, context)
    return await response_stream.result()


def stream_simple(model: ModelSpec, context: Context) -> EventStream[AssistantEvent, AssistantMessage]:
    provider = _resolve_api_provider(model.api)
    return provider.stream_simple(model, context)


async def complete_simple(model: ModelSpec, context: Context) -> AssistantMessage:
    response_stream = stream_simple(model, context)
    return await response_stream.result()
