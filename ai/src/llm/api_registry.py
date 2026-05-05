from __future__ import annotations

from .provider import ApiAdapter

from typing import Protocol

from .models import ModelSpec
from .types import AssistantEvent, AssistantMessage, Context
from .utils.event_stream import EventStream


class ApiProvider(Protocol):
    api: str

    def stream(
            self,
            model: ModelSpec,
            context: Context,
    ) -> EventStream[AssistantEvent, AssistantMessage]:
        ...

    def stream_simple(
        self,
        model: ModelSpec,
        context: Context,
    ) -> EventStream[AssistantEvent, AssistantMessage]:
        ...





_api_registry: dict[str, ApiAdapter] = {}

def register_api_adapter(adapter: ApiAdapter) -> None:
    _api_registry[adapter.api] = adapter


def get_api_adapter(api: str) -> ApiAdapter | None:
    return _api_registry.get(api)


def list_api_adapters() -> list[ApiAdapter]:
    return list(_api_registry.values())


def clear_api_adapters() -> None:
    _api_registry.clear()