from __future__ import annotations

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
    ) -> EventStream[AssistantEvent, AssistantMessage]: ...

    def stream_simple(
        self,
        model: ModelSpec,
        context: Context,
    ) -> EventStream[AssistantEvent, AssistantMessage]: ...


_api_registry: dict[str, ApiProvider] = {}


def register_api_provider(provider: ApiProvider) -> None:
    _api_registry[provider.api] = provider


def get_api_provider(api: str) -> ApiProvider | None:
    return _api_registry.get(api)


def list_api_providers() -> list[ApiProvider]:
    return list(_api_registry.values())


def clear_api_providers() -> None:
    _api_registry.clear()


register_api_adapter = register_api_provider
get_api_adapter = get_api_provider
list_api_adapters = list_api_providers
clear_api_adapters = clear_api_providers
