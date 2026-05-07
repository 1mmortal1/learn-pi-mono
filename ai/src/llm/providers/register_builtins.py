from __future__ import annotations

from ..api_registry import register_api_provider
from .dummy import DummyProvider

_registered = False


def register_builtins() -> None:
    global _registered

    if _registered:
        return

    register_api_provider(DummyProvider())

    try:
        from .openai_responses import OpenAIResponsesProvider
    except ModuleNotFoundError:
        pass
    else:
        register_api_provider(OpenAIResponsesProvider())

    try:
        from .openai_chat_completions import OpenAIChatCompletionsProvider
    except ModuleNotFoundError:
        pass
    else:
        register_api_provider(OpenAIChatCompletionsProvider())

    _registered = True
