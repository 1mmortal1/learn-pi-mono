from __future__ import annotations

import asyncio
import time

from ai.src.llm.client import LLMClient
from ai.src.llm.adapters.dummy import DummyAdapter
from ai.src.llm.models import ModelSpec, get_model, register_model
from ai.src.llm.api_registry import register_api_adapter
from ai.src.llm.types import Context, UserMessage, AssistantMessage


async def main() -> None:
    model = ModelSpec(
        id="gpt-4.1-mini",
        name="GPT-4.1 mini",
        api="openai-responses",
        provider="openai",
        base_url="https://api.openai.com/v1",
        reasoning=False,
    )

    register_model(model)
    register_api_adapter(DummyAdapter())

    saved_model = get_model("gpt-4.1-mini")
    print("saved model:", saved_model)

    if saved_model is None:
        raise RuntimeError("Model was not registered correctly.")

    context = Context(
    system_prompt="You are a helpful assistant.",
    messages=[
        UserMessage(content="你好", timestamp=1),
        AssistantMessage(
            content="你好，我在。",
            model="gpt-4.1-mini",
            provider="openai",
            api="openai-responses",
            timestamp=2,
        ),
        UserMessage(content="解释一下 registry", timestamp=3),
    ],
)


    client = LLMClient()

    print("\nstream demo:")
    stream = client.stream(saved_model, context)
    async for event in stream:
        print("event:", event.type, event.model_dump())

    final_message = await stream.result()
    print("stream final message:", final_message)

    print("\ncomplete demo:")
    reply = await client.complete(saved_model, context)
    print("complete final message:", reply)


if __name__ == "__main__":
    asyncio.run(main())
