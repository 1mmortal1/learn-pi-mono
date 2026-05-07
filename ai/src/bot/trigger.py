from __future__ import annotations

from dataclasses import dataclass

from .types import BotMessageEvent


@dataclass(frozen=True)
class TriggerPolicy:
    wake_prefixes: tuple[str, ...] = ()
    private_chat_always: bool = True
    group_requires_mention_or_prefix: bool = True
    process_replies_to_bot: bool = True
    ignore_bot_messages: bool = True


def prepare_event_for_agent(
    event: BotMessageEvent,
    policy: TriggerPolicy | None = None,
) -> BotMessageEvent | None:
    policy = policy or TriggerPolicy()

    if policy.ignore_bot_messages and event.sender is not None and event.sender.is_bot:
        return None

    prefix_event = _strip_wake_prefix(event, policy.wake_prefixes)
    if prefix_event is not None:
        return prefix_event

    if event.conversation_type == "private" and policy.private_chat_always:
        return event

    if event.is_mention:
        return event

    if policy.process_replies_to_bot and _is_reply_to_bot(event):
        return event

    if (
        event.conversation_type in {"group", "channel", "thread"}
        and policy.group_requires_mention_or_prefix
    ):
        return None

    return event


def _strip_wake_prefix(
    event: BotMessageEvent,
    wake_prefixes: tuple[str, ...],
) -> BotMessageEvent | None:
    text = event.text.lstrip()
    for prefix in wake_prefixes:
        if text.startswith(prefix):
            metadata = dict(event.metadata)
            metadata["wake_prefix"] = prefix
            return event.model_copy(
                update={
                    "text": text[len(prefix) :].lstrip(),
                    "metadata": metadata,
                }
            )
    return None


def _is_reply_to_bot(event: BotMessageEvent) -> bool:
    if event.reply_to_message_id is None:
        return False
    return bool(
        event.metadata.get("reply_to_bot") or event.metadata.get("is_reply_to_bot")
    )
