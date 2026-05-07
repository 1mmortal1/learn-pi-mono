from __future__ import annotations

from .types import BotAttachment, BotMessageEvent


def format_event_for_agent(event: BotMessageEvent) -> str:
    lines: list[str] = []

    sender_name = _sender_label(event)
    if sender_name:
        lines.append(f"[{sender_name}]: {event.text}")
    else:
        lines.append(event.text)

    context_lines = _context_lines(event)
    if context_lines:
        lines.append("")
        lines.extend(context_lines)

    attachment_lines = _attachment_lines(event.attachments)
    if attachment_lines:
        lines.append("")
        lines.append("<attachments>")
        lines.extend(attachment_lines)
        lines.append("</attachments>")

    return "\n".join(lines).strip()


def _sender_label(event: BotMessageEvent) -> str:
    if event.sender is None:
        return event.user_id
    return (
        event.sender.display_name
        or event.sender.name
        or event.sender.id
        or event.user_id
    )


def _context_lines(event: BotMessageEvent) -> list[str]:
    lines = [
        f"platform: {event.platform}",
        f"conversation: {event.conversation_type}:{event.conversation_id}",
        f"message_id: {event.message_id}",
    ]

    if event.reply_to_message_id:
        lines.append(f"reply_to: {event.reply_to_message_id}")

    if event.is_mention:
        lines.append("mentioned_bot: true")

    if event.mentioned_user_ids:
        lines.append(f"mentioned_users: {', '.join(event.mentioned_user_ids)}")

    return [f"[message context] {line}" for line in lines]


def _attachment_lines(attachments: list[BotAttachment]) -> list[str]:
    lines: list[str] = []

    for index, attachment in enumerate(attachments, start=1):
        parts = [f"{index}. type={attachment.type}"]
        if attachment.filename:
            parts.append(f"filename={attachment.filename}")
        if attachment.mime_type:
            parts.append(f"mime={attachment.mime_type}")
        if attachment.local_path:
            parts.append(f"path={attachment.local_path}")
        elif attachment.url:
            parts.append(f"url={attachment.url}")
        if attachment.size is not None:
            parts.append(f"size={attachment.size}")
        lines.append(" ".join(parts))

    return lines
