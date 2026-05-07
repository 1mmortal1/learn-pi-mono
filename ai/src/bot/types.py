from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ConversationType = Literal["private", "group", "channel", "thread", "unknown"]
AttachmentType = Literal["image", "audio", "video", "file", "unknown"]


class BotSender(BaseModel):
    id: str
    name: str | None = None
    display_name: str | None = None
    is_bot: bool = False


class BotAttachment(BaseModel):
    id: str | None = None
    type: AttachmentType = "unknown"
    filename: str | None = None
    mime_type: str | None = None
    local_path: str | None = None
    url: str | None = None
    size: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BotMessageEvent(BaseModel):
    platform: str
    conversation_id: str
    conversation_type: ConversationType = "unknown"
    user_id: str
    message_id: str
    text: str
    raw_text: str | None = None
    sender: BotSender | None = None
    reply_to_message_id: str | None = None
    is_mention: bool = False
    mentioned_user_ids: list[str] = Field(default_factory=list)
    attachments: list[BotAttachment] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def conversation_key(self) -> str:
        return f"{self.platform}:{self.conversation_id}"


class BotResponse(BaseModel):
    platform: str
    conversation_id: str
    message_id: str
    text: str
    stop_reason: str
    error_message: str | None = None
