from __future__ import annotations

from pydantic import BaseModel, Field

from ..llm.types import Message


class SessionNode(BaseModel):
    id: str
    parent_id: str | None = None
    children_ids: list[str] = Field(default_factory=list)
    messages: list[Message] = Field(default_factory=list)
    created_at: int


class Session(BaseModel):
    id: str
    current_node_id: str | None = None
    nodes: dict[str, SessionNode] = Field(default_factory=dict)
    created_at: int
