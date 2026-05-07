from __future__ import annotations

import time
import uuid

from ..llm.types import Message
from .types import Session, SessionNode


class SessionManager:
    def create_session(self) -> Session:
        return Session(
            id=self._new_id(),
            current_node_id=None,
            nodes={},
            created_at=self._now(),
        )

    def get_current_node(self, session: Session) -> SessionNode | None:
        if session.current_node_id is None:
            return None
        return session.nodes.get(session.current_node_id)

    def get_current_messages(self, session: Session) -> list[Message]:
        current_node = self.get_current_node(session)
        if current_node is None:
            return []
        return list(current_node.messages)

    def create_root_node(
        self,
        session: Session,
        messages: list[Message],
    ) -> SessionNode:
        if session.current_node_id is not None:
            raise RuntimeError("Session already has a root node.")

        node = SessionNode(
            id=self._new_id(),
            parent_id=None,
            children_ids=[],
            messages=list(messages),
            created_at=self._now(),
        )

        session.nodes[node.id] = node
        session.current_node_id = node.id
        return node

    def create_child_node(
        self,
        session: Session,
        messages: list[Message],
    ) -> SessionNode:
        current_node = self.get_current_node(session)
        if current_node is None:
            raise RuntimeError("Cannot create child node without a current node.")

        node = SessionNode(
            id=self._new_id(),
            parent_id=current_node.id,
            children_ids=[],
            messages=list(messages),
            created_at=self._now(),
        )

        session.nodes[node.id] = node
        current_node.children_ids.append(node.id)
        session.current_node_id = node.id
        return node

    def set_current_node(self, session: Session, node_id: str) -> None:
        if node_id not in session.nodes:
            raise RuntimeError(f"Node '{node_id}' not found.")
        session.current_node_id = node_id

    def get_node(self, session: Session, node_id: str) -> SessionNode | None:
        return session.nodes.get(node_id)

    def list_children(self, session: Session, node_id: str) -> list[SessionNode]:
        node = self.get_node(session, node_id)
        if node is None:
            raise RuntimeError(f"Node '{node_id}' not found.")

        children: list[SessionNode] = []
        for child_id in node.children_ids:
            child = session.nodes.get(child_id)
            if child is not None:
                children.append(child)

        return children

    def list_nodes(self, session: Session) -> list[SessionNode]:
        return sorted(session.nodes.values(), key=lambda node: node.created_at)

    def _new_id(self) -> str:
        return str(uuid.uuid4())

    def _now(self) -> int:
        return int(time.time() * 1000)

    def checkout_node(self, session: Session, node_id: str) -> None:
        if node_id not in session.nodes:
            raise RuntimeError(f"Node '{node_id}' not found.")
        session.current_node_id = node_id

    def fork_from(
        self,
        session: Session,
        parent_node_id: str,
        messages: list[Message],
    ) -> SessionNode:
        parent_node = session.nodes.get(parent_node_id)
        if parent_node is None:
            raise RuntimeError(f"Parent node '{parent_node_id}' not found.")

        node = SessionNode(
            id=self._new_id(),
            parent_id=parent_node.id,
            children_ids=[],
            messages=list(messages),
            created_at=self._now(),
        )

        session.nodes[node.id] = node
        parent_node.children_ids.append(node.id)
        session.current_node_id = node.id
        return node

    def get_path_to_current(self, session: Session) -> list[SessionNode]:
        current_node = self.get_current_node(session)
        if current_node is None:
            return []

        path: list[SessionNode] = []
        node: SessionNode | None = current_node

        while node is not None:
            path.append(node)
            if node.parent_id is None:
                break
            node = session.nodes.get(node.parent_id)

        path.reverse()
        return path
