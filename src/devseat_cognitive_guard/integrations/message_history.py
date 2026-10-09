from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from ..context_models import ContextItem


class MessageHistoryAdapter:
    """Portable adapter for chat/tool history represented as message dicts.

    The adapter preserves original message objects and only uses serialized
    copies for token selection. Recent messages and selected roles can be
    protected as required context.
    """

    def __init__(
        self,
        *,
        keep_last: int = 6,
        required_roles: Sequence[str] = ("system",),
        role_priorities: Mapping[str, int] | None = None,
        content_field: str = "content",
        role_field: str = "role",
    ) -> None:
        self.keep_last = max(0, int(keep_last))
        self.required_roles = {str(role) for role in required_roles}
        self.role_priorities = {
            "system": 100,
            "user": 80,
            "assistant": 70,
            "tool": 40,
        }
        self.role_priorities.update(
            {str(k): int(v) for k, v in (role_priorities or {}).items()}
        )
        self.content_field = content_field
        self.role_field = role_field

    def to_items(
        self,
        payload: Sequence[Mapping[str, Any]],
    ) -> tuple[ContextItem, ...]:
        required_from = max(0, len(payload) - self.keep_last)
        items: list[ContextItem] = []
        for index, message in enumerate(payload):
            role = str(message.get(self.role_field, "unknown"))
            required = index >= required_from or role in self.required_roles
            content = message.get(self.content_field)
            rendered = (
                content
                if isinstance(content, str)
                else json.dumps(
                    message,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )
            )
            items.append(
                ContextItem(
                    key=f"message:{index:06d}",
                    content=rendered,
                    priority=100 if required else self.role_priorities.get(role, 50),
                    required=required,
                    stable=False,
                    source=f"message-role:{role}",
                )
            )
        return tuple(items)

    def from_items(
        self,
        original: Sequence[Mapping[str, Any]],
        selected: Sequence[ContextItem],
    ) -> list[Mapping[str, Any]]:
        selected_indices = {
            int(item.key.split(":", 1)[1])
            for item in selected
            if item.key.startswith("message:")
        }
        return [
            message
            for index, message in enumerate(original)
            if index in selected_indices
        ]
