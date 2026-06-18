"""Anthropic provider utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import AnthropicMessage


def validate_message(message: AnthropicMessage) -> None:
    """Raise if the message has no content blocks."""
    if not message.content:
        raise ValueError(
            f"Anthropic message {message.id} has no content blocks "
            f"(stop_reason={message.stop_reason})"
        )
