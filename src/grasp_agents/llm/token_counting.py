"""
Token counting for the model-facing view (our :class:`InputItem`s).

Sizes the view *before* generation — counting user inputs, tool outputs,
images, and reasoning traces with the model's tokenizer (via litellm) — rather
than waiting for a response's reported usage. Images are counted with a default
per-image cost (never fetched); encrypted reasoning is sized from its blob; a
chars-per-token estimate is the fallback when no tokenizer is available.
"""

import logging
from collections.abc import Sequence
from typing import Any

from grasp_agents.llm.model_info import count_tokens
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputFile,
    InputImage,
    InputItem,
    InputMessageItem,
    OutputMessageItem,
    ReasoningItem,
)

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4
_FALLBACK_TOKENS_PER_IMAGE = 1024
_FALLBACK_TOKENS_PER_FILE = 2048

_TEXT_ITEM_TYPES = (InputMessageItem, FunctionToolOutputItem, OutputMessageItem)
_IMAGE_ITEM_TYPES = (InputMessageItem, FunctionToolOutputItem)
_FILE_ITEM_TYPES = (InputMessageItem, FunctionToolOutputItem)


def _item_text(item: InputItem) -> str:
    if isinstance(item, FunctionToolCallItem):
        return f"{item.name} {item.arguments}"
    return item.text if isinstance(item, _TEXT_ITEM_TYPES) else ""


def _images(item: InputItem) -> list[InputImage]:
    return item.images if isinstance(item, _IMAGE_ITEM_TYPES) else []


def _files(item: InputItem) -> list[InputFile]:
    return item.files if isinstance(item, _FILE_ITEM_TYPES) else []


def _file_tokens(file: InputFile) -> int:
    """
    Estimate a file's token cost. Inlined files carry base64 ``file_data``, so
    size it from the decoded byte length (~chars/token) — a rough, deliberately
    conservative heuristic (the real cost is provider/format-dependent).
    By-reference files (``file_id`` / ``file_url``, no inline data) fall back to
    a flat estimate.
    """
    data = file.file_data
    if data is not None and data:
        return (len(data) * 3 // 4) // _CHARS_PER_TOKEN
    return _FALLBACK_TOKENS_PER_FILE


def _reasoning_tokens(item: InputItem) -> int:
    """
    Estimate a reasoning item's token cost — the chat-text count ignores it.

    A reasoning turn can dominate the context. OpenAI returns the trace as an
    opaque encrypted blob (``encrypted_content``) with only a short summary, so
    size that blob like an inline file (base64 → bytes → chars/token), which
    tracks the underlying reasoning-token count closely. When the trace is
    plaintext instead (``content_text``, else just the ``summary``), count it.
    """
    if not isinstance(item, ReasoningItem):
        return 0
    if item.encrypted_content:
        return (len(item.encrypted_content) * 3 // 4) // _CHARS_PER_TOKEN
    text = item.content_text or item.summary_text
    return len(text) // _CHARS_PER_TOKEN


def _to_chat_messages(items: Sequence[InputItem]) -> list[dict[str, Any]]:
    """
    Render items to OpenAI-style chat messages for litellm counting. Role is
    immaterial to the count; each item becomes one message with its text and any
    image parts. Files are counted separately (not representable here).
    """
    messages: list[dict[str, Any]] = []
    for item in items:
        text = _item_text(item)
        images = _images(item)
        if images:
            content: list[dict[str, Any]] = []
            if text:
                content.append({"type": "text", "text": text})
            content.extend(
                {"type": "image_url", "image_url": {"url": img.image_url or "data:,"}}
                for img in images
            )
            messages.append({"role": "user", "content": content})
        elif text:
            messages.append({"role": "user", "content": text})
    return messages


def _rough_token_estimate(items: Sequence[InputItem]) -> int:
    total = 0
    for item in items:
        total += len(_item_text(item)) // _CHARS_PER_TOKEN
        total += len(_images(item)) * _FALLBACK_TOKENS_PER_IMAGE
        total += sum(_file_tokens(f) for f in _files(item))
        total += _reasoning_tokens(item)
    return total


def count_input_tokens(model: str, items: Sequence[InputItem]) -> int:
    """
    Token cost of a view (our :class:`InputItem`s) for ``model``, via litellm —
    counting text and images, plus separate estimates for files and encrypted
    reasoning blobs (which aren't representable as chat text). Falls back to a
    chars-per-token estimate when no tokenizer is available; logs which path was
    taken.
    """
    messages = _to_chat_messages(items)
    counted = count_tokens(model, messages=messages) if messages else 0
    if counted > 0:
        logger.debug("counted %d input tokens for %r via tokenizer", counted, model)
        # Files and encrypted reasoning aren't representable as chat text — size
        # them separately and add to the tokenizer's text + image count.
        extra = sum(_file_tokens(f) for item in items for f in _files(item))
        extra += sum(_reasoning_tokens(item) for item in items)
        return counted + extra
    rough = _rough_token_estimate(items)
    logger.debug("no tokenizer for %r; rough estimate of %d input tokens", model, rough)
    return rough
