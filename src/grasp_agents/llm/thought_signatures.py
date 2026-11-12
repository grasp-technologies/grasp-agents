"""
Thought signatures carried on message and tool-call items.

Gemini signs the parts of a model turn rather than its reasoning, so its
signatures live in ``provider_specific_fields`` of messages and tool calls.
Native Gemini items carry one under ``thought_signature``. LiteLLM reports a
text answer's signatures under ``thought_signatures`` and also encodes a tool
call's signature into the call id.
"""

from __future__ import annotations

from itertools import groupby
from typing import TYPE_CHECKING, Any

from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    OutputMessageItem,
    ReasoningItem,
    WebSearchCallItem,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from grasp_agents.types.items import InputItem

THOUGHT_SIGNATURE_KEY = "thought_signature"
_LITELLM_SIGNATURES_KEY = "thought_signatures"
_SIGNATURE_KEYS = (THOUGHT_SIGNATURE_KEY, _LITELLM_SIGNATURES_KEY)

_SIGNED_CALL_ID_MARKER = "__thought__"
_GEMINI = "gemini"


def has_thought_signature(item: OutputMessageItem | FunctionToolCallItem) -> bool:
    fields = item.provider_specific_fields or {}
    return any(key in fields for key in _SIGNATURE_KEYS)


def without_thought_signature[T: OutputMessageItem | FunctionToolCallItem](
    item: T,
) -> T:
    fields = {
        key: value
        for key, value in (item.provider_specific_fields or {}).items()
        if key not in _SIGNATURE_KEYS
    }
    return item.model_copy(update={"provider_specific_fields": fields or None})


def split_signed_call_id(call_id: str) -> tuple[str, str | None]:
    plain_id, marker, signature = call_id.partition(_SIGNED_CALL_ID_MARKER)
    if not marker or not signature:
        return call_id, None
    return plain_id, signature


def tool_call_id_and_fields(
    call_id: str, provider_specific_fields: Mapping[str, Any] | None
) -> tuple[str, dict[str, Any] | None]:
    """
    Plain call id and provider fields for a tool call LiteLLM returned.

    Other providers reject an id carrying a signature (too long, invalid
    characters), so the signature moves into ``provider_specific_fields``
    unless one is already there; other fields on the call are kept.
    """
    plain_id, id_signature = split_signed_call_id(call_id)
    fields = dict(provider_specific_fields or {})
    if id_signature:
        fields.setdefault(THOUGHT_SIGNATURE_KEY, id_signature)
    return plain_id, fields or None


def normalize_litellm_gemini_items(items: Sequence[InputItem]) -> list[InputItem]:
    """
    Plain call ids, and Gemini attribution, for model turns LiteLLM produced
    for Gemini that a transcript holds without a native provider name.

    Such a turn is recognized by a tool call id carrying a signature or a
    message carrying LiteLLM's ``thought_signatures``. Its ids are split, the
    signature moving to the call's fields, and its untagged items are
    attributed to Gemini, so other providers never receive its signatures.
    Returns copies; the given items are never modified.
    """
    normalized: list[InputItem] = []
    for is_turn, group in groupby(items, key=_is_model_turn_item):
        turn = list(group)
        if is_turn and _is_litellm_gemini_turn(turn):
            normalized.extend(_attributed_to_gemini(item) for item in turn)
        else:
            normalized.extend(_with_plain_call_id(item) for item in turn)
    return normalized


def _is_model_turn_item(item: InputItem) -> bool:
    return isinstance(
        item,
        (ReasoningItem, OutputMessageItem, FunctionToolCallItem, WebSearchCallItem),
    )


def _is_litellm_gemini_turn(turn: Sequence[InputItem]) -> bool:
    for item in turn:
        if (
            isinstance(item, FunctionToolCallItem)
            and split_signed_call_id(item.call_id)[1]
        ):
            return True
        if isinstance(item, OutputMessageItem) and _LITELLM_SIGNATURES_KEY in (
            item.provider_specific_fields or {}
        ):
            return True
    return False


def _attributed_to_gemini(item: InputItem) -> InputItem:
    if not isinstance(item, (ReasoningItem, OutputMessageItem, FunctionToolCallItem)):
        return item
    update: dict[str, Any] = {}
    if isinstance(item, FunctionToolCallItem):
        call_id, fields = tool_call_id_and_fields(
            item.call_id, item.provider_specific_fields
        )
        if call_id != item.call_id:
            update |= {"call_id": call_id, "provider_specific_fields": fields}
    if item.native_provider_name is None:
        update["native_provider_name"] = _GEMINI
    return item.model_copy(update=update) if update else item


def _with_plain_call_id(item: InputItem) -> InputItem:
    if not isinstance(item, FunctionToolOutputItem):
        return item
    call_id, _ = split_signed_call_id(item.call_id)
    if call_id == item.call_id:
        return item
    return item.model_copy(update={"call_id": call_id})
