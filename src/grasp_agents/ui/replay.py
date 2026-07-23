"""
Persisted transcript items → displayable events.

The live loop promotes each completed item to an item event; a persisted
transcript holds the items themselves. This rebuilds the matching events so a
UI relaunching over an existing session renders history through the exact same
path as a live run. Textual-free, like ``_event_render``.

Not everything replays: per-generation usage (not persisted), streaming
partials, and the ephemeral system prompt (not part of the transcript) are
gone; turn boundaries are re-derived — one turn per model response.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from grasp_agents.types.events import (
    Event,
    OutputMessageItemEvent,
    ReasoningItemEvent,
    SystemMessageEvent,
    ToolCallItemEvent,
    ToolOutputItemEvent,
    TurnInfo,
    TurnStartEvent,
    UserMessageEvent,
    WebSearchCallItemEvent,
)
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
    OutputMessageItem,
    ReasoningItem,
    WebSearchCallItem,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from grasp_agents.types.items import InputItem

__all__ = ["history_events"]

# Items a model response produces — a run of these is one generation, headed by
# a synthesized turn boundary on replay.
_GENERATED = (OutputMessageItem, ReasoningItem, FunctionToolCallItem, WebSearchCallItem)


def history_events(
    messages: Sequence[InputItem], *, agent: str, first_turn: int = 1
) -> list[Event[Any]]:
    """
    Rebuild the item events a transcript's messages were promoted from.

    ``agent`` stamps each event's ``source``/``destination`` the way the live
    loop does, so pane routing matches. A ``TurnStartEvent`` is synthesized
    before each generation block (numbered from ``first_turn``); a tool
    output's tool name is recovered from its call.
    """
    events: list[Event[Any]] = []
    call_names: dict[str, str] = {}
    turn = first_turn - 1
    in_generation = False
    for item in messages:
        if isinstance(item, _GENERATED) and not in_generation:
            turn += 1
            events.append(TurnStartEvent(source=agent, data=TurnInfo(turn=turn)))
        in_generation = isinstance(item, _GENERATED)

        if isinstance(item, InputMessageItem):
            if item.role == "user":
                events.append(UserMessageEvent(destination=agent, data=item))
            else:
                events.append(SystemMessageEvent(source=agent, data=item))
        elif isinstance(item, OutputMessageItem):
            events.append(OutputMessageItemEvent(source=agent, data=item))
        elif isinstance(item, ReasoningItem):
            events.append(ReasoningItemEvent(source=agent, data=item))
        elif isinstance(item, FunctionToolCallItem):
            call_names[item.call_id] = item.name
            events.append(ToolCallItemEvent(source=agent, data=item))
        elif isinstance(item, FunctionToolOutputItem):
            tool = call_names.get(item.call_id, "tool")
            events.append(
                ToolOutputItemEvent(source=tool, destination=agent, data=item)
            )
        elif isinstance(item, WebSearchCallItem):
            events.append(WebSearchCallItemEvent(source=agent, data=item))
    return events
