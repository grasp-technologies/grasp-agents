"""Persisted transcript items → replayed item events (``ui/replay.py``)."""

from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.events import (
    OutputMessageItemEvent,
    ReasoningItemEvent,
    SystemMessageEvent,
    ToolCallItemEvent,
    ToolOutputItemEvent,
    TurnStartEvent,
    UserMessageEvent,
)
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputItem,
    InputMessageItem,
    OutputMessageItem,
    ReasoningItem,
)
from grasp_agents.ui.replay import history_events


def _answer(text: str) -> OutputMessageItem:
    return OutputMessageItem(status="completed", content=[OutputMessageText(text=text)])


def test_replays_items_with_turn_boundaries_and_tool_names() -> None:
    items: list[InputItem] = [
        InputMessageItem.from_text("question"),
        ReasoningItem.from_reasoning_content("hmm"),
        FunctionToolCallItem(call_id="c1", name="search", arguments="{}"),
        FunctionToolOutputItem.from_tool_result("c1", "result"),
        _answer("answer"),
    ]

    events = history_events(items, agent="lead")

    # One synthesized turn per model response: the reasoning+call block, then
    # (after the tool result) the final answer.
    assert [type(e) for e in events] == [
        UserMessageEvent,
        TurnStartEvent,
        ReasoningItemEvent,
        ToolCallItemEvent,
        ToolOutputItemEvent,
        TurnStartEvent,
        OutputMessageItemEvent,
    ]
    turns = [e.data.turn for e in events if isinstance(e, TurnStartEvent)]
    assert turns == [1, 2]

    user = events[0]
    assert isinstance(user, UserMessageEvent)
    assert user.source is None
    assert user.destination == "lead"

    tool_out = next(e for e in events if isinstance(e, ToolOutputItemEvent))
    assert tool_out.source == "search"  # recovered from the call by call_id
    assert tool_out.destination == "lead"


def test_system_role_and_unknown_call_ids() -> None:
    items: list[InputItem] = [
        InputMessageItem.from_text("be brief", role="system"),
        FunctionToolOutputItem.from_tool_result("orphan", "late result"),
    ]

    events = history_events(items, agent="lead")

    assert isinstance(events[0], SystemMessageEvent)
    assert events[0].source == "lead"
    out = events[1]
    assert isinstance(out, ToolOutputItemEvent)
    assert out.source == "tool"  # call not in the replayed slice


def test_first_turn_offsets_the_numbering() -> None:
    events = history_events([_answer("hi")], agent="a", first_turn=5)
    assert isinstance(events[0], TurnStartEvent)
    assert events[0].data.turn == 5
