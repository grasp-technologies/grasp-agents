"""
Live LLM-token / tool-output streaming in the TUI (headless Pilot).

When the agent streams, deltas accumulate into a single live widget that is
finalised by the matching item event (no duplicate); when it doesn't stream, the
item events render as before.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("textual")

from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.events import (
    LLMStreamEvent,
    OutputMessageItemEvent,
    ToolOutputItemEvent,
    ToolStreamEvent,
    TurnInfo,
    TurnStartEvent,
)
from grasp_agents.types.items import FunctionToolOutputItem, OutputMessageItem
from grasp_agents.types.llm_events import OutputMessageTextPartTextDelta
from grasp_agents.ui.app import GraspAgentsApp, _SelectableStatic


def _llm_delta(text: str, n: int) -> LLMStreamEvent:
    return LLMStreamEvent(
        data=OutputMessageTextPartTextDelta(
            item_id="m1",
            content_index=0,
            output_index=0,
            sequence_number=n,
            delta=text,
        ),
        source="analyst",
    )


def _rendered(widget: _SelectableStatic) -> str:
    return "\n".join(widget.render_line(y).text for y in range(widget.size.height))


@pytest.mark.asyncio
async def test_llm_tokens_accumulate_while_streaming() -> None:
    async def stream():
        yield TurnStartEvent(data=TurnInfo(turn=0), source="analyst")
        yield _llm_delta("Hello ", 0)
        yield _llm_delta("world", 1)  # no final item yet

    app = GraspAgentsApp(stream())
    async with app.run_test() as pilot:
        await app.wait_for_stream()
        await pilot.pause()
        assert app._ga_stream_msg_text.get("analyst") == "Hello world"
        assert "analyst" in app._ga_stream_msg


@pytest.mark.asyncio
async def test_llm_stream_finalizes_into_one_widget() -> None:
    async def stream():
        yield TurnStartEvent(data=TurnInfo(turn=0), source="analyst")
        yield _llm_delta("Hello ", 0)
        yield _llm_delta("world", 1)
        yield OutputMessageItemEvent(
            data=OutputMessageItem(
                content_parts=[OutputMessageText(text="Hello world")],
                status="completed",
            ),
            source="analyst",
        )

    app = GraspAgentsApp(stream())
    async with app.run_test() as pilot:
        await app.wait_for_stream()
        await pilot.pause()
        # finalised: live tracker cleared, and exactly ONE message widget (the
        # streamed one updated in place — not a second mounted on completion)
        assert app._ga_stream_msg == {}
        msgs = list(app.query(".ga-msg"))
        assert len(msgs) == 1, msgs
        assert "Hello world" in _rendered(msgs[0])  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_tool_output_streams_and_finalizes() -> None:
    async def stream():
        yield TurnStartEvent(data=TurnInfo(turn=0), source="analyst")
        yield ToolStreamEvent(data="line1\n", source="RunPython")
        yield ToolStreamEvent(data="line2\n", source="RunPython")
        yield ToolOutputItemEvent(
            data=FunctionToolOutputItem.from_tool_result(
                call_id="r1", output=json.dumps({"result": "done"})
            ),
            source="RunPython",
            destination="analyst",
        )

    app = GraspAgentsApp(stream())
    async with app.run_test() as pilot:
        await app.wait_for_stream()
        await pilot.pause()
        assert app._ga_stream_tool == {}  # finalised by the tool result


@pytest.mark.asyncio
async def test_tool_output_accumulates_while_streaming() -> None:
    async def stream():
        yield TurnStartEvent(data=TurnInfo(turn=0), source="analyst")
        yield ToolStreamEvent(data="line1\n", source="RunPython")
        yield ToolStreamEvent(data="line2\n", source="RunPython")

    app = GraspAgentsApp(stream())
    async with app.run_test() as pilot:
        await app.wait_for_stream()
        await pilot.pause()
        assert app._ga_stream_tool_text.get("analyst") == "line1\nline2\n"
        assert "analyst" in app._ga_stream_tool
