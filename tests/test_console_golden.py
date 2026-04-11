"""
Golden file snapshot tests for EventConsole visual output.

Run with --update-golden to regenerate the .expected files:
    uv run pytest tests/test_console_golden.py --update-golden

The golden files live in tests/golden/ and are plain text (ANSI stripped).
Review them in a diff tool after regenerating to verify visual changes.
"""

import json
import re
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from grasp_agents.console import EventConsole
from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.events import (
    BackgroundTaskCompletedEvent,
    BackgroundTaskInfo,
    BackgroundTaskLaunchedEvent,
    Event,
    GenerationEndEvent,
    LLMStreamEvent,
    LLMStreamingErrorData,
    LLMStreamingErrorEvent,
    OutputMessageItemEvent,
    SystemMessageEvent,
    ToolCallItemEvent,
    ToolErrorEvent,
    ToolErrorInfo,
    ToolResultEvent,
    TurnEndEvent,
    TurnEndInfo,
    TurnInfo,
    TurnStartEvent,
    UserMessageEvent,
)
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
    OutputMessageItem,
    ReasoningItem,
)
from grasp_agents.types.llm_events import (
    FunctionCallArgumentsDelta,
    OutputItemAdded,
    OutputItemDone,
    OutputMessageTextPartTextDelta,
    ReasoningSummaryPartTextDelta,
    ResponseCompleted,
)
from grasp_agents.types.response import (
    InputTokensDetails,
    OutputTokensDetails,
    Response,
    ResponseUsage,
)

GOLDEN_DIR = Path(__file__).parent / "golden"


def _strip_ansi(text: str) -> str:
    """Remove all ANSI escape sequences for stable comparison."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


@pytest.fixture
def update_golden(request: pytest.FixtureRequest) -> bool:
    val = request.config.getoption("--update-golden")
    assert isinstance(val, bool)
    return val


def _make_console(**kwargs: Any) -> tuple[EventConsole, StringIO]:
    buf = StringIO()
    console = Console(file=buf, no_color=True, highlight=False, width=80)
    ec = EventConsole(console=console, **kwargs)
    return ec, buf


async def _collect(ec: EventConsole, events: list[Event[Any]]) -> None:
    async def gen():  # noqa: RUF029
        for e in events:
            yield e

    async for _ in ec.stream(gen()):
        pass


def _assert_golden(
    actual: str, name: str, *, update: bool
) -> None:
    """Compare actual output against golden file, or update it."""
    golden_path = GOLDEN_DIR / f"{name}.expected"
    clean = _strip_ansi(actual)

    if update:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(clean)
        pytest.skip(f"Updated golden file: {golden_path}")

    if not golden_path.exists():
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(clean)
        msg = (
            f"Golden file created: {golden_path}\n"
            "Review it and re-run the test."
        )
        pytest.fail(msg)

    expected = golden_path.read_text()
    if clean != expected:
        import difflib  # noqa: PLC0415

        diff = difflib.unified_diff(
            expected.splitlines(keepends=True),
            clean.splitlines(keepends=True),
            fromfile=f"golden/{name}.expected",
            tofile="actual output",
        )
        diff_text = "".join(diff)
        msg = (
            f"Console output doesn't match golden file.\n"
            f"Run with --update-golden to accept changes.\n\n"
            f"{diff_text}"
        )
        pytest.fail(msg)


# ── Scenario builders ──


def _full_turn_events() -> list[Event[Any]]:
    """A realistic multi-turn scenario covering every visual element."""
    text_item = OutputMessageItem(
        content_parts=[OutputMessageText(text="Let me research that for you.")],
        status="completed",
    )
    tool_item_1 = FunctionToolCallItem(
        call_id="tc_1",
        name="web_search",
        arguments='{"query": "history of the internet", "max_results": 5}',
    )
    tool_result_1 = FunctionToolOutputItem.from_tool_result(
        call_id="tc_1",
        output=(
            "1. ARPANET (1969): First network to use packet switching\n"
            "2. TCP/IP (1983): Standard protocol adopted\n"
            "3. World Wide Web (1991): Tim Berners-Lee at CERN"
        ),
    )
    tool_item_2 = FunctionToolCallItem(
        call_id="tc_2",
        name="write_document",
        arguments=json.dumps({
            "title": "Internet History",
            "content": "The Internet traces its origins to ARPANET...",
            "format": "markdown",
        }),
    )
    tool_result_2 = FunctionToolOutputItem.from_tool_result(
        call_id="tc_2",
        output="Document saved: internet_history.md",
    )
    tool_item_error = FunctionToolCallItem(
        call_id="tc_3",
        name="publish",
        arguments='{"target": "blog"}',
    )

    resp_1 = Response(
        model="claude-sonnet-4-5",
        output_items=[text_item, tool_item_1],
        usage_with_cost=ResponseUsage(
            input_tokens=1200,
            input_tokens_details=InputTokensDetails(cached_tokens=800),
            output_tokens=350,
            output_tokens_details=OutputTokensDetails(reasoning_tokens=200),
            total_tokens=1550,
            cost=0.0043,
        ),
    )
    resp_2 = Response(
        model="claude-sonnet-4-5",
        output_items=[tool_item_2],
        usage_with_cost=ResponseUsage(
            input_tokens=2100,
            input_tokens_details=InputTokensDetails(cached_tokens=1500),
            output_tokens=180,
            total_tokens=2280,
            cost=0.0067,
        ),
    )

    events: list[Event[Any]] = [
        # ── Turn 1 ──
        TurnStartEvent(data=TurnInfo(turn=0), source="coordinator"),
        # Thinking (streaming)
        LLMStreamEvent(
            data=OutputItemAdded(
                item=ReasoningItem(), output_index=0, sequence_number=1
            ),
            source="coordinator",
            exec_id="c1",
        ),
        LLMStreamEvent(
            data=ReasoningSummaryPartTextDelta(
                delta="The user wants to know about internet history.\n",
                summary_index=0,
                output_index=0,
                sequence_number=2,
                item_id="i1",
            ),
            source="coordinator",
            exec_id="c1",
        ),
        LLMStreamEvent(
            data=ReasoningSummaryPartTextDelta(
                delta="I should search for authoritative sources first.",
                summary_index=0,
                output_index=0,
                sequence_number=3,
                item_id="i1",
            ),
            source="coordinator",
            exec_id="c1",
        ),
        LLMStreamEvent(
            data=OutputItemDone(
                item=ReasoningItem(), output_index=0, sequence_number=4
            ),
            source="coordinator",
            exec_id="c1",
        ),
        # Agent text (streaming)
        LLMStreamEvent(
            data=OutputMessageTextPartTextDelta(
                content_index=0,
                delta="Let me research that for you.",
                output_index=1,
                sequence_number=5,
                item_id="i2",
                logprobs=[],
            ),
            source="coordinator",
            exec_id="c1",
        ),
        OutputMessageItemEvent(data=text_item, source="coordinator", exec_id="c1"),
        LLMStreamEvent(
            data=OutputItemDone(item=text_item, output_index=1, sequence_number=6),
            source="coordinator",
            exec_id="c1",
        ),
        # Tool call 1: web_search
        LLMStreamEvent(
            data=OutputItemAdded(
                item=tool_item_1, output_index=2, sequence_number=7
            ),
            source="coordinator",
            exec_id="c1",
        ),
        LLMStreamEvent(
            data=FunctionCallArgumentsDelta(
                delta='{"query": "history of the internet", "max_results": 5}',
                output_index=2,
                sequence_number=8,
                item_id="i3",
            ),
            source="coordinator",
            exec_id="c1",
        ),
        ToolCallItemEvent(data=tool_item_1, source="coordinator", exec_id="c1"),
        LLMStreamEvent(
            data=OutputItemDone(
                item=tool_item_1, output_index=2, sequence_number=9
            ),
            source="coordinator",
            exec_id="c1",
        ),
        LLMStreamEvent(
            data=ResponseCompleted(response=resp_1, sequence_number=10),
            source="coordinator",
            exec_id="c1",
        ),
        # Generation end with usage
        GenerationEndEvent(data=resp_1, source="coordinator", exec_id="c1"),
        # Tool result 1
        ToolResultEvent(data=tool_result_1, source="coordinator", exec_id="c1"),
        # ── Turn 2 ──
        TurnStartEvent(data=TurnInfo(turn=1), source="coordinator"),
        # Tool call 2: write_document (no streaming, direct)
        ToolCallItemEvent(data=tool_item_2, source="coordinator", exec_id="c2"),
        GenerationEndEvent(data=resp_2, source="coordinator", exec_id="c2"),
        ToolResultEvent(data=tool_result_2, source="coordinator", exec_id="c2"),
        # Tool call 3: publish (will error)
        ToolCallItemEvent(
            data=tool_item_error, source="coordinator", exec_id="c3"
        ),
        ToolErrorEvent(
            data=ToolErrorInfo(
                tool_name="publish",
                error="Authentication failed: invalid API key",
                timed_out=False,
            ),
            source="coordinator",
            exec_id="c3",
        ),
        # Turn end
        TurnEndEvent(
            data=TurnEndInfo(turn=1, had_tool_calls=True, stop_reason="max_turns"),  # type: ignore[arg-type]
            source="coordinator",
        ),
    ]
    return events


def _input_messages_events() -> list[Event[Any]]:
    """Scenario for user/system message display."""
    return [
        TurnStartEvent(data=TurnInfo(turn=0), source="tutor"),
        UserMessageEvent(
            data=InputMessageItem.from_text(
                "Explain quantum computing in simple terms.", role="user"
            ),
            source="tutor",
            exec_id="c1",
        ),
        SystemMessageEvent(
            data=InputMessageItem.from_text(
                "You are a helpful tutor. Keep explanations under 3 sentences.",
                role="system",
            ),
            source="tutor",
            exec_id="c1",
        ),
    ]


def _background_task_events() -> list[Event[Any]]:
    """Scenario for background task lifecycle."""
    return [
        TurnStartEvent(data=TurnInfo(turn=0), source="agent"),
        BackgroundTaskLaunchedEvent(
            data=BackgroundTaskInfo(
                task_id="bg_1",
                tool_name="long_running_analysis",
                tool_call_id="tc_bg1",
            ),
            source="agent",
            exec_id="c1",
        ),
        LLMStreamEvent(
            data=OutputMessageTextPartTextDelta(
                content_index=0,
                delta="Working on your request while the analysis runs...",
                output_index=0,
                sequence_number=1,
                item_id="i1",
                logprobs=[],
            ),
            source="agent",
            exec_id="c1",
        ),
        BackgroundTaskCompletedEvent(
            data=BackgroundTaskInfo(
                task_id="bg_1",
                tool_name="long_running_analysis",
                tool_call_id="tc_bg1",
            ),
            source="agent",
            exec_id="c1",
        ),
    ]


def _error_events() -> list[Event[Any]]:
    """Scenario for streaming error display."""
    return [
        TurnStartEvent(data=TurnInfo(turn=0), source="agent"),
        LLMStreamingErrorEvent(
            data=LLMStreamingErrorData(
                error=RuntimeError("Connection reset by peer"),
                model_name="claude-sonnet-4-5",
            ),
            source="agent",
            exec_id="c1",
        ),
    ]


# ── Tests ──


class TestGoldenFullTurn:
    @pytest.mark.asyncio
    async def test_full_turn(self, update_golden: bool) -> None:
        """Complete multi-turn scenario with all element types."""
        ec, buf = _make_console(show_thinking=True)
        await _collect(ec, _full_turn_events())
        _assert_golden(buf.getvalue(), "full_turn", update=update_golden)

    @pytest.mark.asyncio
    async def test_full_turn_no_thinking(self, update_golden: bool) -> None:
        """Same scenario with thinking hidden (default)."""
        ec, buf = _make_console(show_thinking=False)
        await _collect(ec, _full_turn_events())
        _assert_golden(
            buf.getvalue(), "full_turn_no_thinking", update=update_golden
        )

    @pytest.mark.asyncio
    async def test_input_messages(self, update_golden: bool) -> None:
        """User and system message panels."""
        ec, buf = _make_console(show_input_messages=True)
        await _collect(ec, _input_messages_events())
        _assert_golden(buf.getvalue(), "input_messages", update=update_golden)

    @pytest.mark.asyncio
    async def test_background_tasks(self, update_golden: bool) -> None:
        """Background task launch and completion."""
        ec, buf = _make_console()
        await _collect(ec, _background_task_events())
        _assert_golden(buf.getvalue(), "background_tasks", update=update_golden)

    @pytest.mark.asyncio
    async def test_streaming_error(self, update_golden: bool) -> None:
        """LLM streaming error display."""
        ec, buf = _make_console()
        await _collect(ec, _error_events())
        _assert_golden(buf.getvalue(), "streaming_error", update=update_golden)
