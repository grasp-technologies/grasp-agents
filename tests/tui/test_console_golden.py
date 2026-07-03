"""
Golden file snapshot tests for EventConsole visual output.

Run with --update-golden to regenerate the .expected files:
    uv run pytest tests/tui/test_console_golden.py --update-golden

The golden files live in tests/tui/golden/ and are plain text (ANSI stripped).
Review them in a diff tool after regenerating to verify visual changes.
"""

import re
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.events import (
    BackgroundTaskCompletedEvent,
    BackgroundTaskInfo,
    BackgroundTaskLaunchedEvent,
    Event,
    LLMStreamEvent,
    LLMStreamingErrorData,
    LLMStreamingErrorEvent,
    OutputMessageItemEvent,
    SystemMessageEvent,
    TurnInfo,
    TurnStartEvent,
    UserMessageEvent,
)
from grasp_agents.types.items import InputMessageItem, OutputMessageItem
from grasp_agents.types.llm_events import OutputMessageTextPartTextDelta
from grasp_agents.ui.console import EventConsole
from grasp_agents.ui.demo import console_demo_events

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
    async def gen():
        for e in events:
            yield e

    async for _ in ec.stream(gen()):
        pass


def _assert_golden(actual: str, name: str, *, update: bool) -> None:
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
        msg = f"Golden file created: {golden_path}\nReview it and re-run the test."
        pytest.fail(msg)

    expected = golden_path.read_text()
    if clean != expected:
        import difflib

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
        # The message completes (its promoted event) — under the markdown=True
        # default the finished text renders here, not the live delta above.
        OutputMessageItemEvent(
            data=OutputMessageItem(
                id="i1",
                content=[
                    OutputMessageText(
                        text="Working on your request while the analysis runs..."
                    )
                ],
                status="completed",
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
        await _collect(ec, console_demo_events())
        _assert_golden(buf.getvalue(), "full_turn", update=update_golden)

    @pytest.mark.asyncio
    async def test_full_turn_no_thinking(self, update_golden: bool) -> None:
        """Same scenario with thinking hidden (default)."""
        ec, buf = _make_console(show_thinking=False)
        await _collect(ec, console_demo_events())
        _assert_golden(buf.getvalue(), "full_turn_no_thinking", update=update_golden)

    @pytest.mark.asyncio
    async def test_input_messages(self, update_golden: bool) -> None:
        """User and system message panels."""
        ec, buf = _make_console(show_input_messages=True, show_system_messages=True)
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
