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

from grasp_agents.types.content import OutputMessageText, ReasoningSummary
from grasp_agents.types.events import (
    BackgroundTaskCompletedEvent,
    BackgroundTaskInfo,
    BackgroundTaskLaunchedEvent,
    GenerationEndEvent,
    LLMStreamEvent,
    OutputMessageItemEvent,
    ReasoningItemEvent,
    ToolCallItemEvent,
    ToolOutputItemEvent,
    ToolStreamEvent,
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
    OutputMessageTextPartTextDelta,
    ReasoningSummaryPartTextDelta,
    ResponseFallback,
    ResponseRetrying,
)
from grasp_agents.types.response import Response
from grasp_agents.ui.app import GraspAgentsApp, _pane_id, _PromptArea, _SelectableStatic


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


def _think_delta(text: str, n: int) -> LLMStreamEvent:
    return LLMStreamEvent(
        data=ReasoningSummaryPartTextDelta(
            item_id="r1",
            summary_index=0,
            output_index=0,
            sequence_number=n,
            delta=text,
        ),
        source="analyst",
    )


def _retry(n: int) -> LLMStreamEvent:
    return LLMStreamEvent(
        data=ResponseRetrying(attempt=1, error="boom", sequence_number=n),
        source="analyst",
    )


def _fallback(n: int) -> LLMStreamEvent:
    return LLMStreamEvent(
        data=ResponseFallback(
            failed_model="gpt-x",
            fallback_model="claude-y",
            error_type="RateLimit",
            attempt=1,
            sequence_number=n,
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
async def test_thinking_tokens_accumulate_while_streaming() -> None:
    async def stream():
        yield TurnStartEvent(data=TurnInfo(turn=0), source="analyst")
        yield _think_delta("Let me ", 0)
        yield _think_delta("think", 1)  # no final reasoning item yet

    app = GraspAgentsApp(stream())
    async with app.run_test() as pilot:
        await app.wait_for_stream()
        await pilot.pause()
        assert app._ga_stream_think_text.get("analyst") == "Let me think"
        assert "analyst" in app._ga_stream_think


@pytest.mark.asyncio
async def test_thinking_stream_finalizes_into_one_widget() -> None:
    async def stream():
        yield TurnStartEvent(data=TurnInfo(turn=0), source="analyst")
        yield _think_delta("Let me ", 0)
        yield _think_delta("think", 1)
        yield ReasoningItemEvent(
            data=ReasoningItem(
                summary_parts=[ReasoningSummary(text="Let me think")],
                status="completed",
            ),
            source="analyst",
        )

    app = GraspAgentsApp(stream())
    async with app.run_test() as pilot:
        await app.wait_for_stream()
        await pilot.pause()
        # finalised: live tracker cleared, exactly one (thinking) widget
        assert app._ga_stream_think == {}
        msgs = list(app.query(".ga-msg"))
        assert len(msgs) == 1, msgs
        assert "Let me think" in _rendered(msgs[0])  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_response_retrying_discards_partial_thinking() -> None:
    async def stream():
        yield TurnStartEvent(data=TurnInfo(turn=0), source="analyst")
        yield _think_delta("dropped thought", 0)
        yield _retry(1)

    app = GraspAgentsApp(stream())
    async with app.run_test() as pilot:
        await app.wait_for_stream()
        await pilot.pause()
        assert "analyst" not in app._ga_stream_think
        assert "analyst" not in app._ga_stream_think_text


@pytest.mark.asyncio
async def test_response_retrying_discards_partial_message() -> None:
    """A failed attempt's partial deltas are dropped on ResponseRetrying."""

    async def stream():
        yield TurnStartEvent(data=TurnInfo(turn=0), source="analyst")
        yield _llm_delta("dropped partial", 0)
        yield _retry(1)

    app = GraspAgentsApp(stream())
    async with app.run_test() as pilot:
        await app.wait_for_stream()
        await pilot.pause()
        assert "analyst" not in app._ga_stream_msg
        assert "analyst" not in app._ga_stream_msg_text
        assert list(app.query(".ga-msg")) == []
        # a visible retry notice replaces the silently-cleared text
        notices = list(app.query(".ga-notice"))
        assert len(notices) == 1
        assert "retrying" in _rendered(notices[0])  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_response_fallback_shows_notice_and_discards_partial() -> None:
    async def stream():
        yield TurnStartEvent(data=TurnInfo(turn=0), source="analyst")
        yield _llm_delta("partial", 0)
        yield _fallback(1)

    app = GraspAgentsApp(stream())
    async with app.run_test() as pilot:
        await app.wait_for_stream()
        await pilot.pause()
        assert list(app.query(".ga-msg")) == []
        notices = list(app.query(".ga-notice"))
        assert len(notices) == 1
        assert "falling back" in _rendered(notices[0])  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_retry_then_fresh_stream_shows_only_retried_content() -> None:
    async def stream():
        yield TurnStartEvent(data=TurnInfo(turn=0), source="analyst")
        yield _llm_delta("will be ", 0)
        yield _llm_delta("dropped", 1)
        yield _retry(2)
        yield _llm_delta("the real ", 3)
        yield _llm_delta("answer", 4)
        yield OutputMessageItemEvent(
            data=OutputMessageItem(
                content_parts=[OutputMessageText(text="the real answer")],
                status="completed",
            ),
            source="analyst",
        )

    app = GraspAgentsApp(stream())
    async with app.run_test() as pilot:
        await app.wait_for_stream()
        await pilot.pause()
        msgs = list(app.query(".ga-msg"))
        assert len(msgs) == 1, msgs
        rendered = _rendered(msgs[0])  # type: ignore[arg-type]
        assert "the real answer" in rendered
        assert "dropped" not in rendered


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


def _pane_widget_texts(app: GraspAgentsApp, source: str) -> list[str]:
    """Plain text of each child widget in *source*'s pane, in DOM (mount) order."""
    from rich.console import Console
    from textual.widgets import Static

    console = Console(width=100, no_color=True)
    pane = app.query_one(f"#{_pane_id(source)}")
    texts: list[str] = []
    for w in pane.query(Static):
        with console.capture() as cap:
            console.print(getattr(w, "_Static__content", ""))
        texts.append(cap.get())
    return texts


@pytest.mark.asyncio
async def test_backgrounded_tool_stream_does_not_displace_next_tool() -> None:
    """
    A backgrounded tool's drained live output ends with a background-completion
    notice (not a tool-result item), so its stream widget is never finalised by
    a result event. The next tool's output must still render below its own call
    box — not overwrite the leaked widget sitting above it (regression: a
    subagent's RunPython result rendered before its call, after a backgrounded
    Bash job).
    """

    async def stream():
        # turn 0: call Bash, which backgrounds (its result is a launch note)
        yield TurnStartEvent(data=TurnInfo(turn=0), source="de")
        yield ToolCallItemEvent(
            data=FunctionToolCallItem(call_id="b1", name="Bash", arguments="{}"),
            source="de",
            destination="Bash",
        )
        yield GenerationEndEvent(data=Response(output_items=[], model="m"), source="de")
        yield BackgroundTaskLaunchedEvent(
            data=BackgroundTaskInfo(
                task_id="bg_1",
                tool_name="Bash",
                tool_call_id="b1",
                output_name="b1.log",
            ),
            source="de",
        )
        yield ToolOutputItemEvent(
            data=FunctionToolOutputItem.from_tool_result(
                call_id="b1", output="moved to background (id: bg_1)"
            ),
            source="Bash",
            destination="de",
        )
        yield TurnEndEvent(data=TurnEndInfo(turn=0, had_tool_calls=True), source="de")
        # turn 1 drain: Bash's live output + completion (no tool-result item)
        yield ToolStreamEvent(data="batch1\n", source="Bash")
        yield ToolStreamEvent(data="batch2\n", source="Bash")
        yield BackgroundTaskCompletedEvent(
            data=BackgroundTaskInfo(
                task_id="bg_1", tool_name="Bash", tool_call_id="b1"
            ),
            source="de",
        )
        yield UserMessageEvent(
            data=InputMessageItem.from_text("<task_notification/>", role="user"),
            source="Bash",
            destination="de",
        )
        # turn 1 act: a fresh streaming tool call
        yield TurnStartEvent(data=TurnInfo(turn=1), source="de")
        yield ToolCallItemEvent(
            data=FunctionToolCallItem(call_id="p1", name="RunPython", arguments="{}"),
            source="de",
            destination="RunPython",
        )
        yield GenerationEndEvent(data=Response(output_items=[], model="m"), source="de")
        yield ToolStreamEvent(data="loading csv\n", source="RunPython")
        yield ToolOutputItemEvent(
            data=FunctionToolOutputItem.from_tool_result(
                call_id="p1", output=json.dumps({"result": "RESULT_RUNPYTHON"})
            ),
            source="RunPython",
            destination="de",
        )

    app = GraspAgentsApp(stream())
    async with app.run_test():
        await app.wait_for_stream()
        texts = _pane_widget_texts(app, "de")
        call_idx = next((i for i, t in enumerate(texts) if "→ RunPython" in t), None)
        out_idx = next(
            (i for i, t in enumerate(texts) if "RESULT_RUNPYTHON" in t), None
        )
        assert call_idx is not None
        assert out_idx is not None
        # RunPython's output renders below its own call box ...
        assert call_idx < out_idx, texts
        # ... and the backgrounded Bash output it streamed is preserved, not
        # overwritten in place by the next tool's result.
        bash_stream = next((t for t in texts if "batch1" in t and "batch2" in t), None)
        assert bash_stream is not None, texts
        # ... and it is headed by its log file (not dressed up as a real
        # `de ← Bash` tool result the agent received).
        assert "b1.log" in bash_stream  # task_log_name(call_id="b1")
        assert "← Bash" not in bash_stream


@pytest.mark.asyncio
async def test_tool_stream_routes_to_destination_not_last_agent() -> None:
    """
    A backgrounded tool's stream event is stamped with its owning agent
    (``BaseTool.run_stream`` sets ``destination``). The UI must route it to that
    agent's pane by destination even when a *different* agent generated most
    recently — not fall back to "last active agent" (the bug for nested /
    parallel sub-agents, where a bubbled stream arrives decoupled in time).
    """

    async def stream():
        # 'de' generates and launches a backgrounded tool (becomes a known agent)
        yield TurnStartEvent(data=TurnInfo(turn=0), source="de")
        yield ToolCallItemEvent(
            data=FunctionToolCallItem(call_id="b1", name="Bash", arguments="{}"),
            source="de",
            destination="Bash",
        )
        yield GenerationEndEvent(
            data=Response(output_items=[], model="m"), source="de"
        )
        # 'analyst' generates AFTER 'de', so it is now the most-recent agent
        yield TurnStartEvent(data=TurnInfo(turn=0), source="analyst")
        yield GenerationEndEvent(
            data=Response(output_items=[], model="m"), source="analyst"
        )
        # 'de's backgrounded output bubbles now, stamped with destination='de'
        yield ToolStreamEvent(data="DE_STDOUT\n", source="Bash", destination="de")

    app = GraspAgentsApp(stream())
    async with app.run_test():
        await app.wait_for_stream()
        de_texts = _pane_widget_texts(app, "de")
        analyst_texts = _pane_widget_texts(app, "analyst")
        # routed by destination ('de'), not by last-active agent ('analyst')
        assert any("DE_STDOUT" in t for t in de_texts), de_texts
        assert not any("DE_STDOUT" in t for t in analyst_texts), analyst_texts


async def _noop_submit(_text: str):
    # an async generator (has `yield`) that yields nothing — a no-op on_submit
    for _ in ():
        yield


@pytest.mark.asyncio
async def test_paste_grows_prompt_height() -> None:
    from textual import events

    app = GraspAgentsApp(on_submit=_noop_submit)
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", _PromptArea)
        # deliver the paste once, as the driver does (posting the bubbling Paste
        # to the widget loops via the app's paste-forwarding)
        await prompt._on_paste(events.Paste("alpha\nbeta\ngamma"))
        await pilot.pause()
        assert prompt.document.line_count == 3
        assert int(prompt.styles.height.value) == 3  # box grew to fit the paste


@pytest.mark.asyncio
async def test_alt_backspace_deletes_word() -> None:
    app = GraspAgentsApp(on_submit=_noop_submit)
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", _PromptArea)
        prompt.text = "hello world"
        prompt.move_cursor(prompt.document.end)
        await pilot.press("alt+backspace")
        await pilot.pause()
        assert prompt.text == "hello "
