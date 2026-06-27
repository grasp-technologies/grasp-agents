"""Headless tests for the context-token meter (input tokens / window)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

pytest.importorskip("textual")

from textual.widgets import Static

from grasp_agents.types.events import (
    CompactionEvent,
    CompactionInfo,
    Event,
    GenerationEndEvent,
    TurnEndEvent,
    TurnEndInfo,
    TurnInfo,
    TurnStartEvent,
)
from grasp_agents.types.response import (
    InputTokensDetails,
    OutputTokensDetails,
    Response,
    ResponseUsage,
)
from grasp_agents.ui.app import GraspAgentsApp, PromptArea


def _gen(model: str, input_tokens: int) -> GenerationEndEvent:
    return GenerationEndEvent(
        source="analyst",
        exec_id="e",
        data=Response(
            model=model,
            output_items=[],
            usage_with_cost=ResponseUsage(
                input_tokens=input_tokens,
                input_tokens_details=InputTokensDetails(cached_tokens=0),
                output_tokens=10,
                output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                total_tokens=input_tokens + 10,
                cost=0.0,
            ),
        ),
    )


def _agent(extra: list[Event[Any]]):
    async def _on_submit(text: str) -> AsyncIterator[Event[Any]]:
        del text
        yield TurnStartEvent(data=TurnInfo(turn=0), source="analyst")
        for event in extra:
            yield event
        yield TurnEndEvent(
            data=TurnEndInfo(turn=0, had_tool_calls=False, stop_reason="final_answer"),  # type: ignore[arg-type]
            source="analyst",
        )

    return _on_submit


def _render_to_text(renderable: object) -> str:
    from rich.console import Console

    console = Console(width=200, record=True)
    console.print(renderable)
    return console.export_text()


def _widget_renderable(widget: object) -> object:
    # Textual wraps a widget's content in a RichVisual; unwrap to the Rich
    # renderable it was given so we can render it to text.
    shown = widget.render()  # type: ignore[attr-defined]
    return getattr(shown, "_renderable", shown)


def _meter_text(app: GraspAgentsApp) -> str:
    shown = _widget_renderable(app.query_one("#context-meter", Static))
    return shown.plain if hasattr(shown, "plain") else _render_to_text(shown)


async def _submit(pilot: Any, app: GraspAgentsApp, text: str = "hi") -> None:
    app.query_one("#prompt", PromptArea).text = text
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    await pilot.pause()


@pytest.mark.asyncio
async def test_meter_shows_tokens_over_window() -> None:
    app = GraspAgentsApp(on_submit=_agent([_gen("m", 1000)]), main_agent="analyst")
    app.seed_context_window("analyst", 2000)  # else the window is inferred
    async with app.run_test() as pilot:
        await pilot.pause()
        await _submit(pilot, app)
        assert app._ga_input_tokens["analyst"] == 1000
        text = _meter_text(app)
        assert "1,000 / 2,000 tokens" in text
        assert "50%" in text


@pytest.mark.asyncio
async def test_meter_updates_to_post_fold_size_on_compaction() -> None:
    extra: list[Event[Any]] = [
        _gen("m", 1900),
        CompactionEvent(
            source="analyst",
            exec_id="e",
            data=CompactionInfo(
                folded_turns=5,
                preserved_turns=2,
                context_tokens=400,
                context_window=2000,
            ),
        ),
    ]
    app = GraspAgentsApp(on_submit=_agent(extra), main_agent="analyst")
    async with app.run_test() as pilot:
        await pilot.pause()
        await _submit(pilot, app)
        # the compaction event's post-fold size supersedes the pre-fold generation
        assert app._ga_input_tokens["analyst"] == 400
        assert "400 / 2,000 tokens" in _meter_text(app)


@pytest.mark.asyncio
async def test_meter_without_window_shows_bare_count() -> None:
    # No override and an unknown model → no denominator, just the token count.
    app = GraspAgentsApp(
        on_submit=_agent([_gen("totally-unknown-model-xyz", 1234)]),
        main_agent="analyst",
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await _submit(pilot, app)
        text = _meter_text(app)
        assert "1,234 tokens" in text
        assert "/" not in text


def _widget_text(widget: object) -> str:
    return _render_to_text(_widget_renderable(widget))


@pytest.mark.asyncio
async def test_compaction_event_renders_summary_in_pane() -> None:
    from grasp_agents.ui._widgets import SelectableStatic

    extra: list[Event[Any]] = [
        CompactionEvent(
            source="analyst",
            exec_id="e",
            data=CompactionInfo(
                folded_turns=3,
                preserved_turns=2,
                context_tokens=400,
                context_window=2000,
                summary="KEPT THE GOAL AND KEY FACTS",
            ),
        ),
    ]
    app = GraspAgentsApp(on_submit=_agent(extra), main_agent="analyst")
    async with app.run_test() as pilot:
        await pilot.pause()
        await _submit(pilot, app)
        from rich.align import Align

        pane = app._ga_panes["analyst"]
        compaction = [
            w
            for w in pane.query(SelectableStatic)
            if "context compacted" in _widget_text(w)
        ]
        assert len(compaction) == 1
        widget = compaction[0]
        assert "KEPT THE GOAL AND KEY FACTS" in _widget_text(widget)
        # incoming message (summary injected into context) → right-aligned
        assert isinstance(_widget_renderable(widget), Align)
