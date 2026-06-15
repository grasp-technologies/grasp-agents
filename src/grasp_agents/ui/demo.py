"""
Deterministic demo event streams — fixed event lists that exercise every render
path, shared as test fixtures and runnable visual checks.

* :func:`demo_event_list` — a coordinator delegating to two subagents, a
  background task, and a tool returning an image. Multi-source, so it drives the
  pane-per-agent ``tui`` app. Fixture for the headless ``tests/tui`` suite
  (pilot + SVG snapshot).
* :func:`console_demo_events` — a single agent streaming text + thinking through
  tool calls, results, an error, and usage. Exercises the ``console`` linear
  stream (token deltas, thinking gutter). Fixture for the golden tests and the
  ``python -m grasp_agents.ui.demo`` visual check.

For a real (LLM + sandbox) end-to-end demo see
``grasp_agents.ui.examples.data_copilot``.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from grasp_agents.types.content import OutputMessageText, ReasoningSummary
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
    ReasoningItemEvent,
    SystemMessageEvent,
    ToolCallItemEvent,
    ToolErrorEvent,
    ToolErrorInfo,
    ToolOutputItemEvent,
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

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _resp(model: str, items: list[Any], *, cost: float) -> Response:
    return Response(
        model=model,
        output_items=items,
        usage_with_cost=ResponseUsage(
            input_tokens=1500,
            input_tokens_details=InputTokensDetails(cached_tokens=900),
            output_tokens=220,
            output_tokens_details=OutputTokensDetails(reasoning_tokens=120),
            total_tokens=1720,
            cost=cost,
        ),
    )


def _make_sample_image() -> str | None:
    try:
        from PIL import Image, ImageDraw  # noqa: PLC0415
    except ImportError:
        return None
    path = str(Path(tempfile.gettempdir()) / "grasp_tui_demo_chart.png")
    img = Image.new("RGB", (240, 150), "#10131c")
    draw = ImageDraw.Draw(img)
    bars = [40, 95, 60, 130, 80]
    colors = ["#AAACFA", "#BEE4F7", "#3BBF69", "#BFB53B", "#FCA9A9"]
    for i, (h, c) in enumerate(zip(bars, colors, strict=False)):
        x = 20 + i * 44
        draw.rectangle([x, 140 - h, x + 32, 140], fill=c)
    img.save(path)
    return path


def demo_event_list(*, with_image: bool = True) -> list[Event[Any]]:
    """
    The fixed event sequence (used by the demo and by snapshot tests).

    ``with_image=False`` drops the chart image — snapshot tests use it so the
    rendered image (chafa symbol-art when installed, else a half-block fallback)
    can't make the committed SVG vary across environments.
    """
    img_path = _make_sample_image() if with_image else None
    chart_out: dict[str, Any] = {"chart": "bar", "bars": 5}
    if img_path:
        chart_out["image_path"] = img_path

    search = FunctionToolCallItem(
        call_id="r1",
        name="web_search",
        arguments='{"query": "ARPANET history milestones", "max_results": 5}',
    )
    chart = FunctionToolCallItem(
        call_id="w1",
        name="make_chart",
        arguments='{"kind": "bar", "series": "adoption"}',
    )

    return [
        SystemMessageEvent(
            data=InputMessageItem.from_text(
                "You are a coordinator. Delegate research to the researcher "
                "subagent and writing to the writer subagent.",
                role="system",
            ),
            source="coordinator",
            exec_id="c1",
        ),
        UserMessageEvent(
            data=InputMessageItem.from_text(
                "Research internet history and produce a short illustrated brief.",
                role="user",
            ),
            source="User",
            destination="coordinator",
            exec_id="c1",
        ),
        # ── coordinator turn 0 ──
        TurnStartEvent(data=TurnInfo(turn=0), source="coordinator", exec_id="c1"),
        ReasoningItemEvent(
            data=ReasoningItem(
                summary_parts=[
                    ReasoningSummary(text="I'll delegate research first, then writing.")
                ],
                status="completed",
            ),
            source="coordinator",
            exec_id="c1",
        ),
        OutputMessageItemEvent(
            data=OutputMessageItem(
                content_parts=[OutputMessageText(text="Delegating to the researcher…")],
                status="completed",
            ),
            source="coordinator",
            exec_id="c1",
        ),
        ToolCallItemEvent(
            data=FunctionToolCallItem(
                call_id="c_r",
                name="researcher",
                arguments='{"topic": "internet history"}',
            ),
            source="coordinator",
            exec_id="c1",
        ),
        GenerationEndEvent(
            data=_resp("claude-sonnet-4-5", [search], cost=0.0041),
            source="coordinator",
            exec_id="c1",
        ),
        # ── researcher subagent ──
        TurnStartEvent(data=TurnInfo(turn=0), source="researcher", exec_id="r1"),
        ToolCallItemEvent(data=search, source="researcher", exec_id="r1"),
        GenerationEndEvent(
            data=_resp("claude-sonnet-4-5", [search], cost=0.0033),
            source="researcher",
            exec_id="r1",
        ),
        ToolOutputItemEvent(
            data=FunctionToolOutputItem.from_tool_result(
                call_id="r1",
                output=(
                    "1. ARPANET (1969)\n2. TCP/IP (1983)\n"
                    "3. World Wide Web (1991)\n4. Mobile internet (2007)"
                ),
            ),
            source="web_search",
            destination="researcher",
            exec_id="r1",
        ),
        OutputMessageItemEvent(
            data=OutputMessageItem(
                content_parts=[
                    OutputMessageText(text="Found 4 key milestones; handing back.")
                ],
                status="completed",
            ),
            source="researcher",
            exec_id="r1",
        ),
        TurnEndEvent(
            data=TurnEndInfo(turn=0, had_tool_calls=True, stop_reason="final_answer"),  # type: ignore[arg-type]
            source="researcher",
            exec_id="r1",
        ),
        ToolOutputItemEvent(
            data=FunctionToolOutputItem.from_tool_result(
                call_id="c_r", output="4 milestones: ARPANET, TCP/IP, WWW, mobile."
            ),
            source="researcher",
            destination="coordinator",
            exec_id="c1",
        ),
        # ── coordinator turn 1: background task + delegate writing ──
        TurnStartEvent(data=TurnInfo(turn=1), source="coordinator", exec_id="c1"),
        BackgroundTaskLaunchedEvent(
            data=BackgroundTaskInfo(
                task_id="bg1", tool_name="index_sources", tool_call_id="c_bg"
            ),
            source="coordinator",
            exec_id="c1",
        ),
        ToolCallItemEvent(
            data=FunctionToolCallItem(
                call_id="c_w", name="writer", arguments='{"format": "brief"}'
            ),
            source="coordinator",
            exec_id="c1",
        ),
        GenerationEndEvent(
            data=_resp("claude-sonnet-4-5", [chart], cost=0.0028),
            source="coordinator",
            exec_id="c1",
        ),
        # ── writer subagent (returns an image) ──
        TurnStartEvent(data=TurnInfo(turn=0), source="writer", exec_id="w1"),
        ToolCallItemEvent(data=chart, source="writer", exec_id="w1"),
        GenerationEndEvent(
            data=_resp("claude-sonnet-4-5", [chart], cost=0.0019),
            source="writer",
            exec_id="w1",
        ),
        ToolOutputItemEvent(
            data=FunctionToolOutputItem.from_tool_result(
                call_id="w1", output=json.dumps(chart_out)
            ),
            source="make_chart",
            destination="writer",
            exec_id="w1",
        ),
        OutputMessageItemEvent(
            data=OutputMessageItem(
                content_parts=[OutputMessageText(text="Chart rendered; brief ready.")],
                status="completed",
            ),
            source="writer",
            exec_id="w1",
        ),
        TurnEndEvent(
            data=TurnEndInfo(turn=0, had_tool_calls=True, stop_reason="final_answer"),  # type: ignore[arg-type]
            source="writer",
            exec_id="w1",
        ),
        ToolOutputItemEvent(
            data=FunctionToolOutputItem.from_tool_result(
                call_id="c_w", output="Brief with one chart produced."
            ),
            source="writer",
            destination="coordinator",
            exec_id="c1",
        ),
        BackgroundTaskCompletedEvent(
            data=BackgroundTaskInfo(
                task_id="bg1", tool_name="index_sources", tool_call_id="c_bg"
            ),
            source="coordinator",
            exec_id="c1",
        ),
        OutputMessageItemEvent(
            data=OutputMessageItem(
                content_parts=[
                    OutputMessageText(
                        text=(
                            "## Brief ready\n\n"
                            "Researched **4 milestones** and produced an "
                            "illustrated brief:\n\n"
                            "1. ARPANET (1969)\n"
                            "2. TCP/IP (1983)\n"
                            "3. World Wide Web (1991)\n\n"
                            "See the `chart` in the writer pane."
                        )
                    )
                ],
                status="completed",
            ),
            source="coordinator",
            exec_id="c1",
        ),
        TurnEndEvent(
            data=TurnEndInfo(turn=1, had_tool_calls=True, stop_reason="final_answer"),  # type: ignore[arg-type]
            source="coordinator",
            exec_id="c1",
        ),
    ]


async def build_demo_events(
    delay: float = 0.12, *, with_image: bool = True
) -> AsyncIterator[Event[Any]]:
    for event in demo_event_list(with_image=with_image):
        if delay:
            await asyncio.sleep(delay)
        yield event


def console_demo_events() -> list[Event[Any]]:
    """
    A single agent streaming text + thinking through tool calls, results, an
    error, and usage — covers the ``console`` linear stream's render paths
    (token deltas, the thinking gutter, panels, cumulative cost).
    """
    text = OutputMessageItem(
        content_parts=[OutputMessageText(text="Let me research that for you.")],
        status="completed",
    )
    search = FunctionToolCallItem(
        call_id="tc_1",
        name="web_search",
        arguments='{"query": "history of the internet", "max_results": 5}',
    )
    search_out = FunctionToolOutputItem.from_tool_result(
        call_id="tc_1",
        output=(
            "1. ARPANET (1969): first packet-switching network\n"
            "2. TCP/IP (1983): standard protocol adopted\n"
            "3. World Wide Web (1991): Tim Berners-Lee at CERN"
        ),
    )
    write = FunctionToolCallItem(
        call_id="tc_2",
        name="write_document",
        arguments=json.dumps(
            {
                "title": "Internet History",
                "content": "The Internet traces its origins to ARPANET...",
                "format": "markdown",
            }
        ),
    )
    write_out = FunctionToolOutputItem.from_tool_result(
        call_id="tc_2", output="Document saved: internet_history.md"
    )
    publish = FunctionToolCallItem(
        call_id="tc_3", name="publish", arguments='{"target": "blog"}'
    )

    def _delta(item_id: str, delta: str, idx: int, seq: int) -> LLMStreamEvent:
        return LLMStreamEvent(
            data=OutputMessageTextPartTextDelta(
                content_index=0,
                delta=delta,
                output_index=idx,
                sequence_number=seq,
                item_id=item_id,
                logprobs=[],
            ),
            source="coordinator",
            exec_id="c1",
        )

    def _think(delta: str, seq: int) -> LLMStreamEvent:
        return LLMStreamEvent(
            data=ReasoningSummaryPartTextDelta(
                delta=delta,
                summary_index=0,
                output_index=0,
                sequence_number=seq,
                item_id="i1",
            ),
            source="coordinator",
            exec_id="c1",
        )

    return [
        UserMessageEvent(
            data=InputMessageItem.from_text(
                "Research internet history and write a brief summary.", role="user"
            ),
            source="coordinator",
            exec_id="c1",
        ),
        SystemMessageEvent(
            data=InputMessageItem.from_text(
                "You are a research assistant. Use tools, then write a document.",
                role="system",
            ),
            source="coordinator",
            exec_id="c1",
        ),
        # ── Turn 1: think (streamed) → text (streamed) → search ──
        TurnStartEvent(data=TurnInfo(turn=0), source="coordinator", exec_id="c1"),
        LLMStreamEvent(
            data=OutputItemAdded(
                item=ReasoningItem(), output_index=0, sequence_number=1
            ),
            source="coordinator",
            exec_id="c1",
        ),
        _think("The user wants a summary of internet history.\n", 2),
        _think("I'll search authoritative sources, then write it up.", 3),
        LLMStreamEvent(
            data=OutputItemDone(
                item=ReasoningItem(), output_index=0, sequence_number=4
            ),
            source="coordinator",
            exec_id="c1",
        ),
        _delta("i2", "Let me research that for you.", 1, 5),
        OutputMessageItemEvent(data=text, source="coordinator", exec_id="c1"),
        LLMStreamEvent(
            data=OutputItemDone(item=text, output_index=1, sequence_number=6),
            source="coordinator",
            exec_id="c1",
        ),
        ToolCallItemEvent(data=search, source="coordinator", exec_id="c1"),
        LLMStreamEvent(
            data=ResponseCompleted(
                response=_resp("claude-sonnet-4-5", [text, search], cost=0.0043),
                sequence_number=7,
            ),
            source="coordinator",
            exec_id="c1",
        ),
        GenerationEndEvent(
            data=_resp("claude-sonnet-4-5", [text, search], cost=0.0043),
            source="coordinator",
            exec_id="c1",
        ),
        ToolOutputItemEvent(data=search_out, source="coordinator", exec_id="c1"),
        # ── Turn 2: background task + write + a failing publish ──
        TurnStartEvent(data=TurnInfo(turn=1), source="coordinator", exec_id="c1"),
        BackgroundTaskLaunchedEvent(
            data=BackgroundTaskInfo(
                task_id="bg_1", tool_name="index_sources", tool_call_id="tc_bg1"
            ),
            source="coordinator",
            exec_id="c1",
        ),
        ToolCallItemEvent(data=write, source="coordinator", exec_id="c1"),
        GenerationEndEvent(
            data=_resp("claude-sonnet-4-5", [write], cost=0.0067),
            source="coordinator",
            exec_id="c1",
        ),
        ToolOutputItemEvent(data=write_out, source="coordinator", exec_id="c1"),
        BackgroundTaskCompletedEvent(
            data=BackgroundTaskInfo(
                task_id="bg_1", tool_name="index_sources", tool_call_id="tc_bg1"
            ),
            source="coordinator",
            exec_id="c1",
        ),
        ToolCallItemEvent(data=publish, source="coordinator", exec_id="c1"),
        GenerationEndEvent(
            data=_resp("claude-sonnet-4-5", [publish], cost=0.0032),
            source="coordinator",
            exec_id="c1",
        ),
        ToolErrorEvent(
            data=ToolErrorInfo(
                tool_name="publish",
                error="Authentication failed: invalid API key",
                timed_out=False,
            ),
            source="coordinator",
            exec_id="c1",
        ),
        # ── Turn 3: a streaming error ──
        TurnStartEvent(data=TurnInfo(turn=2), source="coordinator", exec_id="c1"),
        LLMStreamingErrorEvent(
            data=LLMStreamingErrorData(
                error=RuntimeError("Connection reset by peer"),
                model_name="claude-sonnet-4-5",
            ),
            source="coordinator",
            exec_id="c1",
        ),
        TurnEndEvent(
            data=TurnEndInfo(turn=2, had_tool_calls=True, stop_reason="max_turns"),  # type: ignore[arg-type]
            source="coordinator",
            exec_id="c1",
        ),
    ]


def main() -> None:
    """
    Visual demo in a real terminal:

    * ``python -m grasp_agents.ui.demo`` — the console stream (add ``--thinking``
      to show streamed reasoning).
    * ``python -m grasp_agents.ui.demo --tui`` — the multi-pane Textual app
      (needs the ``tui`` extra).
    """
    import sys  # noqa: PLC0415

    if "--tui" in sys.argv:
        from . import run_tui  # noqa: PLC0415

        run_tui(build_demo_events())
        return

    from .console import EventConsole  # noqa: PLC0415

    ec = EventConsole(
        show_thinking="--thinking" in sys.argv,
        show_input_messages=True,
        show_usage=True,
        show_tool_args=True,
    )

    async def _run() -> None:
        async def _gen() -> AsyncIterator[Event[Any]]:  # noqa: RUF029
            for event in console_demo_events():
                yield event

        async for _ in ec.stream(_gen()):
            pass

    print("\n  EventConsole visual demo")
    print("  ═══════════════════════\n")
    asyncio.run(_run())
    print()


if __name__ == "__main__":
    main()
