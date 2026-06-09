"""
Deterministic demo event stream — a fixed list of events (a coordinator
delegating to two subagents, a background task, a tool returning an image) that
exercises every render path. This is the fixture for the headless ``tests/tui``
suite (pilot + SVG snapshot); the runnable demo is the interactive
``grasp_agents.tui.examples.data_copilot``.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..types.content import OutputMessageText, ReasoningSummary
from ..types.events import (
    BackgroundTaskCompletedEvent,
    BackgroundTaskInfo,
    BackgroundTaskLaunchedEvent,
    Event,
    GenerationEndEvent,
    OutputMessageItemEvent,
    ReasoningItemEvent,
    SystemMessageEvent,
    ToolCallItemEvent,
    ToolOutputItemEvent,
    TurnEndEvent,
    TurnEndInfo,
    TurnInfo,
    TurnStartEvent,
    UserMessageEvent,
)
from ..types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
    OutputMessageItem,
    ReasoningItem,
)
from ..types.response import (
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
