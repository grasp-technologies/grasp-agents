"""Pure renderer tests — no Textual needed (rich is a core dep)."""

from __future__ import annotations

import json

import pytest
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from grasp_agents.types.content import OutputMessageText, ReasoningSummary
from grasp_agents.types.events import (
    GenerationEndEvent,
    OutputMessageItemEvent,
    ReasoningItemEvent,
    ToolCallItemEvent,
    ToolErrorEvent,
    ToolErrorInfo,
    ToolOutputItemEvent,
    TurnEndEvent,
    TurnEndInfo,
    TurnInfo,
    TurnStartEvent,
)
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    OutputMessageItem,
    ReasoningItem,
)
from grasp_agents.types.response import (
    InputTokensDetails,
    OutputTokensDetails,
    Response,
    ResponseUsage,
)
from grasp_agents.ui._event_render import render_event, render_image, truncate_lines


def test_turn_start_is_rule() -> None:
    ev = TurnStartEvent(data=TurnInfo(turn=0), source="agent")
    assert isinstance(render_event(ev), Rule)


def test_turn_end_final_answer_is_silent() -> None:
    ev = TurnEndEvent(
        data=TurnEndInfo(turn=0, had_tool_calls=False, stop_reason="final_answer"),  # type: ignore[arg-type]
        source="agent",
    )
    assert render_event(ev) is None


def test_turn_end_other_reason_renders() -> None:
    ev = TurnEndEvent(
        data=TurnEndInfo(turn=0, had_tool_calls=True, stop_reason="max_turns"),  # type: ignore[arg-type]
        source="agent",
    )
    assert isinstance(render_event(ev), Text)


def test_tool_call_is_panel() -> None:
    ev = ToolCallItemEvent(
        data=FunctionToolCallItem(
            call_id="1", name="web_search", arguments='{"q": "x"}'
        ),
        source="agent",
    )
    assert isinstance(render_event(ev), Panel)


def _value_cells(panel: Panel) -> list[object]:
    # the args/result kv-table is the panel body; its 2nd column holds the values
    from rich.table import Table

    table = panel.renderable
    assert isinstance(table, Table)
    return list(table.columns[1].cells)


def test_tool_call_code_arg_is_syntax_highlighted() -> None:
    from rich.syntax import Syntax

    ev = ToolCallItemEvent(
        data=FunctionToolCallItem(
            call_id="1",
            name="RunPython",
            arguments=json.dumps(
                {"code": "import pandas as pd\ndf = pd.DataFrame()\nprint(df)", "n": 3}
            ),
        ),
        source="agent",
    )
    panel = render_event(ev)
    assert isinstance(panel, Panel)
    # the code value cell highlights; the scalar (n) stays plain — same table
    syntaxes = [c for c in _value_cells(panel) if isinstance(c, Syntax)]
    assert syntaxes, "code argument was not rendered as a syntax block"
    assert "pandas" in syntaxes[0].code
    assert syntaxes[0].lexer is not None  # "python" resolved to a real lexer


def test_tool_call_scalar_args_stay_tabular() -> None:
    from rich.syntax import Syntax

    ev = ToolCallItemEvent(
        data=FunctionToolCallItem(
            call_id="1",
            name="web_search",
            arguments=json.dumps({"query": "internet history", "max_results": 5}),
        ),
        source="agent",
    )
    panel = render_event(ev)
    assert isinstance(panel, Panel)
    assert not any(isinstance(c, Syntax) for c in _value_cells(panel))


def test_tool_output_markdown_report_renders_as_markdown() -> None:
    from rich.markdown import Markdown

    md = "## Summary\n\nFindings:\n\n- alpha\n- beta\n"
    ev = ToolOutputItemEvent(
        data=FunctionToolOutputItem.from_tool_result(call_id="1", output=md),
        source="research_agent",
        destination="analyst",
    )
    panel = render_event(ev)
    assert isinstance(panel, Panel)
    assert isinstance(panel.renderable, Markdown)


def test_tool_output_plain_text_stays_text() -> None:
    from rich.markdown import Markdown

    ev = ToolOutputItemEvent(
        data=FunctionToolOutputItem.from_tool_result(
            call_id="1", output="mean=5.2\nstd=1.3\n"
        ),
        source="RunPython",
        destination="analyst",
    )
    panel = render_event(ev)
    assert isinstance(panel, Panel)
    assert not isinstance(panel.renderable, Markdown)
    assert isinstance(panel.renderable, Text)


def test_message_is_markdown() -> None:
    from rich.markdown import Markdown

    ev = OutputMessageItemEvent(
        data=OutputMessageItem(
            content_parts=[OutputMessageText(text="hello")], status="completed"
        ),
        source="agent",
    )
    assert isinstance(render_event(ev), Markdown)


def test_reasoning_is_panel() -> None:
    ev = ReasoningItemEvent(
        data=ReasoningItem(
            summary_parts=[ReasoningSummary(text="thinking about it")],
            status="completed",
        ),
        source="agent",
    )
    assert isinstance(render_event(ev), Panel)


def test_tool_error_is_panel() -> None:
    ev = ToolErrorEvent(data=ToolErrorInfo(tool_name="t", error="boom"))
    assert isinstance(render_event(ev), Panel)


def test_generation_end_usage_is_text() -> None:
    resp = Response(
        model="m",
        output_items=[],
        usage_with_cost=ResponseUsage(
            input_tokens=10,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens=5,
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
            total_tokens=15,
            cost=0.001,
        ),
    )
    assert isinstance(render_event(GenerationEndEvent(data=resp, source="agent")), Text)


def test_tool_output_with_image_path_embeds_image(tmp_path) -> None:
    from PIL import Image

    png = tmp_path / "chart.png"
    Image.new("RGB", (10, 10), "#abcdef").save(png)
    ev = ToolOutputItemEvent(
        data=FunctionToolOutputItem.from_tool_result(
            call_id="1", output=json.dumps({"k": "v", "image_path": str(png)})
        ),
        source="make_chart",
        destination="agent",
    )
    rendered = render_event(ev)
    assert isinstance(rendered, Panel)
    assert isinstance(rendered.renderable, Group)  # table + spacer + image


def test_render_image_never_raises_on_bad_path() -> None:
    assert render_image("/no/such/image.png") is not None


def test_chafa_symbol_art_when_available(tmp_path) -> None:
    # When the chafa binding is installed, an image renders as symbol-art: an
    # ANSI string parsed into a Rich Text (not the rich-pixels Pixels fallback).
    pytest.importorskip("chafa")
    from PIL import Image

    png = tmp_path / "c.png"
    Image.new("RGB", (16, 10), "#10b981").save(png)
    out = render_image(str(png))
    # chafa present → ANSI symbol-art parsed into a styled Text (color spans),
    # not the rich-pixels Pixels fallback nor the bare "[image]" label. A solid
    # fill renders as colour-on-space cells, so assert on the spans, not .plain.
    assert isinstance(out, Text)
    assert out.spans, "expected styled symbol-art spans from chafa"


def test_truncate_lines() -> None:
    out = truncate_lines("a\nb\nc\nd", 2)
    assert "a" in out
    assert "more lines" in out


def test_tool_output_inputimage_parts_render() -> None:
    import base64 as b64
    import io as _io

    from PIL import Image as PILImage

    from grasp_agents.types.content import InputImage, InputText

    buf = _io.BytesIO()
    PILImage.new("RGB", (6, 6), "#3366cc").save(buf, format="PNG")
    data_uri = "data:image/png;base64," + b64.b64encode(buf.getvalue()).decode()
    item = FunctionToolOutputItem(
        call_id="1",
        output_parts=[InputText(text="chart ready"), InputImage(image_url=data_uri)],
    )
    ev = ToolOutputItemEvent(data=item, source="make_chart", destination="agent")
    rendered = render_event(ev)
    assert isinstance(rendered, Panel)
    assert isinstance(rendered.renderable, Group)
