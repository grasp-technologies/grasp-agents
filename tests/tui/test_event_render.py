"""Pure renderer tests — no Textual needed (rich is a core dep)."""

from __future__ import annotations

import json

import pytest
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
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
from grasp_agents.ui._event_render import (
    PALETTE,
    _value_cell,
    render_event,
    render_image,
    render_tool_stream,
    truncate_lines,
)


def test_errored_tool_result_has_red_border() -> None:
    # A failed tool result (is_error, set by the loop for a ToolErrorInfo) gets
    # the red error border; a normal result keeps the neutral one.
    err = ToolOutputItemEvent(
        data=FunctionToolOutputItem.from_tool_result(
            call_id="1", output="boom: exit 1", is_error=True
        ),
        source="Bash",
        destination="agent",
    )
    panel = render_event(err)
    assert isinstance(panel, Panel)
    assert panel.border_style == PALETTE["error"]

    ok = ToolOutputItemEvent(
        data=FunctionToolOutputItem.from_tool_result(call_id="1", output="done"),
        source="Bash",
        destination="agent",
    )
    panel = render_event(ok)
    assert isinstance(panel, Panel)
    assert panel.border_style == PALETTE["border_tool_result"]


def test_notebook_edit_source_highlighting() -> None:
    # NotebookEdit's `new_source` arg renders as a syntax-highlighted code block,
    # keyed to the cell type: python for a code cell, markdown for a markdown one.
    code = "import numpy as np\narr = np.arange(10)\nprint(arr.mean())"
    md = "# Title\n\nSome **bold** text\n\n- a\n- b\n- c"
    code_cell = _value_cell("new_source", code, {"cell_type": "code"})
    md_cell = _value_cell("new_source", md, {"cell_type": "markdown"})
    assert isinstance(code_cell, Syntax)
    assert code_cell.lexer is not None
    assert code_cell.lexer.name == "Python"
    assert isinstance(md_cell, Syntax)
    assert md_cell.lexer is not None
    assert md_cell.lexer.name == "Markdown"


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


def test_pure_json_output_renders_like_tool_args() -> None:
    from rich.table import Table

    # A structured (pure-JSON object) answer renders as the same key/value
    # table used for tool-call arguments — not as prose.
    ev = OutputMessageItemEvent(
        data=OutputMessageItem(
            content_parts=[OutputMessageText(text='{"verdict": "pass", "score": 9}')],
            status="completed",
        ),
        source="grader",
    )
    assert isinstance(render_event(ev), Table)


def test_prose_output_with_leading_brace_stays_markdown() -> None:
    from rich.markdown import Markdown

    # Prose that merely starts with "{" is not valid JSON → stays Markdown.
    ev = OutputMessageItemEvent(
        data=OutputMessageItem(
            content_parts=[OutputMessageText(text="{not json} here is the answer")],
            status="completed",
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


def test_empty_reasoning_item_not_rendered() -> None:
    # A finalized reasoning item with no summary text carries nothing to show
    # (e.g. low effort, or encrypted server-side reasoning) — it must not render
    # a stray "thinking…" box.
    ev = ReasoningItemEvent(
        data=ReasoningItem(summary_parts=[], status="completed"), source="agent"
    )
    assert render_event(ev) is None


def test_skill_invocation_user_message_shows_raw_content() -> None:
    from rich.console import Console

    from grasp_agents.types.events import UserMessageEvent
    from grasp_agents.types.items import InputMessageItem

    raw = (
        '<system-reminder note="user invoked skill proofread">\n'
        "Proofread this sentence please.\n"
        "</system-reminder>"
    )
    ev = UserMessageEvent(
        data=InputMessageItem.from_text(raw, role="user"),
        source=None,
        destination="agent",
    )
    panel = render_event(ev)
    assert isinstance(panel, Panel)
    # distinct "skill" frame, carrying the skill name
    assert "skill" in str(panel.title)
    assert "/proofread" in str(panel.title)
    # the body is the raw text the agent sees — wrapper tags included, verbatim
    console = Console(width=80, record=True)
    console.print(panel)
    rendered = console.export_text()
    assert "user invoked skill proofread" in rendered
    assert "Proofread this sentence please." in rendered


def test_tool_error_event_has_no_display() -> None:
    # A tool failure renders only via its ToolOutputItemEvent(is_error=True)
    # (see test_errored_tool_result_has_red_border). The raw ToolErrorEvent is
    # the tool's terminal event and would only duplicate it, so it renders None.
    ev = ToolErrorEvent(data=ToolErrorInfo(tool_name="t", error="boom"))
    assert render_event(ev) is None


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


def test_tool_stream_strips_trailing_newline() -> None:
    # tool output usually ends in a newline; the streaming box must not turn it
    # into a blank line above the panel's own bottom padding, so a trailing
    # newline yields the same body as without one.
    for background in (True, False):
        with_nl = render_tool_stream("a", "Bash", "x\ny\n", background=background)
        without_nl = render_tool_stream("a", "Bash", "x\ny", background=background)
        assert isinstance(with_nl, Panel)
        assert isinstance(without_nl, Panel)
        body = with_nl.renderable
        assert isinstance(body, Text)
        assert body.plain == "x\ny"
        assert without_nl.renderable.plain == "x\ny"  # type: ignore[union-attr]


def test_tool_output_preserves_blank_lines() -> None:
    ev = ToolOutputItemEvent(
        data=FunctionToolOutputItem.from_tool_result(
            call_id="1", output="line 1\n\n\nline 2\n"
        ),
        source="Bash",
        destination="agent",
    )
    panel = render_event(ev)
    assert isinstance(panel, Panel)
    body = panel.renderable
    assert isinstance(body, Text)
    # internal blank lines are kept (only the trailing newline is dropped)
    assert body.plain == "line 1\n\n\nline 2"


def test_background_tool_stream_uses_lighter_palette() -> None:
    from grasp_agents.ui._event_render import PALETTE, render_tool_stream

    panel = render_tool_stream("agent", "Bash", "batch 1\n", background=True)
    assert isinstance(panel, Panel)
    # background progress is styled distinctly from a real tool result
    assert panel.border_style == PALETTE["border_bg_tool"]
    assert panel.border_style != PALETTE["border_tool_result"]


def _render_markdown_text(md_text: str, width: int = 40) -> str:
    from rich.console import Console

    from grasp_agents.ui._event_render import _Markdown

    console = Console(width=width, record=True)
    console.print(_Markdown(md_text))
    return console.export_text()


def test_markdown_headings_are_left_aligned() -> None:
    # Rich centers headings by default; the TUI overrides them to the left.
    out = _render_markdown_text("## A Subheading")
    line = next(ln for ln in out.splitlines() if "A Subheading" in ln)
    assert line.lstrip() == line  # no leading padding => left-aligned, not centered


def test_markdown_renders_xml_block() -> None:
    # Rich drops raw HTML/XML blocks entirely; the TUI renders them as XML.
    md = "Before\n\n<answer>\n  <step>do</step>\n</answer>\n\nAfter"
    out = _render_markdown_text(md, width=60)
    assert "<answer>" in out
    assert "<step>do</step>" in out


def test_markdown_keeps_inline_xml_tags() -> None:
    # inline tags would otherwise be stripped, leaving only the inner text
    out = _render_markdown_text("the <result>42</result> value", width=60)
    assert "<result>42</result>" in out


def test_tool_output_xml_payload_is_highlighted() -> None:
    from rich.syntax import Syntax

    notif = (
        "<task_notification>\n<task_id>bg_6</task_id>\n"
        "<status>completed</status>\n</task_notification>"
    )
    ev = ToolOutputItemEvent(
        data=FunctionToolOutputItem.from_tool_result(call_id="1", output=notif),
        source="Bash",
        destination="data_engineer",
    )
    panel = render_event(ev)
    assert isinstance(panel, Panel)
    assert isinstance(panel.renderable, Syntax)


def test_tool_output_plain_log_stays_text() -> None:
    # a "<" mid-line must not flip plain stdout into an XML block
    ev = ToolOutputItemEvent(
        data=FunctionToolOutputItem.from_tool_result(
            call_id="1", output="started <worker> ok\ndone\n"
        ),
        source="Bash",
        destination="agent",
    )
    panel = render_event(ev)
    assert isinstance(panel, Panel)
    assert isinstance(panel.renderable, Text)


def test_completed_task_notification_is_a_plain_box() -> None:
    from grasp_agents.types.events import UserMessageEvent
    from grasp_agents.types.items import InputMessageItem

    notif = (
        "<task_notification>\n<task_id>bg_1</task_id>\n"
        "<tool_name>Bash</tool_name>\n<status>completed</status>\n"
        "</task_notification>"
    )
    ev = UserMessageEvent(
        data=InputMessageItem.from_text(notif, role="user"),
        source="Bash",
        destination="data_engineer",
    )
    box = render_event(ev)
    # A completed task adds no status line of its own (its
    # BackgroundTaskCompletedEvent already shows "✓ … completed" above the box) —
    # just a box titled "<tool> → <agent>" (recipient = destination).
    assert isinstance(box, Panel)
    assert "Bash → data_engineer" in str(box.title)


def test_interrupted_task_notification_has_status_line_above_box() -> None:
    from rich.console import Group

    from grasp_agents.types.events import UserMessageEvent
    from grasp_agents.types.items import InputMessageItem

    notif = (
        "<task_notification>\n<task_id>bg_1</task_id>\n"
        "<tool_name>Bash</tool_name>\n<status>interrupted</status>\n"
        "</task_notification>"
    )
    ev = UserMessageEvent(
        data=InputMessageItem.from_text(notif, role="user"),
        source="Bash",
        destination="data_engineer",
    )
    result = render_event(ev)
    # Interrupted/failed (no completion event) get a "✗ … (id)" line ABOVE the
    # box — never inside it.
    assert isinstance(result, Group)
    parts = list(result.renderables)
    assert isinstance(parts[0], Text)
    assert "✗" in parts[0].plain and "Bash interrupted" in parts[0].plain
    assert "bg_1" in parts[0].plain
    assert isinstance(parts[1], Panel)
    assert "Bash → data_engineer" in str(parts[1].title)


def test_user_message_xml_payload_is_highlighted() -> None:
    from rich.syntax import Syntax

    from grasp_agents.types.events import UserMessageEvent
    from grasp_agents.types.items import InputMessageItem

    # A non-notice XML payload still renders syntax-highlighted.
    xml = "<data>\n  <item>x</item>\n</data>"
    ev = UserMessageEvent(
        data=InputMessageItem.from_text(xml, role="user"),
        source="upstream",
        destination="agent",
    )
    panel = render_event(ev)
    assert isinstance(panel, Panel)
    assert isinstance(panel.renderable, Syntax)


def test_user_message_prose_stays_text() -> None:
    from grasp_agents.types.events import UserMessageEvent
    from grasp_agents.types.items import InputMessageItem

    ev = UserMessageEvent(
        data=InputMessageItem.from_text("run the <script> now please", role="user"),
        source="User",
        destination="agent",
    )
    panel = render_event(ev)
    assert isinstance(panel, Panel)
    assert isinstance(panel.renderable, Text)


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
