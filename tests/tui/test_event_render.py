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
    WebSearchCallItemEvent,
)
from grasp_agents.types.items import (
    FindInPageAction,
    FunctionToolCallItem,
    FunctionToolOutputItem,
    OpenPageAction,
    OutputMessageItem,
    ReasoningItem,
    SearchAction,
    SearchSource,
    WebSearchCallItem,
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


def _render_to_text(renderable: object, width: int = 80) -> str:
    from rich.console import Console

    console = Console(width=width, record=True)
    console.print(renderable)
    return console.export_text()


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


def test_untrusted_tool_result_unwrapped_to_kv_panel() -> None:
    # An <untrusted_content>-fenced JSON result renders as a key/value panel
    # (not the raw boundary tags), with provenance moved to the title.
    from grasp_agents.context.untrusted_content import wrap_untrusted

    raw = FunctionToolOutputItem.from_tool_result(
        call_id="1", output={"stdout": "line one\nline two", "returncode": 0}
    ).output
    item = FunctionToolOutputItem(
        call_id="1", output=wrap_untrusted(raw, source="Bash")
    )
    ev = ToolOutputItemEvent(data=item, source="Bash", destination="analyst")
    out = _render_to_text(render_event(ev, inline_images=False))

    assert "analyst ← Bash" in out
    assert "[untrusted]" in out
    assert "untrusted_content" not in out  # the raw XML tags are stripped
    assert "stdout" in out  # KV keys present
    assert "returncode" in out
    # stdout's "\n" is rendered as a real newline: two separate output lines
    lines = out.splitlines()
    assert any("line one" in ln and "line two" not in ln for ln in lines)
    assert any("line two" in ln for ln in lines)


def test_ansi_in_tool_result_keeps_border_aligned() -> None:
    # Colorized tool output (e.g. `ls --color`) is parsed into Rich styles, so
    # the panel border stays aligned. Raw escape bytes left in the string would
    # inflate Rich's width measurement and shove the right border inward on the
    # colored lines only (a ragged edge).
    from grasp_agents.context.untrusted_content import wrap_untrusted

    color, reset = "\x1b[01;34m", "\x1b[0m"
    stdout = f"a.txt\n{color}subdir{reset}\nb.txt"  # one colored line among plain
    raw = FunctionToolOutputItem.from_tool_result(
        call_id="1", output={"stdout": stdout, "returncode": 0}
    ).output
    item = FunctionToolOutputItem(
        call_id="1", output=wrap_untrusted(raw, source="Bash")
    )
    ev = ToolOutputItemEvent(data=item, source="Bash", destination="agent")
    out = _render_to_text(render_event(ev, inline_images=False), width=60)

    assert "subdir" in out
    assert "[01;34m" not in out  # no leftover escape-code text in the cell
    bordered = [ln for ln in out.splitlines() if ln.startswith("│")]
    assert bordered
    assert len({len(ln) for ln in bordered}) == 1  # every border line same width


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

    # markdown=False disables it in panels too — the report renders as plain text.
    panel_plain = render_event(ev, markdown=False)
    assert isinstance(panel_plain, Panel)
    assert not isinstance(panel_plain.renderable, Markdown)


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
            content=[OutputMessageText(text="hello")], status="completed"
        ),
        source="agent",
    )
    assert isinstance(render_event(ev), Markdown)


def test_markdown_links_plain_without_hyperlinks_osc8_with() -> None:
    # hyperlinks=False (notebooks/pipes): links render as "text (url)" plain text,
    # NOT OSC-8 (which leaks as escape garbage there); the URL is shown for the
    # frontend to auto-linkify. hyperlinks=True (a real terminal): clean OSC-8.
    url = "https://www.techradar.com/pro/microsoft-forced-to-turn-to-aws"
    ev = OutputMessageItemEvent(
        data=OutputMessageItem(
            content=[OutputMessageText(text=f"AWS ([techradar.com]({url})).")],
            status="completed",
        ),
        source="agent",
    )
    plain = _render_raw(render_event(ev, hyperlinks=False), width=200)
    assert "]8;" not in plain  # no OSC-8 escape
    assert url in _render_to_text(render_event(ev, hyperlinks=False), width=200)

    linked = _render_raw(render_event(ev, hyperlinks=True), width=200)
    assert "]8;" in linked  # OSC-8 hyperlink emitted for a real terminal


def test_pure_json_output_renders_like_tool_args() -> None:
    from rich.table import Table

    # A structured (pure-JSON object) answer renders as the same key/value
    # table used for tool-call arguments — not as prose.
    ev = OutputMessageItemEvent(
        data=OutputMessageItem(
            content=[OutputMessageText(text='{"verdict": "pass", "score": 9}')],
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
            content=[OutputMessageText(text="{not json} here is the answer")],
            status="completed",
        ),
        source="agent",
    )
    assert isinstance(render_event(ev), Markdown)


def test_reasoning_is_gutter() -> None:
    # Reasoning renders as a left-border gutter (not a box), matching the
    # streaming console / TUI thinking style.
    ev = ReasoningItemEvent(
        data=ReasoningItem(
            summary=[ReasoningSummary(text="thinking about it")],
            status="completed",
        ),
        source="agent",
    )
    rend = render_event(ev)
    assert not isinstance(rend, Panel)
    out = _render_to_text(rend)
    assert "┌ thinking" in out
    assert "│ thinking about it" in out
    assert "└" in out


def test_empty_reasoning_item_not_rendered() -> None:
    # A finalized reasoning item with no summary text carries nothing to show
    # (e.g. low effort, or encrypted server-side reasoning) — it must not render
    # a stray "thinking…" box.
    ev = ReasoningItemEvent(
        data=ReasoningItem(summary=[], status="completed"), source="agent"
    )
    assert render_event(ev) is None


def test_skill_invocation_user_message_shows_raw_content() -> None:
    from rich.console import Console

    from grasp_agents.types.events import UserMessageEvent
    from grasp_agents.types.items import InputMessageItem

    raw = (
        '<system-reminder subject="user invoked skill proofread">\n'
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
    assert "<proofread>" in str(panel.title)
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
        output=[],
        usage=ResponseUsage(
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
        output=[InputText(text="chart ready"), InputImage(image_url=data_uri)],
    )
    ev = ToolOutputItemEvent(data=item, source="make_chart", destination="agent")
    rendered = render_event(ev)
    assert isinstance(rendered, Panel)
    assert isinstance(rendered.renderable, Group)


# ── Web search call items (server-side search/fetch) ──


def _render_raw(renderable: object, width: int = 80) -> str:
    """Render WITH ANSI/escape sequences kept (export_text() would strip them)."""
    import io

    from rich.console import Console

    buf = io.StringIO()
    Console(file=buf, force_terminal=True, color_system="256", width=width).print(
        renderable
    )
    return buf.getvalue()


def test_web_search_search_action_rendered() -> None:
    # Without hyperlinks (notebook): queries, source titles, real URLs as plain
    # text, and page age are all surfaced.
    ev = WebSearchCallItemEvent(
        data=WebSearchCallItem(
            action=SearchAction(
                queries=["history of the internet"],
                sources=[
                    SearchSource(
                        url="https://ex.com", title="Internet", page_age="2 days ago"
                    )
                ],
            ),
            status="completed",
        ),
        source="searcher",
    )
    panel = render_event(ev, hyperlinks=False)
    assert isinstance(panel, Panel)
    assert "web_search" in str(panel.title)
    text = _render_to_text(panel)
    assert "history of the internet" in text
    assert "Internet" in text  # source title surfaced
    assert "https://ex.com" in text  # real URL surfaced (plain text)
    assert "2 days ago" in text  # page age surfaced


def test_web_search_sources_osc8_only_with_hyperlinks() -> None:
    # hyperlinks=False (notebooks/pipes): URLs are plain text, no OSC-8 (which
    # would leak as escape garbage). hyperlinks=True (a real terminal): the title
    # is a clickable OSC-8 link.
    ev = WebSearchCallItemEvent(
        data=WebSearchCallItem(
            action=SearchAction(
                queries=["q"],
                sources=[SearchSource(url="https://ex.com/article", title="Title")],
            ),
            status="completed",
        ),
        source="searcher",
    )
    plain_ev = render_event(ev, hyperlinks=False)
    assert "]8;" not in _render_raw(plain_ev)  # no OSC-8 sequence
    assert "https://ex.com/article" in _render_to_text(plain_ev)  # plain URL

    linked = _render_raw(render_event(ev, hyperlinks=True))
    assert "]8;" in linked  # title rendered as a clickable OSC-8 link


def test_web_search_hides_grounding_redirect_url() -> None:
    # Gemini grounding sources carry vertexaisearch.cloud.google.com/grounding-api-
    # redirect/... URLs, not the real page — long and opaque, pure clutter. They're
    # dropped (the title is the real domain); the source itself is still listed.
    redirect = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbC123"
    ev = WebSearchCallItemEvent(
        data=WebSearchCallItem(
            action=SearchAction(
                queries=["q"],
                sources=[SearchSource(url=redirect, title="taiwannews.com.tw")],
            ),
            status="completed",
        ),
        source="searcher",
    )
    # The redirect is never shown NOR used as a link target — in either mode.
    text = _render_to_text(render_event(ev, hyperlinks=False), width=200)
    assert "taiwannews.com.tw" in text  # domain/title shown
    assert "vertexaisearch" not in text  # opaque redirect URL dropped
    assert "grounding-api-redirect" not in text
    # With hyperlinks on, the title must NOT become an OSC-8 link to the redirect.
    raw_linked = _render_raw(render_event(ev, hyperlinks=True), width=200)
    assert "vertexaisearch" not in raw_linked
    assert "grounding-api-redirect" not in raw_linked


def test_web_search_open_page_rendered() -> None:
    ev = WebSearchCallItemEvent(
        data=WebSearchCallItem(
            action=OpenPageAction(url="https://example.com/page"), status="completed"
        ),
        source="a",
    )
    text = _render_to_text(render_event(ev))
    assert "open page" in text
    assert "https://example.com/page" in text


def test_web_search_find_in_page_rendered() -> None:
    ev = WebSearchCallItemEvent(
        data=WebSearchCallItem(
            action=FindInPageAction(url="https://example.com/page", pattern="needle"),
            status="completed",
        ),
        source="a",
    )
    text = _render_to_text(render_event(ev))
    assert "find in page" in text
    assert "needle" in text


def test_renders_compaction_notice() -> None:
    from grasp_agents.types.events import CompactionEvent, CompactionInfo

    ev = CompactionEvent(
        source="a",
        data=CompactionInfo(
            folded_turns=7,
            preserved_turns=3,
            context_tokens=1850,
            context_window=2000,
            summary="Kept the goal and the key results discovered so far.",
        ),
    )
    text = _render_to_text(render_event(ev))
    assert "compacted" in text
    assert "7 turns" in text
    assert "3 recent turns kept" in text
    assert "1,850" in text
    assert "2,000" in text
    assert "Kept the goal" in text  # the summary itself is shown
    # shown raw — the exact wrapped message the agent receives (like a skill call)
    assert "system-reminder" in text
    assert "summarized" in text  # the injected subject is visible


def test_compaction_notice_singular_turn() -> None:
    from grasp_agents.types.events import CompactionEvent, CompactionInfo

    ev = CompactionEvent(
        source="a",
        data=CompactionInfo(folded_turns=1, preserved_turns=1, context_tokens=900),
    )
    text = _render_to_text(render_event(ev))
    assert "folded 1 turn" in text
    assert "1 turns" not in text


def test_bg_launched_notice_cites_log_only_when_present() -> None:
    from grasp_agents.types.events import (
        BackgroundTaskInfo,
        BackgroundTaskLaunchedEvent,
    )

    with_log = render_event(
        BackgroundTaskLaunchedEvent(
            data=BackgroundTaskInfo(
                task_id="bg_1",
                tool_name="Bash",
                tool_call_id="c1",
                output_name="call_c1.log",
            ),
            source="lead",
        )
    )
    assert isinstance(with_log, Text)
    assert "· call_c1.log" in with_log.plain

    without_log = render_event(
        BackgroundTaskLaunchedEvent(
            data=BackgroundTaskInfo(
                task_id="bg_1", tool_name="Bash", tool_call_id="c1"
            ),
            source="lead",
        )
    )
    assert isinstance(without_log, Text)
    assert "call_c1.log" not in without_log.plain
