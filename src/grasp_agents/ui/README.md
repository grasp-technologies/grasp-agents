# grasp_agents.ui — terminal UIs for agent event streams

Two front-ends over one Textual-free renderer (`grasp_agents.ui._event_render`),
both consuming the same typed `AsyncIterator[Event]`:

- **`console`** — a light, linear ANSI stream (`EventConsole` / `render_events`).
  Needs only `rich` (a core dependency); works in any terminal, a pipe, or a
  notebook. Try `python -m grasp_agents.ui.demo`.
- **`app`** — a full-screen **Textual** app (the rest of this doc): **one pane
  per agent/subagent**, a **bottom tab bar** to switch between them (status glyph
  per tab; `↳` marks a subagent), a follow-latest toggle, switchable themes, and
  inline images. Two modes: **monitor** a stream, or **interactive** (an input
  box drives the agent each turn). Needs the `tui` extra.

Both share the renderable builders in `grasp_agents.ui._event_render`, so the
linear stream and the multi-pane app stay visually consistent.

## Install

```bash
pip install 'grasp_agents[tui]'        # textual + textual-image + rich-pixels
```

## Run it (terminal)

The TUI is a full-screen terminal app — run it in a **real terminal** (not a
notebook; see below for those).

For a runnable end-to-end demo, see **Real demo: data-analysis copilot** below.

**Monitor** your own run — save as `monitor.py`, then `python monitor.py`:

```python
from grasp_agents.ui import run_tui

run_tui(agent.run_stream("your prompt", ctx=ctx))   # blocks until you quit
```

**Interactive** — type messages, the agent runs each turn:

```python
from grasp_agents.ui import run_tui_interactive

run_tui_interactive(agent)   # rollback, skills palette, and token meter inferred
```

`run_tui*` start the app's own event loop, so call them at module top level (not
inside `asyncio.run`).

Keys: `q` quit · `f` toggle follow-latest · `esc` interrupt the running turn ·
click a tab (or `←`/`→`) to focus an agent · `ctrl+p` command palette (theme
switch).

## Tool approvals

Wire a `TuiApprovalStore` onto the run context and register the approval gate;
the app then pops a small dialog whenever a gated tool call needs a decision —
**once**, **session** (skip re-prompting for this tool), **always**, or **deny**
(`esc` denies). With no dialog open, `esc` interrupts the whole turn instead.

Pass `persist_path` to keep **always** decisions across restarts — they're
written to that JSON file and reloaded on next launch (the store is otherwise
in-memory).

```python
from pathlib import Path

from grasp_agents import RunContext
from grasp_agents.agent.approval_store import build_store_approval
from grasp_agents.ui import TuiApprovalStore, run_tui_interactive

store = TuiApprovalStore(persist_path=Path(".grasp/approvals.json"))
ctx = RunContext(approval_store=store, session_key="user-1")
agent = LLMAgent(name="assistant", ctx=ctx, llm=llm, tools=[...])
agent.add_before_tool_hook(build_store_approval(tool_names={"delete_record"}))
run_tui_interactive(agent)   # main_agent + ctx (with the approval store) inferred
```

Runnable demo (`grasp_agents.examples.tui.approval_copilot`): an ops assistant
whose `delete_record` / `update_record` calls require approval.

```bash
python -m grasp_agents.examples.tui.approval_copilot   # needs OPENAI_API_KEY
```

## Real demo: data-analysis copilot (LLM + sandbox)

A non-trivial, runnable example (`grasp_agents.examples.tui.data_copilot`): an
*analyst* agent that delegates to two **sandboxed** specialists —
`data_engineer` (generates/inspects data with numpy) and `viz_specialist`
(computes stats and renders matplotlib charts shown inline) — each running
Python in a confined local sandbox, sharing one workspace.

```bash
# needs OPENAI_API_KEY in .env, the `notebook-exec` extra, and the `srt` CLI
python -m grasp_agents.examples.tui.data_copilot
```

Type e.g. *"Generate 120 daily sales values and plot the 7-day moving
average"* — watch the analyst delegate (bottom tabs switch as each subagent
runs), the specialists execute code in the sandbox, and the chart render inline.

Uses **srt** confinement: the sandbox Jupyter kernel needs loopback, which
`seatbelt` blocks but `srt` permits (via `allowLocalBinding`) while still
confining egress. Verified end-to-end in `tests/integration/`.

## Notebooks

The interactive app needs a terminal (Textual has no Jupyter driver). For
notebooks, use the helpers that reuse the same renderers:

```python
from grasp_agents.ui import render_events_inline, display_screenshot

# live: render each event inline as it streams (real images via IPython)
async for _ in render_events_inline(agent.run_stream("…", ctx=ctx)):
    pass

# static: a one-shot SVG snapshot of the multi-pane layout
await display_screenshot(agent.run_stream("…", ctx=ctx))
```

`render_events_inline` needs only `rich` (no Textual); `display_screenshot`
renders the app headless and embeds an SVG. For a plain linear stream,
`console.EventConsole` also works in notebooks unchanged.

## Multi-subagent model

Panes and tabs are keyed by `event.source` (each agent/subagent emits its own
source). Tool results / user input route to their `destination` (the addressed
agent). Parent→child edges come from `ToolCallItemEvent` /
`BackgroundTaskLaunchedEvent` (and show as a `↳` prefix). Status: working `●` →
done `✓` / error `✗`.

## Images

Inline images render as **chafa** symbol-art — colored Unicode glyphs that live
in the cell grid, so they scroll cleanly and never blank. (Terminal-graphics
protocols like Kitty/Sixel fight a scrolling pane's compositor, which is what
makes them flicker or disappear there.) **Click any inline image** to open a
full-resolution zoom on a dedicated, non-scrolling modal screen, where
**textual-image** drives the terminal's native graphics protocol (TGP on
Kitty/Ghostty) — the one place those protocols are reliable. Press `esc`/`q` or
click to close.

Both inline tool-output images (e.g. a sandbox running matplotlib, returned as
`InputImage` parts) and a tool result whose JSON carries an `image_path` are
shown.

chafa needs its Python binding — the wheel bundles `libchafa`, so no system
package is required:

```bash
pip install 'chafa.py>=1.2'      # already in the [tui] extra
```

Without it, inline images fall back to **rich-pixels** half-blocks (lower-res
but dependency-light); the zoom modal still works via textual-image.

## Testing (no manual UI inspection)

```bash
uv run pytest tests/tui                                       # all layers
uv run pytest tests/tui/test_snapshot.py --snapshot-update    # refresh SVG baseline
```

- `test_event_render.py` — pure renderer unit tests (no Textual).
- `test_app.py` / `test_interactive.py` — headless `App.run_test()` pilots:
  assert the panes / subagent nesting / status / pane-switching and the
  interactive input→run flow, all from fixed event streams.
- `test_notebook.py` — the inline renderer + SVG screenshot helpers.
- `test_snapshot.py` — `pytest-textual-snapshot` renders the whole screen to a
  committed SVG and auto-diffs it, so visual regressions fail CI without anyone
  eyeballing widgets. The demo stream is deterministic (no clock, no
  machine-specific paths) so the baseline is portable.
