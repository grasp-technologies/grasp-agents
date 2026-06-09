"""
Integration: the data-copilot pipeline runs against a real LLM + sandbox.

Deselected by default; run with::

    uv run pytest -m integration tests/integration/test_tui_data_copilot.py -s
"""

from __future__ import annotations

from pathlib import Path

import pytest

from grasp_agents.types.events import (
    BackgroundTaskLaunchedEvent,
    ProcPacketOutEvent,
    RunPacketOutEvent,
    ToolOutputItemEvent,
    UserMessageEvent,
)
from grasp_agents.ui.examples.data_copilot import build_copilot

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_data_copilot_pipeline(tmp_path: Path) -> None:
    analyst, _ctx = build_copilot(tmp_path, confinement="srt")
    prompt = (
        "Generate a small synthetic dataset of 120 daily sales values, then "
        "plot the 7-day moving average over time."
    )
    sources: set[str] = set()
    user_dests: set[str | None] = set()
    bg_tools: list[str] = []
    images = 0
    final: object | None = None
    async for ev in analyst.run_stream(prompt):
        if ev.source:
            sources.add(ev.source)
        if isinstance(ev, UserMessageEvent):
            user_dests.add(ev.destination)
        if isinstance(ev, BackgroundTaskLaunchedEvent):
            bg_tools.append(ev.data.tool_name)
        if isinstance(ev, ToolOutputItemEvent) and ev.data.images:
            images += 1
        if (
            isinstance(ev, (ProcPacketOutEvent, RunPacketOutEvent))
            and ev.source == "analyst"
            and ev.data.payloads
        ):
            final = ev.data.payloads[0]

    subagents = {"data_engineer", "viz_specialist"} & sources
    print(
        f"\nsubagents_run={subagents} inline_images={images} "
        f"sources={sources} user_msg_dests={user_dests}"
    )
    assert subagents, f"no specialist subagent ran; sources={sources}"
    assert final is not None, "analyst produced no final answer"
    # subagents receive (and the UI shows) their input as a user message
    assert {"data_engineer", "viz_specialist"} & user_dests, user_dests
    # When the engineer backgrounds a Bash job, its streamed output is mirrored
    # to a crash-recoverable log under the shared workspace (turned on by the
    # ctx's checkpoint store + file backend). The mechanism is covered
    # deterministically in tests/sandbox/test_bash_polish.py; here we verify the
    # end-to-end wiring whenever the LLM actually backgrounds a command.
    task_logs = list((tmp_path / ".grasp" / "tasks").glob("*.log"))
    print(f"bg_tools={bg_tools} bg_task_logs={[p.name for p in task_logs]}")
    if not bg_tools:
        pytest.skip("LLM did not background a Bash command this run")
    assert task_logs, "a task backgrounded but no log under .grasp/tasks/"
    assert any(p.read_text().strip() for p in task_logs), "bg-task log(s) empty"


@pytest.mark.asyncio
async def test_runpython_displays_inline_image(tmp_path: Path) -> None:
    """Mechanism check (no LLM): plt.show() in the sandbox yields an InputImage."""
    from grasp_agents import RunContext
    from grasp_agents.sandbox import local_environment
    from grasp_agents.tools.code_interpreter import (
        RunPython,
        RunPythonInput,
    )
    from grasp_agents.types.content import InputImage

    env = local_environment(
        allowed_roots=[tmp_path],
        confinement="srt",
        env={"MPLCONFIGDIR": str(tmp_path / ".mpl")},
    )
    ctx = RunContext[None](state=None, environment=env)
    code = (
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\n"
        "ax.plot([1, 2, 3], [1, 4, 9])\n"
        "plt.show()\n"
    )
    out = await RunPython()._run(RunPythonInput(code=code), ctx=ctx)
    images = [p for p in out if isinstance(p, InputImage)]
    print(f"\nout_parts={len(out)} images={len(images)}")
    assert images, "no inline image from plt.show() (inline backend on at startup)"


@pytest.mark.asyncio
async def test_runpython_inside_textual_event_loop(tmp_path: Path) -> None:
    """Reproduce the interactive-TUI condition: RunPython inside Textual's loop."""
    from textual import work
    from textual.app import App

    from grasp_agents import RunContext
    from grasp_agents.sandbox import local_environment
    from grasp_agents.tools.code_interpreter import RunPython, RunPythonInput

    env = local_environment(
        allowed_roots=[tmp_path],
        confinement="srt",
        env={"MPLCONFIGDIR": str(tmp_path / ".mpl")},
    )
    ctx = RunContext[None](state=None, environment=env)
    captured: dict[str, object] = {}

    class _Probe(App[None]):
        def on_mount(self) -> None:
            self._probe()

        @work
        async def _probe(self) -> None:
            try:
                captured["out"] = await RunPython()._run(
                    RunPythonInput(code="print(40 + 2)"), ctx=ctx
                )
            except Exception as exc:
                captured["err"] = repr(exc)

    async with _Probe().run_test() as pilot:
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
    print(f"\ncaptured={captured}")
    assert captured.get("out"), captured


@pytest.mark.asyncio
async def test_interactive_data_copilot_end_to_end(tmp_path: Path) -> None:
    """The full interactive path the user runs: drive the analyst via the TUI."""
    from textual.widgets import Static

    from grasp_agents.ui.app import GraspAgentsApp, _PromptArea

    analyst, _ctx = build_copilot(tmp_path, confinement="srt")
    app = GraspAgentsApp(on_submit=analyst.run_stream, main_agent=analyst.name)

    def titles(source: str) -> list[str]:
        # Static stores its renderable in the name-mangled __content; read the
        # panel titles back to see what rendered in each pane.
        if source not in app._ga_panes:
            return []
        return [
            str(getattr(getattr(w, "_Static__content", None), "title", ""))
            for w in app._ga_panes[source].query(Static)
        ]

    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#prompt", _PromptArea).text = (
            "Generate 60 random sales values and plot a histogram."
        )
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        # capture inside the context — the widget tree is torn down on app exit
        panes = set(app._ga_panes)
        de_titles, an_titles = titles("data_engineer"), titles("analyst")
        viz_imgs = (
            len(app._ga_panes["viz_specialist"].query(".ga-img"))
            if "viz_specialist" in app._ga_panes
            else 0
        )

    print(f"\npanes={panes}  viz_image_widgets={viz_imgs}")
    print(f"data_engineer titles={de_titles}\nanalyst titles={an_titles}")
    assert {"data_engineer", "viz_specialist"} <= panes, panes
    # tools (RunPython/Bash) must NOT get their own pane/tab — only agents do
    assert "RunPython" not in panes, panes
    assert "Bash" not in panes, panes
    # a subagent's own tool calls render in ITS pane, not the parent analyst's
    assert any("RunPython" in t for t in de_titles), de_titles
    assert not any("RunPython" in t for t in an_titles), an_titles
    assert any("data_engineer" in t for t in an_titles), an_titles
