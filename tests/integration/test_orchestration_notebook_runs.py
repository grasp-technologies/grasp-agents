"""
Smoke-execute the orchestration & durability demo notebook end-to-end (real
LLM + srt sandboxes), so the shipped ``.ipynb`` is known to actually run. The
demo exercises composable workflows, a Runner team, four crash/resume
scenarios on a ``FileCheckpointStore``, and a confined Bash + RunPython
analysis — we assert the on-disk traces each section leaves behind.

``integration``-gated; needs ``OPENAI_API_KEY``, the ``srt`` CLI, and
``nbclient``. Run unsandboxed:

    uv run pytest -m integration tests/integration/test_orchestration_notebook_runs.py
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

_NB = (
    Path(__file__).resolve().parents[2]
    / "src/grasp_agents/examples/notebooks/orchestration_durability_demo.ipynb"
)


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY")
@pytest.mark.skipif(shutil.which("srt") is None, reason="needs srt")
def test_orchestration_demo_notebook_executes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")

    # srt forces TMPDIR to $CLAUDE_CODE_TMPDIR (set inside Claude Code, outside
    # the sandbox); drop it so the in-notebook srt kernels use their own temp.
    monkeypatch.delenv("CLAUDE_CODE_TMPDIR", raising=False)
    monkeypatch.delenv("CLAUDE_TMPDIR", raising=False)

    nb = nbformat.read(str(_NB), as_version=4)
    client = nbclient.NotebookClient(
        nb,
        timeout=420,  # per cell; the slowest is the sandboxed analyst run
        kernel_name="python3",
        # Run with `tmp_path` as cwd so ./orchestration_workdir lands there.
        resources={"metadata": {"path": str(tmp_path)}},
    )
    client.execute()  # raises CellExecutionError if any cell errors

    # The resume demos prove completed work is restored, not re-executed —
    # their counter printouts are the actual claim, so pin them. Strip ANSI:
    # the EventConsole renders in color (and syntax-highlights XML notices
    # word-by-word), so colour codes would otherwise split phrases we match.
    import re

    raw_stdout = "\n".join(
        "".join(out.get("text", ""))
        for cell in nb.cells
        if cell.cell_type == "code"
        for out in cell.get("outputs", [])
        if out.get("output_type") == "stream"
    )
    stdout = re.sub(r"\x1b\[[0-9;]*m", "", raw_stdout)
    assert "add-tool calls on resume: 1" in stdout
    assert "drafter LLM calls after resume: 1" in stdout
    assert "picker LLM calls after resume: 1" in stdout

    ws = tmp_path / "orchestration_workdir"

    # Section 3 left checkpoints for every session on disk.
    checkpoints = ws / "checkpoints"
    sessions = {p.name for p in checkpoints.iterdir() if p.is_dir()}
    assert {
        "solo-agent", "workflow-demo", "team-demo", "bg-demo", "team-research"
    } <= sessions

    # Section 3.4: a Bash bg task isn't resurrected, but its streamed output is
    # durably saved to .grasp/tasks/ — partial (interrupted before the batch
    # finished). On resume the agent reads that log and processes only the
    # REMAINING items, completing the batch without redoing finished ones.
    bg_logs = list((ws / "bg_workspace" / ".grasp" / "tasks").glob("*.log"))
    assert bg_logs, "no background task log was saved"
    all_bg = "\n".join(p.read_text() for p in bg_logs)
    assert "item 1 of 5 done" in all_bg  # the first run processed item 1...
    # ...and the earliest (first-run) log is partial — it did NOT finish the batch.
    first_log = min(bg_logs, key=lambda p: p.stat().st_mtime)
    assert "item 5 of 5 done" not in first_log.read_text()
    # On resume the agent finished the remaining items, completing the batch.
    assert "item 5 of 5 done" in stdout + all_bg
    # ...and on resume the agent was notified (the framing always streams).
    assert "Resumed from a checkpoint" in stdout, (
        "resume should stream the framing notice to the agent"
    )

    # Section 3.5: the resumable sub-agents drove to completion after the crash
    # (the re-spawn mechanism is asserted rigorously in
    # tests/integration/test_durable_subagents_live.py).
    assert "coordinator briefing after resume" in stdout

    # Section 4: the sandboxed analyst downloaded the dataset and saved the
    # chart it displayed.
    assert (ws / "data_workspace" / "penguins.csv").is_file()
    chart = ws / "data_workspace" / "penguins_mass.png"
    assert chart.is_file()
    # ...and the chart is not blank. A blank figure is still a valid non-zero
    # PNG, so check the pixels: a real bar chart (axes, ticks, bars) has plenty
    # of non-white content; an empty white canvas (saving after plt.show()
    # closed the figure) does not.
    pytest.importorskip("numpy")
    mpimg = pytest.importorskip("matplotlib.image")
    img = mpimg.imread(str(chart))
    non_white = float((img[..., :3] < 0.95).any(axis=-1).mean())
    assert non_white > 0.02, (
        f"penguins_mass.png looks blank (non-white fraction={non_white:.4f})"
    )
    # ...quietly: `curl -fsSL` keeps curl's progress meter (a stderr write on
    # every byte) out of the agent's tool output.
    assert "Dload  Upload" not in stdout, "curl progress meter leaked into output"
    # The deliberate crash/resume demos must read cleanly: their on-purpose
    # failures are caught and narrated ("crashed as planned"), never dumped as
    # a raw logged traceback (the Runner cell wraps its crash in
    # expected_crash() to hush the EventBus ERROR log).
    assert "Traceback (most recent call last)" not in stdout, (
        "a deliberate-crash demo leaked a logged traceback into the output"
    )
    # The background-task demo command must not lead with a bare `sleep` (the
    # Bash anti-spin guard rejects that); leading with real work keeps it
    # long-running without tripping the guard.
    assert "stalls the agent loop" not in stdout, (
        "the bg-task command tripped Bash's leading-sleep guard"
    )
