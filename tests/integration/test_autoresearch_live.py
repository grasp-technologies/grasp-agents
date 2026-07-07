"""
Live end-to-end test for the auto-research demo (``examples/tui/autoresearch.py``):
a real LLM runs a short but complete research session — venv setup, dataset
download through the srt network allowlist, notebook-driven modeling, hidden
holdout scoring — and the session state must survive a simulated restart.

Slow (several minutes) and ``integration``-gated; needs ``OPENAI_API_KEY``
and the ``srt`` CLI. Run unsandboxed, foreground:

    uv run pytest -m integration tests/integration/test_autoresearch_live.py -s
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from grasp_agents.examples.tui.autoresearch import (
    ResearchState,
    build_researcher,
    prepare_workspace,
    run_headless,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("srt") is None, reason="srt not installed"),
]

_SHORT_GOAL = """\
Short research session — be quick and frugal:
1. Download the Titanic data and split it.
2. ONE simple baseline, 5-fold CV, record_experiment.
3. ONE improvement attempt, CV, record_experiment.
4. Submit your best model's holdout predictions exactly once.
5. Finish with a 5-line research report. No plots, small cells."""


def _model() -> str:
    return os.environ.get("AUTORESEARCH_TEST_MODEL", "gpt-5.5")


def test_prepare_workspace_is_idempotent(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    prepare_workspace(ws)
    # Lays out the workspace; the venv itself is provisioned by the env later.
    assert (ws / "data").is_dir()
    assert (ws / "submissions").is_dir()
    notebook = json.loads((ws / "research.ipynb").read_text())
    assert notebook["nbformat"] == 4

    (ws / "research.ipynb").write_text(json.dumps(notebook | {"marker": True}))
    prepare_workspace(ws)
    # An existing notebook is never overwritten.
    assert json.loads((ws / "research.ipynb").read_text()).get("marker") is True


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY")
@pytest.mark.asyncio
async def test_short_research_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # srt forces TMPDIR to $CLAUDE_CODE_TMPDIR when set (outside the sandbox);
    # drop it so the kernel uses srt's own writable temp.
    monkeypatch.delenv("CLAUDE_CODE_TMPDIR", raising=False)
    monkeypatch.delenv("CLAUDE_TMPDIR", raising=False)

    ws = tmp_path / "ws"
    researcher, ctx = build_researcher(ws, model=_model(), max_turns=30)

    final = await run_headless(researcher, goal=_SHORT_GOAL)

    assert final, "agent produced no final report"
    print(f"\n=== final report ===\n{final}")

    # The research protocol ran: split done, experiments logged, one scored
    # submission against the hidden holdout.
    state = ctx.state
    assert state.holdout_labels, "split_dataset was never called"
    assert any(e.cv_accuracy is not None for e in state.experiments), (
        "no CV experiment was recorded"
    )
    assert state.submissions_used >= 1, "no holdout submission was made"
    assert state.best_holdout is not None, "no submission was scored"
    assert state.best_holdout > 0.7, f"holdout accuracy too low: {state.best_holdout}"
    assert any(e.holdout_balanced_accuracy is not None for e in state.experiments)

    # Real work happened in the notebook: executed code cells with outputs.
    nb = json.loads((ws / "research.ipynb").read_text())
    executed = [
        c for c in nb["cells"] if c["cell_type"] == "code" and c.get("execution_count")
    ]
    assert executed, "no executed code cells in research.ipynb"

    # The split artifacts exist and the holdout is unlabeled on disk.
    assert (ws / "data" / "train.csv").is_file()
    holdout_header = (ws / "data" / "holdout.csv").read_text().splitlines()[0]
    assert "Survived" not in holdout_header

    # --- Simulated restart: fresh agent + ctx over the same checkpoints ---
    researcher2, ctx2 = build_researcher(ws, model=_model(), max_turns=30)
    assert ctx2.state == ResearchState()  # fresh state before loading
    async with researcher2:
        # Run-start order: the session restore (ctx.state) precedes the
        # agent's own reload — run_stream does both; a direct load must too.
        await ctx2.load_checkpoint()
        checkpoint = await researcher2.load_checkpoint()
    assert checkpoint is not None, "no agent checkpoint on disk"
    # serialize_state=True rehydrates the experiment log + hidden labels.
    assert ctx2.state.experiments == state.experiments
    assert ctx2.state.holdout_labels == state.holdout_labels
    assert ctx2.state.submissions_used == state.submissions_used
