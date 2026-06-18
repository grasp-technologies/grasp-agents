"""
Smoke-execute the code-interpreter demo notebook end-to-end (real LLM + srt
kernel), so the shipped ``.ipynb`` is known to actually run, not just parse.
The demo drives a multi-step persistent session — two RunPython steps where the
second reuses the first's variables, then a Bash step — and we assert the
figure the agent saves in step 2 (``cumsum.png``) lands in the workspace.

``integration``-gated; needs ``OPENAI_API_KEY``, the ``srt`` CLI, and
``nbclient``. Run unsandboxed:

    uv run pytest -m integration tests/integration/test_demo_notebook_runs.py
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

_NB = (
    Path(__file__).resolve().parents[2]
    / "src/grasp_agents/examples/notebooks/code_interpreter.ipynb"
)


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY")
@pytest.mark.skipif(shutil.which("srt") is None, reason="needs srt")
def test_code_interpreter_demo_notebook_executes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")

    # srt forces TMPDIR to $CLAUDE_CODE_TMPDIR (set inside Claude Code, outside
    # the sandbox); drop it so the in-notebook srt kernel uses its own temp.
    monkeypatch.delenv("CLAUDE_CODE_TMPDIR", raising=False)
    monkeypatch.delenv("CLAUDE_TMPDIR", raising=False)

    nb = nbformat.read(str(_NB), as_version=4)
    client = nbclient.NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        # Run with `tmp_path` as cwd so the notebook's ./ci_workdir lands there.
        resources={"metadata": {"path": str(tmp_path)}},
    )
    client.execute()  # raises CellExecutionError if any cell errors
    # Step 2's RunPython saves cumsum.png in the workspace (./ci_workdir under
    # the notebook's cwd = tmp_path) — confirms the persistent multi-step
    # session ran and did real work.
    assert (tmp_path / "ci_workdir" / "cumsum.png").is_file(), (
        "expected the agent to save cumsum.png via the persistent RunPython session"
    )
