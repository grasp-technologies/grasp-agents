"""
Tests for ``RunPython`` (the code-interpreter tool).

The artifact-collection logic and guard paths run kernel-free (no integration
marker). The end-to-end execution tests spawn a real kernel under **srt**
confinement (loopback ZMQ + write confinement); they are gated behind the
``integration`` marker and ``srt`` being installed, and run via
``uv run pytest -m integration``.
"""

from __future__ import annotations

import base64
import shutil
from typing import TYPE_CHECKING, Any

import pytest

from grasp_agents.agent.agent_context import AgentContext
from grasp_agents.agent.background_tasks import BackgroundTaskManager
from grasp_agents.agent.llm_agent_transcript import LLMAgentTranscript
from grasp_agents.run_context import RunContext
from grasp_agents.sandbox import local_environment
from grasp_agents.tools.code_interpreter import RunPython, RunPythonInput, _human_size
from grasp_agents.tools.file_backend import LocalFileBackend
from grasp_agents.types.content import InputImage, InputText
from grasp_agents.types.events import ToolErrorInfo

if TYPE_CHECKING:
    from pathlib import Path

# A 1x1 transparent PNG — lets us assert image handling without matplotlib.
_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="  # noqa: E501
_PNG_BYTES = base64.b64decode(_PNG_B64)

_needs_srt = pytest.mark.skipif(shutil.which("srt") is None, reason="srt not installed")


@pytest.fixture(autouse=True)
def _drop_claude_tmpdir(monkeypatch: pytest.MonkeyPatch) -> None:
    # srt forces TMPDIR to $CLAUDE_CODE_TMPDIR (set inside Claude Code, outside
    # the sandbox) → kernel temp fails. Drop it so srt uses its own writable
    # temp. No-op in a normal shell; harmless for the kernel-free tests.
    monkeypatch.delenv("CLAUDE_CODE_TMPDIR", raising=False)
    monkeypatch.delenv("CLAUDE_TMPDIR", raising=False)


def _error_message(result: Any) -> str:
    assert isinstance(result, ToolErrorInfo), (
        f"Expected a ToolErrorInfo, got {type(result).__name__}: {result!r}"
    )
    return result.error


def _agent_ctx() -> AgentContext:
    transcript = LLMAgentTranscript()
    return AgentContext.create(
        transcript=transcript,
        tools={},
        bg_tasks=BackgroundTaskManager(
            agent_name="test", transcript=transcript, tools={}
        ),
    )


def _text(parts: list[InputText | InputImage]) -> str:
    return "\n".join(p.text for p in parts if isinstance(p, InputText))


def _images(parts: list[InputText | InputImage]) -> list[InputImage]:
    return [p for p in parts if isinstance(p, InputImage)]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_human_size() -> None:
    assert _human_size(0) == "0 B"
    assert _human_size(512) == "512 B"
    assert _human_size(1536) == "1.5 KB"
    assert _human_size(5 * 1024 * 1024) == "5.0 MB"


# ---------------------------------------------------------------------------
# Guard paths (no kernel spawned)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_python_requires_exec_backend(tmp_path: Path) -> None:
    backend = LocalFileBackend(allowed_roots=[tmp_path])
    ctx: RunContext[Any] = RunContext(file_backend=backend, session_key="s")
    result = await RunPython().run(
        RunPythonInput(code="x = 1"), ctx=ctx, agent_ctx=_agent_ctx()
    )
    assert "exec_backend" in _error_message(result)


# ---------------------------------------------------------------------------
# Artifact collection (kernel-free — files staged on disk directly).
# Artifacts are surfaced as references (path / mime / size), never inlined.
# ---------------------------------------------------------------------------


async def _collect(tmp_path: Path, paths: list[str], **tool_kwargs: Any) -> list[str]:
    env = local_environment(allowed_roots=[tmp_path])
    ctx: RunContext[Any] = RunContext(environment=env)
    return await RunPython(**tool_kwargs)._collect_artifacts(ctx, _agent_ctx(), paths)


@pytest.mark.asyncio
async def test_collect_artifacts_image_is_referenced_not_inlined(
    tmp_path: Path,
) -> None:
    (tmp_path / "shot.png").write_bytes(_PNG_BYTES)
    lines = await _collect(tmp_path, ["shot.png"])
    assert any("shot.png" in line and "image/png" in line for line in lines)


@pytest.mark.asyncio
async def test_collect_artifacts_file_reference(tmp_path: Path) -> None:
    (tmp_path / "data.csv").write_text("a,b\n1,2\n")
    lines = await _collect(tmp_path, ["data.csv"])
    assert any("data.csv" in line and "text/csv" in line for line in lines)


@pytest.mark.asyncio
async def test_collect_artifacts_directory_expands(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "a.txt").write_text("a")
    (out / "b.txt").write_text("bb")
    lines = await _collect(tmp_path, ["out"])
    assert sum("a.txt" in line for line in lines) == 1
    assert sum("b.txt" in line for line in lines) == 1


@pytest.mark.asyncio
async def test_collect_artifacts_absolute_path(tmp_path: Path) -> None:
    target = tmp_path / "abs.csv"
    target.write_text("x")
    lines = await _collect(tmp_path, [str(target)])
    assert any("abs.csv" in line for line in lines)


@pytest.mark.asyncio
async def test_collect_artifacts_missing(tmp_path: Path) -> None:
    lines = await _collect(tmp_path, ["nope.bin"])
    assert any("not accessible" in line for line in lines)


@pytest.mark.asyncio
async def test_collect_artifacts_no_file_backend() -> None:
    ctx: RunContext[Any] = RunContext(session_key="s")
    lines = await RunPython()._collect_artifacts(ctx, _agent_ctx(), ["x.png"])
    assert any("no file_backend" in line for line in lines)


# ---------------------------------------------------------------------------
# End-to-end execution — real kernel under srt confinement (integration).
# ---------------------------------------------------------------------------


def _srt_ctx(tmp_path: Path) -> RunContext[Any]:
    env = local_environment(allowed_roots=[tmp_path], confinement="srt")
    return RunContext(environment=env)


async def _run(
    ctx: RunContext[Any], agent_ctx: AgentContext, inp: RunPythonInput
) -> list[InputText | InputImage]:
    result = await RunPython().run(inp, ctx=ctx, agent_ctx=agent_ctx)
    assert not isinstance(result, ToolErrorInfo), result
    return result  # type: ignore[return-value]


async def _close(agent_ctx: AgentContext) -> None:
    await agent_ctx.nb_kernel_holder.close()
    if agent_ctx.ipy_kernel_holder is not None:
        await agent_ctx.ipy_kernel_holder.close()


@_needs_srt
@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_python_executes_and_returns_value(tmp_path: Path) -> None:
    ctx = _srt_ctx(tmp_path)
    agent_ctx = _agent_ctx()
    try:
        parts = await _run(ctx, agent_ctx, RunPythonInput(code="print('hi'); 40 + 2"))
        text = _text(parts)
        assert "hi" in text
        assert "42" in text
        assert "status=ok" in text
    finally:
        await _close(agent_ctx)


@_needs_srt
@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_python_state_persists(tmp_path: Path) -> None:
    ctx = _srt_ctx(tmp_path)
    agent_ctx = _agent_ctx()  # create() wires a persistent code kernel
    try:
        await _run(ctx, agent_ctx, RunPythonInput(code="value = 41"))
        parts = await _run(ctx, agent_ctx, RunPythonInput(code="value + 1"))
        assert "42" in _text(parts)
    finally:
        await _close(agent_ctx)


@_needs_srt
@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_python_all_output_modalities(tmp_path: Path) -> None:
    """Stdout + execute_result + every viewable image MIME + an error."""
    ctx = _srt_ctx(tmp_path)
    agent_ctx = _agent_ctx()
    code = (
        "import base64\n"
        "from IPython.display import display\n"
        f"_b64 = '{_PNG_B64}'\n"
        "print('modality-stdout')\n"
        "for _m in ('image/png', 'image/jpeg', 'image/gif', 'image/webp'):\n"
        "    display({_m: _b64}, raw=True)\n"
        "21 * 2\n"
    )
    try:
        parts = await _run(ctx, agent_ctx, RunPythonInput(code=code))
        text = _text(parts)
        assert "modality-stdout" in text  # stream
        assert "42" in text  # execute_result
        mimes = {img.mime_type for img in _images(parts)}
        assert mimes == {"image/png", "image/jpeg", "image/gif", "image/webp"}
    finally:
        await _close(agent_ctx)


@_needs_srt
@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_python_error_output(tmp_path: Path) -> None:
    ctx = _srt_ctx(tmp_path)
    agent_ctx = _agent_ctx()
    try:
        parts = await _run(ctx, agent_ctx, RunPythonInput(code="1 / 0"))
        text = _text(parts)
        assert "ZeroDivisionError" in text
        assert "status=error" in text
    finally:
        await _close(agent_ctx)


@_needs_srt
@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_python_inline_plot(tmp_path: Path) -> None:
    ctx = _srt_ctx(tmp_path)
    agent_ctx = _agent_ctx()
    code = (
        "import base64\n"
        "from IPython.display import Image, display\n"
        f"display(Image(data=base64.b64decode('{_PNG_B64}'), format='png'))\n"
    )
    try:
        parts = await _run(ctx, agent_ctx, RunPythonInput(code=code))
        assert _images(parts), f"expected an inline image, got {parts}"
    finally:
        await _close(agent_ctx)


@_needs_srt
@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_python_artifacts_are_references(tmp_path: Path) -> None:
    """A saved figure + CSV come back as references (not inlined images)."""
    ctx = _srt_ctx(tmp_path)
    agent_ctx = _agent_ctx()
    code = (
        "import base64\n"
        f"open('out.png', 'wb').write(base64.b64decode('{_PNG_B64}'))\n"
        "open('data.csv', 'w').write('a,b\\n1,2\\n')\n"
    )
    try:
        parts = await _run(
            ctx, agent_ctx, RunPythonInput(code=code, artifacts=["out.png", "data.csv"])
        )
        text = _text(parts)
        assert "out.png" in text
        assert "image/png" in text
        assert "data.csv" in text
        assert "text/csv" in text
        assert not _images(parts), "saved files must be referenced, not inlined"
    finally:
        await _close(agent_ctx)


@_needs_srt
@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_python_displays_saved_image(tmp_path: Path) -> None:
    """The model can view a saved image by displaying it (the display channel)."""
    ctx = _srt_ctx(tmp_path)
    agent_ctx = _agent_ctx()
    code = (
        "import base64\n"
        f"open('fig.png', 'wb').write(base64.b64decode('{_PNG_B64}'))\n"
        "from IPython.display import Image, display\n"
        "display(Image(filename='fig.png'))\n"
    )
    try:
        parts = await _run(ctx, agent_ctx, RunPythonInput(code=code))
        assert _images(parts), "displaying a saved image should inline it"
    finally:
        await _close(agent_ctx)


@_needs_srt
@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_python_kernel_separate_from_run_cell(tmp_path: Path) -> None:
    """RunPython and RunCell are independent kernels — state does not leak."""
    import nbformat
    from nbformat import v4

    from grasp_agents.tools.file_edit import (
        NotebookReadInput,
        NotebookReadResult,
        NotebookReadTool,
    )
    from grasp_agents.tools.notebook_exec import RunCell, RunCellInput

    ctx = _srt_ctx(tmp_path)
    agent_ctx = _agent_ctx()

    nb = v4.new_notebook()
    nb.cells.append(v4.new_code_cell("marker"))
    p = tmp_path / "nb.ipynb"
    p.write_text(nbformat.writes(nb))

    try:
        # Define a variable in RunPython's kernel …
        await _run(ctx, agent_ctx, RunPythonInput(code="marker = 99"))
        # … and confirm a notebook cell (RunCell's kernel) does NOT see it.
        read = await NotebookReadTool().run(
            NotebookReadInput(path=str(p)), ctx=ctx, agent_ctx=agent_ctx
        )
        assert isinstance(read, NotebookReadResult)
        result = await RunCell().run(
            RunCellInput(notebook_path=str(p), cell_id=read.cells[0].id),
            ctx=ctx,
            agent_ctx=agent_ctx,
        )
        assert not isinstance(result, ToolErrorInfo), result
        assert "NameError" in _text(result)  # type: ignore[arg-type]
    finally:
        await _close(agent_ctx)
