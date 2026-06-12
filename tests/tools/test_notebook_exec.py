"""
Tests for ``RunCell`` (notebook execution) + its output conversions.

Pure-function conversions and the guard paths run unsandboxed-free (no kernel);
the end-to-end execution tests spawn a real kernel (loopback ZMQ) and are gated
behind the ``integration`` marker (``uv run pytest -m integration``).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import nbformat
import pytest
from nbformat import v4

from grasp_agents.agent.agent_context import AgentContext
from grasp_agents.agent.background_tasks import BackgroundTaskManager
from grasp_agents.agent.llm_agent_transcript import LLMAgentTranscript
from grasp_agents.run_context import RunContext
from grasp_agents.sandbox import local_environment
from grasp_agents.sandbox.kernel import CellOutput
from grasp_agents.tools.bash_common import ShellState
from grasp_agents.tools.bash_session import BashSessionHolder
from grasp_agents.tools.cell_output import (
    render_outputs_as_parts,
    sanitize_output_data,
)
from grasp_agents.tools.file_backend import LocalFileBackend
from grasp_agents.tools.file_edit import (
    FileEditSessionState,
    NotebookReadInput,
    NotebookReadResult,
    NotebookReadTool,
)
from grasp_agents.tools.file_edit.notebook import make_output
from grasp_agents.tools.notebook_exec import (
    KernelHolder,
    RunCell,
    RunCellInput,
    cell_output_to_nbformat,
)
from grasp_agents.types.content import InputImage, InputText
from grasp_agents.types.events import ToolErrorInfo

if TYPE_CHECKING:
    from pathlib import Path

_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="  # noqa: E501


def _error_message(result: Any) -> str:
    assert isinstance(result, ToolErrorInfo), (
        f"Expected a ToolErrorInfo, got {type(result).__name__}: {result!r}"
    )
    return result.error


def _agent_ctx() -> AgentContext:
    transcript = LLMAgentTranscript()
    return AgentContext(
        transcript=transcript,
        tools={},
        file_edit_state=FileEditSessionState(),
        bg_tasks=BackgroundTaskManager(
            agent_name="test", transcript=transcript, tools={}
        ),
        session_holder=BashSessionHolder(),
        nb_kernel_holder=KernelHolder(),
        shell_state=ShellState(),
    )


def _write_code_nb(path: Path, *sources: str) -> None:
    nb = v4.new_notebook()
    for src in sources:
        nb.cells.append(v4.new_code_cell(src))
    path.write_text(nbformat.writes(nb))


# ---------------------------------------------------------------------------
# Pure conversions (no kernel)
# ---------------------------------------------------------------------------


def test_render_outputs_text_and_image() -> None:
    outputs = [
        make_output("stream", name="stdout", text="hello\n"),
        make_output(
            "execute_result", data={"text/plain": "42"}, metadata={}, execution_count=2
        ),
        make_output(
            "display_data",
            data={"image/png": _PNG_B64, "text/plain": "<Figure>"},
            metadata={},
        ),
    ]
    parts = render_outputs_as_parts(outputs, header="[execution_count=2 status=ok]")
    assert isinstance(parts[0], InputText)
    assert "hello" in parts[0].text
    assert "42" in parts[0].text
    assert "execution_count=2" in parts[0].text
    images = [p for p in parts if isinstance(p, InputImage)]
    assert len(images) == 1
    assert images[0].mime_type == "image/png"


def test_render_outputs_error_strips_ansi() -> None:
    outputs = [
        make_output(
            "error",
            ename="ValueError",
            evalue="bad",
            traceback=["\x1b[31mTraceback\x1b[0m", "ValueError: bad"],
        )
    ]
    parts = render_outputs_as_parts(outputs, header="[status=error]")
    assert isinstance(parts[0], InputText)
    assert "\x1b[" not in parts[0].text
    assert "Traceback" in parts[0].text
    assert "status=error" in parts[0].text


def test_render_outputs_caps_text() -> None:
    outputs = [make_output("stream", name="stdout", text="x" * 100)]
    parts = render_outputs_as_parts(outputs, max_text_chars=20)
    assert isinstance(parts[0], InputText)
    assert "[... output truncated ...]" in parts[0].text


def test_render_outputs_truncation_keeps_tail() -> None:
    # A traceback's most informative line is its last — truncation must
    # keep the tail, not just the head.
    text = "start " + "x" * 200 + " ValueError: the actual error"
    outputs = [make_output("stream", name="stdout", text=text)]
    parts = render_outputs_as_parts(outputs, max_text_chars=80)
    assert isinstance(parts[0], InputText)
    assert parts[0].text.startswith("start")
    assert "ValueError: the actual error" in parts[0].text


def test_render_outputs_multiformat_images() -> None:
    outputs = [
        make_output("display_data", data={"image/jpeg": _PNG_B64}, metadata={}),
        make_output("display_data", data={"image/gif": _PNG_B64}, metadata={}),
    ]
    parts = render_outputs_as_parts(outputs)
    mimes = {p.mime_type for p in parts if isinstance(p, InputImage)}
    assert mimes == {"image/jpeg", "image/gif"}


def test_sanitize_drops_javascript_mime() -> None:
    data, modified = sanitize_output_data({"application/javascript": "alert(1)"})
    assert modified
    assert "application/javascript" not in data
    assert data["text/plain"] == "[executable output removed by sanitizer]"


def test_sanitize_strips_html_script_and_handlers() -> None:
    html = (
        '<div onclick="steal()">hi</div>'
        "<script>evil()</script>"
        '<a href="javascript:bad()">x</a>'
    )
    data, modified = sanitize_output_data({"text/html": html})
    assert modified
    out = data["text/html"]
    assert "<script>" not in out
    assert "evil()" not in out
    assert "onclick" not in out
    assert "javascript:" not in out
    assert "hi" in out  # benign content preserved


def test_sanitize_leaves_benign_html_untouched() -> None:
    data, modified = sanitize_output_data(
        {"text/html": "<b>bold</b>", "text/plain": "bold"}
    )
    assert not modified
    assert data["text/html"] == "<b>bold</b>"


def test_cell_output_to_nbformat_sanitizes_by_default() -> None:
    output = CellOutput(
        output_type="display_data",
        data={"application/javascript": "x()", "text/plain": "r"},
    )
    sanitized = cell_output_to_nbformat(output)
    assert "application/javascript" not in sanitized["data"]
    # Opt-out keeps the raw payload.
    raw = cell_output_to_nbformat(output, sanitize=False)
    assert "application/javascript" in raw["data"]


def test_cell_output_to_nbformat_shapes() -> None:
    stream = cell_output_to_nbformat(
        CellOutput(output_type="stream", name="stdout", text="x")
    )
    assert stream["output_type"] == "stream"
    assert stream["text"] == "x"

    img = cell_output_to_nbformat(
        CellOutput(output_type="display_data", data={"image/png": _PNG_B64})
    )
    assert img["output_type"] == "display_data"
    assert "image/png" in img["data"]
    assert "transient" not in img  # not part of the nbformat schema

    err = cell_output_to_nbformat(
        CellOutput(output_type="error", ename="E", evalue="v", traceback=("a", "b"))
    )
    assert err["output_type"] == "error"
    assert err["ename"] == "E"
    assert list(err["traceback"]) == ["a", "b"]


# ---------------------------------------------------------------------------
# Guard paths (no kernel spawned)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_cell_requires_exec_backend(tmp_path: Path) -> None:
    # File backend only, no exec backend → no kernel.
    backend = LocalFileBackend(allowed_roots=[tmp_path])
    ctx: RunContext[Any] = RunContext(file_backend=backend, session_key="s")
    p = tmp_path / "nb.ipynb"
    _write_code_nb(p, "x = 1")
    result = await RunCell().run(
        RunCellInput(notebook_path=str(p), cell_id="anything"),
        ctx=ctx,
        agent_ctx=_agent_ctx(),
    )
    assert "exec_backend" in _error_message(result)


@pytest.mark.asyncio
async def test_run_cell_rejects_markdown(tmp_path: Path) -> None:
    env = local_environment(allowed_roots=[tmp_path])
    ctx: RunContext[Any] = RunContext(environment=env)
    agent_ctx = _agent_ctx()

    nb = v4.new_notebook()
    nb.cells.append(v4.new_markdown_cell("# just prose"))
    p = tmp_path / "nb.ipynb"
    p.write_text(nbformat.writes(nb))

    read = await NotebookReadTool().run(
        NotebookReadInput(path=str(p)), ctx=ctx, agent_ctx=agent_ctx
    )
    assert isinstance(read, NotebookReadResult)
    md_id = read.cells[0].id

    result = await RunCell().run(
        RunCellInput(notebook_path=str(p), cell_id=md_id), ctx=ctx, agent_ctx=agent_ctx
    )
    assert "only code cells" in _error_message(result)


@pytest.mark.asyncio
async def test_run_cell_unknown_cell(tmp_path: Path) -> None:
    env = local_environment(allowed_roots=[tmp_path])
    ctx: RunContext[Any] = RunContext(environment=env)
    agent_ctx = _agent_ctx()
    p = tmp_path / "nb.ipynb"
    _write_code_nb(p, "x = 1")
    await NotebookReadTool().run(
        NotebookReadInput(path=str(p)), ctx=ctx, agent_ctx=agent_ctx
    )
    result = await RunCell().run(
        RunCellInput(notebook_path=str(p), cell_id="ghost"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    assert "No cell with id" in _error_message(result)


# ---------------------------------------------------------------------------
# End-to-end execution (real kernel — integration)
# ---------------------------------------------------------------------------


async def _run_cell(
    ctx: RunContext[Any], agent_ctx: AgentContext, path: Path, cell_id: str
) -> list[InputText | InputImage]:
    result = await RunCell().run(
        RunCellInput(notebook_path=str(path), cell_id=cell_id),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    assert not isinstance(result, ToolErrorInfo), result
    return result  # type: ignore[return-value]


async def _first_cell_id(
    ctx: RunContext[Any], agent_ctx: AgentContext, path: Path
) -> str:
    read = await NotebookReadTool().run(
        NotebookReadInput(path=str(path)), ctx=ctx, agent_ctx=agent_ctx
    )
    assert isinstance(read, NotebookReadResult)
    return read.cells[0].id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_cell_executes_and_writes_back(tmp_path: Path) -> None:
    env = local_environment(allowed_roots=[tmp_path])
    ctx: RunContext[Any] = RunContext(environment=env)
    agent_ctx = _agent_ctx()
    p = tmp_path / "nb.ipynb"
    _write_code_nb(p, "print('hi'); 40 + 2")
    try:
        cid = await _first_cell_id(ctx, agent_ctx, p)
        parts = await _run_cell(ctx, agent_ctx, p, cid)
        text = "".join(p.text for p in parts if isinstance(p, InputText))
        assert "hi" in text
        assert "42" in text

        # Outputs + execution_count written back into the .ipynb.
        nb = json.loads(p.read_text())
        cell = nb["cells"][0]
        assert cell["execution_count"] == 1
        assert cell["outputs"]
    finally:
        await agent_ctx.nb_kernel_holder.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_cell_image_part(tmp_path: Path) -> None:
    env = local_environment(allowed_roots=[tmp_path])
    ctx: RunContext[Any] = RunContext(environment=env)
    agent_ctx = _agent_ctx()
    p = tmp_path / "nb.ipynb"
    _write_code_nb(
        p,
        "import base64\n"
        "from IPython.display import Image, display\n"
        f"display(Image(data=base64.b64decode('{_PNG_B64}'), format='png'))",
    )
    try:
        cid = await _first_cell_id(ctx, agent_ctx, p)
        parts = await _run_cell(ctx, agent_ctx, p, cid)
        images = [part for part in parts if isinstance(part, InputImage)]
        assert images, f"expected an image part, got {parts}"
        # And the image is persisted in the notebook.
        nb = json.loads(p.read_text())
        outs = nb["cells"][0]["outputs"]
        assert any("image/png" in o.get("data", {}) for o in outs)
    finally:
        await agent_ctx.nb_kernel_holder.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_cell_state_persists_across_calls(tmp_path: Path) -> None:
    env = local_environment(allowed_roots=[tmp_path])
    ctx: RunContext[Any] = RunContext(environment=env)
    agent_ctx = _agent_ctx()  # one nb_kernel_holder → one persistent kernel
    p = tmp_path / "nb.ipynb"
    _write_code_nb(p, "value = 5", "value * 2")
    try:
        read = await NotebookReadTool().run(
            NotebookReadInput(path=str(p)), ctx=ctx, agent_ctx=agent_ctx
        )
        assert isinstance(read, NotebookReadResult)
        await _run_cell(ctx, agent_ctx, p, read.cells[0].id)
        parts = await _run_cell(ctx, agent_ctx, p, read.cells[1].id)
        text = "".join(part.text for part in parts if isinstance(part, InputText))
        assert "10" in text
    finally:
        await agent_ctx.nb_kernel_holder.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_cell_error_written_back(tmp_path: Path) -> None:
    env = local_environment(allowed_roots=[tmp_path])
    ctx: RunContext[Any] = RunContext(environment=env)
    agent_ctx = _agent_ctx()
    p = tmp_path / "nb.ipynb"
    _write_code_nb(p, "1 / 0")
    try:
        cid = await _first_cell_id(ctx, agent_ctx, p)
        parts = await _run_cell(ctx, agent_ctx, p, cid)
        text = "".join(part.text for part in parts if isinstance(part, InputText))
        assert "ZeroDivisionError" in text
        nb = json.loads(p.read_text())
        outs = nb["cells"][0]["outputs"]
        assert any(o["output_type"] == "error" for o in outs)
    finally:
        await agent_ctx.nb_kernel_holder.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_cell_sanitizes_executable_output(tmp_path: Path) -> None:
    """A cell emitting JS/script HTML is sanitized before write-back."""
    env = local_environment(allowed_roots=[tmp_path])
    ctx: RunContext[Any] = RunContext(environment=env)
    agent_ctx = _agent_ctx()
    p = tmp_path / "nb.ipynb"
    _write_code_nb(
        p,
        "from IPython.display import HTML, display\n"
        "display(HTML('<div onclick=\"x()\">hi</div><script>evil()</script>'))",
    )
    try:
        cid = await _first_cell_id(ctx, agent_ctx, p)
        await _run_cell(ctx, agent_ctx, p, cid)
        nb = json.loads(p.read_text())
        outs = nb["cells"][0]["outputs"]
        htmls = [o.get("data", {}).get("text/html", "") for o in outs]
        joined = "".join("".join(h) if isinstance(h, list) else h for h in htmls)
        assert "<script>" not in joined
        assert "evil()" not in joined
        assert "onclick" not in joined
        assert "hi" in joined  # benign content survived
    finally:
        await agent_ctx.nb_kernel_holder.close()
