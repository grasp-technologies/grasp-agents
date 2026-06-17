"""
E2B compatibility for the notebook tools.

N1 (NotebookRead / NotebookEdit) is backend-agnostic — it routes through
``ctx.file_backend`` — so it works over an E2B sandbox's filesystem unchanged.
N2 (RunCell) needs a ``KernelCapable`` exec backend; the base ``e2b`` backend is
shell + files only (no in-sandbox Jupyter kernel), so RunCell refuses with a
clear error rather than silently failing — remote-kernel execution on E2B is a
separate (deferred) backend.

Live, gated on ``e2b`` + ``E2B_API_KEY`` and run unsandboxed:
``uv run pytest -m integration tests/sandbox/test_notebook_e2b.py``.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import nbformat
import pytest
from nbformat import v4

from grasp_agents.agent.agent_context import AgentContext
from grasp_agents.agent.background_tasks import BackgroundTaskManager
from grasp_agents.agent.llm_agent_transcript import LLMAgentTranscript
from grasp_agents.run_context import RunContext
from grasp_agents.sandbox import e2b_environment
from grasp_agents.tools.bash_common import ShellState
from grasp_agents.tools.bash_session import BashSessionHolder
from grasp_agents.tools.file_edit import (
    FileEditSessionState,
    NotebookEditInput,
    NotebookEditResult,
    NotebookEditTool,
    NotebookReadInput,
    NotebookReadResult,
    NotebookReadTool,
)
from grasp_agents.tools.notebook_exec import KernelHolder, RunCell, RunCellInput
from grasp_agents.types.events import ToolErrorInfo

if TYPE_CHECKING:
    from grasp_agents.file_backend.base import FileBackend

pytestmark = pytest.mark.asyncio

_WS = "/home/user/workspace"
_HAS_E2B = importlib.util.find_spec("e2b") is not None
# bool, not the key itself — a raw key in a module global can surface in logs.
_HAS_E2B_KEY = bool(os.getenv("E2B_API_KEY"))
_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="  # noqa: E501
_live = pytest.mark.skipif(
    not (_HAS_E2B and _HAS_E2B_KEY), reason="needs e2b installed + E2B_API_KEY"
)


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
        ipy_kernel_holder=KernelHolder(),
        shell_state=ShellState(),
    )


async def _write_notebook(fb: FileBackend, path: Path, source: str) -> None:
    nb = v4.new_notebook()
    nb.cells.append(v4.new_code_cell(source))
    await fb.write_bytes(path, nbformat.writes(nb).encode("utf-8"), mode=0o644)


def _src(cell: Any) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s


async def _kernel_stdout(kernel: Any, code: str) -> str:
    """Run ``code`` in the kernel; return its concatenated text output."""
    from grasp_agents.sandbox.kernel import CellResult

    text = ""
    async for item in kernel.execute(code):
        if isinstance(item, CellResult):
            continue
        if item.output_type == "stream":
            text += item.text or ""
        elif item.output_type == "error":
            text += (item.evalue or "") + " ".join(item.traceback)
        elif item.data:
            text += str(item.data.get("text/plain", ""))
    return text


@pytest.mark.integration
@_live
async def test_notebook_read_edit_on_e2b() -> None:
    """NotebookRead/NotebookEdit operate on a notebook in the E2B sandbox FS."""
    async with e2b_environment(allowed_roots=[_WS]) as env:
        ctx: RunContext[Any] = RunContext(environment=env)
        agent_ctx = _agent_ctx()
        path = Path(f"{_WS}/demo.ipynb")
        await _write_notebook(env.file_backend, path, "x = 1")

        read = await NotebookReadTool().run(
            NotebookReadInput(path=str(path)), ctx=ctx, agent_ctx=agent_ctx
        )
        assert isinstance(read, NotebookReadResult)
        assert read.total_cells == 1
        cid = read.cells[0].id

        edited = await NotebookEditTool().run(
            NotebookEditInput(notebook_path=str(path), cell_id=cid, new_source="x = 2"),
            ctx=ctx,
            agent_ctx=agent_ctx,
        )
        assert isinstance(edited, NotebookEditResult)

        text, _ = await env.file_backend.read_text(path)
        after = json.loads(text)
        assert _src(after["cells"][0]) == "x = 2"


@pytest.mark.integration
@_live
async def test_run_cell_refuses_without_code_interpreter() -> None:
    """On a non-code-interpreter E2B sandbox, RunCell refuses clearly."""
    async with e2b_environment(allowed_roots=[_WS]) as env:
        ctx: RunContext[Any] = RunContext(environment=env)
        agent_ctx = _agent_ctx()
        path = Path(f"{_WS}/demo.ipynb")
        await _write_notebook(env.file_backend, path, "x = 1")

        read = await NotebookReadTool().run(
            NotebookReadInput(path=str(path)), ctx=ctx, agent_ctx=agent_ctx
        )
        assert isinstance(read, NotebookReadResult)

        result = await RunCell().run(
            RunCellInput(notebook_path=str(path), cell_id=read.cells[0].id),
            ctx=ctx,
            agent_ctx=agent_ctx,
        )
        assert isinstance(result, ToolErrorInfo)
        assert "code-interpreter" in result.error


@pytest.mark.integration
@_live
async def test_run_cell_on_e2b_code_interpreter() -> None:
    """RunCell executes against an E2B code-interpreter kernel (state + plot)."""
    from grasp_agents.types.content import InputImage, InputText

    async with e2b_environment(allowed_roots=[_WS], code_interpreter=True) as env:
        ctx: RunContext[Any] = RunContext(environment=env)
        agent_ctx = _agent_ctx()
        path = Path(f"{_WS}/demo.ipynb")

        nb = v4.new_notebook()
        nb.cells.append(v4.new_code_cell("y = 21"))
        nb.cells.append(v4.new_code_cell("print('hi'); y * 2"))
        nb.cells.append(
            v4.new_code_cell(
                "import base64\n"
                "from IPython.display import Image, display\n"
                f"display(Image(data=base64.b64decode('{_PNG_B64}'), format='png'))"
            )
        )
        await env.file_backend.write_bytes(
            path, nbformat.writes(nb).encode("utf-8"), mode=0o644
        )

        read = await NotebookReadTool().run(
            NotebookReadInput(path=str(path)), ctx=ctx, agent_ctx=agent_ctx
        )
        assert isinstance(read, NotebookReadResult)
        ids = [c.id for c in read.cells]

        try:
            # cell 0 sets state; cell 1 uses it -> persistence across RunCell calls
            await RunCell().run(
                RunCellInput(notebook_path=str(path), cell_id=ids[0]),
                ctx=ctx,
                agent_ctx=agent_ctx,
            )
            parts = await RunCell().run(
                RunCellInput(notebook_path=str(path), cell_id=ids[1]),
                ctx=ctx,
                agent_ctx=agent_ctx,
            )
            assert not isinstance(parts, ToolErrorInfo), parts
            text = "".join(p.text for p in parts if isinstance(p, InputText))
            assert "hi" in text
            assert "42" in text

            # a plot cell -> an image part + image persisted in the notebook
            plot = await RunCell().run(
                RunCellInput(notebook_path=str(path), cell_id=ids[2]),
                ctx=ctx,
                agent_ctx=agent_ctx,
            )
            assert not isinstance(plot, ToolErrorInfo), plot
            assert any(isinstance(p, InputImage) for p in plot)
            after = json.loads((await env.file_backend.read_text(path))[0])
            outs = after["cells"][2]["outputs"]
            assert any("image/png" in o.get("data", {}) for o in outs)
        finally:
            await agent_ctx.nb_kernel_holder.close()


@pytest.mark.integration
@_live
async def test_run_python_on_e2b_code_interpreter() -> None:
    """RunPython executes against an E2B code-interpreter kernel (state + plot)."""
    from grasp_agents.tools.code_interpreter import RunPython, RunPythonInput
    from grasp_agents.types.content import InputImage, InputText

    async with e2b_environment(allowed_roots=[_WS], code_interpreter=True) as env:
        ctx: RunContext[Any] = RunContext(environment=env)
        agent_ctx = _agent_ctx()
        try:
            # State persists across RunPython calls (its own code kernel).
            await RunPython().run(
                RunPythonInput(code="y = 21"), ctx=ctx, agent_ctx=agent_ctx
            )
            parts = await RunPython().run(
                RunPythonInput(code="print('hi'); y * 2"),
                ctx=ctx,
                agent_ctx=agent_ctx,
            )
            assert not isinstance(parts, ToolErrorInfo), parts
            text = "".join(p.text for p in parts if isinstance(p, InputText))
            assert "hi" in text
            assert "42" in text

            # A displayed plot comes back as an image part.
            plot = await RunPython().run(
                RunPythonInput(
                    code=(
                        "import base64\n"
                        "from IPython.display import Image, display\n"
                        f"display(Image(data=base64.b64decode('{_PNG_B64}'), "
                        "format='png'))"
                    )
                ),
                ctx=ctx,
                agent_ctx=agent_ctx,
            )
            assert not isinstance(plot, ToolErrorInfo), plot
            assert any(isinstance(p, InputImage) for p in plot)
        finally:
            await agent_ctx.nb_kernel_holder.close()
            if agent_ctx.ipy_kernel_holder is not None:
                await agent_ctx.ipy_kernel_holder.close()


@pytest.mark.integration
@_live
async def test_kernel_context_rebind_preserves_state_on_e2b() -> None:
    """
    Re-attaching a new kernel to a persisted ``context_id`` keeps its in-memory
    variables — the mechanism that lets a session resumed from a checkpoint keep
    its kernel state instead of starting fresh. Kernel-level (no LLM needed).
    """
    from grasp_agents.sandbox.kernel import KernelCapable

    async with e2b_environment(allowed_roots=[_WS], code_interpreter=True) as env:
        backend = env.exec_backend
        assert isinstance(backend, KernelCapable)

        # Session 1: set state; capture the context id (what a checkpoint stores).
        k1 = await backend.open_kernel()
        async for _ in k1.execute("session_var = 1234"):
            pass
        cid = k1.context_id
        assert cid is not None
        await k1.close()  # close must NOT tear the context down

        # Resume: a brand-new kernel re-bound to the id still sees the variable.
        k2 = await backend.open_kernel(context_id=cid)
        assert k2.context_id == cid
        assert "1234" in await _kernel_stdout(k2, "print(session_var)")
        await k2.close()

        # The KernelHolder seed (what AgentContext.create threads on resume)
        # routes the id through open_kernel — the same live context, so a mutation
        # builds on the persisted value.
        holder = KernelHolder(context_id=cid)
        assert holder.context_id == cid
        k3 = await holder.get(backend)
        assert k3.context_id == cid
        assert "1235" in await _kernel_stdout(k3, "print(session_var + 1)")
        await holder.close()

        # Control: a fresh kernel (no id) must NOT see it — proving the re-attach
        # is what carried the state, not a shared sandbox-wide global.
        k4 = await backend.open_kernel()
        assert k4.context_id != cid
        out = await _kernel_stdout(
            k4, "print('present' if 'session_var' in globals() else 'absent')"
        )
        assert "absent" in out
        await k4.close()
