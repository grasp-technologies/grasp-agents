"""
Unit tests for :class:`NotebookReadTool` / :class:`NotebookEditTool` and the
``.ipynb`` redirect in ``Read`` / ``Write`` / ``Edit``.

Focus: cell-id addressing, the three edit modes, the read-before-write +
mtime invariants (inherited from the file-edit family), output-clearing on a
code-cell replace, stable cell-id minting for legacy id-less notebooks, and
the generic-tool redirects. The ``agent_ctx`` / ``state`` fixtures come from
``conftest.py``.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING, Any

import nbformat
import pytest
from nbformat import v4

from grasp_agents.run_context import RunContext
from grasp_agents.tools import FileToolkit
from grasp_agents.tools.file_backend import LocalFileBackend
from grasp_agents.tools.file_edit import (
    EditInput,
    EditTool,
    NotebookEditInput,
    NotebookEditResult,
    NotebookEditTool,
    NotebookReadInput,
    NotebookReadResult,
    NotebookReadTool,
    NullRedactor,
    ReadInput,
    ReadResult,
    ReadTool,
    WriteInput,
    WriteTool,
)
from grasp_agents.types.content import InputImage, InputText
from grasp_agents.types.events import ToolErrorInfo

# 1x1 PNG; other formats use dummy base64 (the tool wraps, doesn't validate bytes).
_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

if TYPE_CHECKING:
    from pathlib import Path

    from grasp_agents.agent.agent_context import AgentContext

pytestmark = pytest.mark.asyncio

TEST_KEY = "test"


def _error_message(result: Any) -> str:
    assert isinstance(result, ToolErrorInfo), (
        f"Expected a ToolErrorInfo, got {type(result).__name__}: {result!r}"
    )
    return result.error


def _src(cell: Any) -> str:
    """Nbformat stores source as str or list-of-lines on disk."""
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def _write_modern_nb(path: Path, cells: list[tuple[str, str]]) -> None:
    """Write an nbformat 4.5 notebook (cells carry native ids)."""
    nb = v4.new_notebook()
    for cell_type, source in cells:
        nb.cells.append(
            v4.new_code_cell(source)
            if cell_type == "code"
            else v4.new_markdown_cell(source)
        )
    path.write_text(nbformat.writes(nb))


def _write_legacy_nb(path: Path, cells: list[tuple[str, str]]) -> None:
    """Write a pre-4.5 notebook JSON with no cell ids."""
    payload: dict[str, Any] = {
        "cells": [
            {"cell_type": ct, "source": src, "metadata": {}}
            | ({"outputs": [], "execution_count": None} if ct == "code" else {})
            for ct, src in cells
        ],
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "nbformat": 4,
        "nbformat_minor": 2,
    }
    path.write_text(json.dumps(payload))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx(tmp_path: Path) -> RunContext[Any]:
    backend = LocalFileBackend(allowed_roots=[tmp_path])
    return RunContext[Any](file_backend=backend, session_key=TEST_KEY)


@pytest.fixture
def nb_read() -> NotebookReadTool:
    return NotebookReadTool()


@pytest.fixture
def nb_edit() -> NotebookEditTool:
    return NotebookEditTool()


async def _read_cells(
    nb_read: NotebookReadTool,
    path: Path,
    ctx: RunContext[Any],
    agent_ctx: AgentContext,
) -> NotebookReadResult:
    result = await nb_read.run(
        NotebookReadInput(path=str(path)), ctx=ctx, agent_ctx=agent_ctx
    )
    assert isinstance(result, NotebookReadResult), result
    return result


# ---------------------------------------------------------------------------
# NotebookRead
# ---------------------------------------------------------------------------


async def test_read_modern_notebook(
    tmp_path: Path,
    ctx: RunContext[Any],
    agent_ctx: AgentContext,
    nb_read: NotebookReadTool,
) -> None:
    p = tmp_path / "nb.ipynb"
    _write_modern_nb(p, [("markdown", "# Title"), ("code", "x = 1")])

    result = await _read_cells(nb_read, p, ctx, agent_ctx)
    assert result.total_cells == 2
    assert [c.cell_type for c in result.cells] == ["markdown", "code"]
    assert result.cells[1].source == "x = 1"
    assert all(c.id for c in result.cells)


async def test_read_output_preview_and_has_output(
    tmp_path: Path,
    ctx: RunContext[Any],
    agent_ctx: AgentContext,
    nb_read: NotebookReadTool,
) -> None:
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "source": "print('hi')",
                "execution_count": 3,
                "metadata": {},
                "id": "cE",
                "outputs": [
                    {"output_type": "stream", "name": "stdout", "text": "hi\n"},
                    {
                        "output_type": "display_data",
                        "data": {"image/png": "iVBORw0KGgo="},
                        "metadata": {},
                    },
                ],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    p = tmp_path / "out.ipynb"
    p.write_text(json.dumps(nb))

    result = await _read_cells(nb_read, p, ctx, agent_ctx)
    cell = result.cells[0]
    assert cell.execution_count == 3
    assert cell.has_output is True
    assert cell.output_preview is not None
    assert "hi" in cell.output_preview
    # Image output is noted, not inlined as bytes, in the multi-cell view.
    assert "[image/png output]" in cell.output_preview


async def test_read_single_cell_by_id(
    tmp_path: Path,
    ctx: RunContext[Any],
    agent_ctx: AgentContext,
    nb_read: NotebookReadTool,
) -> None:
    p = tmp_path / "nb.ipynb"
    _write_modern_nb(p, [("markdown", "a"), ("code", "b"), ("markdown", "c")])
    full = await _read_cells(nb_read, p, ctx, agent_ctx)
    target = full.cells[1].id

    # include_images=False keeps the structured result for a single cell.
    one = await nb_read.run(
        NotebookReadInput(path=str(p), cell_id=target, include_images=False),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    assert isinstance(one, NotebookReadResult)
    assert len(one.cells) == 1
    assert one.cells[0].id == target
    assert one.total_cells == 3  # total reflects the whole notebook


async def test_read_single_cell_missing_id(
    tmp_path: Path,
    ctx: RunContext[Any],
    agent_ctx: AgentContext,
    nb_read: NotebookReadTool,
) -> None:
    p = tmp_path / "nb.ipynb"
    _write_modern_nb(p, [("code", "x")])
    result = await nb_read.run(
        NotebookReadInput(path=str(p), cell_id="nope"), ctx=ctx, agent_ctx=agent_ctx
    )
    assert "No cell with id" in _error_message(result)


async def test_legacy_ids_stable_across_reads(
    tmp_path: Path,
    ctx: RunContext[Any],
    agent_ctx: AgentContext,
    nb_read: NotebookReadTool,
) -> None:
    """A read does not write; ids are deterministic so two reads agree."""
    p = tmp_path / "legacy.ipynb"
    _write_legacy_nb(p, [("markdown", "a"), ("code", "b")])
    before = p.read_text()

    first = await _read_cells(nb_read, p, ctx, agent_ctx)
    second = await _read_cells(nb_read, p, ctx, agent_ctx)
    assert [c.id for c in first.cells] == [c.id for c in second.cells]
    assert all(c.id for c in first.cells)
    # Read is side-effect free on disk.
    assert p.read_text() == before


def _nb_with_image(image_mime: str) -> dict[str, Any]:
    return {
        "cells": [
            {
                "cell_type": "code",
                "source": "plot()",
                "execution_count": 1,
                "id": "c0",
                "metadata": {},
                "outputs": [
                    {
                        "output_type": "display_data",
                        "data": {image_mime: _PNG_B64, "text/plain": "<Figure>"},
                        "metadata": {},
                    }
                ],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


async def test_read_single_cell_surfaces_image_parts(
    tmp_path: Path,
    ctx: RunContext[Any],
    agent_ctx: AgentContext,
    nb_read: NotebookReadTool,
) -> None:
    """A cell_id read returns content parts including the cell's stored image."""
    p = tmp_path / "plot.ipynb"
    p.write_text(json.dumps(_nb_with_image("image/png")))

    result = await nb_read.run(
        NotebookReadInput(path=str(p), cell_id="c0"), ctx=ctx, agent_ctx=agent_ctx
    )
    assert isinstance(result, list)
    texts = [x for x in result if isinstance(x, InputText)]
    images = [x for x in result if isinstance(x, InputImage)]
    assert texts
    assert "id=c0" in texts[0].text
    assert len(images) == 1
    assert images[0].mime_type == "image/png"


async def test_read_surfaces_non_png_formats(
    tmp_path: Path,
    ctx: RunContext[Any],
    agent_ctx: AgentContext,
    nb_read: NotebookReadTool,
) -> None:
    p = tmp_path / "j.ipynb"
    p.write_text(json.dumps(_nb_with_image("image/jpeg")))
    result = await nb_read.run(
        NotebookReadInput(path=str(p), cell_id="c0"), ctx=ctx, agent_ctx=agent_ctx
    )
    assert isinstance(result, list)
    images = [x for x in result if isinstance(x, InputImage)]
    assert len(images) == 1
    assert images[0].mime_type == "image/jpeg"


async def test_read_whole_notebook_omits_images_by_default(
    tmp_path: Path,
    ctx: RunContext[Any],
    agent_ctx: AgentContext,
    nb_read: NotebookReadTool,
) -> None:
    """Whole-notebook read stays structured (placeholders) to bound context."""
    p = tmp_path / "w.ipynb"
    p.write_text(json.dumps(_nb_with_image("image/png")))
    result = await nb_read.run(
        NotebookReadInput(path=str(p)), ctx=ctx, agent_ctx=agent_ctx
    )
    assert isinstance(result, NotebookReadResult)
    preview = result.cells[0].output_preview
    assert preview is not None
    assert "[image/png" in preview


async def test_read_whole_notebook_with_include_images(
    tmp_path: Path,
    ctx: RunContext[Any],
    agent_ctx: AgentContext,
    nb_read: NotebookReadTool,
) -> None:
    """include_images=True surfaces figures across the whole notebook."""
    p = tmp_path / "wi.ipynb"
    p.write_text(json.dumps(_nb_with_image("image/png")))
    result = await nb_read.run(
        NotebookReadInput(path=str(p), include_images=True),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    assert isinstance(result, list)
    assert any(isinstance(x, InputImage) for x in result)


# ---------------------------------------------------------------------------
# NotebookEdit — replace
# ---------------------------------------------------------------------------


async def test_edit_replace_clears_code_outputs(
    tmp_path: Path,
    ctx: RunContext[Any],
    agent_ctx: AgentContext,
    nb_read: NotebookReadTool,
    nb_edit: NotebookEditTool,
) -> None:
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "source": "x = 1",
                "execution_count": 7,
                "metadata": {},
                "outputs": [{"output_type": "stream", "name": "stdout", "text": "z"}],
                "id": "c0",
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    p = tmp_path / "nb.ipynb"
    p.write_text(json.dumps(nb))
    await _read_cells(nb_read, p, ctx, agent_ctx)

    result = await nb_edit.run(
        NotebookEditInput(notebook_path=str(p), cell_id="c0", new_source="x = 2"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    assert isinstance(result, NotebookEditResult)
    assert result.edit_mode == "replace"
    assert result.cell_type == "code"

    after = json.loads(p.read_text())
    cell = after["cells"][0]
    assert _src(cell) == "x = 2"
    assert cell["execution_count"] is None
    assert cell["outputs"] == []


async def test_edit_replace_markdown_keeps_type(
    tmp_path: Path,
    ctx: RunContext[Any],
    agent_ctx: AgentContext,
    nb_read: NotebookReadTool,
    nb_edit: NotebookEditTool,
) -> None:
    p = tmp_path / "nb.ipynb"
    _write_modern_nb(p, [("markdown", "old")])
    r = await _read_cells(nb_read, p, ctx, agent_ctx)
    cid = r.cells[0].id

    result = await nb_edit.run(
        NotebookEditInput(notebook_path=str(p), cell_id=cid, new_source="new"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    assert isinstance(result, NotebookEditResult)
    assert result.cell_type == "markdown"
    after = json.loads(p.read_text())
    assert _src(after["cells"][0]) == "new"
    assert after["cells"][0]["cell_type"] == "markdown"


async def test_edit_replace_unknown_cell_id(
    tmp_path: Path,
    ctx: RunContext[Any],
    agent_ctx: AgentContext,
    nb_read: NotebookReadTool,
    nb_edit: NotebookEditTool,
) -> None:
    p = tmp_path / "nb.ipynb"
    _write_modern_nb(p, [("code", "x")])
    await _read_cells(nb_read, p, ctx, agent_ctx)

    result = await nb_edit.run(
        NotebookEditInput(notebook_path=str(p), cell_id="ghost", new_source="y"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    assert "No cell with id" in _error_message(result)


async def test_edit_replace_requires_cell_id(
    tmp_path: Path,
    ctx: RunContext[Any],
    agent_ctx: AgentContext,
    nb_read: NotebookReadTool,
    nb_edit: NotebookEditTool,
) -> None:
    p = tmp_path / "nb.ipynb"
    _write_modern_nb(p, [("code", "x")])
    await _read_cells(nb_read, p, ctx, agent_ctx)
    result = await nb_edit.run(
        NotebookEditInput(notebook_path=str(p), new_source="y"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    assert "cell_id is required" in _error_message(result)


# ---------------------------------------------------------------------------
# NotebookEdit — insert
# ---------------------------------------------------------------------------


async def test_edit_insert_after_cell(
    tmp_path: Path,
    ctx: RunContext[Any],
    agent_ctx: AgentContext,
    nb_read: NotebookReadTool,
    nb_edit: NotebookEditTool,
) -> None:
    p = tmp_path / "nb.ipynb"
    _write_modern_nb(p, [("markdown", "first"), ("code", "second")])
    r = await _read_cells(nb_read, p, ctx, agent_ctx)
    first_id = r.cells[0].id

    result = await nb_edit.run(
        NotebookEditInput(
            notebook_path=str(p),
            cell_id=first_id,
            new_source="inserted",
            cell_type="code",
            edit_mode="insert",
        ),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    assert isinstance(result, NotebookEditResult)
    assert result.total_cells == 3
    after = json.loads(p.read_text())
    assert [_src(c) for c in after["cells"]] == ["first", "inserted", "second"]
    assert after["cells"][1]["cell_type"] == "code"
    assert after["cells"][1]["id"] == result.cell_id


async def test_edit_insert_at_start_without_cell_id(
    tmp_path: Path,
    ctx: RunContext[Any],
    agent_ctx: AgentContext,
    nb_read: NotebookReadTool,
    nb_edit: NotebookEditTool,
) -> None:
    p = tmp_path / "nb.ipynb"
    _write_modern_nb(p, [("code", "body")])
    await _read_cells(nb_read, p, ctx, agent_ctx)

    result = await nb_edit.run(
        NotebookEditInput(
            notebook_path=str(p),
            new_source="# header",
            cell_type="markdown",
            edit_mode="insert",
        ),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    assert isinstance(result, NotebookEditResult)
    after = json.loads(p.read_text())
    assert [_src(c) for c in after["cells"]] == ["# header", "body"]


async def test_edit_insert_requires_cell_type(
    tmp_path: Path,
    ctx: RunContext[Any],
    agent_ctx: AgentContext,
    nb_read: NotebookReadTool,
    nb_edit: NotebookEditTool,
) -> None:
    p = tmp_path / "nb.ipynb"
    _write_modern_nb(p, [("code", "x")])
    await _read_cells(nb_read, p, ctx, agent_ctx)
    result = await nb_edit.run(
        NotebookEditInput(notebook_path=str(p), new_source="y", edit_mode="insert"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    assert "cell_type is required" in _error_message(result)


# ---------------------------------------------------------------------------
# NotebookEdit — delete
# ---------------------------------------------------------------------------


async def test_edit_delete(
    tmp_path: Path,
    ctx: RunContext[Any],
    agent_ctx: AgentContext,
    nb_read: NotebookReadTool,
    nb_edit: NotebookEditTool,
) -> None:
    p = tmp_path / "nb.ipynb"
    _write_modern_nb(p, [("code", "keep"), ("code", "drop")])
    r = await _read_cells(nb_read, p, ctx, agent_ctx)
    drop_id = r.cells[1].id

    result = await nb_edit.run(
        NotebookEditInput(notebook_path=str(p), cell_id=drop_id, edit_mode="delete"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    assert isinstance(result, NotebookEditResult)
    assert result.total_cells == 1
    after = json.loads(p.read_text())
    assert [_src(c) for c in after["cells"]] == ["keep"]


# ---------------------------------------------------------------------------
# Read-before-write + staleness (inherited family invariants)
# ---------------------------------------------------------------------------


async def test_edit_refuses_without_prior_read(
    tmp_path: Path,
    ctx: RunContext[Any],
    agent_ctx: AgentContext,
    nb_edit: NotebookEditTool,
) -> None:
    p = tmp_path / "nb.ipynb"
    _write_modern_nb(p, [("code", "x")])
    nb = json.loads(p.read_text())
    cid = nb["cells"][0]["id"]

    result = await nb_edit.run(
        NotebookEditInput(notebook_path=str(p), cell_id=cid, new_source="y"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    assert "Must Read" in _error_message(result)


async def test_edit_refuses_on_stale_mtime(
    tmp_path: Path,
    ctx: RunContext[Any],
    agent_ctx: AgentContext,
    nb_read: NotebookReadTool,
    nb_edit: NotebookEditTool,
) -> None:
    p = tmp_path / "nb.ipynb"
    _write_modern_nb(p, [("code", "x")])
    r = await _read_cells(nb_read, p, ctx, agent_ctx)
    cid = r.cells[0].id

    await asyncio.sleep(0.01)
    _write_modern_nb(p, [("code", "tampered")])
    os.utime(p, None)

    result = await nb_edit.run(
        NotebookEditInput(notebook_path=str(p), cell_id=cid, new_source="y"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    assert "modified since you last read" in _error_message(result)


async def test_consecutive_edits_do_not_trip_staleness(
    tmp_path: Path,
    ctx: RunContext[Any],
    agent_ctx: AgentContext,
    nb_read: NotebookReadTool,
    nb_edit: NotebookEditTool,
) -> None:
    p = tmp_path / "nb.ipynb"
    _write_modern_nb(p, [("code", "a"), ("code", "b")])
    r = await _read_cells(nb_read, p, ctx, agent_ctx)
    id0, id1 = r.cells[0].id, r.cells[1].id

    await nb_edit.run(
        NotebookEditInput(notebook_path=str(p), cell_id=id0, new_source="A"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    # Second edit in the same session: the first edit refreshed the mtime
    # record, so this does not trip the staleness check. The id is stable
    # because cell 1 carried a native id preserved across the rewrite.
    result = await nb_edit.run(
        NotebookEditInput(notebook_path=str(p), cell_id=id1, new_source="B"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    assert isinstance(result, NotebookEditResult)
    after = json.loads(p.read_text())
    assert [_src(c) for c in after["cells"]] == ["A", "B"]


async def test_edit_refuses_missing_file(
    tmp_path: Path,
    ctx: RunContext[Any],
    agent_ctx: AgentContext,
    nb_edit: NotebookEditTool,
) -> None:
    p = tmp_path / "ghost.ipynb"
    result = await nb_edit.run(
        NotebookEditInput(notebook_path=str(p), cell_id="x", new_source="y"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    err = _error_message(result)
    assert "does not exist" in err or "not found" in err.lower()


# ---------------------------------------------------------------------------
# .ipynb redirect from generic Read / Write / Edit
# ---------------------------------------------------------------------------


async def test_generic_read_renders_cell_view(
    tmp_path: Path,
    ctx: RunContext[Any],
    agent_ctx: AgentContext,
) -> None:
    p = tmp_path / "nb.ipynb"
    _write_modern_nb(p, [("markdown", "# Heading"), ("code", "compute()")])

    read_tool = ReadTool(redactor=NullRedactor())
    result = await read_tool.run(ReadInput(path=str(p)), ctx=ctx, agent_ctx=agent_ctx)
    assert isinstance(result, ReadResult)
    # Cell-structured text, not raw notebook JSON.
    assert result.content.startswith("Notebook:")
    assert "# Heading" in result.content
    assert "compute()" in result.content
    assert '"cell_type"' not in result.content  # no raw JSON leaked


async def test_generic_read_then_notebook_edit(
    tmp_path: Path,
    ctx: RunContext[Any],
    agent_ctx: AgentContext,
    nb_edit: NotebookEditTool,
) -> None:
    """The generic Read on an .ipynb records the read, satisfying read-before-edit."""
    p = tmp_path / "nb.ipynb"
    _write_modern_nb(p, [("code", "x")])
    cid = json.loads(p.read_text())["cells"][0]["id"]

    read_tool = ReadTool(redactor=NullRedactor())
    await read_tool.run(ReadInput(path=str(p)), ctx=ctx, agent_ctx=agent_ctx)

    result = await nb_edit.run(
        NotebookEditInput(notebook_path=str(p), cell_id=cid, new_source="y"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    assert isinstance(result, NotebookEditResult)


async def test_generic_write_refuses_ipynb(
    tmp_path: Path, ctx: RunContext[Any], agent_ctx: AgentContext
) -> None:
    p = tmp_path / "nb.ipynb"
    _write_modern_nb(p, [("code", "x")])
    write_tool = WriteTool()
    result = await write_tool.run(
        WriteInput(path=str(p), content="{}"), ctx=ctx, agent_ctx=agent_ctx
    )
    msg = _error_message(result)
    assert "NotebookEdit" in msg
    # File untouched.
    assert "cells" in p.read_text()


async def test_generic_edit_refuses_ipynb(
    tmp_path: Path, ctx: RunContext[Any], agent_ctx: AgentContext
) -> None:
    p = tmp_path / "nb.ipynb"
    _write_modern_nb(p, [("code", "x")])
    edit_tool = EditTool()
    result = await edit_tool.run(
        EditInput(path=str(p), old_string="x", new_string="y"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    assert "NotebookEdit" in _error_message(result)


# ---------------------------------------------------------------------------
# Toolkit wiring
# ---------------------------------------------------------------------------


async def test_toolkit_excludes_notebook_by_default() -> None:
    tk = FileToolkit(redactor=NullRedactor())
    names = {t.name for t in tk.tools()}
    assert "NotebookRead" not in names
    assert "NotebookEdit" not in names


async def test_toolkit_includes_notebook_when_enabled() -> None:
    tk = FileToolkit(redactor=NullRedactor(), include_notebook=True)
    names = [t.name for t in tk.tools()]
    assert "NotebookRead" in names
    assert "NotebookEdit" in names
    assert "NotebookRead" in {t.name for t in tk.read_only_tools()}
    assert "NotebookEdit" not in {t.name for t in tk.read_only_tools()}
