"""Tests for :class:`CopyTool` (``Copy``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from grasp_agents.file_backend import CompositeFileBackend, InMemoryFileBackend, Mount
from grasp_agents.session_context import SessionContext
from grasp_agents.tools import FileToolkit
from grasp_agents.tools.file_edit import (
    CopyInput,
    CopyResult,
    CopyTool,
    EditInput,
    EditResult,
    FileEditSessionState,
    NullRedactor,
    ReadInput,
)
from grasp_agents.types.events import ToolErrorInfo
from tests._helpers import _make_agent_ctx

pytestmark = pytest.mark.asyncio

NOTES = Path("/user/notes")
WORK = Path("/work")
LIBRARY = Path("/library")


def _setup(
    **notes: str,
) -> tuple[
    CompositeFileBackend, SessionContext[Any], Any, dict[str, InMemoryFileBackend]
]:
    children = {
        "notes": InMemoryFileBackend(
            [NOTES], files={str(NOTES / k): v for k, v in notes.items()}
        ),
        "work": InMemoryFileBackend([WORK]),
        "library": InMemoryFileBackend(
            [LIBRARY], files={str(LIBRARY / "ref.md"): "ref"}
        ),
    }
    backend = CompositeFileBackend(
        [
            Mount(NOTES, children["notes"]),
            Mount(WORK, children["work"]),
            Mount(LIBRARY, children["library"], read_only=True),
        ],
        scratch=WORK,
    )
    ctx: SessionContext[Any] = SessionContext(file_backend=backend, session_key="s")
    agent_ctx = _make_agent_ctx(file_edit_state=FileEditSessionState())
    return backend, ctx, agent_ctx, children


async def _copy(ctx: SessionContext[Any], agent_ctx: Any, **kwargs: Any) -> Any:
    return await CopyTool().run(CopyInput(**kwargs), ctx=ctx, agent_ctx=agent_ctx)


def _error(result: Any) -> str:
    assert isinstance(result, ToolErrorInfo), result
    return result.error


async def test_copies_a_file_across_mounts() -> None:
    _, ctx, agent_ctx, children = _setup(**{"a.md": "alpha"})

    result = await _copy(
        ctx, agent_ctx, source=str(NOTES / "a.md"), destination=str(WORK / "a.md")
    )

    assert isinstance(result, CopyResult)
    assert (result.files_copied, result.bytes_copied) == (1, 5)
    assert (await children["work"].read_text(WORK / "a.md"))[0] == "alpha"
    assert (await children["notes"].read_text(NOTES / "a.md"))[0] == "alpha"


async def test_copies_a_file_into_an_existing_directory() -> None:
    _, ctx, agent_ctx, children = _setup(**{"a.md": "alpha"})
    await children["work"].mkdir(WORK / "in")

    result = await _copy(
        ctx, agent_ctx, source=str(NOTES / "a.md"), destination=str(WORK / "in")
    )

    assert isinstance(result, CopyResult)
    assert result.destination == str(WORK / "in/a.md")
    assert await children["work"].exists(WORK / "in/a.md")


async def test_copies_a_directory_tree() -> None:
    _, ctx, agent_ctx, children = _setup(
        **{"a.md": "1", "inbox/b.md": "22", "inbox/deep/c.md": "333"}
    )

    result = await _copy(
        ctx, agent_ctx, source=str(NOTES), destination=str(WORK / "notes")
    )

    assert isinstance(result, CopyResult)
    assert (result.files_copied, result.bytes_copied) == (3, 6)
    assert (await children["work"].read_text(WORK / "notes/inbox/deep/c.md"))[
        0
    ] == "333"


async def test_refuses_to_overwrite_without_permission() -> None:
    _, ctx, agent_ctx, children = _setup(**{"a.md": "new", "b.md": "new"})
    await children["work"].write_bytes(WORK / "out/a.md", b"old", mode=0o644)

    result = await _copy(
        ctx, agent_ctx, source=str(NOTES), destination=str(WORK / "out")
    )

    assert "already exist" in _error(result)
    assert "/work/out/a.md" in _error(result)
    assert not await children["work"].exists(WORK / "out/b.md")
    assert (await children["work"].read_text(WORK / "out/a.md"))[0] == "old"


async def test_overwrite_replaces_existing_files() -> None:
    _, ctx, agent_ctx, children = _setup(**{"a.md": "new"})
    await children["work"].write_bytes(WORK / "a.md", b"old", mode=0o644)

    result = await _copy(
        ctx,
        agent_ctx,
        source=str(NOTES / "a.md"),
        destination=str(WORK / "a.md"),
        overwrite=True,
    )

    assert isinstance(result, CopyResult)
    assert (await children["work"].read_text(WORK / "a.md"))[0] == "new"


async def test_overwrite_refuses_a_destination_changed_since_read() -> None:
    _, ctx, agent_ctx, children = _setup(**{"a.md": "new"})
    await children["work"].write_bytes(WORK / "a.md", b"old", mode=0o644)
    await FileToolkit(redactor=NullRedactor()).read.run(
        ReadInput(path=str(WORK / "a.md")), ctx=ctx, agent_ctx=agent_ctx
    )
    await children["work"].write_bytes(WORK / "a.md", b"someone else", mode=0o644)

    result = await _copy(
        ctx,
        agent_ctx,
        source=str(NOTES / "a.md"),
        destination=str(WORK / "a.md"),
        overwrite=True,
    )

    assert "modified since you last read it" in _error(result)
    assert (await children["work"].read_text(WORK / "a.md"))[0] == "someone else"


async def test_refuses_a_read_only_destination() -> None:
    _, ctx, agent_ctx, _ = _setup(**{"a.md": "x"})

    result = await _copy(
        ctx, agent_ctx, source=str(NOTES / "a.md"), destination=str(LIBRARY / "a.md")
    )

    assert "read-only" in _error(result)


async def test_copied_files_count_as_known_to_the_agent() -> None:
    _, ctx, agent_ctx, _ = _setup(**{"a.md": "alpha"})
    await _copy(
        ctx, agent_ctx, source=str(NOTES / "a.md"), destination=str(WORK / "a.md")
    )

    edit = await FileToolkit().edit.run(
        EditInput(path=str(WORK / "a.md"), old_string="alpha", new_string="beta"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )

    assert isinstance(edit, EditResult)


async def test_enforces_the_file_cap_before_writing() -> None:
    _, ctx, agent_ctx, children = _setup(**{"a.md": "1", "b.md": "2", "c.md": "3"})

    result = await CopyTool(max_files=2).run(
        CopyInput(source=str(NOTES), destination=str(WORK / "n")),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )

    assert "3 files" in _error(result)
    assert not await children["work"].exists(WORK / "n")


async def test_enforces_the_byte_cap_before_writing() -> None:
    _, ctx, agent_ctx, children = _setup(**{"a.md": "12345"})

    result = await CopyTool(max_bytes=4).run(
        CopyInput(source=str(NOTES / "a.md"), destination=str(WORK / "a.md")),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )

    assert "bytes" in _error(result)
    assert not await children["work"].exists(WORK / "a.md")


async def test_refuses_to_copy_onto_itself_or_into_itself() -> None:
    _, ctx, agent_ctx, _ = _setup(**{"a.md": "x"})

    onto = await _copy(
        ctx, agent_ctx, source=str(NOTES / "a.md"), destination=str(NOTES / "a.md")
    )
    into = await _copy(
        ctx, agent_ctx, source=str(NOTES), destination=str(NOTES / "copy")
    )

    assert "itself" in _error(onto)
    assert "itself" in _error(into)


async def test_single_file_needs_an_existing_parent() -> None:
    _, ctx, agent_ctx, _ = _setup(**{"a.md": "x"})

    result = await _copy(
        ctx,
        agent_ctx,
        source=str(NOTES / "a.md"),
        destination=str(WORK / "no/such/a.md"),
    )

    assert "Parent directory" in _error(result)


async def test_toolkit_offers_copy_only_when_asked() -> None:
    assert "Copy" not in [t.name for t in FileToolkit().tools()]
    assert "Copy" in [t.name for t in FileToolkit(include_copy=True).tools()]
    assert "Copy" not in [
        t.name for t in FileToolkit(include_copy=True).read_only_tools()
    ]
