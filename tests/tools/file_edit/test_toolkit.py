"""
Integration tests for :class:`FileEditToolkit`.

Checks that the toolkit wires its three tools to a single backing store,
per-session reset clears the right slot, and the end-to-end Read → Write
and Read → Edit flows compose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from grasp_agents.tools.file_edit import (
    EditTool,
    FileEditToolkit,
    NullRedactor,
    ReadInput,
    ReadTool,
    WriteInput,
    WriteTool,
)
from grasp_agents.types.events import ToolErrorInfo


def _error_message(result: Any) -> str:
    """Unwrap a ``ToolErrorInfo`` returned from ``.run(...)``."""
    assert isinstance(result, ToolErrorInfo), (
        f"Expected a ToolErrorInfo, got {type(result).__name__}: {result!r}"
    )
    return result.error


def test_toolkit_default_root_is_cwd() -> None:
    tk = FileEditToolkit()
    # Private-member access intentional: asserting internal wiring.
    assert tk._allowed_roots == [Path.cwd()]  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]


def test_toolkit_returns_three_tools(tmp_path: Path) -> None:
    tk = FileEditToolkit(allowed_roots=[tmp_path])
    tools = tk.tools()
    assert len(tools) == 3
    assert isinstance(tools[0], ReadTool)
    assert isinstance(tools[1], WriteTool)
    assert isinstance(tools[2], EditTool)


def test_tools_share_store(tmp_path: Path) -> None:
    """
    All tools must point at the same backing store; otherwise
    read-before-write can never succeed.
    """
    tk = FileEditToolkit(allowed_roots=[tmp_path])
    shared = tk.store
    # Private-member access intentional: asserting the invariant.
    assert tk.read._store is shared  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    assert tk.write._store is shared  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    assert tk.edit._store is shared  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_read_then_write_composes(tmp_path: Path) -> None:
    tk = FileEditToolkit(allowed_roots=[tmp_path], redactor=NullRedactor())
    f = tmp_path / "a.txt"
    f.write_text("original")

    await tk.read.run(ReadInput(path=str(f)))
    await tk.write.run(WriteInput(path=str(f), content="updated"))

    assert f.read_text() == "updated"


@pytest.mark.asyncio
async def test_reset_session_clears_everything(tmp_path: Path) -> None:
    tk = FileEditToolkit(allowed_roots=[tmp_path], redactor=NullRedactor())
    f = tmp_path / "a.txt"
    f.write_text("x")
    await tk.read.run(ReadInput(path=str(f)))
    await tk.allow_dotfile(tmp_path / ".env")

    # State exists pre-reset.
    state_before = await tk.store.get_session_state(tk.default_session_key)
    assert state_before.read_file_state
    assert state_before.dotfile_overrides

    await tk.reset_session()

    # Post-reset get_session_state returns a freshly-created state.
    state_after = await tk.store.get_session_state(tk.default_session_key)
    assert state_after.read_file_state == {}
    assert state_after.dotfile_overrides == set()


@pytest.mark.asyncio
async def test_reset_session_invalidates_read_before_write(
    tmp_path: Path,
) -> None:
    """
    After reset_session the Write tool sees no prior read, so writes to
    existing files are refused until re-read.
    """
    tk = FileEditToolkit(allowed_roots=[tmp_path], redactor=NullRedactor())
    f = tmp_path / "a.txt"
    f.write_text("x")
    await tk.read.run(ReadInput(path=str(f)))

    await tk.reset_session()

    result = await tk.write.run(WriteInput(path=str(f), content="y"))
    assert "Must Read" in _error_message(result)


@pytest.mark.asyncio
async def test_allow_dotfile_adds_resolved_path(tmp_path: Path) -> None:
    tk = FileEditToolkit(allowed_roots=[tmp_path])
    await tk.allow_dotfile(tmp_path / ".env")
    state = await tk.store.get_session_state(tk.default_session_key)
    assert (tmp_path / ".env").resolve() in state.dotfile_overrides


@pytest.mark.asyncio
async def test_reset_session_isolated_per_key(tmp_path: Path) -> None:
    """
    ``reset_session(key)`` affects only the given key; other sessions'
    state survives.
    """
    tk = FileEditToolkit(allowed_roots=[tmp_path])
    await tk.allow_dotfile(tmp_path / ".env", session_key="alice")
    await tk.allow_dotfile(tmp_path / ".env", session_key="bob")

    await tk.reset_session(session_key="alice")

    alice = await tk.store.get_session_state("alice")
    bob = await tk.store.get_session_state("bob")
    assert alice.dotfile_overrides == set()
    assert bob.dotfile_overrides  # Bob unaffected
