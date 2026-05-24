"""
Unit tests for :class:`DeleteTool`.

Focus: read-before-delete + mtime-staleness invariants and the
directory-refusal guard.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pytest

from grasp_agents.tools.file_edit import (
    DeleteInput,
    DeleteResult,
    DeleteTool,
    InMemoryFileEditStore,
    NullRedactor,
    ReadInput,
    ReadTool,
)
from grasp_agents.types.events import ToolErrorInfo

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.asyncio

TEST_KEY = "test"


def _error_message(result: Any) -> str:
    assert isinstance(result, ToolErrorInfo), (
        f"Expected a ToolErrorInfo, got {type(result).__name__}: {result!r}"
    )
    return result.error


@pytest.fixture
def store() -> InMemoryFileEditStore:
    return InMemoryFileEditStore()


@pytest.fixture
def read_tool(store: InMemoryFileEditStore, tmp_path: Path) -> ReadTool:
    return ReadTool(
        store=store,
        allowed_roots=[tmp_path],
        redactor=NullRedactor(),
        default_session_key=TEST_KEY,
    )


@pytest.fixture
def delete_tool(store: InMemoryFileEditStore, tmp_path: Path) -> DeleteTool:
    return DeleteTool(
        store=store,
        allowed_roots=[tmp_path],
        default_session_key=TEST_KEY,
    )


async def test_delete_succeeds_after_read(
    read_tool: ReadTool, delete_tool: DeleteTool, tmp_path: Path
) -> None:
    f = tmp_path / "victim.txt"
    f.write_text("bye")

    await read_tool.run(ReadInput(path=str(f)))
    result = await delete_tool.run(DeleteInput(path=str(f)))

    assert isinstance(result, DeleteResult)
    assert result.deleted
    assert not f.exists()


async def test_delete_refuses_without_prior_read(
    delete_tool: DeleteTool, tmp_path: Path
) -> None:
    f = tmp_path / "victim.txt"
    f.write_text("bye")

    result = await delete_tool.run(DeleteInput(path=str(f)))
    msg = _error_message(result)
    assert "Read" in msg
    assert f.exists(), "File must remain when read-before-delete fails"


async def test_delete_refuses_on_mtime_drift(
    read_tool: ReadTool, delete_tool: DeleteTool, tmp_path: Path
) -> None:
    f = tmp_path / "drifty.txt"
    f.write_text("v1")

    await read_tool.run(ReadInput(path=str(f)))
    # Bump mtime past the recorded value — simulates an external edit.
    st = f.stat()
    os.utime(f, (st.st_atime, st.st_mtime + 5))

    result = await delete_tool.run(DeleteInput(path=str(f)))
    msg = _error_message(result)
    assert "modified" in msg.lower() or "re-read" in msg.lower()
    assert f.exists()


async def test_delete_refuses_directory(
    delete_tool: DeleteTool, tmp_path: Path
) -> None:
    d = tmp_path / "subdir"
    d.mkdir()
    # The fact that Delete refuses dirs is independent of read-before-delete;
    # if we hit the dir check, we expect a clear message.
    result = await delete_tool.run(DeleteInput(path=str(d)))
    msg = _error_message(result)
    # Either of "directories" or the read-before-delete path are acceptable
    # — both block the dangerous operation.
    assert (
        "directories" in msg.lower()
        or "directory" in msg.lower()
        or "Read" in msg
    )
    assert d.is_dir()


async def test_delete_clears_read_state(
    read_tool: ReadTool,
    delete_tool: DeleteTool,
    store: InMemoryFileEditStore,
    tmp_path: Path,
) -> None:
    f = tmp_path / "ephemeral.txt"
    f.write_text("temp")

    await read_tool.run(ReadInput(path=str(f)))
    await delete_tool.run(DeleteInput(path=str(f)))

    state = await store.get_session_state(TEST_KEY)
    # After deletion, the read record is cleared so a future Write to
    # the same path treats it as a fresh create.
    assert f.resolve() not in state.read_file_state


async def test_delete_outside_root_refused(
    delete_tool: DeleteTool, tmp_path: Path
) -> None:
    # Escape attempt — must hit the allowed-roots check, not the
    # read-before-delete check.
    result = await delete_tool.run(DeleteInput(path="/etc/hosts"))
    msg = _error_message(result)
    assert "outside" in msg.lower() or "sensitive" in msg.lower()
