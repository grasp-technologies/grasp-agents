"""Tests for :class:`InMemoryFileBackend`."""

from __future__ import annotations

import errno
import stat as stat_module
import time
from pathlib import Path
from typing import Any

import pytest

from grasp_agents.file_backend import (
    InMemoryFileBackend,
    PathAccessError,
    StaleFileError,
)
from grasp_agents.session_context import SessionContext
from grasp_agents.tools import FileToolkit
from grasp_agents.tools.file_edit import (
    EditInput,
    EditResult,
    FileEditSessionState,
    NullRedactor,
    ReadInput,
    ReadResult,
    WriteInput,
    WriteResult,
)
from grasp_agents.tools.file_search import (
    GlobInput,
    GlobResult,
    GrepInput,
    GrepResult,
)
from tests._helpers import _make_agent_ctx, max_loop_gap

pytestmark = pytest.mark.asyncio

ROOT = Path("/mem")


def _backend(**files: str) -> InMemoryFileBackend:
    return InMemoryFileBackend([ROOT], files={f"/mem/{k}": v for k, v in files.items()})


async def test_write_then_read_round_trips() -> None:
    backend = _backend()
    mtime = await backend.write_bytes(ROOT / "a.md", b"hello", mode=0o644)

    text, read_mtime = await backend.read_text(ROOT / "a.md")

    assert text == "hello"
    assert read_mtime == mtime
    assert (await backend.stat(ROOT / "a.md")).size == 5


async def test_mtime_strictly_increases_per_write() -> None:
    backend = _backend()
    first = await backend.write_bytes(ROOT / "a.md", b"1", mode=0o644)
    second = await backend.write_bytes(ROOT / "a.md", b"2", mode=0o644)
    third = await backend.write_bytes(ROOT / "b.md", b"3", mode=0o644)

    assert first < second < third


async def test_seeded_files_create_implicit_directories() -> None:
    backend = _backend(**{"notes/inbox/a.md": "x"})

    assert await backend.exists(ROOT / "notes")
    assert await backend.exists(ROOT / "notes/inbox")
    assert stat_module.S_ISDIR((await backend.stat(ROOT / "notes")).mode)
    assert stat_module.S_ISREG((await backend.stat(ROOT / "notes/inbox/a.md")).mode)
    assert await backend.parent_exists(ROOT / "notes/inbox/b.md")
    assert not await backend.parent_exists(ROOT / "other/b.md")


async def test_roots_exist_even_when_empty() -> None:
    backend = _backend()

    assert await backend.exists(ROOT)
    assert await backend.parent_exists(ROOT / "a.md")
    assert await backend.list_dir(ROOT) == []


async def test_list_dir_direct_and_recursive() -> None:
    backend = _backend(**{"a.md": "1", "notes/b.md": "2", "notes/deep/c.md": "3"})

    direct = await backend.list_dir(ROOT)
    recursive = await backend.list_dir(ROOT, recursive=True)

    assert sorted((e.name, e.is_dir) for e in direct) == [
        ("a.md", False),
        ("notes", True),
    ]
    assert sorted(str(e.path) for e in recursive) == [
        "/mem/a.md",
        "/mem/notes",
        "/mem/notes/b.md",
        "/mem/notes/deep",
        "/mem/notes/deep/c.md",
    ]


async def test_mkdir_creates_an_empty_directory() -> None:
    backend = _backend()
    await backend.mkdir(ROOT / "x/y")

    assert await backend.exists(ROOT / "x/y")
    assert [e.name for e in await backend.list_dir(ROOT / "x")] == ["y"]


async def test_validate_path_confines_to_roots() -> None:
    backend = _backend(**{"a.md": "1"})

    assert await backend.validate_path(Path("a.md"), must_exist=True) == ROOT / "a.md"
    assert (
        await backend.validate_path(Path("/mem/x/../a.md"), must_exist=True)
        == ROOT / "a.md"
    )
    with pytest.raises(PathAccessError, match="outside"):
        await backend.validate_path(Path("/mem/../etc/passwd"), must_exist=False)
    with pytest.raises(PathAccessError, match="does not exist"):
        await backend.validate_path(ROOT / "missing.md", must_exist=True)


async def test_overwrite_false_refuses_existing_file() -> None:
    backend = _backend(**{"a.md": "1"})

    with pytest.raises(FileExistsError):
        await backend.write_bytes(ROOT / "a.md", b"2", mode=0o644, overwrite=False)


async def test_expected_mtime_guards_writes() -> None:
    backend = _backend(**{"a.md": "v1"})
    _, mtime = await backend.read_text(ROOT / "a.md")

    await backend.write_bytes(ROOT / "a.md", b"v2", mode=0o644, expected_mtime=mtime)
    with pytest.raises(StaleFileError):
        await backend.write_bytes(
            ROOT / "a.md", b"v3", mode=0o644, expected_mtime=mtime
        )
    with pytest.raises(StaleFileError):
        await backend.write_bytes(
            ROOT / "new.md", b"x", mode=0o644, expected_mtime=mtime
        )

    assert (await backend.read_text(ROOT / "a.md"))[0] == "v2"
    assert not await backend.exists(ROOT / "new.md")


async def test_expected_mtime_guards_deletes() -> None:
    backend = _backend(**{"a.md": "v1"})
    _, mtime = await backend.read_text(ROOT / "a.md")
    await backend.write_bytes(ROOT / "a.md", b"v2", mode=0o644)

    with pytest.raises(StaleFileError):
        await backend.delete(ROOT / "a.md", expected_mtime=mtime)
    assert await backend.exists(ROOT / "a.md")

    _, current = await backend.read_text(ROOT / "a.md")
    await backend.delete(ROOT / "a.md", expected_mtime=current)
    assert not await backend.exists(ROOT / "a.md")


async def test_a_conditional_delete_of_a_vanished_file_is_stale() -> None:
    backend = _backend(**{"a.md": "v1"})
    _, mtime = await backend.read_text(ROOT / "a.md")
    await backend.delete(ROOT / "a.md")

    with pytest.raises(StaleFileError):
        await backend.delete(ROOT / "a.md", expected_mtime=mtime)
    with pytest.raises(FileNotFoundError):
        await backend.delete(ROOT / "a.md")


@pytest.mark.parametrize(
    ("files", "message"),
    [
        ({"notes.md": "x"}, "absolute"),
        ({"/elsewhere/x.md": "x"}, "outside"),
        ({"/mem/../elsewhere/x.md": "x"}, "outside"),
        ({"/mem": "x"}, "root"),
        ({"/mem/a": "x", "/mem/a/b.md": "y"}, "inside the file"),
    ],
)
async def test_seed_paths_are_validated(files: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        InMemoryFileBackend([ROOT], files=files)


async def test_seed_paths_are_normalized() -> None:
    backend = InMemoryFileBackend([ROOT], files={"/mem/x/../a.md": "x"})

    assert await backend.exists(ROOT / "a.md")
    assert not await backend.exists(ROOT / "x")


async def test_nothing_is_created_inside_a_file() -> None:
    backend = _backend(a="x")

    with pytest.raises(NotADirectoryError):
        await backend.write_bytes(ROOT / "a/b.md", b"y", mode=0o644)
    with pytest.raises(NotADirectoryError):
        await backend.append_bytes(ROOT / "a/b.md", b"y", mode=0o644)
    with pytest.raises(NotADirectoryError):
        await backend.mkdir(ROOT / "a/sub")
    assert [str(e.path) for e in await backend.list_dir(ROOT, recursive=True)] == [
        "/mem/a"
    ]


async def test_size_limits_refuse_oversized_writes() -> None:
    backend = InMemoryFileBackend([ROOT], max_file_bytes=10, max_total_bytes=25)
    await backend.write_bytes(ROOT / "a", b"x" * 10, mode=0o644)
    await backend.write_bytes(ROOT / "b", b"x" * 10, mode=0o644)

    with pytest.raises(OSError, match="per file") as too_big:
        await backend.write_bytes(ROOT / "c", b"x" * 11, mode=0o644)
    with pytest.raises(OSError, match="per file"):
        await backend.append_bytes(ROOT / "a", b"x", mode=0o644)
    with pytest.raises(OSError, match="in total") as full:
        await backend.write_bytes(ROOT / "c", b"x" * 6, mode=0o644)

    assert too_big.value.errno == errno.EFBIG
    assert full.value.errno == errno.ENOSPC
    assert not await backend.exists(ROOT / "c")
    await backend.write_bytes(ROOT / "a", b"x" * 4, mode=0o644)
    await backend.write_bytes(ROOT / "c", b"x" * 10, mode=0o644)
    await backend.delete(ROOT / "b")
    await backend.append_bytes(ROOT / "a", b"x" * 6, mode=0o644)


async def test_seeds_must_fit_the_size_limits() -> None:
    with pytest.raises(ValueError, match="limit"):
        InMemoryFileBackend([ROOT], files={"/mem/a": "x" * 11}, max_file_bytes=10)
    with pytest.raises(ValueError, match="limit"):
        InMemoryFileBackend(
            [ROOT], files={"/mem/a": "x" * 6, "/mem/b": "x" * 6}, max_total_bytes=10
        )


async def test_directories_cannot_be_read_or_deleted() -> None:
    backend = _backend(**{"notes/a.md": "1"})

    with pytest.raises(IsADirectoryError):
        await backend.read_text(ROOT / "notes")
    with pytest.raises(IsADirectoryError):
        await backend.delete(ROOT / "notes")
    with pytest.raises(FileNotFoundError):
        await backend.read_text(ROOT / "missing.md")


async def test_append_bytes_extends_the_file() -> None:
    backend = _backend()
    await backend.append_bytes(ROOT / "log.txt", b"a", mode=0o644)
    await backend.append_bytes(ROOT / "log.txt", b"b", mode=0o644)

    assert (await backend.read_text(ROOT / "log.txt"))[0] == "ab"


async def test_find_files_globs_newest_first() -> None:
    backend = _backend()
    await backend.write_bytes(ROOT / "old.md", b"1", mode=0o644)
    await backend.write_bytes(ROOT / "notes/new.md", b"2", mode=0o644)
    await backend.write_bytes(ROOT / "skip.txt", b"3", mode=0o644)

    matched, truncated = await backend.find_files(ROOT, "**/*.md")

    assert [str(e.path) for e in matched] == ["/mem/notes/new.md", "/mem/old.md"]
    assert not truncated


async def test_grep_uses_the_shared_text_grep() -> None:
    backend = _backend(**{"a.md": "one\nTODO two\n", "b.md": "none\n"})

    result = await backend.grep(ROOT, "TODO", output_mode="content")

    assert result.lines == ["/mem/a.md:2:TODO two"]
    assert result.num_files_matched == 1


async def test_a_catastrophic_pattern_does_not_stall_the_event_loop() -> None:
    backend = _backend(**{"a.md": "a" * 24 + "!\n"})

    result, gap = await max_loop_gap(backend.grep(ROOT, "(a+)+$"))

    assert result.files == []
    assert gap < 0.5


async def test_grep_stops_at_the_backends_time_budget() -> None:
    backend = InMemoryFileBackend(
        [ROOT], files={"/mem/a.md": "x" * 5_000 + "\n"}, grep_time_budget=0.3
    )

    started = time.monotonic()
    result = await backend.grep(ROOT, "(x+x+)+y")

    assert time.monotonic() - started < 2
    assert result.files == []
    assert "stopped after 0.3s" in result.notes[0]


async def test_file_tools_work_end_to_end() -> None:
    backend = _backend(**{"memory/MEMORY.md": "# Memory\n- [a](a.md)\n"})
    ctx: SessionContext[Any] = SessionContext(file_backend=backend, session_key="s")
    tk = FileToolkit(redactor=NullRedactor())
    agent_ctx = _make_agent_ctx(file_edit_state=FileEditSessionState())

    read = await tk.read.run(
        ReadInput(path="/mem/memory/MEMORY.md"), ctx=ctx, agent_ctx=agent_ctx
    )
    edit = await tk.edit.run(
        EditInput(
            path="/mem/memory/MEMORY.md",
            old_string="- [a](a.md)",
            new_string="- [a](a.md)\n- [b](b.md)",
        ),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    write = await tk.write.run(
        WriteInput(path="/mem/memory/b.md", content="body"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    glob = await tk.glob.run(
        GlobInput(pattern="**/*.md", path="/mem"), ctx=ctx, agent_ctx=agent_ctx
    )
    grep = await tk.grep.run(
        GrepInput(pattern="b\\.md", path="/mem"), ctx=ctx, agent_ctx=agent_ctx
    )

    assert isinstance(read, ReadResult)
    assert isinstance(edit, EditResult)
    assert isinstance(write, WriteResult)
    assert write.created
    assert isinstance(glob, GlobResult)
    assert set(glob.files) == {"/mem/memory/MEMORY.md", "/mem/memory/b.md"}
    assert isinstance(grep, GrepResult)
    assert grep.output == "/mem/memory/MEMORY.md"
