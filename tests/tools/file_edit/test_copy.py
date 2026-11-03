"""Tests for :class:`CopyTool` (``Copy``)."""

from __future__ import annotations

import contextlib
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from grasp_agents.file_backend import (
    CompositeFileBackend,
    FileStat,
    InMemoryFileBackend,
    LocalFileBackend,
    Mount,
)
from grasp_agents.file_backend.paths import AccessMode
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

# cspell:ignore mkfifo WRONLY

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
    await FileToolkit(redactor=NullRedactor()).read.run(
        ReadInput(path=str(WORK / "a.md")), ctx=ctx, agent_ctx=agent_ctx
    )

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


async def test_overwrite_needs_a_read_of_each_existing_file() -> None:
    _, ctx, agent_ctx, children = _setup(**{"a.md": "new", "b.md": "new"})
    await children["work"].write_bytes(WORK / "out/a.md", b"old", mode=0o644)
    await children["work"].write_bytes(WORK / "out/b.md", b"old", mode=0o644)
    await FileToolkit(redactor=NullRedactor()).read.run(
        ReadInput(path=str(WORK / "out/a.md")), ctx=ctx, agent_ctx=agent_ctx
    )

    result = await _copy(
        ctx, agent_ctx, source=str(NOTES), destination=str(WORK / "out"), overwrite=True
    )

    assert "Read" in _error(result)
    assert "/work/out/b.md" in _error(result)
    assert "/work/out/a.md" not in _error(result)
    assert (await children["work"].read_text(WORK / "out/a.md"))[0] == "old"


async def test_a_file_where_a_directory_is_needed_refuses_before_writing() -> None:
    _, ctx, agent_ctx, children = _setup(**{"0.md": "zero", "a/b.md": "bee"})
    await children["work"].write_bytes(WORK / "dst/a", b"a file", mode=0o644)

    result = await _copy(
        ctx, agent_ctx, source=str(NOTES), destination=str(WORK / "dst")
    )

    assert "/work/dst/a is a file" in _error(result)
    assert not await children["work"].exists(WORK / "dst/0.md")


class _Counting(InMemoryFileBackend):
    def __init__(self, files: dict[str, str]) -> None:
        super().__init__(["/w"], files=files)
        self.calls: Counter[str] = Counter()

    async def stat(self, path: Path) -> FileStat:
        self.calls["stat"] += 1
        return await super().stat(path)

    async def validate_path(
        self,
        path: Path,
        *,
        must_exist: bool,
        access: AccessMode = "read",
        dotfile_overrides: set[Path] | None = None,
    ) -> Path:
        self.calls["validate_path"] += 1
        return await super().validate_path(
            path,
            must_exist=must_exist,
            access=access,
            dotfile_overrides=dotfile_overrides,
        )


async def test_the_file_cap_refuses_before_checking_each_file() -> None:
    backend = _Counting({f"/w/src/f{i}.txt": "x" for i in range(50)})
    ctx: SessionContext[Any] = SessionContext(file_backend=backend, session_key="s")

    result = await CopyTool(max_files=10).run(
        CopyInput(source="/w/src", destination="/w/dst"),
        ctx=ctx,
        agent_ctx=_make_agent_ctx(file_edit_state=FileEditSessionState()),
    )

    assert "50 files" in _error(result)
    assert backend.calls["validate_path"] == 2
    assert backend.calls["stat"] == 1


class _NoSizes(InMemoryFileBackend):
    """Reports size 0 for every file, as some MCP servers do."""

    async def stat(self, path: Path) -> FileStat:
        st = await super().stat(path)
        return FileStat(mtime=st.mtime, mode=st.mode, size=0)


async def test_the_byte_cap_counts_the_bytes_actually_read() -> None:
    backend = _NoSizes(["/w"], files={"/w/src/a": "x" * 6, "/w/src/b": "y" * 6})
    ctx: SessionContext[Any] = SessionContext(file_backend=backend, session_key="s")

    result = await CopyTool(max_bytes=10).run(
        CopyInput(source="/w/src", destination="/w/dst"),
        ctx=ctx,
        agent_ctx=_make_agent_ctx(file_edit_state=FileEditSessionState()),
    )

    assert "bytes" in _error(result)
    assert not await backend.exists(Path("/w/dst"))


def _local_setup(
    tmp_path: Path, *, deny_read: list[Path] | None = None
) -> tuple[Path, InMemoryFileBackend, SessionContext[Any], Any]:
    work = tmp_path.resolve() / "work"
    (work / "src").mkdir(parents=True)
    scratch = InMemoryFileBackend(["/scratch"])
    backend = CompositeFileBackend(
        [
            Mount(work, LocalFileBackend(allowed_roots=[work], deny_read=deny_read)),
            Mount(Path("/scratch"), scratch),
        ],
        scratch=Path("/scratch"),
    )
    ctx: SessionContext[Any] = SessionContext(file_backend=backend, session_key="s")
    return work, scratch, ctx, _make_agent_ctx(file_edit_state=FileEditSessionState())


async def test_a_directory_copy_skips_what_it_may_not_read(tmp_path: Path) -> None:
    work, scratch, ctx, agent_ctx = _local_setup(
        tmp_path, deny_read=[tmp_path.resolve() / "work/src/secrets"]
    )
    src = work / "src"
    outside = tmp_path.resolve() / "outside"
    outside.mkdir()
    (outside / "id_rsa").write_text("outside the roots")
    (src / "secrets").mkdir()
    (src / "secrets/key.txt").write_text("denied")
    (src / "notes.md").write_text("ok")
    (src / ".env").write_text("TOKEN=1")
    (src / "outside_link.txt").symlink_to(outside / "id_rsa")
    (src / "inside_link.md").symlink_to(src / "notes.md")

    result = await _copy(ctx, agent_ctx, source=str(src), destination="/scratch/x")

    assert isinstance(result, CopyResult), result
    assert result.files_copied == 1
    copied = await scratch.list_dir(Path("/scratch/x"), recursive=True)
    assert [str(e.path) for e in copied] == ["/scratch/x/notes.md"]
    assert result.files_skipped == 4
    assert sorted(Path(line.split(": ")[0]).name for line in result.skipped) == [
        ".env",
        "inside_link.md",
        "key.txt",
        "outside_link.txt",
    ]


async def test_a_directory_copy_skips_special_files(tmp_path: Path) -> None:
    work, scratch, ctx, agent_ctx = _local_setup(tmp_path)
    (work / "src/a.txt").write_text("a")
    fifo = work / "src/pipe"
    os.mkfifo(fifo)

    try:
        result = await CopyTool(timeout=5).run(
            CopyInput(source=str(work / "src"), destination="/scratch/x"),
            ctx=ctx,
            agent_ctx=agent_ctx,
        )
    finally:
        # Unblock a reader stuck opening the FIFO, so the run can end.
        with contextlib.suppress(OSError):
            os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))

    assert isinstance(result, CopyResult), result
    assert result.files_copied == 1
    assert result.skipped == [f"{fifo}: not a regular file"]
    assert await scratch.exists(Path("/scratch/x/a.txt"))


async def test_skipped_entries_are_listed_up_to_a_limit(tmp_path: Path) -> None:
    work, _, ctx, agent_ctx = _local_setup(tmp_path)
    (work / "src/real.txt").write_text("x")
    for i in range(25):
        (work / f"src/link{i:02}.txt").symlink_to(work / "src/real.txt")

    result = await _copy(
        ctx, agent_ctx, source=str(work / "src"), destination="/scratch/x"
    )

    assert isinstance(result, CopyResult), result
    assert result.files_copied == 1
    assert result.files_skipped == 25
    assert len(result.skipped) == 20
