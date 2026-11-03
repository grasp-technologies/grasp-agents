"""Tests for :class:`CompositeFileBackend` and :class:`Mount`."""

from __future__ import annotations

import stat as stat_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from grasp_agents.agent.task_progress import spill_if_large
from grasp_agents.file_backend import (
    CompositeFileBackend,
    GrepRawResult,
    InMemoryFileBackend,
    LocalFileBackend,
    Mount,
    PathAccessError,
    StaleFileError,
)
from grasp_agents.memory import MemoryProvider
from grasp_agents.sandbox import local_environment
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
)
from grasp_agents.tools.file_search import GlobInput, GlobResult, GrepInput, GrepResult
from grasp_agents.types.events import ToolErrorInfo
from tests._helpers import _make_agent_ctx

if TYPE_CHECKING:
    from grasp_agents.file_backend.base import GrepOutputMode

pytestmark = pytest.mark.asyncio

MEMORY = Path("/user/memory")
NOTES = Path("/user/notes")
SCRATCH = Path("/scratch")


def _mem(root: Path, **files: str) -> InMemoryFileBackend:
    return InMemoryFileBackend(
        [root], files={str(root / k): v for k, v in files.items()}
    )


def _composite(*, notes_read_only: bool = False) -> CompositeFileBackend:
    return CompositeFileBackend(
        [
            Mount(
                MEMORY,
                _mem(MEMORY, **{"MEMORY.md": "# Memory\n"}),
                description="memdir",
            ),
            Mount(
                NOTES,
                _mem(NOTES, **{"inbox/a.md": "TODO read the paper\n"}),
                read_only=notes_read_only,
                description="user notes",
            ),
            Mount(SCRATCH, _mem(SCRATCH)),
        ],
        scratch=SCRATCH,
    )


# ---- construction -----------------------------------------------------------


@pytest.mark.parametrize(
    ("mounts", "message"),
    [
        ([Mount(Path("rel"), _mem(Path("/rel")))], "absolute"),
        ([Mount(Path("/a/../b"), _mem(Path("/b")))], "normalized"),
        (
            [Mount(Path("/a"), _mem(Path("/a"))), Mount(Path("/a"), _mem(Path("/a")))],
            "duplicate",
        ),
        (
            [
                Mount(Path("/a"), _mem(Path("/a"))),
                Mount(Path("/a/b"), _mem(Path("/a/b"))),
            ],
            "nested",
        ),
        ([Mount(Path("/y"), _mem(Path("/x")))], "does not serve"),
    ],
)
async def test_invalid_mounts_are_rejected(mounts: list[Mount], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CompositeFileBackend(mounts)


async def test_scratch_must_be_a_writable_mount() -> None:
    with pytest.raises(ValueError, match="scratch"):
        CompositeFileBackend([Mount(MEMORY, _mem(MEMORY))], scratch=SCRATCH)
    with pytest.raises(ValueError, match="scratch"):
        CompositeFileBackend(
            [Mount(SCRATCH, _mem(SCRATCH), read_only=True)], scratch=SCRATCH
        )


async def test_scratch_defaults_to_the_first_writable_mount() -> None:
    docs = Path("/docs")
    backend = CompositeFileBackend(
        [Mount(docs, _mem(docs), read_only=True), Mount(SCRATCH, _mem(SCRATCH))]
    )

    assert backend.scratch.path == SCRATCH
    assert backend.allowed_roots == [SCRATCH, docs]
    with pytest.raises(ValueError, match="writable"):
        CompositeFileBackend([Mount(docs, _mem(docs), read_only=True)])


async def test_scratch_comes_first_in_allowed_roots() -> None:
    backend = _composite()

    assert backend.allowed_roots == [SCRATCH, MEMORY, NOTES]
    assert [m.path for m in backend.mounts] == [MEMORY, NOTES, SCRATCH]


# ---- routing and validation -------------------------------------------------


async def test_validate_path_routes_and_normalizes() -> None:
    backend = _composite()

    assert backend.mount_for(MEMORY / "a.md") is backend.mounts[0]
    assert backend.mount_for(Path("/user")) is None
    assert (
        await backend.validate_path(
            Path("/user/memory/../notes/x.md"), must_exist=False
        )
        == NOTES / "x.md"
    )
    assert (
        await backend.validate_path(Path("tmp.txt"), must_exist=False)
        == SCRATCH / "tmp.txt"
    )


async def test_paths_outside_mounts_are_refused() -> None:
    backend = _composite()

    with pytest.raises(PathAccessError, match="/user/memory"):
        await backend.validate_path(Path("/etc/passwd"), must_exist=False)
    with pytest.raises(PathAccessError):
        await backend.validate_path(Path("/user/../etc/passwd"), must_exist=False)


async def test_read_only_mounts_refuse_writes() -> None:
    backend = _composite(notes_read_only=True)

    assert await backend.validate_path(NOTES / "inbox/a.md", must_exist=True)
    with pytest.raises(PathAccessError, match="read-only"):
        await backend.validate_path(
            NOTES / "inbox/a.md", must_exist=True, access="write"
        )


async def test_virtual_dirs_are_readable_not_writable() -> None:
    backend = _composite()

    assert await backend.validate_path(Path("/user"), must_exist=True) == Path("/user")
    assert await backend.validate_path(Path("/"), must_exist=True) == Path("/")
    with pytest.raises(PathAccessError, match="not inside a mount"):
        await backend.validate_path(Path("/user"), must_exist=True, access="write")
    with pytest.raises(PathAccessError, match="outside every mount"):
        await backend.validate_path(
            Path("/user/x.md"), must_exist=False, access="write"
        )


async def test_file_operations_route_to_the_owning_mount() -> None:
    memory = _mem(MEMORY)
    backend = CompositeFileBackend(
        [Mount(MEMORY, memory), Mount(SCRATCH, _mem(SCRATCH))]
    )

    mtime = await backend.write_bytes(MEMORY / "a.md", b"hello", mode=0o644)

    assert (await memory.read_text(MEMORY / "a.md"))[0] == "hello"
    assert (await backend.read_text(MEMORY / "a.md")) == ("hello", mtime)
    assert (await backend.stat(MEMORY / "a.md")).size == 5
    assert await backend.exists(MEMORY / "a.md")
    assert await backend.parent_exists(MEMORY / "b.md")
    await backend.append_bytes(MEMORY / "a.md", b"!", mode=0o644)
    assert (await backend.read_bytes(MEMORY / "a.md"))[0] == b"hello!"
    await backend.mkdir(MEMORY / "topics")
    assert await backend.exists(MEMORY / "topics")
    await backend.delete(MEMORY / "a.md")
    assert not await backend.exists(MEMORY / "a.md")


async def test_expected_mtime_reaches_the_child() -> None:
    backend = _composite()
    _, mtime = await backend.read_text(MEMORY / "MEMORY.md")
    await backend.write_bytes(MEMORY / "MEMORY.md", b"changed", mode=0o644)

    with pytest.raises(StaleFileError):
        await backend.write_bytes(
            MEMORY / "MEMORY.md", b"x", mode=0o644, expected_mtime=mtime
        )
    with pytest.raises(StaleFileError):
        await backend.delete(MEMORY / "MEMORY.md", expected_mtime=mtime)


async def test_mkdir_of_a_mount_root_reaches_its_backend(tmp_path: Path) -> None:
    root = tmp_path.resolve() / "new"
    backend = CompositeFileBackend(
        [Mount(root, LocalFileBackend(allowed_roots=[root]))]
    )

    await backend.mkdir(root)
    await backend.mkdir(root.parent)

    assert root.is_dir()


async def test_virtual_dirs_exist_and_list_their_mounts() -> None:
    backend = _composite()

    assert await backend.exists(Path("/user"))
    assert stat_module.S_ISDIR((await backend.stat(Path("/user"))).mode)
    assert await backend.parent_exists(MEMORY)
    with pytest.raises(IsADirectoryError):
        await backend.read_text(Path("/user"))

    root_names = sorted((e.name, e.is_dir) for e in await backend.list_dir(Path("/")))
    user_names = sorted(e.name for e in await backend.list_dir(Path("/user")))
    recursive = {
        str(e.path) for e in await backend.list_dir(Path("/user"), recursive=True)
    }

    assert root_names == [("scratch", True), ("user", True)]
    assert user_names == ["memory", "notes"]
    assert {
        "/user/memory",
        "/user/memory/MEMORY.md",
        "/user/notes",
        "/user/notes/inbox",
        "/user/notes/inbox/a.md",
    } <= recursive


async def test_find_files_spans_mounts_newest_first() -> None:
    backend = _composite()
    await backend.write_bytes(MEMORY / "late.md", b"x", mode=0o644)

    matched, truncated = await backend.find_files(Path("/user"), "**/*.md")

    assert str(matched[0].path) == "/user/memory/late.md"
    assert {str(e.path) for e in matched} == {
        "/user/memory/MEMORY.md",
        "/user/memory/late.md",
        "/user/notes/inbox/a.md",
    }
    assert not truncated


async def test_find_files_at_a_virtual_dir_keeps_the_newest() -> None:
    old, new = InMemoryFileBackend(["/a"]), InMemoryFileBackend(["/b"])
    for i in range(10):
        await old.write_bytes(Path(f"/a/old{i}.md"), b"x", mode=0o644)
    for i in range(3):
        await new.write_bytes(Path(f"/b/new{i}.md"), b"x", mode=0o644)
    backend = CompositeFileBackend([Mount(Path("/a"), old), Mount(Path("/b"), new)])

    matched, truncated = await backend.find_files(Path("/"), "**/*.md", head_limit=5)

    assert [e.name for e in matched] == [
        "new2.md",
        "new1.md",
        "new0.md",
        "old9.md",
        "old8.md",
    ]
    assert truncated


async def test_find_files_at_a_virtual_dir_skips_build_dirs() -> None:
    app = Path("/w/app")
    backend = CompositeFileBackend(
        [Mount(app, _mem(app, **{"a.js": "x", "node_modules/p/i.js": "x"}))]
    )

    matched, _ = await backend.find_files(Path("/w"), "**/*.js")

    assert [str(e.path) for e in matched] == ["/w/app/a.js"]


async def test_find_files_inside_one_mount_delegates() -> None:
    backend = _composite()

    matched, _ = await backend.find_files(NOTES, "inbox/*.md")

    assert [str(e.path) for e in matched] == ["/user/notes/inbox/a.md"]


class _NoGrep(InMemoryFileBackend):
    async def grep(self, root: Path, pattern: str, **kwargs: Any) -> GrepRawResult:
        raise NotImplementedError("this backend cannot grep")


class _FailingGrep(InMemoryFileBackend):
    def __init__(self, root: Path, error: Exception) -> None:
        super().__init__([root], files={str(root / "a.md"): "TODO\n"})
        self.error = error

    async def grep(self, root: Path, pattern: str, **kwargs: Any) -> GrepRawResult:
        raise self.error


GREP_FAILURES = [
    ValueError("unrecognized file type: cs"),
    RuntimeError("index unavailable"),
    OSError("rg not found"),
]


async def test_grep_spans_mounts() -> None:
    backend = _composite()
    await backend.write_bytes(MEMORY / "todo.md", b"TODO memory\n", mode=0o644)

    result = await backend.grep(Path("/user"), "TODO", output_mode="content")

    assert result.lines == [
        "/user/memory/todo.md:1:TODO memory",
        "/user/notes/inbox/a.md:1:TODO read the paper",
    ]
    assert result.num_matches == 2
    assert result.num_files_matched == 2
    assert result.notes == []


@pytest.mark.parametrize("mode", ["files_with_matches", "count", "content"])
async def test_grep_reports_mounts_it_could_not_search(mode: GrepOutputMode) -> None:
    backend = CompositeFileBackend(
        [
            Mount(MEMORY, _NoGrep([MEMORY], files={str(MEMORY / "a.md"): "TODO\n"})),
            Mount(NOTES, _mem(NOTES, **{"b.md": "TODO\n"})),
        ]
    )

    result = await backend.grep(Path("/user"), "TODO", output_mode=mode)

    assert result.num_files_matched == 1
    assert len(result.notes) == 1
    assert "/user/memory" in result.notes[0]


@pytest.mark.parametrize("error", GREP_FAILURES, ids=lambda e: type(e).__name__)
async def test_grep_turns_a_failing_mount_into_a_note(error: Exception) -> None:
    backend = CompositeFileBackend(
        [
            Mount(MEMORY, _FailingGrep(MEMORY, error)),
            Mount(NOTES, _mem(NOTES, **{"b.md": "TODO\n"})),
        ]
    )

    result = await backend.grep(Path("/user"), "TODO")

    assert result.files == [NOTES / "b.md"]
    assert result.notes == [f"/user/memory was not searched: {error}"]


@pytest.mark.parametrize("error", GREP_FAILURES, ids=lambda e: type(e).__name__)
async def test_grep_raises_when_every_mount_failed(error: Exception) -> None:
    backend = CompositeFileBackend(
        [
            Mount(MEMORY, _FailingGrep(MEMORY, error)),
            Mount(NOTES, _FailingGrep(NOTES, RuntimeError("second"))),
        ]
    )

    with pytest.raises(type(error), match=str(error)):
        await backend.grep(Path("/user"), "TODO")


async def test_grep_raises_when_no_mount_could_search() -> None:
    backend = CompositeFileBackend(
        [Mount(MEMORY, _NoGrep([MEMORY], files={str(MEMORY / "a.md"): "TODO\n"}))]
    )

    with pytest.raises(NotImplementedError):
        await backend.grep(Path("/user"), "TODO")


async def test_add_allowed_root_only_accepts_mounted_paths() -> None:
    backend = _composite()

    backend.add_allowed_root(MEMORY / "topics")
    with pytest.raises(PathAccessError, match="no mount"):
        backend.add_allowed_root(Path("/elsewhere"))


async def test_includes_mounted_backends() -> None:
    memory = _mem(MEMORY)
    backend = CompositeFileBackend([Mount(MEMORY, memory)])

    assert backend.includes(memory)
    assert backend.includes(backend)
    assert not backend.includes(_mem(MEMORY))


async def test_spills_land_in_the_scratch_mount() -> None:
    backend = _composite()

    _, spill_path = await spill_if_large(backend, name="call_1", text="x" * 500, cap=10)

    assert spill_path is not None
    assert spill_path.startswith("/scratch/")


async def test_local_child_shares_its_real_paths(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    (root / "a.txt").write_text("disk")
    backend = CompositeFileBackend(
        [
            Mount(root, LocalFileBackend(allowed_roots=[root])),
            Mount(SCRATCH, _mem(SCRATCH)),
        ],
        scratch=SCRATCH,
    )

    assert (await backend.read_text(root / "a.txt"))[0] == "disk"
    with pytest.raises(PathAccessError):
        await backend.validate_path(root / ".." / "outside.txt", must_exist=False)


# ---- SessionContext and tools ----------------------------------------------


async def test_session_context_accepts_a_composite_over_the_environment(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    env = local_environment(allowed_roots=[root])
    composite = CompositeFileBackend(
        [Mount(root, env.file_backend), Mount(SCRATCH, _mem(SCRATCH))], scratch=SCRATCH
    )

    ctx: SessionContext[Any] = SessionContext(environment=env, file_backend=composite)

    assert ctx.file_backend is composite
    with pytest.raises(ValueError, match="environment"):
        SessionContext(environment=env, file_backend=_composite())


async def test_memory_provider_works_through_a_composite() -> None:
    backend = _composite()
    ctx: SessionContext[Any] = SessionContext(
        file_backend=backend, memory=MemoryProvider(root=MEMORY)
    )

    assert ctx.memory is not None
    snapshot = await ctx.memory.load()

    assert snapshot.index is not None
    assert snapshot.index.startswith("# Memory")


async def test_memory_bootstraps_a_new_memdir_through_a_composite(
    tmp_path: Path,
) -> None:
    memdir = tmp_path.resolve() / "memory"
    backend = CompositeFileBackend(
        [
            Mount(SCRATCH, _mem(SCRATCH)),
            Mount(memdir, LocalFileBackend(allowed_roots=[memdir])),
        ]
    )
    ctx: SessionContext[Any] = SessionContext(
        file_backend=backend, memory=MemoryProvider(root=memdir)
    )

    assert ctx.memory is not None
    snapshot = await ctx.memory.load()

    assert snapshot.index is not None
    assert snapshot.index.startswith("# Memory")
    assert (memdir / "MEMORY.md").read_text().startswith("# Memory")


async def test_file_tools_work_across_mounts() -> None:
    backend = _composite(notes_read_only=True)
    ctx: SessionContext[Any] = SessionContext(file_backend=backend, session_key="s")
    tk = FileToolkit(redactor=NullRedactor())
    agent_ctx = _make_agent_ctx(file_edit_state=FileEditSessionState())

    read = await tk.read.run(
        ReadInput(path=str(MEMORY / "MEMORY.md")), ctx=ctx, agent_ctx=agent_ctx
    )
    edit = await tk.edit.run(
        EditInput(
            path=str(MEMORY / "MEMORY.md"),
            old_string="# Memory",
            new_string="# Memory\n- [a](a.md)",
        ),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    refused = await tk.write.run(
        WriteInput(path=str(NOTES / "new.md"), content="x"),
        ctx=ctx,
        agent_ctx=agent_ctx,
    )
    glob = await tk.glob.run(
        GlobInput(pattern="**/*.md", path="/user"), ctx=ctx, agent_ctx=agent_ctx
    )
    grep = await tk.grep.run(
        GrepInput(pattern="TODO", path="/"), ctx=ctx, agent_ctx=agent_ctx
    )

    assert isinstance(read, ReadResult)
    assert isinstance(edit, EditResult)
    assert isinstance(refused, ToolErrorInfo)
    assert "read-only" in refused.error
    assert isinstance(glob, GlobResult)
    assert set(glob.files) == {"/user/memory/MEMORY.md", "/user/notes/inbox/a.md"}
    assert isinstance(grep, GrepResult)
    assert grep.output == "/user/notes/inbox/a.md"


async def test_grep_tool_renders_notes() -> None:
    backend = CompositeFileBackend(
        [
            Mount(MEMORY, _NoGrep([MEMORY], files={str(MEMORY / "a.md"): "TODO\n"})),
            Mount(NOTES, _mem(NOTES, **{"b.md": "TODO\n"})),
        ]
    )
    ctx: SessionContext[Any] = SessionContext(file_backend=backend, session_key="s")

    result = await FileToolkit().grep.run(
        GrepInput(pattern="TODO", path="/user"), ctx=ctx
    )

    assert isinstance(result, GrepResult)
    assert result.output.splitlines()[0] == "/user/notes/b.md"
    assert "Note:" in result.output
    assert "/user/memory" in result.output
