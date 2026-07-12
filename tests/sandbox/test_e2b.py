"""
Tests for the E2B hosted backend.

The ``e2b`` SDK is an optional extra and a real sandbox needs an API key +
network, so — as with Seatbelt/srt — the deterministic adapter surface (path
translation, mtime / exit-code mapping, command construction, lifecycle wiring)
is unit-tested in-session against a fake sandbox, while the end-to-end round
trip is an ``integration``-marked test guarded by ``e2b`` being installed and
``E2B_API_KEY`` being set.

The fake sandbox is cast to ``AsyncSandbox`` and driven only through the public
backend API; the fake exception classes subclass the real e2b
``TimeoutException`` / ``CommandExitException`` so the adapter's isinstance
checks match. The ``e2b`` SDK is imported at module top (installed in the dev
environment), as in the adapter itself.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import datetime as dt
import importlib.util
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
from e2b import AsyncSandbox, CommandExitException, TimeoutException

from grasp_agents.file_backend.base import FileBackend
from grasp_agents.file_backend.paths import PathAccessError
from grasp_agents.sandbox import (
    E2BEnvironment,
    ExecBackend,
    ExecutionEnvironment,
    NetworkPolicy,
    SandboxPolicy,
    SnapshotCapable,
    e2b_environment,
)
from grasp_agents.sandbox.e2b import _handle as e2b_handle
from grasp_agents.sandbox.e2b import environment as e2b_env
from grasp_agents.sandbox.e2b import file_backend as e2b_file_backend
from grasp_agents.sandbox.exec_backend import ExecChunk, ExecResult, TerminationReason
from grasp_agents.session_context import SessionContext

from ._bg_harness import (
    background,
    drain_notes,
    flush,
    kill,
    make_stack,
    marker_size,
    poll_until_done,
)

pytestmark = pytest.mark.asyncio

_WS = "/home/user/workspace"


# --- fake exception classes -------------------------------------------------


class _FakeTimeout(TimeoutException):
    """Subclasses the real e2b type so the adapter's isinstance checks match."""


class _FakeCommandExit(CommandExitException):
    """Subclasses the real e2b type so the adapter's isinstance checks match."""

    def __init__(self, *, stdout: str = "", stderr: str = "", exit_code: int = 1):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


# --- fake sandbox (cast to AsyncSandbox; driven via the public backend API) -


@dataclass
class _FakeFileType:
    value: str


class _FakeEntry:
    def __init__(
        self,
        *,
        name: str,
        path: str,
        type_value: str = "file",
        size: int = 0,
        mode: int = 0o644,
        mtime_epoch: float = 1000.0,
    ) -> None:
        self.name = name
        self.path = path
        self.type = _FakeFileType(type_value)
        self.size = size
        self.mode = mode
        self.modified_time = dt.datetime.fromtimestamp(mtime_epoch, tz=dt.UTC)


@dataclass
class _FakeResult:
    stdout: str
    exit_code: int


@dataclass
class _FakeSnapshotInfo:
    snapshot_id: str


class _FakeFiles:
    def __init__(self) -> None:
        self.reads: dict[str, Any] = {}
        self.infos: dict[str, _FakeEntry] = {}
        self.listing: dict[str, list[_FakeEntry]] = {}
        self.written: dict[str, Any] = {}
        self.removed: list[str] = []
        self.made_dirs: list[str] = []

    async def read(self, path: str, fmt: str = "text") -> Any:
        data = self.reads[path]
        if fmt == "bytes":
            return bytearray(data if isinstance(data, bytes) else str(data).encode())
        return data

    async def write(self, path: str, data: Any) -> Any:
        self.written[path] = data
        self.infos.setdefault(
            path, _FakeEntry(name=Path(path).name, path=path, mtime_epoch=2000.0)
        )
        return None

    async def get_info(self, path: str) -> Any:
        return self.infos[path]

    async def exists(self, path: str) -> bool:
        return path in self.infos or path in self.reads or path in self.written

    async def remove(self, path: str) -> Any:
        self.removed.append(path)
        return None

    async def make_dir(self, path: str) -> Any:
        self.made_dirs.append(path)
        return True

    async def list(self, path: str, depth: int = 1) -> Any:
        del depth
        return self.listing.get(path, [])


class _FakeHandle:
    def __init__(
        self,
        *,
        exit_code: int = 0,
        chunks: tuple[tuple[str, str], ...] = (),
        raise_exc: BaseException | None = None,
        on_stdout: Any = None,
        on_stderr: Any = None,
    ) -> None:
        self.exit_code = exit_code
        self._chunks = chunks
        self._raise = raise_exc
        self._on_stdout = on_stdout
        self._on_stderr = on_stderr

    async def wait(self) -> Any:
        for stream, data in self._chunks:
            cb = self._on_stdout if stream == "stdout" else self._on_stderr
            if cb is not None:
                cb(data)
        if self._raise is not None:
            raise self._raise
        return None

    async def kill(self) -> Any:
        return True


class _FakeCommands:
    def __init__(
        self,
        *,
        exit_code: int = 0,
        chunks: tuple[tuple[str, str], ...] = (),
        raise_exc: BaseException | None = None,
        run_raises: BaseException | None = None,
        fg_stdout: str = "",
        fg_exit_code: int = 0,
        fg_raises: BaseException | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.stdin_writes: list[tuple[int, str]] = []
        self.killed: list[int] = []
        self._exit_code = exit_code
        self._chunks = chunks
        self._raise = raise_exc
        self._run_raises = run_raises
        self._fg_stdout = fg_stdout
        self._fg_exit_code = fg_exit_code
        self._fg_raises = fg_raises

    async def run(
        self,
        cmd: str,
        *,
        background: bool | None = None,
        cwd: str | None = None,
        envs: dict[str, str] | None = None,
        timeout: float | None = None,
        stdin: bool | None = None,
        on_stdout: Any = None,
        on_stderr: Any = None,
    ) -> Any:
        self.calls.append(
            {
                "cmd": cmd,
                "background": background,
                "cwd": cwd,
                "envs": envs,
                "timeout": timeout,
                "stdin": stdin,
            }
        )
        if background:
            if self._run_raises is not None:
                raise self._run_raises
            return _FakeHandle(
                exit_code=self._exit_code,
                chunks=self._chunks,
                raise_exc=self._raise,
                on_stdout=on_stdout,
                on_stderr=on_stderr,
            )
        if self._fg_raises is not None:
            raise self._fg_raises
        return _FakeResult(self._fg_stdout, self._fg_exit_code)

    async def send_stdin(self, pid: int, data: str) -> Any:
        self.stdin_writes.append((pid, data))

    async def kill(self, pid: int) -> Any:
        self.killed.append(pid)


class _FakeSandbox:
    def __init__(
        self,
        *,
        files: _FakeFiles | None = None,
        commands: _FakeCommands | None = None,
        sandbox_id: str = "sbx_fake",
    ) -> None:
        self.files = files or _FakeFiles()
        self.commands = commands or _FakeCommands()
        self.sandbox_id = sandbox_id
        self.paused = False
        self.killed = False
        self.snapshotted = False

    async def pause(self) -> Any:
        self.paused = True

    async def kill(self) -> Any:
        self.killed = True
        return True

    async def create_snapshot(self) -> Any:
        self.snapshotted = True
        return _FakeSnapshotInfo(snapshot_id=f"snap-{self.sandbox_id}")


def _env(
    fake: _FakeSandbox, *, owns: bool = False, pause_on_exit: bool = False
) -> E2BEnvironment:
    return E2BEnvironment.from_sandbox(
        cast("AsyncSandbox", fake),
        policy=SandboxPolicy(allowed_roots=(Path(_WS),)),
        owns_sandbox=owns,
        pause_on_exit=pause_on_exit,
    )


# --- protocol conformance ----------------------------------------------------


async def test_protocol_conformance() -> None:
    env = e2b_environment(allowed_roots=[_WS])
    assert isinstance(env, ExecutionEnvironment)
    assert isinstance(env, SnapshotCapable)
    assert isinstance(env.file_backend, FileBackend)
    assert isinstance(env.exec_backend, ExecBackend)
    assert env.file_backend.name == "e2b"
    assert env.exec_backend.name == "e2b"


# --- pure helpers (white-box; clearer tested directly) ----------------------


async def test_wrap_stdin() -> None:
    assert e2b_handle.wrap_stdin("echo hi", None) == "echo hi"
    assert e2b_handle.wrap_stdin("echo hi", b"") == "echo hi"
    wrapped = e2b_handle.wrap_stdin("cat", b"hello")
    assert "base64 -d | (cat)" in wrapped
    assert base64.b64encode(b"hello").decode() in wrapped


async def test_build_grep_cmd_content_mode() -> None:
    cmd = e2b_file_backend._build_grep_cmd(
        Path(_WS),
        "foo",
        glob=None,
        file_type="py",
        case_insensitive=True,
        output_mode="content",
        show_line_numbers=True,
        before_context=None,
        after_context=None,
        context=2,
    )
    assert cmd.startswith("grep -r -E -i -n -C2")
    assert "--include=*.py" in cmd
    assert "-e foo" in cmd
    assert cmd.endswith(_WS)


async def test_parse_grep_modes() -> None:
    files = e2b_file_backend._parse_grep("/w/a.py\n/w/b.py\n", "files_with_matches")
    assert files.files == [Path("/w/a.py"), Path("/w/b.py")]

    counts = e2b_file_backend._parse_grep("/w/a.py:3\n/w/b.py:1\n", "count")
    assert counts.counts == [(Path("/w/a.py"), 3), (Path("/w/b.py"), 1)]
    assert counts.num_matches == 4

    content = e2b_file_backend._parse_grep(
        "/w/a.py:1:foo\n--\n/w/a.py:9:foo\n", "content"
    )
    assert content.lines == ["/w/a.py:1:foo", "/w/a.py:9:foo"]  # `--` dropped


# --- file backend ------------------------------------------------------------


async def test_validate_path_containment() -> None:
    backend = _env(_FakeSandbox()).file_backend
    resolved = await backend.validate_path(Path(f"{_WS}/sub/x.txt"), must_exist=False)
    assert str(resolved) == f"{_WS}/sub/x.txt"
    with pytest.raises(PathAccessError, match="outside allowed roots"):
        await backend.validate_path(Path("/etc/passwd"), must_exist=False)


async def test_validate_path_must_exist() -> None:
    fake = _FakeSandbox()
    fake.files.infos[f"{_WS}/there.txt"] = _FakeEntry(
        name="there.txt", path=f"{_WS}/there.txt"
    )
    backend = _env(fake).file_backend
    await backend.validate_path(Path(f"{_WS}/there.txt"), must_exist=True)
    with pytest.raises(PathAccessError, match="does not exist"):
        await backend.validate_path(Path(f"{_WS}/missing.txt"), must_exist=True)


async def test_stat_maps_fields_and_mtime() -> None:
    fake = _FakeSandbox()
    fake.files.infos[f"{_WS}/f"] = _FakeEntry(
        name="f", path=f"{_WS}/f", size=42, mode=0o600, mtime_epoch=999.0
    )
    st = await _env(fake).file_backend.stat(Path(f"{_WS}/f"))
    assert st.size == 42
    assert st.mode == 0o600
    assert st.mtime == 999.0  # datetime -> epoch float


async def test_read_text_and_bytes() -> None:
    fake = _FakeSandbox()
    fake.files.reads[f"{_WS}/f"] = "body"
    fake.files.infos[f"{_WS}/f"] = _FakeEntry(
        name="f", path=f"{_WS}/f", mtime_epoch=777.0
    )
    backend = _env(fake).file_backend
    text, mtime = await backend.read_text(Path(f"{_WS}/f"))
    assert text == "body"
    assert mtime == 777.0
    data, _ = await backend.read_bytes(Path(f"{_WS}/f"))
    assert data == b"body"


async def test_write_bytes_returns_post_write_mtime() -> None:
    fake = _FakeSandbox()
    backend = _env(fake).file_backend
    mtime = await backend.write_bytes(Path(f"{_WS}/new.txt"), b"data", mode=0o644)
    assert fake.files.written[f"{_WS}/new.txt"] == b"data"
    assert mtime == 2000.0  # _FakeFiles.write stamps a fresh get_info mtime


async def test_delete_and_mkdir() -> None:
    fake = _FakeSandbox()
    backend = _env(fake).file_backend
    await backend.delete(Path(f"{_WS}/gone.txt"))
    await backend.mkdir(Path(f"{_WS}/d"))
    assert fake.files.removed == [f"{_WS}/gone.txt"]
    assert f"{_WS}/d" in fake.files.made_dirs


async def test_list_dir_maps_is_dir() -> None:
    fake = _FakeSandbox()
    fake.files.listing[_WS] = [
        _FakeEntry(name="a.py", path=f"{_WS}/a.py"),
        _FakeEntry(name="sub", path=f"{_WS}/sub", type_value="dir"),
    ]
    entries = await _env(fake).file_backend.list_dir(Path(_WS))
    by_name = {e.name: e for e in entries}
    assert by_name["a.py"].is_dir is False
    assert by_name["sub"].is_dir is True


async def test_deny_write_carveout_enforced_on_tool_plane() -> None:
    policy = SandboxPolicy(
        allowed_roots=(Path(_WS),),
        deny_write=(Path(f"{_WS}/protected"),),
    )
    backend = E2BEnvironment.from_sandbox(
        cast("AsyncSandbox", _FakeSandbox()), policy=policy
    ).file_backend
    protected = Path(f"{_WS}/protected/x")
    # reads are allowed; writes to the carved-out region are denied
    await backend.validate_path(protected, must_exist=False, access="read")
    with pytest.raises(PathAccessError):
        await backend.validate_path(protected, must_exist=False, access="write")


async def test_find_files_globs_over_listing() -> None:
    fake = _FakeSandbox()
    fake.files.listing[_WS] = [
        _FakeEntry(name="a.py", path=f"{_WS}/a.py", mtime_epoch=10.0),
        _FakeEntry(name="b.txt", path=f"{_WS}/b.txt", mtime_epoch=20.0),
    ]
    matched, truncated = await _env(fake).file_backend.find_files(Path(_WS), "*.py")
    assert [e.name for e in matched] == ["a.py"]
    assert truncated is False


# --- grep (shelled out through the co-located exec) -------------------------


async def test_grep_files_mode_builds_command_and_parses() -> None:
    cmds = _FakeCommands(fg_stdout=f"{_WS}/a.py\n{_WS}/b.py\n", fg_exit_code=0)
    backend = _env(_FakeSandbox(commands=cmds)).file_backend
    res = await backend.grep(Path(_WS), "needle", glob="*.py")
    assert res.files == [Path(f"{_WS}/a.py"), Path(f"{_WS}/b.py")]
    grep_cmd = cmds.calls[0]["cmd"]
    assert grep_cmd.startswith("grep -r -E -l")
    assert "--include=*.py" in grep_cmd
    assert "-e needle" in grep_cmd


async def test_grep_no_match_returns_empty() -> None:
    # grep exits 1 on no match — a CommandExitException the backend treats as empty.
    cmds = _FakeCommands(fg_raises=_FakeCommandExit(exit_code=1))
    backend = _env(_FakeSandbox(commands=cmds)).file_backend
    res = await backend.grep(Path(_WS), "needle")
    assert res.files == []


async def test_grep_multiline_unsupported() -> None:
    backend = _env(_FakeSandbox()).file_backend
    with pytest.raises(NotImplementedError, match="multiline"):
        await backend.grep(Path(_WS), "x", multiline=True)


# --- exec backend ------------------------------------------------------------


async def test_execute_joins_streamed_output() -> None:
    cmds = _FakeCommands(
        exit_code=0, chunks=(("stdout", "hello\n"), ("stderr", "warn"))
    )
    res = await _env(_FakeSandbox(commands=cmds)).exec_backend.execute("echo hi")
    assert isinstance(res, ExecResult)
    assert res.stdout == "hello\n"
    assert res.stderr == "warn"
    assert res.returncode == 0
    assert res.reason is TerminationReason.EXIT
    assert res.backend == "e2b"


async def test_nonzero_exit_is_data_not_raised() -> None:
    cmds = _FakeCommands(
        chunks=(("stdout", "partial"),),
        raise_exc=_FakeCommandExit(exit_code=7, stderr="boom"),
    )
    res = await _env(_FakeSandbox(commands=cmds)).exec_backend.execute("false")
    assert res.returncode == 7  # CommandExitException mapped to data, not raised
    assert res.stdout == "partial"
    assert res.reason is TerminationReason.EXIT


async def test_timeout_maps_to_overall_timeout() -> None:
    cmds = _FakeCommands(raise_exc=_FakeTimeout())
    res = await _env(_FakeSandbox(commands=cmds)).exec_backend.execute("sleep 999")
    assert res.reason is TerminationReason.OVERALL_TIMEOUT
    assert res.returncode == -1


async def test_spawn_failure_maps_to_spawn_error() -> None:
    cmds = _FakeCommands(run_raises=RuntimeError("sandbox not found"))
    res = await _env(_FakeSandbox(commands=cmds)).exec_backend.execute("echo hi")
    assert res.reason is TerminationReason.SPAWN_ERROR
    assert "sandbox not found" in res.stderr


async def test_stdin_is_base64_wrapped() -> None:
    cmds = _FakeCommands(chunks=(("stdout", "hello"),))
    await _env(_FakeSandbox(commands=cmds)).exec_backend.execute("cat", stdin=b"hello")
    assert "base64 -d" in cmds.calls[0]["cmd"]


async def test_cwd_defaults_to_first_root_and_validates() -> None:
    cmds = _FakeCommands()
    backend = _env(_FakeSandbox(commands=cmds)).exec_backend
    await backend.execute("pwd")
    assert cmds.calls[0]["cwd"] == _WS
    with pytest.raises(PathAccessError, match="outside"):
        await backend.execute("pwd", cwd=Path("/etc"))


async def test_output_truncation() -> None:
    big = "x" * 1_000_010  # over the per-stream cap
    cmds = _FakeCommands(chunks=(("stdout", big),))
    res = await _env(_FakeSandbox(commands=cmds)).exec_backend.execute("cat big")
    assert res.truncated is True
    assert len(res.stdout) == 1_000_000


async def test_stream_yields_chunks_then_terminal() -> None:
    cmds = _FakeCommands(chunks=(("stdout", "a"), ("stderr", "b")), exit_code=0)
    backend = _env(_FakeSandbox(commands=cmds)).exec_backend
    items = [item async for item in backend.stream("echo")]
    chunks = [i for i in items if isinstance(i, ExecChunk)]
    terminals = [i for i in items if isinstance(i, ExecResult)]
    assert [(c.stream, c.data) for c in chunks] == [("stdout", "a"), ("stderr", "b")]
    assert len(terminals) == 1


# --- environment lifecycle ---------------------------------------------------


async def test_enter_makes_dirs_and_exit_kills() -> None:
    fake = _FakeSandbox()
    async with _env(fake, owns=True):
        assert _WS in fake.files.made_dirs
    assert fake.killed is True


async def test_exit_does_not_kill_when_not_owned() -> None:
    fake = _FakeSandbox()
    async with _env(fake, owns=False):
        pass
    assert fake.killed is False


async def test_pause_on_exit() -> None:
    fake = _FakeSandbox()
    async with _env(fake, owns=True, pause_on_exit=True):
        pass
    assert fake.paused is True
    assert fake.killed is False


async def test_snapshot_creates_persistent_snapshot() -> None:
    # snapshot() is a real create_snapshot (not pause): returns the snapshot id.
    fake = _FakeSandbox(sandbox_id="sbx_snap")
    ref = await _env(fake).snapshot()
    assert ref == "snap-sbx_snap"
    assert fake.snapshotted is True
    assert fake.paused is False


async def test_pause_returns_id() -> None:
    fake = _FakeSandbox(sandbox_id="sbx_p")
    ref = await _env(fake).pause()
    assert ref == "sbx_p"
    assert fake.paused is True
    assert fake.snapshotted is False


async def test_restore_spawns_fresh_sandbox_from_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: dict[str, Any] = {}
    new_sb = _FakeSandbox(sandbox_id="sbx_new")

    class _FakeAsyncSandbox:
        @staticmethod
        async def create(**kwargs: Any) -> _FakeSandbox:
            created.update(kwargs)
            return new_sb

    monkeypatch.setattr(e2b_env, "AsyncSandbox", _FakeAsyncSandbox)

    old = _FakeSandbox(sandbox_id="sbx_old")
    env = E2BEnvironment.from_sandbox(
        cast("AsyncSandbox", old),
        policy=SandboxPolicy(allowed_roots=(Path(_WS),)),
        owns_sandbox=True,
    )
    await env.restore("snap-xyz")
    assert created["template"] == "snap-xyz"  # spawned from the snapshot
    assert old.killed is True  # previous sandbox killed
    # the holder swapped under the backends: I/O now hits the new sandbox
    await env.file_backend.mkdir(Path(f"{_WS}/probe"))
    assert f"{_WS}/probe" in new_sb.files.made_dirs


async def test_resume_reconnects_to_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connected: dict[str, Any] = {}
    resumed = _FakeSandbox(sandbox_id="sbx_resumed")

    class _FakeAsyncSandbox:
        @staticmethod
        async def connect(sandbox_id: str) -> _FakeSandbox:
            connected["id"] = sandbox_id
            return resumed

    monkeypatch.setattr(e2b_env, "AsyncSandbox", _FakeAsyncSandbox)

    env = E2BEnvironment.from_sandbox(
        cast("AsyncSandbox", _FakeSandbox(sandbox_id="sbx_orig")),
        policy=SandboxPolicy(allowed_roots=(Path(_WS),)),
    )
    await env.resume()  # defaults to the currently-bound id
    assert connected["id"] == "sbx_orig"


# --- factory + create path ---------------------------------------------------


async def test_enter_creates_sandbox_with_mapped_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: dict[str, Any] = {}
    fake_sb = _FakeSandbox()

    class _FakeAsyncSandbox:
        @staticmethod
        async def create(**kwargs: Any) -> _FakeSandbox:
            created.update(kwargs)
            return fake_sb

    monkeypatch.setattr(e2b_env, "AsyncSandbox", _FakeAsyncSandbox)

    env = e2b_environment(
        allowed_roots=[_WS],
        network=NetworkPolicy.ALL,
        template="base",
        api_key="k",
    )
    async with env:
        pass

    assert created["allow_internet_access"] is True
    assert created["template"] == "base"
    assert created["api_key"] == "k"
    assert _WS in fake_sb.files.made_dirs


async def test_factory_network_none_is_default() -> None:
    env = e2b_environment(allowed_roots=[_WS], network=NetworkPolicy.NONE)
    assert env.policy.network is NetworkPolicy.NONE


@pytest.mark.parametrize("net", [NetworkPolicy.ALLOWLIST, NetworkPolicy.LOOPBACK])
async def test_factory_rejects_unsupported_network(net: NetworkPolicy) -> None:
    with pytest.raises(NotImplementedError):
        e2b_environment(allowed_roots=[_WS], network=net)


# --- live integration --------------------------------------------------------
#
# Run (unsandboxed) with `uv sync --extra e2b`, then:
#   uv run pytest -m integration tests/sandbox/test_e2b.py
# E2B_API_KEY is picked up from .env via tests/conftest.py + the e2b SDK.

_HAS_E2B = importlib.util.find_spec("e2b") is not None
# bool, not the key itself — a raw key in a module global can surface in logs.
_HAS_E2B_KEY = bool(os.getenv("E2B_API_KEY"))
_live = pytest.mark.skipif(
    not (_HAS_E2B and _HAS_E2B_KEY), reason="needs e2b installed + E2B_API_KEY"
)


@pytest.mark.integration
@_live
async def test_e2b_live_fs_and_exec() -> None:
    async with e2b_environment(allowed_roots=[_WS]) as env:
        fb = env.file_backend
        xb = env.exec_backend
        assert xb is not None

        # write / read / stat roundtrip
        target = Path(f"{_WS}/hi.txt")
        await fb.write_bytes(target, b"hello", mode=0o644)
        text, mtime = await fb.read_text(target)
        assert text == "hello"
        assert mtime > 0
        assert (await fb.stat(target)).size == 5

        # mkdir + list_dir + find_files (glob over a real recursive listing)
        await fb.mkdir(Path(f"{_WS}/pkg"))
        await fb.write_bytes(Path(f"{_WS}/pkg/m.py"), b"x = 1\n", mode=0o644)
        names = {e.name for e in await fb.list_dir(Path(_WS))}
        assert {"hi.txt", "pkg"} <= names
        matched, _ = await fb.find_files(Path(_WS), "**/*.py")
        assert any(e.name == "m.py" for e in matched)

        # grep shelled out inside the sandbox (co-located exec)
        res = await fb.grep(Path(_WS), "x = 1", output_mode="files_with_matches")
        assert any(p.name == "m.py" for p in res.files)

        # exec: success, non-zero-as-data, stdin pipe, cwd, streaming
        ok = await xb.execute("echo from-sandbox")
        assert ok.stdout.strip() == "from-sandbox"
        assert ok.returncode == 0

        bad = await xb.execute("exit 7")
        assert bad.returncode == 7  # returned as data, not raised

        piped = await xb.execute("cat", stdin=b"piped-in")
        assert piped.stdout == "piped-in"

        cwd = await xb.execute("pwd")
        assert cwd.stdout.strip() == _WS

        chunks = [c async for c in xb.stream(r"printf 'a\nb\n'")]
        out = "".join(
            c.data for c in chunks if isinstance(c, ExecChunk) and c.stream == "stdout"
        )
        assert out == "a\nb\n"

        await fb.delete(target)
        assert not await fb.exists(target)


@pytest.mark.integration
@_live
async def test_e2b_live_snapshot_restore() -> None:
    # Real persistent snapshot: restore() spawns a fresh sandbox from the
    # snapshot, so changes made AFTER the snapshot are gone (true rewind).
    ref: str | None = None
    async with e2b_environment(allowed_roots=[_WS]) as env:
        fb = env.file_backend
        await fb.write_bytes(Path(f"{_WS}/before.txt"), b"before", mode=0o644)

        ref = await env.snapshot()  # persistent create_snapshot
        assert ref

        # mutate state AFTER the snapshot, then rewind
        await fb.write_bytes(Path(f"{_WS}/after.txt"), b"after", mode=0o644)
        await env.restore(ref)  # fresh sandbox from the snapshot

        assert await fb.exists(Path(f"{_WS}/before.txt"))  # captured
        assert not await fb.exists(Path(f"{_WS}/after.txt"))  # post-snapshot → gone
        again = await env.exec_backend.execute("echo resumed")
        assert again.stdout.strip() == "resumed"

    # best-effort: drop the persistent snapshot template the test created
    if ref:
        with contextlib.suppress(Exception):
            e2b_mod = importlib.import_module("e2b")
            await e2b_mod.AsyncSandbox.delete_snapshot(ref)


@pytest.mark.integration
@_live
async def test_e2b_live_pause_resume() -> None:
    # Suspend/resume the SAME sandbox (state preserved, not a rewind).
    async with e2b_environment(allowed_roots=[_WS]) as env:
        marker = Path(f"{_WS}/kept.txt")
        await env.file_backend.write_bytes(marker, b"kept", mode=0o644)

        sandbox_id = await env.pause()
        assert sandbox_id
        await env.resume(sandbox_id)

        text, _ = await env.file_backend.read_text(marker)
        assert text == "kept"
        assert (await env.exec_backend.execute("echo up")).stdout.strip() == "up"


@pytest.mark.integration
@_live
async def test_e2b_live_project_workflow() -> None:
    # A realistic file<->exec loop: write a project, compute via Bash, read the
    # result back through the file backend (co-location), then edit + re-run.
    async with e2b_environment(allowed_roots=[_WS]) as env:
        fb = env.file_backend
        xb = env.exec_backend
        assert xb is not None

        await fb.mkdir(Path(f"{_WS}/proj"))
        await fb.write_bytes(Path(f"{_WS}/proj/data.txt"), b"3\n4\n5\n", mode=0o644)
        await fb.write_bytes(
            Path(f"{_WS}/proj/sum.py"),
            b"print(sum(int(x) for x in open('data.txt')))\n",
            mode=0o644,
        )

        run = await xb.execute("python3 sum.py > out.txt", cwd=Path(f"{_WS}/proj"))
        assert run.returncode == 0
        out, _ = await fb.read_text(Path(f"{_WS}/proj/out.txt"))
        assert out.strip() == "12"  # exec wrote it; file backend reads it back

        # edit the data through the file backend, re-run, observe the change
        await fb.write_bytes(Path(f"{_WS}/proj/data.txt"), b"10\n20\n", mode=0o644)
        run2 = await xb.execute("python3 sum.py", cwd=Path(f"{_WS}/proj"))
        assert run2.stdout.strip() == "30"

        # grep + glob over the real project tree
        hits = await fb.grep(Path(_WS), "sum", output_mode="files_with_matches")
        assert any(p.name == "sum.py" for p in hits.files)
        matched, _ = await fb.find_files(Path(_WS), "**/*.py")
        assert {e.name for e in matched} == {"sum.py"}


@pytest.mark.integration
@_live
async def test_e2b_live_exec_matrix() -> None:
    async with e2b_environment(allowed_roots=[_WS]) as env:
        xb = env.exec_backend
        assert xb is not None

        # per-call env vars
        env_res = await xb.execute("echo $MYVAR", env={"MYVAR": "hello-env"})
        assert env_res.stdout.strip() == "hello-env"

        # cwd into a subdir
        await xb.execute(f"mkdir -p {_WS}/sub")
        cwd = await xb.execute("pwd", cwd=Path(f"{_WS}/sub"))
        assert cwd.stdout.strip() == f"{_WS}/sub"

        # arbitrary non-zero exit codes preserved as data
        assert (await xb.execute("exit 42")).returncode == 42

        # multi-line stdin via the base64 pipe
        counted = await xb.execute("wc -l", stdin=b"a\nb\nc\n")
        assert int(counted.stdout.strip()) == 3

        # stderr captured independently of stdout
        err = await xb.execute("echo oops >&2")
        assert err.stdout == ""
        assert "oops" in err.stderr

        # real timeout -> OVERALL_TIMEOUT (not a raised exception)
        slow = await xb.execute("sleep 5", timeout=2)
        assert slow.reason is TerminationReason.OVERALL_TIMEOUT

        # streaming delivers the full output in order
        chunks = [c async for c in xb.stream("for i in 1 2 3; do echo $i; done")]
        out = "".join(
            c.data for c in chunks if isinstance(c, ExecChunk) and c.stream == "stdout"
        )
        assert out == "1\n2\n3\n"


@pytest.mark.integration
@_live
async def test_e2b_live_file_matrix() -> None:
    async with e2b_environment(allowed_roots=[_WS]) as env:
        fb = env.file_backend

        # binary roundtrip (non-utf8 bytes)
        blob = bytes(range(256))
        await fb.write_bytes(Path(f"{_WS}/blob.bin"), blob, mode=0o644)
        data, _ = await fb.read_bytes(Path(f"{_WS}/blob.bin"))
        assert data == blob

        # large file
        big = b"x" * (256 * 1024)
        await fb.write_bytes(Path(f"{_WS}/big.bin"), big, mode=0o644)
        back, _ = await fb.read_bytes(Path(f"{_WS}/big.bin"))
        assert len(back) == len(big)
        assert (await fb.stat(Path(f"{_WS}/big.bin"))).size == len(big)

        # unicode text
        text = "日本語 → 世界 🌍"
        await fb.write_bytes(Path(f"{_WS}/u.txt"), text.encode("utf-8"), mode=0o644)
        got, _ = await fb.read_text(Path(f"{_WS}/u.txt"))
        assert got == text

        # empty file
        await fb.write_bytes(Path(f"{_WS}/empty.txt"), b"", mode=0o644)
        empty, _ = await fb.read_text(Path(f"{_WS}/empty.txt"))
        assert empty == ""

        # nested mkdir + recursive listing
        await fb.mkdir(Path(f"{_WS}/a/b/c"))
        await fb.write_bytes(Path(f"{_WS}/a/b/c/deep.txt"), b"deep", mode=0o644)
        recursive = await fb.list_dir(Path(_WS), recursive=True)
        assert any(e.name == "deep.txt" for e in recursive)

        # delete clears existence
        await fb.delete(Path(f"{_WS}/empty.txt"))
        assert not await fb.exists(Path(f"{_WS}/empty.txt"))


@pytest.mark.integration
@_live
async def test_e2b_live_grep_matrix() -> None:
    async with e2b_environment(allowed_roots=[_WS]) as env:
        fb = env.file_backend
        await fb.write_bytes(
            Path(f"{_WS}/a.py"), b"import os\nTODO: x\ntodo: y\n", mode=0o644
        )
        await fb.write_bytes(Path(f"{_WS}/b.md"), b"TODO: z\n", mode=0o644)

        # content mode with line numbers
        content = await fb.grep(Path(_WS), "import", output_mode="content")
        assert any("a.py" in ln and "import os" in ln for ln in content.lines)

        # count mode
        counts = await fb.grep(Path(_WS), "TODO", output_mode="count")
        by_file = {p.name: n for p, n in counts.counts}
        assert by_file.get("a.py") == 1
        assert by_file.get("b.md") == 1

        # case-insensitive widens the match
        ci = await fb.grep(
            Path(_WS), "todo", output_mode="count", case_insensitive=True
        )
        assert {p.name: n for p, n in ci.counts}.get("a.py") == 2

        # glob filter restricts to *.py
        only_py = await fb.grep(
            Path(_WS), "TODO", glob="*.py", output_mode="files_with_matches"
        )
        assert {p.name for p in only_py.files} == {"a.py"}


@pytest.mark.integration
@_live
async def test_e2b_live_network_none_blocks_egress() -> None:
    async with e2b_environment(allowed_roots=[_WS], network=NetworkPolicy.NONE) as env:
        xb = env.exec_backend
        assert xb is not None
        code = "import urllib.request as u; u.urlopen('https://example.com', timeout=8)"
        probe = await xb.execute(f'python3 -c "{code}"', timeout=20)
        assert probe.returncode != 0  # egress denied


@pytest.mark.integration
@_live
async def test_e2b_live_network_all_allows_egress() -> None:
    async with e2b_environment(allowed_roots=[_WS], network=NetworkPolicy.ALL) as env:
        xb = env.exec_backend
        assert xb is not None
        code = (
            "import urllib.request as u; "
            "print(u.urlopen('https://example.com', timeout=15).status)"
        )
        probe = await xb.execute(f'python3 -c "{code}"', timeout=30)
        assert probe.returncode == 0
        assert probe.stdout.strip() == "200"


@pytest.mark.integration
@_live
async def test_e2b_live_deny_write_two_plane() -> None:
    # The FS carve-out is enforced on the TOOL plane (validate_path); the remote
    # OS plane treats the whole sandbox as the boundary, so Bash can still write.
    protected = f"{_WS}/protected"
    async with e2b_environment(allowed_roots=[_WS], deny_write=[protected]) as env:
        fb = env.file_backend
        xb = env.exec_backend
        assert xb is not None

        target = Path(f"{protected}/secret.txt")
        # tool plane: write validation refused
        with pytest.raises(PathAccessError):
            await fb.validate_path(target, must_exist=False, access="write")
        # but reads under it validate fine (deny_write is write-only)
        await fb.validate_path(target, must_exist=False, access="read")

        # OS plane: Bash is not constrained by the tool-plane carve-out
        wrote = await xb.execute(f"mkdir -p {protected} && echo hi > {target}")
        assert wrote.returncode == 0
        assert (await xb.execute(f"cat {target}")).stdout.strip() == "hi"


@pytest.mark.integration
@_live
async def test_e2b_live_concurrent_commands() -> None:
    # Many commands run concurrently in one sandbox without interfering.
    async with e2b_environment(allowed_roots=[_WS]) as env:
        xb = env.exec_backend
        assert xb is not None
        results = await asyncio.gather(*(xb.execute(f"echo n{i}") for i in range(6)))
        assert [r.stdout.strip() for r in results] == [f"n{i}" for i in range(6)]
        assert all(r.returncode == 0 for r in results)


@pytest.mark.integration
@_live
async def test_e2b_live_persistent_session() -> None:
    from grasp_agents.sandbox.exec_backend import SessionCapable

    async def _collect(
        session: Any, command: str, **kw: float
    ) -> tuple[str, str, ExecResult]:
        out: list[str] = []
        err: list[str] = []
        terminal: ExecResult | None = None
        async for item in session.run(command, **kw):
            if isinstance(item, ExecResult):
                terminal = item
            else:
                (out if item.stream == "stdout" else err).append(item.data)
        assert terminal is not None
        return "".join(out), "".join(err), terminal

    async with e2b_environment(allowed_roots=[_WS]) as env:
        backend = env.exec_backend
        assert isinstance(backend, SessionCapable)
        session = await backend.open_session(cwd=Path(_WS))
        try:
            # cd and env persist across commands (one long-lived shell).
            await _collect(session, "mkdir -p sub && cd sub && export FOO=bar")
            out, _, r = await _collect(session, "pwd && echo $FOO")
            assert r.returncode == 0
            assert "sub" in out
            assert "bar" in out
            # stdout / stderr stay separate.
            out2, err2, r2 = await _collect(session, "echo OUT; echo ERR 1>&2")
            assert r2.returncode == 0
            assert "OUT" in out2
            assert "ERR" in err2
            # non-zero exit propagates and does not break the session.
            _, _, rf = await _collect(session, "false")
            assert rf.returncode == 1
            # timeout kills the shell and closes the session (no SIGINT on E2B).
            _, _, rt = await _collect(session, "sleep 30", timeout=1.0)
            assert rt.timed_out
            assert session.closed
        finally:
            await session.close()


# --- live backgrounding (manager + Bash + KillTask) --------------------------
#
# The same backgrounding flow test_bash_polish exercises against a local
# subprocess, here against a real remote E2B container: a long Bash command
# outlives its deadline → the BackgroundTaskManager sidelines it → it completes
# (output read off its buffered events) / is killed.


@pytest.mark.integration
@_live
async def test_e2b_live_background_poll_and_complete() -> None:
    async with e2b_environment(allowed_roots=[_WS]) as env:
        ctx: SessionContext[None] = SessionContext(environment=env)
        agent_ctx, mgr = make_stack()

        note, task_id = await background(
            mgr,
            ctx,
            agent_ctx,
            "echo early && sleep 3 && echo late",
            abg=0.5,
            timeout=60,
        )
        assert "moved to the background" in note  # outlived the deadline
        assert task_id is not None

        collected, out = await poll_until_done(mgr, task_id, tries=80, delay=0.5)
        assert out.status == "completed"
        assert out.result is not None  # the terminal BashResult
        assert out.result.returncode == 0
        # output produced before *and* after backgrounding is read off the buffer
        assert "early" in collected
        assert "late" in collected
        # poll_until_done only waits/reads; the task is dropped by drain, not by
        # the read, so it is still tracked here (delivered + dropped on drain).
        assert task_id in mgr._tasks  # pyright: ignore[reportPrivateUsage]


@pytest.mark.integration
@_live
async def test_e2b_live_background_kill_terminates_remote_command() -> None:
    # KillTask cancels the manager task → the E2B backend kills the remote
    # command (handle.kill) → the marker file stops growing in the sandbox.
    async with e2b_environment(allowed_roots=[_WS]) as env:
        ctx: SessionContext[None] = SessionContext(environment=env)
        agent_ctx, mgr = make_stack()

        marker = f"{_WS}/ticks.txt"
        _note, task_id = await background(
            mgr,
            ctx,
            agent_ctx,
            f"while true; do echo tick >> {marker}; sleep 0.2; done",
            abg=0.5,
            timeout=30,
        )
        assert task_id is not None

        killed = await kill(mgr, task_id)
        assert killed.status == "cancelled"  # killed mid-run, not finished

        # the remote process is actually gone: the marker stops growing
        size_a = await marker_size(env, marker)
        await asyncio.sleep(2.0)
        size_b = await marker_size(env, marker)
        assert size_a == size_b

        with pytest.raises(ValueError, match="Unknown background task id"):
            await kill(mgr, task_id)


@pytest.mark.integration
@_live
async def test_e2b_live_background_small_result_inlined() -> None:
    # A finished background command's output is delivered inline in its
    # completion note and the task is dropped.
    async with e2b_environment(allowed_roots=[_WS]) as env:
        ctx: SessionContext[None] = SessionContext(environment=env)
        agent_ctx, mgr = make_stack()

        _note, task_id = await background(
            mgr, ctx, agent_ctx, "echo hello && sleep 3", abg=0.5, timeout=60
        )
        assert task_id is not None

        await mgr.wait_idle()
        notes = await drain_notes(mgr, ctx)
        assert len(notes) == 1
        assert "completed" in notes[0]
        assert "hello" in notes[0]  # inlined directly
        assert "omitted" not in notes[0]  # small result inlined whole
        assert mgr._tasks == {}  # pyright: ignore[reportPrivateUsage]


@pytest.mark.integration
@_live
async def test_e2b_live_background_large_result_excerpted() -> None:
    # Cap-and-defer end-to-end on a real sandbox: a large backgrounded result is
    # excerpted in the completion note, which points at the task's .grasp log
    # (in the remote FS) holding the full output.
    async with e2b_environment(allowed_roots=[_WS]) as env:
        ctx: SessionContext[None] = SessionContext(environment=env)
        agent_ctx, mgr = make_stack()

        _note, task_id = await background(
            mgr,
            ctx,
            agent_ctx,
            "head -c 5000 /dev/zero | tr '\\0' 'A'; echo END; sleep 3",
            abg=0.5,
            timeout=60,
            max_inline_result_chars=200,
        )
        assert task_id is not None

        await mgr.wait_idle()
        notes = await drain_notes(mgr, ctx)
        assert len(notes) == 1
        assert "completed" in notes[0]
        assert "chars omitted" in notes[0]  # excerpted
        assert "<log_file>" in notes[0]  # points at the .grasp log
        assert mgr._tasks == {}  # pyright: ignore[reportPrivateUsage]

        # The full, untruncated output is in the .grasp log (in the remote FS).
        match = re.search(r"<log_file>(.+?)</log_file>", notes[0], re.DOTALL)
        assert match is not None
        log_text, _ = await env.file_backend.read_text(Path(match.group(1).strip()))
        assert log_text.count("A") == 5000


@pytest.mark.integration
@_live
async def test_e2b_live_background_writes_greppable_log() -> None:
    # A backgrounded command's streamed output is mirrored to an agent-readable
    # .grasp/tasks log inside the *remote* sandbox FS (real path translation),
    # indexed on the TaskRecord.
    from grasp_agents.durability.checkpoint_store import InMemoryCheckpointStore
    from grasp_agents.durability.store_keys import task_prefix
    from grasp_agents.durability.task_record import TaskRecord

    async with e2b_environment(allowed_roots=[_WS]) as env:
        store = InMemoryCheckpointStore()
        ctx: SessionContext[None] = SessionContext(
            environment=env, checkpoint_store=store, session_key="s1"
        )
        agent_ctx, mgr = make_stack()

        _note, task_id = await background(
            mgr, ctx, agent_ctx, "echo HELLO && sleep 5", abg=0.5, timeout=60
        )
        assert task_id is not None

        await asyncio.sleep(1.0)  # let HELLO stream from the sandbox
        await flush(mgr, ctx)

        keys = await store.list_keys(task_prefix("s1"))
        rec = TaskRecord.model_validate_json((await store.load(keys[0])) or b"{}")
        assert rec.output_path is not None
        assert ".grasp/tasks" in rec.output_path
        # The log lives in the remote sandbox FS and holds the streamed output.
        content, _ = await env.file_backend.read_text(Path(rec.output_path))
        assert "HELLO" in content

        await kill(mgr, task_id)


# ---------- created sandbox not orphaned on setup failure ----------


class _SetupFailFiles:
    async def make_dir(self, path: str) -> None:
        del path
        msg = "make_dir failed"
        raise RuntimeError(msg)


class _SetupFailSandbox:
    def __init__(self) -> None:
        self.files = _SetupFailFiles()
        self.killed = False

    @classmethod
    async def create(cls, **kwargs: Any) -> _SetupFailSandbox:
        del kwargs
        return cls()

    async def kill(self) -> None:
        self.killed = True


class TestE2BSetupFailure:
    async def test_created_sandbox_killed_when_setup_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        del tmp_path
        e2b_env_mod = pytest.importorskip("grasp_agents.sandbox.e2b.environment")

        created: list[_SetupFailSandbox] = []

        class _Recorder(_SetupFailSandbox):
            @classmethod
            async def create(cls, **kwargs: Any) -> _SetupFailSandbox:
                sandbox = await super().create(**kwargs)
                created.append(sandbox)  # type: ignore[arg-type]
                return sandbox

        monkeypatch.setattr(e2b_env_mod, "_sandbox_cls", lambda _ci: _Recorder)

        from grasp_agents.sandbox.e2b._handle import SandboxHandle
        from grasp_agents.sandbox.e2b.exec import E2BExecBackend
        from grasp_agents.sandbox.e2b.file_backend import E2BFileBackend

        policy = SandboxPolicy(allowed_roots=(Path("/home/user/workspace"),))
        holder = SandboxHandle(None)
        env = e2b_env_mod.E2BEnvironment(
            policy=policy,
            holder=holder,
            file_backend=E2BFileBackend(holder, policy=policy),
            exec_backend=E2BExecBackend(holder, policy=policy),
            create_params={},
            owns_sandbox=True,
        )

        with pytest.raises(RuntimeError, match="make_dir failed"):
            await env.__aenter__()

        assert len(created) == 1
        assert created[0].killed is True
        assert holder.sandbox is None
