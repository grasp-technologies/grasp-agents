"""
Test doubles for the E2B backend: a fake ``AsyncSandbox`` with its ``files``
and ``commands`` APIs, a persistent-shell fake and a code-interpreter fake.

Cast a fake to ``AsyncSandbox`` and drive it only through the public backend
API. The fake exception classes subclass the real e2b ``TimeoutException`` /
``CommandExitException`` so the adapter's isinstance checks match.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from e2b import CommandExitException, TimeoutException

WS = "/home/user/workspace"


# --- fake exception classes -------------------------------------------------


class FakeTimeout(TimeoutException):
    """Subclasses the real e2b type so the adapter's isinstance checks match."""


class FakeCommandExit(CommandExitException):
    """Subclasses the real e2b type so the adapter's isinstance checks match."""

    def __init__(self, *, stdout: str = "", stderr: str = "", exit_code: int = 1):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


# --- fake sandbox (cast to AsyncSandbox; driven via the public backend API) -


@dataclass
class FakeFileType:
    value: str


class FakeEntry:
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
        self.type = FakeFileType(type_value)
        self.size = size
        self.mode = mode
        self.modified_time = dt.datetime.fromtimestamp(mtime_epoch, tz=dt.UTC)


@dataclass
class FakeResult:
    stdout: str
    exit_code: int


@dataclass
class FakeSnapshotInfo:
    snapshot_id: str


class FakeFiles:
    def __init__(self) -> None:
        self.reads: dict[str, Any] = {}
        self.infos: dict[str, FakeEntry] = {}
        self.listing: dict[str, list[FakeEntry]] = {}
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
            path, FakeEntry(name=Path(path).name, path=path, mtime_epoch=2000.0)
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


class FakeHandle:
    def __init__(
        self,
        *,
        exit_code: int = 0,
        chunks: tuple[tuple[str, str], ...] = (),
        raise_exc: BaseException | None = None,
        hang: bool = False,
        deadline: asyncio.TimerHandle | None = None,
        on_stdout: Any = None,
        on_stderr: Any = None,
    ) -> None:
        self.exit_code = exit_code
        self.killed = False
        self.disconnected = False
        self._chunks = chunks
        self._raise = raise_exc
        self._hang = hang
        self._deadline = deadline
        self._stopped = asyncio.Event()
        self._on_stdout = on_stdout
        self._on_stderr = on_stderr

    async def wait(self) -> Any:
        try:
            for stream, data in self._chunks:
                cb = self._on_stdout if stream == "stdout" else self._on_stderr
                if cb is not None:
                    cb(data)
            if self._hang:  # runs until killed / disconnected
                await self._stopped.wait()
                raise FakeCommandExit(exit_code=137)
            if self._raise is not None:
                raise self._raise
            return None
        finally:
            if self._deadline is not None:
                self._deadline.cancel()

    async def kill(self) -> Any:
        self.killed = True
        self._stopped.set()
        return True

    async def disconnect(self) -> None:
        self.disconnected = True
        self._stopped.set()


class FakeCommands:
    def __init__(
        self,
        *,
        exit_code: int = 0,
        chunks: tuple[tuple[str, str], ...] = (),
        raise_exc: BaseException | None = None,
        run_raises: BaseException | None = None,
        hang: bool = False,
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
        self._hang = hang
        self.handles: list[FakeHandle] = []
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
            # Like the SDK, a command timeout cancels the task that called run.
            deadline = None
            if timeout:
                task = asyncio.current_task()
                assert task is not None
                deadline = asyncio.get_running_loop().call_later(timeout, task.cancel)
            handle = FakeHandle(
                exit_code=self._exit_code,
                chunks=self._chunks,
                raise_exc=self._raise,
                hang=self._hang,
                deadline=deadline,
                on_stdout=on_stdout,
                on_stderr=on_stderr,
            )
            self.handles.append(handle)
            return handle
        if self._fg_raises is not None:
            raise self._fg_raises
        return FakeResult(self._fg_stdout, self._fg_exit_code)

    async def send_stdin(self, pid: int, data: str) -> Any:
        self.stdin_writes.append((pid, data))

    async def kill(self, pid: int) -> Any:
        self.killed.append(pid)


class FakeSandbox:
    def __init__(
        self,
        *,
        files: FakeFiles | None = None,
        commands: FakeCommands | None = None,
        sandbox_id: str = "sbx_fake",
    ) -> None:
        self.files = files or FakeFiles()
        self.commands = commands or FakeCommands()
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
        return FakeSnapshotInfo(snapshot_id=f"snap-{self.sandbox_id}")


# --- persistent shells and code contexts ---------------------------------


class ShellHandle:
    """The output stream of a fake persistent shell, as one client holds it."""

    def __init__(self, pid: int, on_stdout: Any, on_stderr: Any) -> None:
        self.pid = pid
        self._on_stdout = on_stdout
        self._on_stderr = on_stderr
        self.disconnected = False

    def deliver(self, stdout: str, stderr: str) -> None:
        if not self.disconnected:
            self._on_stdout(stdout)
            self._on_stderr(stderr)

    async def disconnect(self) -> None:
        self.disconnected = True


class ShellCommands(FakeCommands):
    """
    Fake persistent shells, numbered per sandbox from pid 1: a stdin payload
    echoes its end markers on both streams, as ``/bin/sh`` would. A pid only
    reaches a shell started on the same sandbox.
    """

    def __init__(self) -> None:
        super().__init__()
        self.shells: dict[int, ShellHandle] = {}
        self.shell_timeouts: list[float | None] = []

    async def run(self, cmd: str, **kwargs: Any) -> Any:
        if not kwargs.get("stdin"):
            return await super().run(cmd, **kwargs)
        self.shell_timeouts.append(kwargs.get("timeout"))
        pid = len(self.shells) + 1
        handle = ShellHandle(pid, kwargs["on_stdout"], kwargs["on_stderr"])
        self.shells[pid] = handle
        return handle

    async def send_stdin(self, pid: int, data: str) -> Any:
        await super().send_stdin(pid, data)
        if pid in self.killed:
            raise RuntimeError(f"process {pid} not found")
        found = re.search(r"__GRASP_[0-9a-f]+__", data)
        assert found is not None
        self.shells[pid].deliver(f"{found.group(0)} 0\n", f"{found.group(0)}\n")


class CodeSandbox(FakeSandbox):
    """Code-interpreter double: a context exists only in the sandbox it was made in."""

    def __init__(self, *, sandbox_id: str, contexts: tuple[str, ...] = ()) -> None:
        super().__init__(sandbox_id=sandbox_id)
        self.contexts = set(contexts)
        self.ran: list[tuple[str, str]] = []

    async def create_code_context(self, cwd: str | None = None) -> Any:
        context = SimpleNamespace(id=f"{self.sandbox_id}-ctx{len(self.contexts)}")
        self.contexts.add(context.id)
        return context

    async def list_code_contexts(self) -> list[Any]:
        return [
            SimpleNamespace(id=context_id, language="python", cwd=WS)
            for context_id in sorted(self.contexts)
        ]

    async def run_code(self, code: str, *, context: Any, **kwargs: Any) -> Any:
        if context.id not in self.contexts:
            raise RuntimeError(f"no context {context.id} on {self.sandbox_id}")
        self.ran.append((context.id, code))
        logs = SimpleNamespace(stdout=[], stderr=[])
        return SimpleNamespace(logs=logs, results=[], error=None, execution_count=1)
