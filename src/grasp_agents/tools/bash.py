"""
``Bash`` — run a shell command via ``ctx.exec_backend``.

Opt-in and stateless: it consumes the :class:`ExecBackend` wired onto
:attr:`RunContext.exec_backend` (set it via :func:`local_environment` or by
constructing a backend directly). Without an exec backend the tool refuses to
run — an agent gets no shell access by default.

Non-interactive contract: no TTY, no shell state carried between calls. The
backend owns the real timeout (it kills the process group on expiry); the
per-call ``timeout`` is clamped to ``max_timeout``. Output is captured and
size-capped by the backend; ``truncated`` flags when a cap was hit.

Auto-background migration (long commands → ``BackgroundTaskManager``) and the
streamed-output / heartbeat path are deliberately out of this first cut — see
``docs/roadmap/15-sandbox-and-terminal.md`` step 3.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..types.tool import BaseTool, ToolProgressCallback

if TYPE_CHECKING:
    from ..run_context import RunContext

DEFAULT_BASH_TIMEOUT = 120.0
MAX_BASH_TIMEOUT = 600.0


class BashInput(BaseModel):
    """Input schema for the ``Bash`` tool."""

    command: str = Field(
        description=(
            "The shell command to run, non-interactively, via `/bin/sh -c`. "
            "Quote paths with spaces. Chain steps with `&&`. Do not launch "
            "interactive programs or long-lived servers."
        )
    )
    cwd: str | None = Field(
        default=None,
        description=(
            "Working directory for the command. Must resolve under the "
            "environment's allowed roots. Defaults to the first allowed root."
        ),
    )
    timeout: float | None = Field(
        default=None,
        description=(
            "Overall wall-clock timeout in seconds. Defaults to "
            f"{DEFAULT_BASH_TIMEOUT:g}s; clamped to {MAX_BASH_TIMEOUT:g}s."
        ),
        gt=0,
    )


class BashResult(BaseModel):
    """Output schema for the ``Bash`` tool."""

    stdout: str
    stderr: str
    returncode: int
    # TerminationReason value: "exit" | "overall_timeout" | "no_output_timeout"
    # | "manual_cancel" | "signal" | "spawn_error".
    reason: str
    timed_out: bool = False
    truncated: bool = False
    runtime_ms: float = 0.0
    backend: str = ""


class Bash(BaseTool[BashInput, BashResult, Any]):
    """
    Execute a shell command in the agent's environment via ``ctx.exec_backend``.

    Stateless wrapper around the bound :class:`ExecBackend`; the isolation that
    applies (none / seatbelt / bwrap / docker / ...) is whatever the wired
    backend provides and is reported back in :attr:`BashResult.backend`.
    """

    name = "Bash"
    description = (
        "Run a shell command in the agent's environment and return its "
        "stdout, stderr, and exit code. Use this for running programs, build "
        "and test commands, and git operations. Prefer the dedicated file "
        "tools (Read / Write / Edit / Glob / Grep) over shell equivalents "
        "(cat / sed / find / grep) — they are safer and give better output. "
        "Commands run non-interactively (no prompts, no TTY) and share no "
        "state between calls; pass `cd ... &&` if you need a different "
        "directory in one call, or set `cwd`."
    )

    def __init__(
        self,
        *,
        default_timeout: float = DEFAULT_BASH_TIMEOUT,
        max_timeout: float = MAX_BASH_TIMEOUT,
        timeout: float | None = None,
    ) -> None:
        super().__init__(timeout=timeout)
        self._default_timeout = default_timeout
        self._max_timeout = max_timeout

    async def _run(
        self,
        inp: BashInput,
        *,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
    ) -> BashResult:
        del exec_id, progress_callback

        if ctx is None or ctx.exec_backend is None:
            raise ValueError(
                "Bash requires ctx.exec_backend. Wire an ExecBackend on "
                "RunContext (e.g. via local_environment(...)) before running "
                "the agent."
            )

        requested = inp.timeout if inp.timeout is not None else self._default_timeout
        effective_timeout = min(requested, self._max_timeout)
        cwd = Path(inp.cwd) if inp.cwd is not None else None

        result = await ctx.exec_backend.execute(
            inp.command,
            cwd=cwd,
            timeout=effective_timeout,
        )

        return BashResult(
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
            reason=result.reason.value,
            timed_out=result.timed_out,
            truncated=result.truncated,
            runtime_ms=result.runtime_ms,
            backend=result.backend,
        )
