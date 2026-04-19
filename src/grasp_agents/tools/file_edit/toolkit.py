"""
``FileEditToolkit`` — factory that bundles a ``FileEditStore`` with a
matching set of file-edit tools.

Usage::

    toolkit = FileEditToolkit(allowed_roots=[Path.cwd()])
    agent.tools = [*toolkit.tools(), my_custom_tool]

    # Production: route state through ``RunContext`` so sub-agents and
    # later calls in the same session share read-before-write records.
    ctx = RunContext(
        state=my_state,
        file_edit_store=toolkit.store,
        session_key="conv-42",
    )
    await agent.run(..., ctx=ctx)

    # Teardown a session when the conversation ends.
    await toolkit.reset_session("conv-42")

When callers don't supply a ``ctx.file_edit_store``, the tools fall
back to the toolkit's own store keyed by ``default_session_key`` — this
keeps standalone / test usage working without RunContext plumbing.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .edit import EditTool
from .read import DEFAULT_MAX_READ_CHARS, ReadTool
from .redact import DefaultSecretRedactor, SecretRedactor
from .store import FileEditStore, InMemoryFileEditStore
from .write import DEFAULT_NEW_FILE_MODE, WriteTool

if TYPE_CHECKING:
    from ...types.tool import BaseTool


class FileEditToolkit:
    """
    Build and hold the store + a matching set of file-edit tools.

    The toolkit owns one :class:`FileEditStore` (in-memory by default)
    shared by every tool it constructs. Tools prefer the
    ``RunContext``'s store when the caller wires one up; otherwise they
    read/write state through the toolkit's own store keyed by
    ``default_session_key``.
    """

    def __init__(
        self,
        *,
        allowed_roots: list[Path] | None = None,
        include_dotfiles: bool = True,
        redactor: SecretRedactor | None = None,
        store: FileEditStore | None = None,
        default_session_key: str = "default",
        max_read_chars: int = DEFAULT_MAX_READ_CHARS,
        new_file_mode: int = DEFAULT_NEW_FILE_MODE,
        tool_timeout: float | None = None,
    ) -> None:
        """
        Create a toolkit.

        Args:
            allowed_roots: Directories the tools may read / write under.
                Defaults to ``[Path.cwd()]``. Each entry is expanded and
                resolved at tool-call time.
            include_dotfiles: If True (default), the sensitive-path deny
                list adds common credential-dotfile patterns (``.env``,
                ``~/.ssh``, etc.) on top of the system-path baseline. Set
                False to only block system paths.
            redactor: Secret-redaction strategy for Read output.
                Defaults to :class:`DefaultSecretRedactor` — pass
                :class:`NullRedactor()` to opt out.
            store: Session-keyed backing store. Defaults to
                :class:`InMemoryFileEditStore`. Pass a custom
                implementation to plug in persistence or shared backends.
            default_session_key: Key used when tools are called without
                a ``RunContext`` that provides its own ``file_edit_store``.
                Lets the same toolkit serve standalone / test use.
            max_read_chars: Character cap on the formatted output of a
                single Read call. Default ``100_000``.
            new_file_mode: File permissions applied to freshly-created
                files. Default ``0o644``. Existing files preserve their
                current mode on overwrite.
            tool_timeout: Per-tool async timeout in seconds. ``None``
                disables the timeout.

        """
        self._allowed_roots: list[Path] = (
            list(allowed_roots) if allowed_roots is not None else [Path.cwd()]
        )
        self._include_dotfiles = include_dotfiles
        self._redactor: SecretRedactor = redactor or DefaultSecretRedactor()
        self._store: FileEditStore = store or InMemoryFileEditStore()
        self._default_session_key = default_session_key
        self._max_read_chars = max_read_chars
        self._new_file_mode = new_file_mode
        self._tool_timeout = tool_timeout

        self._read_tool = ReadTool(
            store=self._store,
            allowed_roots=self._allowed_roots,
            redactor=self._redactor,
            default_session_key=default_session_key,
            max_read_chars=self._max_read_chars,
            timeout=tool_timeout,
        )
        self._write_tool = WriteTool(
            store=self._store,
            allowed_roots=self._allowed_roots,
            default_session_key=default_session_key,
            include_dotfiles=self._include_dotfiles,
            new_file_mode=self._new_file_mode,
            timeout=tool_timeout,
        )
        self._edit_tool = EditTool(
            store=self._store,
            allowed_roots=self._allowed_roots,
            default_session_key=default_session_key,
            include_dotfiles=self._include_dotfiles,
            timeout=tool_timeout,
        )

    # ---- Tool accessors ----------------------------------------------------

    @property
    def read(self) -> ReadTool:
        return self._read_tool

    @property
    def write(self) -> WriteTool:
        return self._write_tool

    @property
    def edit(self) -> EditTool:
        return self._edit_tool

    def tools(self) -> list[BaseTool[Any, Any, Any]]:
        """Return the configured tools as a list, ready to attach to an agent."""
        return [self._read_tool, self._write_tool, self._edit_tool]

    # ---- Store + session -----------------------------------------------------

    @property
    def store(self) -> FileEditStore:
        """Expose the backing store for use as ``ctx.file_edit_store``."""
        return self._store

    @property
    def default_session_key(self) -> str:
        return self._default_session_key

    async def reset_session(self, session_key: str | None = None) -> None:
        """
        Drop the state for a session (default: the toolkit's own key).

        After reset, the next tool call for that session starts with a
        fresh :class:`FileEditSessionState`.
        """
        await self._store.reset_session(session_key or self._default_session_key)

    # ---- Dotfile-override management ---------------------------------------

    async def allow_dotfile(
        self, path: Path, *, session_key: str | None = None
    ) -> None:
        """
        Whitelist a specific sensitive dotfile for a session.

        Bypasses the user-dotfile deny list for ``path``. The system-path
        baseline is not overridable. Defaults to the toolkit's own
        ``default_session_key``.
        """
        key = session_key or self._default_session_key
        state = await self._store.get_session_state(key)
        # ``expanduser`` + ``resolve`` here are cheap path manipulations
        # that don't block on disk; the async-path lint doesn't apply.
        state.dotfile_overrides.add(path.expanduser().resolve())  # noqa: ASYNC240
