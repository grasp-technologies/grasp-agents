"""
:class:`MCPFileBackend` — routes file I/O to an MCP server.

Built around two MCP surfaces:

* **resources** — the canonical read-only surface. ``read_text`` calls
  ``resources/read``; ``stat`` / ``exists`` / ``list_dir`` /
  ``find_files`` all query a cached :class:`MCPResourceIndex` populated
  by a single ``resources/list`` per session.

* **tools** — for mutations only. The server SHOULD expose:

  - ``write_file(path, content, mode?) -> {mtime_ms}``
  - ``delete_file(path) -> {}``

  After every mutation the backend invalidates the resource index so
  the next read sees fresh metadata.

Sandbox enforcement (path containment, sensitive-path policy, etc.)
lives on the server side; the backend's :meth:`validate_path` only
enforces ``allowed_roots`` membership in the server's address space.

Read-before-write bookkeeping lives on the *agent* (each
:class:`AgentLoop` owns its own :class:`FileEditSessionState`); the
backend itself is pure I/O.

All paths crossing the public API are :class:`pathlib.Path`; the
backend renders them as POSIX strings for the wire.
"""

from __future__ import annotations

import json
import posixpath
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, cast

from ...mcp.resource_index import AnyUrl, MCPResourceIndex
from .base import FileBackend, FileEntry, FileStat, GrepRawResult
from .local import glob_filter_entries
from .paths import PathAccessError

try:
    from mcp.types import TextContent, TextResourceContents
except ImportError as _err:
    msg = (
        "MCP file backend requires the 'mcp' package. "
        "Install with: pip install grasp-agents[mcp]"
    )
    raise ImportError(msg) from _err

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from mcp.types import ContentBlock

    from ...mcp.client import MCPClient
    from .paths import AccessMode


# Canonical names — the grasp-agents :mod:`examples.mcp_memory_server`
# exposes these out of the box.
DEFAULT_RESOURCE_URI_SCHEME = "file://"
DEFAULT_WRITE_TOOL = "write_file"
DEFAULT_DELETE_TOOL = "delete_file"


class MCPFileBackend(FileBackend):
    """
    :class:`FileBackend` backed by an :class:`MCPClient`.

    Reads use :meth:`ClientSession.read_resource` for content and a
    cached :class:`MCPResourceIndex` for metadata.  Mutations use
    ``write_file`` / ``delete_file`` MCP tools and invalidate the index.

    Pass a *connected* :class:`MCPClient`; the backend re-uses its
    session and the client must remain connected for the backend's
    lifetime.

    Args:
        client: A connected :class:`MCPClient`.
        allowed_roots: Directories the backend will accept paths under,
            in the server's address space (e.g. ``[Path("/memdir")]``).
        resource_uri_scheme: URI scheme used to translate Paths to
            resource URIs. Default ``"file://"``. Must match the
            server's scheme.
        write_tool_name: Override the default ``write_file`` tool name.
        delete_tool_name: Override the default ``delete_file`` tool name.
        index: Optional pre-existing :class:`MCPResourceIndex` to share
            (saves a duplicate ``resources/list``).

    """

    def __init__(
        self,
        client: MCPClient,
        *,
        allowed_roots: list[Path | str],
        resource_uri_scheme: str = DEFAULT_RESOURCE_URI_SCHEME,
        write_tool_name: str = DEFAULT_WRITE_TOOL,
        delete_tool_name: str = DEFAULT_DELETE_TOOL,
        index: MCPResourceIndex | None = None,
    ) -> None:
        from pathlib import Path as _Path  # noqa: PLC0415

        self._client = client
        self.name = f"mcp:{client.name}"
        self._allowed_roots: list[Path] = [_Path(r) for r in allowed_roots]
        self._uri_scheme = resource_uri_scheme
        self._write_tool = write_tool_name
        self._delete_tool = delete_tool_name
        self._index = index or MCPResourceIndex(client, uri_scheme=resource_uri_scheme)

    @property
    def allowed_roots(self) -> list[Path]:
        return list(self._allowed_roots)

    def add_allowed_root(self, root: Path) -> None:
        from pathlib import Path as _Path  # noqa: PLC0415

        resolved = _Path(root)
        if any(resolved == r or r in resolved.parents for r in self._allowed_roots):
            return
        self._allowed_roots.append(resolved)

    @property
    def index(self) -> MCPResourceIndex:
        """Expose the resource index for sharing with other adapters."""
        return self._index

    # ---- Path / URI helpers --------------------------------------------------

    def _posix(self, path: Path) -> PurePosixPath:
        # ``normpath`` collapses ``./`` and ``../`` lexically (safe — MCP
        # paths live in the server's address space, no client-side
        # symlinks), so containment checks cannot be escaped via ``..``.
        return PurePosixPath(posixpath.normpath(PurePosixPath(path).as_posix()))

    def _to_uri(self, path: Path) -> str:
        return self._index.make_uri(self._posix(path))

    def _to_wire(self, path: Path) -> str:
        return str(self._posix(path))

    def _canonical(self, path: Path) -> Path:
        """
        Canonical key for read_file_state / write tracking.

        MCP paths live in the server's address space — there are no
        symlinks to follow client-side. We normalize lexically (collapses
        ``./``, ``../``, trailing slashes) and rebuild in the caller's
        Path flavor so two equivalent paths from different callers
        (e.g. ``Read`` post-validate_path vs. ``MemoryProvider`` loading
        the snapshot) hash to the same entry.
        """
        return type(path)(str(self._posix(path)))

    # ---- FileBackend protocol ------------------------------------------------

    async def validate_path(
        self,
        path: Path,
        *,
        must_exist: bool,
        access: AccessMode = "read",
        dotfile_overrides: set[Path] | None = None,
    ) -> Path:
        del access, dotfile_overrides  # MCP defers fs policy to the server

        if not self._allowed_roots:
            raise PathAccessError("No allowed_roots configured for MCP file backend.")

        candidate = self._posix(path)
        for root in self._allowed_roots:
            root_posix = self._posix(root)
            if candidate == root_posix or root_posix in candidate.parents:
                # Reconstruct in the caller's Path flavor (PosixPath /
                # WindowsPath / etc.) so they keep their preferred type.
                resolved = type(path)(str(candidate))
                break
        else:
            roots = ", ".join(str(r) for r in self._allowed_roots)
            raise PathAccessError(f"Path {path} is outside allowed roots [{roots}]")

        if must_exist and not await self.exists(resolved):
            raise PathAccessError(f"Path does not exist: {resolved}")

        return resolved

    async def stat(self, path: Path) -> FileStat:
        entry = await self._index.get(self._to_uri(path))
        if entry is None:
            return FileStat(mtime=0.0, mode=0, size=-1)
        return FileStat(
            mtime=entry.mtime_seconds,
            mode=0o644,
            size=max(entry.size, 0),
        )

    async def exists(self, path: Path) -> bool:
        return await self._index.get(self._to_uri(path)) is not None

    async def parent_exists(self, path: Path) -> bool:
        # MCP resources are flat; a directory "exists" iff at least one
        # listed resource sits under it. Sufficient for the Write-tool
        # parent-existence check, which only needs to refuse silent
        # parent-creation.
        parent = self._posix(path).parent
        if str(parent) == str(self._posix(path)):
            return True
        if await self._index.get(self._index.make_uri(parent)) is not None:
            return True
        children = await self._index.list_under(parent)
        return len(children) > 0

    async def read_text(self, path: Path) -> tuple[str, float]:
        uri = self._to_uri(path)
        session = self._client.session
        try:
            result = await session.read_resource(AnyUrl(uri))
        except Exception as exc:
            raise OSError(f"MCP resources/read for {uri!r} failed: {exc}") from exc
        for content in result.contents:
            if isinstance(content, TextResourceContents):
                # Report the mtime from the SAME surface the staleness guard
                # later consults via ``stat`` (the list index) — mixing it
                # with the read-meta surface causes spurious edit refusals
                # whenever the two disagree. Read-meta is the fallback for
                # files the index doesn't know yet.
                entry = await self._index.get(uri)
                if entry is not None:
                    mtime = entry.mtime_seconds
                else:
                    meta = content.meta or {}
                    mtime = _ms_to_seconds(meta.get("mtime_ms"))
                return content.text, mtime
        raise OSError(f"MCP resources/read for {uri!r} returned no text content.")

    async def read_bytes(self, path: Path) -> tuple[bytes, float]:
        content, mtime = await self.read_text(path)
        return content.encode("utf-8"), mtime

    async def write_bytes(
        self,
        path: Path,
        data: bytes,
        *,
        mode: int,
        overwrite: bool = True,
    ) -> float:
        del overwrite  # MCP servers always overwrite atomically
        result = await self._call_tool(
            self._write_tool,
            {
                "path": self._to_wire(path),
                "content": data.decode("utf-8"),
                "mode": mode,
            },
        )
        # Server mutation → stale index. Drop it; next read repopulates.
        await self._index.refresh()
        # Same-surface mtime as ``stat`` (see ``read_text``); the write
        # tool's response is the fallback if the index doesn't list the
        # file yet.
        entry = await self._index.get(self._to_uri(path))
        if entry is not None:
            return entry.mtime_seconds
        return _ms_to_seconds(result.get("mtime_ms"))

    async def delete(self, path: Path) -> None:
        await self._call_tool(self._delete_tool, {"path": self._to_wire(path)})
        await self._index.refresh()

    async def mkdir(self, path: Path) -> None:
        # MCP resource namespaces are flat: directories are implicit and
        # come into being when a resource is written under them. Nothing
        # to create.
        del path

    async def list_dir(self, path: Path, *, recursive: bool = False) -> list[FileEntry]:
        entries = await self._index.children_of(self._posix(path), recursive=recursive)
        return [
            FileEntry(
                name=e.path.name,
                path=type(path)(str(e.path)),
                is_dir=e.is_dir,
                mtime=e.mtime_seconds,
            )
            for e in entries
        ]

    async def find_files(
        self,
        root: Path,
        pattern: str,
        *,
        include_hidden: bool = False,
        head_limit: int = 250,
    ) -> tuple[list[FileEntry], bool]:
        """
        Glob over the resource index. MCP has no native glob; we use the
        cached ``resources/list`` (one round-trip per session) and
        filter client-side, sorted newest-first.
        """
        flat_entries = await self.list_dir(root, recursive=True)
        return glob_filter_entries(
            flat_entries,
            root,
            pattern,
            include_hidden=include_hidden,
            head_limit=head_limit,
        )

    async def grep(
        self,
        root: Path,
        pattern: str,
        *,
        glob: str | None = None,
        file_type: str | None = None,
        case_insensitive: bool = False,
        multiline: bool = False,
        output_mode: str = "files_with_matches",
        show_line_numbers: bool = True,
        before_context: int | None = None,
        after_context: int | None = None,
        context: int | None = None,
    ) -> GrepRawResult:
        del (
            root,
            pattern,
            glob,
            file_type,
            case_insensitive,
            multiline,
            output_mode,
            show_line_numbers,
            before_context,
            after_context,
            context,
        )
        raise NotImplementedError(
            "MCPFileBackend does not implement grep yet. The current "
            "file-tool protocol only mandates read/write/delete; servers "
            "that want grep would need to expose a dedicated tool. "
            "Use Glob (via find_files) + Read instead for content "
            "searches over MCP-backed memory directories."
        )

    # ---- Internals ----------------------------------------------------------

    async def _call_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        session = self._client.session
        try:
            result = await session.call_tool(tool_name, args)
        except Exception as exc:
            raise OSError(
                f"MCP tools/call {tool_name} for {args!r} failed: {exc}"
            ) from exc
        if getattr(result, "isError", False):
            raise OSError(
                f"MCP {tool_name} for {args!r} returned error: "
                f"{_text_payload(result.content)}"
            )
        return _parse_json_payload(result.content, tool_name)


def _ms_to_seconds(raw: Any) -> float:
    if raw is None:
        return 0.0
    try:
        return float(raw) / 1000.0
    except (TypeError, ValueError):
        return 0.0


def _text_payload(content: Sequence[ContentBlock]) -> str:
    return "\n".join(block.text for block in content if isinstance(block, TextContent))


def _parse_json_payload(
    content: Sequence[ContentBlock], tool_name: str
) -> dict[str, Any]:
    raw = _text_payload(content).strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OSError(
            f"MCP {tool_name} returned non-JSON content: {raw[:160]!r}"
        ) from exc
    if not isinstance(parsed, dict):
        raise OSError(f"MCP {tool_name} returned non-object JSON: {raw[:160]!r}")
    return cast("dict[str, Any]", parsed)
