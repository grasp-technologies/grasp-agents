"""
:class:`MCPFileBackend` — routes file I/O to an MCP server.

Reads use MCP's native **resources** surface (``resources/read``);
mutations and metadata operations go through MCP **tools** (``write_file``,
``stat_file``, ``delete_file``, ``list_dir``). This mirrors the MCP
protocol's design: resources are the canonical read-only browse surface,
tools carry side effects.

Sandbox enforcement (path containment, sensitive-path policy, etc.)
lives on the server side; the backend's :meth:`validate_path` only
enforces ``allowed_roots`` membership in the server's address space.
Device-path and credential-dotfile checks are local-FS concerns and are
skipped.

MCP file-tool protocol
======================

The server SHOULD expose the following surface. Tool result blocks
contain a JSON object with the named keys.

Resources
---------

For every file under the configured root, expose a resource with URI
``<scheme><path>`` (default ``file://<path>``). ``TextResourceContents``
SHOULD include ``_meta.updated_ms`` so the backend can populate
``mtime`` without a second round-trip.

Tools
-----

``write_file(path: str, content: str, mode?: int) -> {mtime_ms: int}``
    Atomically replace the file content. The server SHOULD ensure
    parent-directory existence (or refuse with a clear error).

``stat_file(path: str) -> {exists, mtime_ms?, mode?, is_dir?, size?}``
    Return file metadata. When ``exists`` is false the other keys MAY be
    absent.

``delete_file(path: str) -> {}``
    Remove the file. Server errors propagate as :class:`OSError`.

``list_dir(path: str, recursive?: bool) -> {entries: [{name, path, is_dir, mtime_ms}]}``
    List a directory. ``recursive`` defaults to false.

Tool / URI prefixes are configurable in case the server namespaces its
surface.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from .backend import FileEntry, FileStat, GrepRawResult, glob_filter_entries
from .paths import PathAccessError

try:
    from mcp.types import TextContent, TextResourceContents
    from pydantic import AnyUrl
except ImportError as _err:
    msg = (
        "MCP file backend requires the 'mcp' package. "
        "Install with: pip install grasp-agents[mcp]"
    )
    raise ImportError(msg) from _err

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mcp import ClientSession

    from ...mcp.client import MCPClient


# Canonical names — the grasp-agents :mod:`examples.mcp_memory_server`
# exposes these out of the box.
DEFAULT_RESOURCE_URI_SCHEME = "file://"
DEFAULT_WRITE_TOOL = "write_file"
DEFAULT_STAT_TOOL = "stat_file"
DEFAULT_DELETE_TOOL = "delete_file"
DEFAULT_LIST_TOOL = "list_dir"


class MCPFileBackend:
    """
    :class:`FileBackend` backed by an :class:`MCPClient`.

    Reads use :meth:`ClientSession.read_resource`; mutations + metadata
    operations use :meth:`ClientSession.call_tool`. Pass a *connected*
    :class:`MCPClient`; the backend re-uses its session and the client
    must remain connected for the backend's lifetime.

    Args:
        client: A connected :class:`MCPClient`.
        resource_uri_scheme: URI scheme prepended to file paths to form
            the resource URI. Default ``"file://"``.
        write_tool_name: Override the default ``write_file`` tool name.
        stat_tool_name: Override the default ``stat_file`` tool name.
        delete_tool_name: Override the default ``delete_file`` tool name.
        list_tool_name: Override the default ``list_dir`` tool name.

    """

    def __init__(
        self,
        client: MCPClient,
        *,
        resource_uri_scheme: str = DEFAULT_RESOURCE_URI_SCHEME,
        write_tool_name: str = DEFAULT_WRITE_TOOL,
        stat_tool_name: str = DEFAULT_STAT_TOOL,
        delete_tool_name: str = DEFAULT_DELETE_TOOL,
        list_tool_name: str = DEFAULT_LIST_TOOL,
    ) -> None:
        self._client = client
        self._uri_scheme = resource_uri_scheme
        self._write_tool = write_tool_name
        self._stat_tool = stat_tool_name
        self._delete_tool = delete_tool_name
        self._list_tool = list_tool_name

    @property
    def name(self) -> str:
        return f"mcp:{self._client.name}"

    async def validate_path(
        self,
        path: str,
        allowed_roots: list[str],
        *,
        must_exist: bool,
        dotfile_overrides: Iterable[str] | None = None,
        include_dotfiles: bool = True,
    ) -> str:
        del dotfile_overrides, include_dotfiles

        if not allowed_roots:
            raise PathAccessError(
                "No allowed_roots configured for MCP file backend."
            )
        if not path:
            raise PathAccessError("Empty path is not allowed.")

        # The server resolves the path in its own filesystem. We only
        # enforce textual containment under one of the configured
        # roots to keep accidental wildcard escapes contained.
        candidate = _normalize(path)
        for root in allowed_roots:
            normalized_root = _normalize(root)
            if candidate == normalized_root or candidate.startswith(
                normalized_root.rstrip("/") + "/"
            ):
                resolved = candidate
                break
        else:
            roots = ", ".join(allowed_roots)
            raise PathAccessError(
                f"Path {path!r} is outside allowed roots [{roots}]"
            )

        if must_exist and not await self.exists(resolved):
            raise PathAccessError(f"Path does not exist: {resolved}")

        return resolved

    async def stat(self, path: str) -> FileStat:
        result = await self._call_tool(self._stat_tool, {"path": path})
        exists = bool(result.get("exists", False))
        if not exists:
            return FileStat(mtime=0.0, mode=0, size=-1)
        return FileStat(
            mtime=_ms_to_seconds(result.get("mtime_ms")),
            mode=int(result.get("mode") or 0o644),
            size=int(result.get("size") or 0),
        )

    async def exists(self, path: str) -> bool:
        result = await self._call_tool(self._stat_tool, {"path": path})
        return bool(result.get("exists", False))

    async def parent_exists(self, path: str) -> bool:
        # POSIX-style parent. Empty / root paths short-circuit to True.
        idx = path.rstrip("/").rfind("/")
        if idx <= 0:
            return True
        parent = path[:idx]
        result = await self._call_tool(self._stat_tool, {"path": parent})
        return bool(result.get("exists", False)) and bool(
            result.get("is_dir", False)
        )

    async def read_text(self, path: str) -> tuple[str, float]:
        uri = f"{self._uri_scheme}{path}"
        session = self._session()
        try:
            result = await session.read_resource(AnyUrl(uri))
        except Exception as exc:
            raise OSError(
                f"MCP resources/read for {uri!r} failed: {exc}"
            ) from exc
        for content in result.contents:
            if isinstance(content, TextResourceContents):
                meta = getattr(content, "meta", None) or {}
                mtime_ms = meta.get("updated_ms") if isinstance(meta, dict) else None
                return content.text, _ms_to_seconds(mtime_ms)
        raise OSError(
            f"MCP resources/read for {uri!r} returned no text content."
        )

    async def read_bytes(self, path: str) -> tuple[bytes, float]:
        content, mtime = await self.read_text(path)
        return content.encode("utf-8"), mtime

    async def write_bytes(
        self, path: str, data: bytes, *, mode: int, overwrite: bool = True
    ) -> float:
        del overwrite  # MCP servers always overwrite atomically
        result = await self._call_tool(
            self._write_tool,
            {"path": path, "content": data.decode("utf-8"), "mode": mode},
        )
        return _ms_to_seconds(result.get("mtime_ms"))

    async def delete(self, path: str) -> None:
        await self._call_tool(self._delete_tool, {"path": path})

    async def list_dir(
        self, path: str, *, recursive: bool = False
    ) -> list[FileEntry]:
        result = await self._call_tool(
            self._list_tool, {"path": path, "recursive": recursive}
        )
        raw_entries = result.get("entries") or []
        if not isinstance(raw_entries, list):
            raise OSError(
                f"MCP {self._list_tool} returned non-list entries for {path!r}."
            )
        entries: list[FileEntry] = []
        for raw in raw_entries:
            if not isinstance(raw, dict):
                continue
            entries.append(
                FileEntry(
                    name=str(raw.get("name", "")),
                    path=str(raw.get("path", "")),
                    is_dir=bool(raw.get("is_dir", False)),
                    mtime=_ms_to_seconds(raw.get("mtime_ms")),
                )
            )
        return entries

    async def find_files(
        self,
        root: str,
        pattern: str,
        *,
        include_hidden: bool = False,
        head_limit: int = 250,
    ) -> tuple[list[FileEntry], bool]:
        """
        Glob walk on the MCP server by listing the directory recursively
        and filtering client-side. Servers MAY ship an optimized
        ``find_files`` tool in the future; this fallback handles the
        common memdir-sized case without one.
        """
        entries = await self.list_dir(root, recursive=True)
        return glob_filter_entries(
            entries,
            root,
            pattern,
            include_hidden=include_hidden,
            head_limit=head_limit,
        )

    async def grep(
        self,
        root: str,
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
            "file-tool protocol only mandates read/write/stat/delete/"
            "list_dir; servers that want grep would need to expose a "
            "dedicated tool. Use Glob (via find_files) + Read instead "
            "for content searches over MCP-backed memory directories."
        )

    # ---- Internals ----------------------------------------------------------

    async def _call_tool(
        self, tool_name: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        session = self._session()
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

    def _session(self) -> ClientSession:
        session: ClientSession | None = (
            self._client._session  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        )
        if session is None:
            msg = (
                f"MCPClient {self._client.name!r} is not connected; "
                "call connect() or use it as an async context manager first."
            )
            raise RuntimeError(msg)
        return session


def _ms_to_seconds(raw: Any) -> float:
    if raw is None:
        return 0.0
    try:
        return float(raw) / 1000.0
    except (TypeError, ValueError):
        return 0.0


def _text_payload(content: Any) -> str:
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for block in content:
        if isinstance(block, TextContent):
            parts.append(block.text)
    return "\n".join(parts)


def _parse_json_payload(content: Any, tool_name: str) -> dict[str, Any]:
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
        raise OSError(
            f"MCP {tool_name} returned non-object JSON: {raw[:160]!r}"
        )
    return parsed


def _normalize(path: str) -> str:
    """POSIX-style normalization — collapse repeated separators and ``.``."""
    is_absolute = path.startswith("/")
    parts = [p for p in path.split("/") if p and p != "."]
    out = "/".join(parts)
    if is_absolute:
        out = "/" + out
    return out or ("/" if is_absolute else ".")
