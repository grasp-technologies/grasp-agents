"""
Example MCP server backing a memdir on disk with the file-tool protocol.

Two surfaces:

* **Resources** — one per file under ``<root>``, with URI ``file://<absolute-path>``.
  The MCP-native read path. ``MEMORY.md`` is exposed alongside topic ``.md``
  files; the resource ``_meta`` field carries ``updated_ms`` (and ``type``
  for typed memories) so :class:`MCPMemoryProvider` can build its snapshot
  in one ``resources/list`` call.

* **Tools** — ``write_file``, ``stat_file``, ``delete_file``, ``list_dir``
  matching the protocol that :class:`MCPFileBackend` consumes. The agent
  reads via resources but writes / inspects via tools (resources are
  read-only by spec).

This is the reference implementation: a Postgres-backed or remote server
implementing the same tool / resource surface would be drop-in compatible.

Run::

    python /path/to/mcp_memory_server.py <root_dir>

The server speaks stdio per the MCP protocol; the demo notebook spawns it
via :class:`grasp_agents.MCPServerStdio` + :class:`grasp_agents.MCPClient`.

The ``examples/`` directory is intentionally NOT a Python package (no
``__init__.py``) — the demo notebook locates this file by path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from mcp.server.lowlevel import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    TextContent,
    Tool,
)
from pydantic import AnyUrl

logger = logging.getLogger(__name__)

INDEX_FILE_NAME = "MEMORY.md"
ALLOWED_TYPES = {"user", "feedback", "project", "reference"}


# ---------------------------------------------------------------------------
# Frontmatter helpers (only used to surface ``type`` in resource ``_meta``)
# ---------------------------------------------------------------------------


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    match = re.match(
        r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?(.*)\Z", text, re.DOTALL
    )
    if match is None:
        return {}, text
    raw_fm = yaml.safe_load(match.group(1)) or {}
    if not isinstance(raw_fm, dict):
        raw_fm = {}
    return raw_fm, match.group(2)


# ---------------------------------------------------------------------------
# Path helpers — keep all server-side paths under ``<root>`` strictly.
# ---------------------------------------------------------------------------


def _resolve_under_root(root: Path, path: str) -> Path:
    """Resolve ``path`` and refuse anything outside ``root``."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        msg = f"Path {path!r} escapes the memdir root {root}."
        raise ValueError(msg) from exc
    return resolved


# ---------------------------------------------------------------------------
# Resource surface
# ---------------------------------------------------------------------------


def _file_uri(path: Path) -> str:
    return f"file://{path.as_posix()}"


def _list_topic_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.glob("*.md") if p.name != INDEX_FILE_NAME)


def _build_resources(root: Path) -> list[Resource]:
    resources: list[Resource] = []
    index_path = root / INDEX_FILE_NAME
    if index_path.is_file():
        mtime_ms = int(index_path.stat().st_mtime * 1000)
        resources.append(
            Resource(
                uri=AnyUrl(_file_uri(index_path)),
                name=INDEX_FILE_NAME,
                description="MEMORY.md — the always-loaded memory index.",
                mimeType="text/markdown",
                _meta={"updated_ms": mtime_ms},
            )
        )
    for path in _list_topic_files(root):
        try:
            fm, _ = _split_frontmatter(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        name = str(fm.get("name") or path.stem)
        description = str(fm.get("description") or name)
        mtime_ms = int(path.stat().st_mtime * 1000)
        meta: dict[str, Any] = {"updated_ms": mtime_ms}
        if fm.get("type") in ALLOWED_TYPES:
            meta["type"] = fm["type"]
        resources.append(
            Resource(
                uri=AnyUrl(_file_uri(path)),
                name=name,
                description=description,
                mimeType="text/markdown",
                _meta=meta,
            )
        )
    return resources


def _read_resource_contents(root: Path, uri: AnyUrl) -> ReadResourceContents:
    uri_str = str(uri)
    if not uri_str.startswith("file://"):
        msg = f"Unknown resource scheme: {uri_str!r}"
        raise ValueError(msg)
    raw_path = uri_str[len("file://") :]
    resolved = _resolve_under_root(root, raw_path)
    if not resolved.is_file():
        msg = f"Resource not found: {uri_str}"
        raise FileNotFoundError(msg)
    text = resolved.read_text(encoding="utf-8")
    mtime_ms = int(resolved.stat().st_mtime * 1000)
    return ReadResourceContents(
        content=text,
        mime_type="text/markdown",
        meta={"updated_ms": mtime_ms},
    )


# ---------------------------------------------------------------------------
# File-tool surface (write, stat, delete, list_dir)
# ---------------------------------------------------------------------------


def _tool_write_file(
    root: Path, *, path: str, content: str, mode: int | None = None
) -> dict[str, Any]:
    resolved = _resolve_under_root(root, path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    if mode is not None:
        resolved.chmod(mode & 0o7777)
    mtime_ms = int(resolved.stat().st_mtime * 1000)
    return {"mtime_ms": mtime_ms}


def _tool_stat_file(root: Path, *, path: str) -> dict[str, Any]:
    try:
        resolved = _resolve_under_root(root, path)
    except ValueError:
        return {"exists": False}
    if not resolved.exists():
        return {"exists": False}
    st = resolved.stat()
    return {
        "exists": True,
        "mtime_ms": int(st.st_mtime * 1000),
        "mode": st.st_mode & 0o7777,
        "is_dir": resolved.is_dir(),
        "size": st.st_size,
    }


def _tool_delete_file(root: Path, *, path: str) -> dict[str, Any]:
    resolved = _resolve_under_root(root, path)
    if not resolved.exists():
        msg = f"File not found: {path!r}"
        raise FileNotFoundError(msg)
    resolved.unlink()
    return {}


def _tool_list_dir(
    root: Path, *, path: str, recursive: bool = False
) -> dict[str, Any]:
    resolved = _resolve_under_root(root, path)
    if not resolved.is_dir():
        msg = f"Not a directory: {path!r}"
        raise NotADirectoryError(msg)

    entries: list[dict[str, Any]] = []
    paths_iter = resolved.rglob("*") if recursive else resolved.iterdir()
    for p in paths_iter:
        try:
            st = p.stat()
        except OSError:
            continue
        entries.append(
            {
                "name": p.name,
                "path": str(p),
                "is_dir": p.is_dir(),
                "mtime_ms": int(st.st_mtime * 1000),
            }
        )
    return {"entries": entries}


# ---------------------------------------------------------------------------
# Server wiring
# ---------------------------------------------------------------------------


def build_server(root: Path) -> Server:
    """Build a low-level MCP :class:`Server` backed by ``root`` on disk."""
    app = Server("grasp-memory-demo")

    @app.list_resources()
    async def list_resources_handler() -> list[Resource]:  # noqa: RUF029
        return _build_resources(root)

    @app.read_resource()
    async def read_resource_handler(uri: AnyUrl) -> list[ReadResourceContents]:  # noqa: RUF029
        return [_read_resource_contents(root, uri)]

    @app.list_tools()
    async def list_tools_handler() -> list[Tool]:  # noqa: RUF029
        return [
            Tool(
                name="write_file",
                description=(
                    "Atomically write a UTF-8 text file. Parent directories "
                    "are created if missing. Returns the new mtime in ms."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "mode": {"type": "integer"},
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="stat_file",
                description=(
                    "Return file metadata: {exists, mtime_ms?, mode?, "
                    "is_dir?, size?}. ``exists`` is false for missing paths."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="delete_file",
                description="Remove a file. Errors when no file exists at the path.",
                inputSchema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="list_dir",
                description=(
                    "List a directory. ``recursive`` defaults to false. "
                    "Returns {entries: [{name, path, is_dir, mtime_ms}]}."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "recursive": {"type": "boolean"},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            ),
        ]

    @app.call_tool()
    async def call_tool_handler(  # noqa: RUF029
        name: str, arguments: dict[str, Any]
    ) -> list[TextContent]:
        if name == "write_file":
            result = _tool_write_file(
                root,
                path=arguments["path"],
                content=arguments["content"],
                mode=arguments.get("mode"),
            )
        elif name == "stat_file":
            result = _tool_stat_file(root, path=arguments["path"])
        elif name == "delete_file":
            result = _tool_delete_file(root, path=arguments["path"])
        elif name == "list_dir":
            result = _tool_list_dir(
                root,
                path=arguments["path"],
                recursive=bool(arguments.get("recursive", False)),
            )
        else:
            msg = f"Unknown tool: {name!r}"
            raise ValueError(msg)
        return [TextContent(type="text", text=json.dumps(result))]

    return app


async def serve(root: Path) -> None:
    app = build_server(root)
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "usage: python -m grasp_agents.examples.mcp_memory_server "
            "<root_dir>",
            file=sys.stderr,
        )
        sys.exit(2)
    root = Path(sys.argv[1]).expanduser().resolve()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    asyncio.run(serve(root))


if __name__ == "__main__":
    main()
