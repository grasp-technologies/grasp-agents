# Files

The file tools let an agent read, write, edit, delete and search files through a pluggable `FileBackend` on `SessionContext.file_backend`: the host filesystem, an MCP server, or an E2B sandbox. The tools enforce read-before-write and staleness checks, match edits fuzzily, redact secrets from what the model reads, and keep notebooks intact. This page covers the tool set, its safety rules, and the backend contract for writing your own backend.

## Key types

| Name | Module | Role |
|---|---|---|
| `FileToolkit` | `tools/file_toolkit.py` | Builds a matching set of file tools (`grasp_agents.tools`). |
| `ReadTool`, `WriteTool`, `EditTool`, `DeleteTool`, `ReadImageTool`, `NotebookReadTool`, `NotebookEditTool` | `tools/file_edit/` | Tool classes (names `Read`, `Write`, `Edit`, `Delete`, `ReadImage`, `NotebookRead`, `NotebookEdit`). |
| `GlobTool`, `GrepTool` | `tools/file_search/` | Search tools (`Glob`, `Grep`). |
| `FileEditSessionState` | `tools/file_edit/session_state.py` | The read-before-write ledger (`agent.agent_ctx.file_edit_state`). |
| `SecretRedactor`, `DefaultSecretRedactor`, `NullRedactor` | `tools/file_edit/redact.py` | Redaction of `Read` output. |
| `FileBackend` (ABC), `FileStat`, `FileEntry`, `GrepRawResult` | `file_backend/base.py` | Backend contract. |
| `LocalFileBackend`, `MCPFileBackend` | `file_backend/local.py`, `file_backend/mcp.py` | Host filesystem; MCP server (`grasp-agents[mcp]`). |
| `E2BFileBackend` | `sandbox/e2b/file_backend.py` | E2B sandbox filesystem. |
| `PathAccessError`, `atomic_write_bytes` | `file_backend/paths.py`, `file_backend/atomic_write.py` | Path policy errors; atomic writes. |

## Wiring

```python
from pathlib import Path
from grasp_agents import LLMAgent, SessionContext
from grasp_agents.file_backend import LocalFileBackend
from grasp_agents.tools import FileToolkit

ctx = SessionContext(file_backend=LocalFileBackend(allowed_roots=[Path("./project")]))
toolkit = FileToolkit(include_image=True, include_notebook=True)
agent = LLMAgent[str, str, None]("editor", llm=llm, ctx=ctx, tools=toolkit.tools())
```

- `toolkit.tools()` returns `Read`, `Write`, `Edit`, `Delete`, `Glob`, `Grep`, plus `NotebookRead` + `NotebookEdit` (`include_notebook`, needs `nbformat`) and `ReadImage` (`include_image`).
- `toolkit.read_only_tools()` returns `Read`, `Glob`, `Grep` (plus the notebook / image readers when enabled).
- With an `ExecutionEnvironment` on `ctx.environment`, the file backend is the environment's, co-located with `Bash` (see [Sandbox](sandbox.md)).

## The tools

| Tool | Behavior |
|---|---|
| `Read` | `cat -n` line-numbered text with 1-based `offset` / `limit` (default 500 lines). Refuses binary extensions and files over 10 MB; truncates a window over 100,000 chars at a line boundary with a notice. Renders `.ipynb` as a cell view. Output is redacted and fenced as untrusted. Records the read. |
| `Write` | Creates or atomically overwrites a file. Creates missing parent directories. New files get mode `0o644`; overwrites keep the existing mode. |
| `Edit` | Replaces `old_string` with `new_string` in an existing file, using the fuzzy chain below. More than one match needs `replace_all=True`; otherwise it refuses and reports the count and strategy. |
| `Delete` | Removes one file (never a directory). |
| `Glob` | Glob search, newest first, capped at 250 results. Hidden directories are skipped unless `FileToolkit(glob_include_hidden=True)`; common build/cache directories are always skipped. |
| `Grep` | Regex search with modes `files_with_matches` (default), `count`, `content` (with `-A` / `-B` / `-C`, line numbers). `LocalFileBackend` runs `rg`, which must be on `PATH`. Paginated with `head_limit` (250) and `offset`. |
| `ReadImage` | Returns a PNG / JPEG / GIF / WebP file as an image the model can see (up to 5 MB). |
| `NotebookRead`, `NotebookEdit` | Cell-structured notebook access (see [Notebooks](notebooks.md)). |

`Write`, `Edit`, `Delete` and `NotebookEdit` declare their path as a concurrency-conflict key, so two writes to the same path in one batch never interleave.

## Read-before-write

`Write` (on an existing file), `Edit`, `Delete`, `NotebookEdit` and `RunCell` refuse unless:

1. the file was read this session (`Read`, `NotebookRead`) — so the model never clobbers content it has not seen; and
2. its current `mtime` equals the one recorded at that read — otherwise the model is told to re-read.

A successful write refreshes the record, so consecutive edits do not trip on their own writes. The ledger lives in the agent's `FileEditSessionState` (on `AgentContext`), not in the backend. It is checkpointed with the agent and rewound by rollback, and each agent and sub-agent has its own. Tools used outside an agent loop (no `agent_ctx`) skip the check.

## Fuzzy edit chain

LLM-written `old_string`s drift: trailing whitespace, tabs vs spaces, smart quotes, double-escaped newlines. `Edit` tries nine strategies in order, strict first, and stops at the first that matches:

1. `exact`
2. `line_trimmed`
3. `whitespace_normalized`
4. `indentation_flexible`
5. `escape_normalized`
6. `trimmed_boundary`
7. `unicode_normalized`
8. `block_anchor` (exact first/last lines, similar middle)
9. `context_aware` (per-line similarity)

Replacements preserve the file's conventions. If the matched region uses curly quotes and `new_string` uses straight ones, the quotes are converted. The replacement's line endings follow the replaced region's (or the file's dominant style), so a CRLF file stays CRLF.

## Path safety and redaction

`LocalFileBackend.validate_path` checks, in order:

1. Blocked device paths (`/dev/zero`, `/dev/stdin`, …).
2. Containment under `allowed_roots` (after `~` expansion and symlink resolution).
3. The sensitive-path deny list: system paths always (`/etc/`, `/boot/`, Docker sockets, …); credential dotfiles (`.ssh/`, `.aws/`, `.gnupg/`, `.kube/`, `.docker/`, `.env*`, `.netrc`, `.pypirc`) unless `include_dotfiles=False`.
4. `deny_read` / `allow_read` / `deny_write` carve-outs; `readonly_roots` are readable and write-denied.

A per-session opt-in bypasses the dotfile list for one path: `agent.agent_ctx.file_edit_state.add_dotfile_override(resolved_path)`. System paths cannot be overridden.

`Read` output passes through a `SecretRedactor` (default `DefaultSecretRedactor`: AWS keys, GitHub tokens, OpenAI / Anthropic keys, JWTs, private-key blocks → `<REDACTED:KIND>`). Pass `FileToolkit(redactor=NullRedactor())` to disable it, or your own callable.

## Backends

| Backend | Notes |
|---|---|
| `LocalFileBackend(allowed_roots=, readonly_roots=, include_dotfiles=, deny_read=, allow_read=, deny_write=)` | Host filesystem. Atomic writes (temp file + `fsync` + `os.replace`); true appends; `Grep` via `rg`. Defaults to `[Path.cwd()]`. |
| `MCPFileBackend(client=, allowed_roots=)` | Reads through MCP resources; writes and deletes through server tools (`write_file`, `delete_file`). The server owns containment; the backend checks only `allowed_roots`. No grep (`NotImplementedError`); use `Glob` + `Read`. |
| `E2BFileBackend` | Built by `e2b_environment` (see [Sandbox](sandbox.md)). Paths are inside the sandbox; grep runs remotely; multiline grep is unsupported. |

Non-local stores are best exposed as an MCP server through `MCPFileBackend`; the framework does not project application state as files.

### Writing a backend

Subclass `FileBackend` and implement every abstract method: `allowed_roots`, `add_allowed_root`, `validate_path`, `stat`, `exists`, `parent_exists`, `read_text`, `read_bytes`, `write_bytes`, `delete`, `mkdir`, `list_dir`, `find_files`, `grep`. Override `append_bytes` if you can append in place; the default rewrites the whole file.

## Contracts

1. Validate before any I/O: every tool resolves its path through `validate_path(path, access=...)` first, and backend I/O methods receive only validated paths (`FileBackend`).
2. `write_bytes` is atomic and creates missing parent directories; it returns the post-write mtime (`FileBackend.write_bytes`).
3. Backends are pure I/O. The read/mtime ledger lives in the agent's `FileEditSessionState`, never in the backend.
4. A backend that cannot search raises `NotImplementedError` from `find_files` / `grep`, so tools can report a clear error.
5. `add_allowed_root(root)` is idempotent; `SessionContext` uses it to admit the memory directory automatically (`SessionContext._validate_memory_under_backend_roots`).
6. `Write` and `Edit` refuse `.ipynb` and point the model at `NotebookEdit`.
7. Read-before-write and staleness checks apply to every mutating file tool inside an agent loop.
8. Skills are read from the host filesystem (`pathlib`), not through `ctx.file_backend` (see [Skills](skills.md)).

## Limitations

- Writes take no backup of the previous content; recovery relies on your VCS or an environment snapshot ([Rollback](rollback.md)).
- `Delete` removes files only; deleting directories is left to the user.
- The ledger holds at most 1,000 read records per agent (oldest evicted), after which an old read must be repeated before writing.
- `MCPFileBackend` cannot grep, and mode bits may be ignored by the server.

## See also

[Sandbox](sandbox.md) · [Notebooks](notebooks.md) · [MCP](mcp.md) · [Memory](memory.md) · [Security](security.md) · code: `tools/file_edit/`, `tools/file_search/`, `tools/file_toolkit.py`, `file_backend/`
