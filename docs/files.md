# Files

The file tools let an agent read, write, edit, copy, delete and search files through a pluggable `FileBackend` on `SessionContext.file_backend`: the host filesystem, an MCP server, an E2B sandbox, process memory, or several of these mounted as one tree. The tools enforce read-before-write and staleness checks, make each write conditional on the file being unchanged since it was read, match edits fuzzily, redact secrets from what the model reads, and keep notebooks intact. This page covers the tool set, its safety rules, composite backends, and the backend contract for writing your own backend.

## Key types

| Name | Module | Role |
|---|---|---|
| `FileToolkit` | `tools/file_toolkit.py` | Builds a matching set of file tools (`grasp_agents.tools`). |
| `ReadTool`, `WriteTool`, `EditTool`, `DeleteTool`, `CopyTool`, `ReadImageTool`, `NotebookReadTool`, `NotebookEditTool` | `tools/file_edit/` | Tool classes (names `Read`, `Write`, `Edit`, `Delete`, `Copy`, `ReadImage`, `NotebookRead`, `NotebookEdit`). |
| `GlobTool`, `GrepTool` | `tools/file_search/` | Search tools (`Glob`, `Grep`). |
| `FileEditSessionState` | `tools/file_edit/session_state.py` | The read-before-write ledger (`agent.agent_ctx.file_edit_state`). |
| `SecretRedactor`, `DefaultSecretRedactor`, `NullRedactor` | `tools/file_edit/redact.py` | Redaction of `Read` output. |
| `FileBackend` (ABC), `FileStat`, `FileEntry`, `GrepRawResult`, `StaleFileError` | `file_backend/base.py` | Backend contract; `StaleFileError` is a refused conditional write. |
| `LocalFileBackend`, `MCPFileBackend`, `InMemoryFileBackend` | `file_backend/local.py`, `file_backend/mcp.py`, `file_backend/memory.py` | Host filesystem; MCP server (`grasp-agents[mcp]`); process memory. |
| `E2BFileBackend` | `sandbox/e2b/file_backend.py` | E2B sandbox filesystem. |
| `CompositeFileBackend`, `Mount` | `file_backend/composite.py` | Several backends served as one tree. |
| `grep_texts`, `grep_texts_async` | `file_backend/text_grep.py` | Pure-Python grep for backends that hold file text. |
| `make_mounts_section` | `context/mounts_section.py` | The `mounts` system-prompt section. |
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

- `toolkit.tools()` returns `Read`, `Write`, `Edit`, `Delete`, `Glob`, `Grep`, plus `NotebookRead` + `NotebookEdit` (`include_notebook`, needs `nbformat`), `ReadImage` (`include_image`) and `Copy` (`include_copy`).
- `toolkit.read_only_tools()` returns `Read`, `Glob`, `Grep` (plus the notebook / image readers when enabled).
- With an `ExecutionEnvironment` on `ctx.environment`, the file backend is the environment's, co-located with `Bash`, or a `CompositeFileBackend` that mounts it (see [Sandbox](sandbox.md)).

## The tools

| Tool | Behavior |
|---|---|
| `Read` | `cat -n` line-numbered text with 1-based `offset` / `limit` (default 500 lines). Refuses binary extensions and files over 10 MB; truncates a window over 100,000 chars at a line boundary with a notice. Renders `.ipynb` as a cell view. Output is redacted and fenced as untrusted. Records the read. |
| `Write` | Creates or atomically overwrites a file. Creates missing parent directories. New files get mode `0o644`; overwrites keep the existing mode. |
| `Edit` | Replaces `old_string` with `new_string` in an existing file, using the fuzzy chain below. More than one match needs `replace_all=True`; otherwise it refuses and reports the count and strategy. |
| `Delete` | Removes one file (never a directory). |
| `Copy` | Copies a file or a directory tree without passing its content through the model (see [Copy](#copy)). Opt-in: `FileToolkit(include_copy=True)`. |
| `Glob` | Glob search returning the 250 most recently modified matches, newest first (`truncated` says whether more matched). Hidden directories are skipped unless `FileToolkit(glob_include_hidden=True)`; common build/cache directories (`node_modules`, `__pycache__`, `dist`, …) below the search path are always skipped. |
| `Grep` | Regex search with modes `files_with_matches` (default), `count`, `content` (with `-A` / `-B` / `-C`, line numbers). `LocalFileBackend` runs `rg`, which must be on `PATH`; `InMemoryFileBackend` searches in Python (`grep_texts`). Notes from the backend (a mount it could not search, a search cut short by its time budget) follow the results. Paginated with `head_limit` (250) and `offset`. |
| `ReadImage` | Returns a PNG / JPEG / GIF / WebP file as an image the model can see (up to 5 MB). |
| `NotebookRead`, `NotebookEdit` | Cell-structured notebook access (see [Notebooks](notebooks.md)). |

`Write`, `Edit`, `Delete` and `NotebookEdit` declare their path as a concurrency-conflict key, and `Copy` its destination, so two writes to the same path in one batch never interleave.

## Read-before-write

`Write` (on an existing file), `Edit`, `Delete`, `NotebookEdit`, `RunCell` and `Copy` (when it overwrites) refuse unless:

1. the file was read this session (`Read`, `NotebookRead`) — so the model never clobbers content it has not seen; and
2. its current `mtime` equals the one recorded at that read — otherwise the model is told to re-read.

The write itself is conditional too. The tool passes the recorded mtime to `write_bytes` / `delete` as `expected_mtime`, and the backend raises `StaleFileError`, changing nothing, when the file changed or vanished after the tool's check; the model is told to re-read. `RunCell` passes the mtime the notebook had before its cell ran, so the outputs never land on a notebook that changed meanwhile.

A successful write refreshes the record, so consecutive edits do not trip on their own writes. The ledger lives in the agent's `FileEditSessionState` (on `AgentContext`), not in the backend. It is checkpointed with the agent and rewound by rollback, and each agent and sub-agent has its own. Tools used outside an agent loop (no `agent_ctx`) skip the check and write unconditionally.

## Copy

`Copy` copies through `ctx.file_backend`, so bytes go from backend to backend and never through the model. Across the mounts of a `CompositeFileBackend`, it is how files get into a sandbox mount and how results come back.

- A file copied onto an existing directory lands inside it; otherwise `destination` is the new file's path, and its parent must exist. A directory's contents land under `destination`, which is created if missing (`a/b.txt` → `destination/a/b.txt`).
- Inside a directory, every file is validated for reading, must resolve to its own path under the source, and must be a regular file. A file that fails (refused by the policy, a symbolic link, a FIFO or device) is skipped and never read: `files_skipped` counts them and `skipped` lists the first 20 with the reason.
- The whole plan is checked before anything is written: at most `max_files` files (1,000) and `max_bytes` bytes (50 MiB), no existing destination file unless `overwrite=True`, and no existing file where the copy needs a directory. Inside an agent, overwriting a file also needs a current read record, as `Write` does, and the write is conditional on it.
- Every source is read before the first write, and the byte cap counts the bytes actually read.
- Copied files are recorded as read, so a following `Edit` needs no `Read`.

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
| `E2BFileBackend` | Built by `e2b_environment` (see [Sandbox](sandbox.md)). Paths are inside the sandbox; grep runs remotely; multiline grep is unsupported. A grep `type` selects the same files as ripgrep 14 for 32 common types (`py`, `js`, `ts`, `md`, `sh`, …) and raises on any other type; filter with `glob` there. `stat` reports file-type bits from the entry type. |
| `InMemoryFileBackend(allowed_roots, files=, max_file_bytes=, max_total_bytes=, grep_time_budget=)` | Files in a dict, gone with the process: a per-session scratch mount or a test double. Directories are implicit, and each root exists from the start. `files=` seeds content; a seed outside the roots, inside another seed or over a limit raises `ValueError`. A write or append past `max_file_bytes` or `max_total_bytes` raises `OSError` (`EFBIG` / `ENOSPC`). Writes are atomic, `expected_mtime` checks are exact, and every write gets a later mtime than the one before. `Grep` runs `grep_texts` within `grep_time_budget` (5 s by default). |
| `CompositeFileBackend(mounts, scratch=)` | Routes each call to a child backend by mount path (see [Composite backends](#composite-backends)). |

To serve your own store (a database, say), implement `FileBackend` or put the store behind an MCP server for `MCPFileBackend`, then mount it next to the sandbox with `CompositeFileBackend`.

### Writing a backend

Subclass `FileBackend` and implement every abstract method: `allowed_roots`, `add_allowed_root`, `validate_path`, `stat`, `exists`, `parent_exists`, `read_text`, `read_bytes`, `write_bytes`, `delete`, `mkdir`, `list_dir`, `find_files`, `grep`. Override `append_bytes` if you can append in place; the default rewrites the whole file.

- **Honor `expected_mtime`.** When `write_bytes` or `delete` gets one, raise `StaleFileError` without changing anything if the file is missing or its mtime differs. Check and write in one step where the store can (a conditional update); otherwise check right before writing. The file tools always pass the argument and rely on the check.
- **Search stored text.** A backend that holds file text can return `await grep_texts_async(files, pattern, root=root, ...)` from `grep`, which builds a `GrepRawResult` shaped like the local `rg` backend's. `find_files` can filter `list_dir(root, recursive=True)` through `glob_filter_entries` (`file_backend/local.py`) in a worker thread.
- **Notes.** Put caveats the model should see in `GrepRawResult.notes`; `Grep` prints them after the results.
- **Wrappers.** A backend that serves other backends overrides `includes`, which `SessionContext` uses to accept it next to an environment.

## Composite backends

A `CompositeFileBackend` serves one tree from several backends, each attached by a `Mount` at an absolute path:

```python
from pathlib import Path
from grasp_agents import MemoryProvider, SessionContext
from grasp_agents.file_backend import CompositeFileBackend, InMemoryFileBackend, Mount

backend = CompositeFileBackend(
    [
        Mount(Path("/user/memory"), memory_backend, description="what you know about the user"),
        Mount(Path("/user/notes"), notes_backend, read_only=True, description="the user's notes"),
        Mount(Path("/scratch"), InMemoryFileBackend(["/scratch"], max_total_bytes=100_000_000)),
    ],
    scratch=Path("/scratch"),
)
ctx = SessionContext(file_backend=backend, memory=MemoryProvider(root="/user/memory"))
```

- **Routing.** Every call goes to the mount whose path contains its target, and the path is passed on unchanged: each mount path must equal or sit under one of its backend's `allowed_roots` (checked at construction). Mount a local directory at its resolved (symlink-free) path. Relative paths resolve against the scratch mount.
- **Virtual directories.** Mounts never nest. Directories above the mount paths (`/` and `/user` in the example) exist and can be listed, globbed and grepped, but never written.
- **Read-only mounts** refuse writes both in `validate_path(access="write")` and in each mutating call. A path that resolves outside its mount, such as a symbolic link into another mount, is refused.
- **Scratch.** `allowed_roots` lists the `scratch` mount first (by default the first writable mount; it must be writable). Spilled tool output and background-task logs land there, and `Glob` / `Grep` calls without a path search it, rather than, say, the user's memory.
- **Searching above mounts.** `Glob` at a virtual directory keeps the newest matches across the mounts below it. `Grep` there searches those mounts concurrently and joins their results in path order. A mount it could not search (a backend without grep, an offline server) is named in a note after the results; the call fails only when every mount failed.
- **Memory.** The memdir must sit inside a mount: the composite's `add_allowed_root` admits no other path. `mkdir` of a mount's own path reaches its backend, so a memdir mounted from a new local directory is created on first use.
- **The prompt.** `LLMAgent` always registers the `mounts` system-prompt section (`make_mounts_section`). For a composite it lists each mount with its description and access, marks the one `Bash` and code execution can see, and points at `Copy` when the agent has it. For any other backend it renders nothing.

Next to an environment, `SessionContext` accepts a composite when a mount includes `environment.file_backend` (`FileBackend.includes`); see [Sandbox](sandbox.md#in-a-composite-backend).

## Contracts

1. Validate before any I/O: every tool resolves its path through `validate_path(path, access=...)` first, and backend I/O methods receive only validated paths (`FileBackend`).
2. `write_bytes` is atomic and creates missing parent directories; it returns the post-write mtime. Given `expected_mtime`, `write_bytes` and `delete` change nothing and raise `StaleFileError` when the file is missing or its mtime differs (`FileBackend.write_bytes`, `FileBackend.delete`).
3. Backends are pure I/O. The read/mtime ledger lives in the agent's `FileEditSessionState`, never in the backend.
4. A backend that cannot search raises `NotImplementedError` from `find_files` / `grep`, so tools can report a clear error.
5. `add_allowed_root(root)` is idempotent; `SessionContext` uses it to admit the memory directory automatically (`SessionContext._validate_memory_under_backend_roots`). A `CompositeFileBackend` adds no roots: it accepts a path inside a mount and raises `PathAccessError` for any other.
6. `Write` and `Edit` refuse `.ipynb` and point the model at `NotebookEdit`.
7. Read-before-write and staleness checks apply to every mutating file tool inside an agent loop, and the tool passes the read record's mtime to the backend as `expected_mtime` (`refuse_if_stale`).
8. Skills are read from the host filesystem (`pathlib`), not through `ctx.file_backend` (see [Skills](skills.md)).
9. A composite routes by mount path without translating it, never nests mounts, and refuses a path that resolves outside its mount (`CompositeFileBackend.validate_path`).
10. A read-only mount refuses every change, in `validate_path(access="write")` and in each mutating call (`CompositeFileBackend`).
11. A composite's `allowed_roots` begin with its writable scratch mount, where framework writes and path-less searches go (`CompositeFileBackend.allowed_roots`).
12. `Copy` checks its whole plan and reads every source before its first write, and reads only files that pass `validate_path` for reading (`CopyTool`).

## Limitations

- Writes take no backup of the previous content; recovery relies on your VCS or an environment snapshot ([Rollback](rollback.md)).
- `Delete` removes files only; deleting directories is left to the user.
- The ledger holds at most 1,000 read records per agent (oldest evicted), after which an old read must be repeated before writing.
- `MCPFileBackend` cannot grep, and mode bits may be ignored by the server.
- The local, E2B and MCP backends check `expected_mtime` right before writing, not in one step with the write, so a change landing in between is still overwritten. Creating a file is never conditional.
- `grep_texts` differs from `rg`: it uses the `regex` module's Python syntax instead of Rust's, ignores `.gitignore`, matches a glob containing `/` against the path under the search root, and knows only a subset of `rg`'s file types (`file_backend/text_grep.py`). It refuses patterns and globs over 1,000 characters and globs with more than 128 brace alternatives.
- A `grep_texts` search stops at its time budget (5 s by default) and returns only the files it finished, with a note saying so. A cancelled caller stops waiting at once, but the worker thread runs until the search ends or the budget runs out.
- Inside a directory, `Copy` skips files it may not read, symbolic links and special files instead of failing; check `skipped`. It holds every source in memory until written, and a write that fails partway (a size limit, a concurrent change) leaves the files already copied.
- `Glob` and `Grep` at a composite's virtual directory search every mount below it. `Glob` there lists each mount in full (a local mount's `.git` and `node_modules` included) before filtering, and leaves out a mount that cannot list its files, logging only a warning.
- `InMemoryFileBackend` keeps nothing past the process.

## See also

[Sandbox](sandbox.md) · [Notebooks](notebooks.md) · [MCP](mcp.md) · [Memory](memory.md) · [Security](security.md) · code: `tools/file_edit/`, `tools/file_search/`, `tools/file_toolkit.py`, `file_backend/`, `context/mounts_section.py`
