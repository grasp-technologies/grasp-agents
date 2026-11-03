# Memory

Cross-session memory is a directory of markdown files (the *memdir*): an
always-loaded `MEMORY.md` index plus topic files with YAML frontmatter. A
`MemoryProvider` reads it through the session's `file_backend` and surfaces it
to the agent: the index in the system prompt, and relevant topic files on each
new input. The agent writes memories with the generic file tools; there are no
dedicated memory tools.

Memory is distinct from the transcript. The transcript (`agent.transcript`) is
one agent's conversation; memory (`SessionContext.memory`) is knowledge that
outlives sessions.

## Key types

| Name | Module | Role |
|---|---|---|
| `MemoryProvider` | `memory/provider.py` (root export) | Read-side view of a memdir over a `FileBackend` |
| `InMemoryMemoryProvider` | `memory/provider.py` | Fixed snapshot, no I/O (tests, notebooks) |
| `MemorySnapshot` | `memory/provider.py` | Frozen index + entries, loaded once |
| `MemoryEntry`, `MemoryFrontmatter`, `MemoryType` | `memory/types.py` | A topic file and its frontmatter (`MemoryEntry` is a root export) |
| `make_llm_relevance_selector` | `memory/selectors.py` | LLM-backed per-turn selector |
| `make_memory_section`, `relevant_memories_attachment` | `memory/injection.py` | The `memory` section and the `relevant_memories` attachment |
| `default_memdir_path` | `memory/default_path.py` | Default location |

## Layout

```
<memdir>/
  MEMORY.md              index: one line per topic file, always in the prompt
  user_role.md           topic files (any depth; dot-dirs are skipped)
  feedback_terse.md
```

A topic file:

```markdown
---
name: feedback_terse
description: The user wants answers without trailing summaries.
type: feedback
---

Don't end replies with a recap. **Why:** ... **How to apply:** ...
```

- `name` (required, at most 64 characters: lowercase letters, digits, `-`, `_`)
  and `description` (required, at most 2048 characters).
- `type` is one of `user`, `feedback`, `project`, `reference`; an unknown value
  is kept but treated as untyped.
- Freshness comes from the file's mtime, not a frontmatter field. Entries
  older than `stale_after` (default 7 days) get a freshness warning.
- The index is capped at 200 lines / 25,000 bytes in the prompt (a truncation
  marker is shown); at most 200 topic files are loaded, newest first.

The model-facing rules for writing memories (what to save, what not to save,
the two-step write: topic file, then index line) are in
`memory/memory_instructions.md`, rendered into the `memory` section.

## Location

`MemoryProvider(root=None)` uses `default_memdir_path()`:
`$GRASP_MEMORY_DIR` if set, else `~/.grasp/projects/<sanitized-cwd>/memory/`.
For an MCP backend, pass the path in the server's address space
(`MemoryProvider(root="/memdir")`).

## Wiring

```python
from grasp_agents import LLMAgent, MemoryProvider, SessionContext
from grasp_agents.file_backend import LocalFileBackend
from grasp_agents.memory import make_llm_relevance_selector

memory = MemoryProvider(root="~/.myapp/memory")
memory.set_selector(make_llm_relevance_selector(small_llm))

ctx = SessionContext[None](
    state=None,
    file_backend=LocalFileBackend(allowed_roots=[]),  # memdir only; None means cwd
    memory=memory,
)
with ctx:
    agent = LLMAgent[str, str, None](name="assistant", llm=llm, enable_memory=True)
```

`SessionContext` requires a `file_backend` when `memory` is set. Its validator
adds the memdir to the backend's allowed roots and binds the backend to the
provider. Any backend works: local, sandboxed, E2B or `MCPFileBackend`.

`enable_memory=True` (default `False`) adds three things:

1. the `memory` system-prompt section: the instructions plus the index in a
   `<memory-index>` block;
2. the `relevant_memories` input attachment;
3. the file tools (`FileToolkit().tools()`: `Read`, `Write`, `Edit`, `Delete`,
   `Glob`, `Grep`), skipped individually if you passed a tool of the same name.

The memdir is snapshotted on first use and the snapshot is cached on the
provider. When `MEMORY.md` is missing and `auto_create_index=True` (default),
a stub index is written; this is best effort and silently skipped on a
read-only backend. The index read is recorded in the agent's read-before-write
ledger, so the agent may `Edit` `MEMORY.md` without reading it first.

## Per-turn relevance

When the provider has a selector (`set_selector`), each new input message gets
a `<system-reminder subject="relevant_memories">` block with the selected
entries: a `### [type] name (updated <timestamp>)` heading, the description, any
freshness warning, and the body. Entries whose file the agent already read this
session (surfaced earlier or opened with `Read`) are filtered out before
selection, so a body is not injected twice. Without a selector there is no
attachment; the index alone tells the agent what exists.

`make_llm_relevance_selector(llm, max_select=5, max_tokens=256)` sends the
selector model one request: the latest user message (text plus images/files)
and a manifest of entry names, types, dates and descriptions — no bodies, no
history. It returns at most `max_select` entries. A failed call or an
unparseable answer selects nothing. Any callable matching the `Selector`
Protocol (`selector.py`) can be used instead.

## Trust

Memory is trusted content. The index and the relevant-memories attachment are
injected without `<untrusted_content>` fencing, with the same authority as the
system prompt. (When the agent opens a topic file with `Read`, that tool
result is fenced like any file read.)

What this implies:

- Anyone who can write to the memdir can instruct every future session. Keep
  it writable only by the agent and trusted operators; never point it at a
  location that users or other services write to.
- In a multi-tenant deployment, give each tenant its own memdir.
- An agent that both reads untrusted content and has memory enabled can be
  talked into saving an injected instruction. Gate writes to the memdir with
  approvals or a before-tool hook when that risk matters.
- `enable_memory` gives the agent file tools over the backend's whole allowed
  root set, not just the memdir. Scope the backend (or a sandbox policy) to
  what the agent should touch.

See [Security](security.md).

## Contracts

1. Memory I/O goes only through `ctx.file_backend` (`MemoryProvider._load_via_backend`).
2. The `memory` section reads the cached snapshot, so it is cache-stable within
   a session (`MemoryProvider.load`).
3. The system-prompt section never consults the selector; selection happens only
   on the input attachment.
4. Topic-file bodies are recorded as read only when actually surfaced
   (`MemoryProvider.fetch_body`), so read-before-write still applies to them.

## Limitations

- The snapshot is not refreshed during a session: memories the agent writes
  are on disk but do not appear in the index section or in selection until
  `await memory.refresh()` or the next session.
- Input attachments receive the transcript *before* the new input message, and
  `make_llm_relevance_selector` reads the latest user message from it. It
  therefore selects against the previous user turn, and selects nothing on the
  first turn of a conversation.
- No automatic consolidation or expiry of memories; stale entries only get a
  warning.

## See also

- [Files](files.md) (backends, read-before-write), [Skills](skills.md),
  [Security](security.md), [Hooks](hooks.md) (sections and attachments).
- Code: `memory/`, `selector.py`, `session_context.py`.
- Examples: `src/grasp_agents/examples/notebooks/memory_skills.ipynb`,
  `src/grasp_agents/examples/notebooks/mcp_memory.ipynb`.
