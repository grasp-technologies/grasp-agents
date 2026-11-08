# Durability

With a `CheckpointStore` on the `SessionContext`, every agent checkpoints at each closed boundary of its loop: after the input, after each tool round, after the answer. A crashed or restarted process resumes the session from the store. This page covers the store contract, what is persisted and when, the resume rules, schema versioning, and **deferred effects** — the way a tool's durable side effect lands together with the checkpoint that records the call.

Every guarantee here holds on the file and in-memory stores. A transactional store (e.g. one you write over your database) makes checkpoints atomic; it is an optional strengthening, never a requirement.

## Key types

| Name | Module | Role |
|---|---|---|
| `CheckpointStore` | `durability/checkpoint_store.py` | Store ABC. Root export. |
| `InMemoryCheckpointStore` | `durability/checkpoint_store.py` | For tests and short-lived sessions. Root export. |
| `FileCheckpointStore` | `durability/file_checkpoint_store.py` | One JSON file per key under a root directory. |
| `AgentCheckpoint`, `StepWatermark`, `AgentContextState` | `durability/checkpoints.py` | An agent's head: position, rollback boundaries, folds, cached output. |
| `SessionCheckpoint` | `durability/checkpoints.py` | The one session-scoped record per `session_key`. |
| `TaskRecord` | `durability/task_record.py` | A background task (see [Background tasks](background-tasks.md)). |
| `make_store_key`, `make_tool_call_path`, `session_prefix`, `task_prefix` | `durability/store_keys.py` | Key construction. |
| `prepare_messages_for_resume`, `InterruptionType` | `durability/resume.py` | Dangling-round repair. |
| `read_agent_histories`, `read_task_records`, `read_pending_messages` | `durability/session_history.py` | Side-effect-free readers over a persisted session. |
| `DeferredEffects`, `AgentContext.defer_effect` | `agent/deferred_effects.py`, `agent/agent_context.py` | The effects outbox. |

## Quick start

```python
from grasp_agents import LLMAgent, SessionContext
from grasp_agents.durability import FileCheckpointStore

ctx = SessionContext(
    state=load_state_from_db("user-42"),
    checkpoint_store=FileCheckpointStore(".grasp/checkpoints"),
    session_key="user-42",
)
agent = LLMAgent[str, str, MyState]("assistant", llm=llm, ctx=ctx)

await agent.run("Hello")   # checkpoints as it goes
# After a crash, in a new process with the same session_key:
await agent.run()          # no input: resume the session
```

## Key layout

Keys are `"<session_key>/<kind>[/<path>]"`, with kinds `agent`, `workflow`, `parallel`, `runner`, `team`, `task`, `session` and `mailbox`. A processor's path is its lineage (`[parent, ..., name]`). A call dispatched as a tool adds a `tc_<call_id>` segment, and parallel replicas use `<subproc>_<idx>`. Examples:

| Record | Key |
|---|---|
| A bare agent | `user-42/agent/assistant` |
| A sub-agent dispatched by `assistant`'s call `c7` | `user-42/agent/assistant/tc_c7` |
| That call's background task | `user-42/task/assistant/tc_c7` |
| The session record | `user-42/session` |

`exec_id` never appears in a key, so keys are stable across resumes. Sub-agents share the parent's `session_key`.

## What is persisted

- **Agent head** (`AgentCheckpoint`):
  - `current`, a `StepWatermark`: committed message count, log version, turn, step and filesystem ref, plus `AgentContextState` — the read-before-write ledger, shell cwd, deferred task-record flips, launch and mail high-water marks, and kernel context ids.
  - `step_watermarks`: rollback boundaries.
  - `folds` and `usage` (the latest response's reported usage, so a resumed agent sizes its context without re-estimating it).
  - `output` (the step's cached answer), `location`, `stop_reason`.
- **Message log**: the transcript, as an append-only JSONL log next to the head. The head stores only a watermark (`message_count`), never the messages.
- **Session record** (`SessionCheckpoint`, one per session): serialized `ctx.state` when `serialize_state=True`, the latest filesystem snapshot ref (`fs_snapshot_policy`), and `session_metadata`. `session_metadata` is write-only: persisted for inspection, never restored.
- **Task records** and **mailbox records**.
- Not persisted: the initial context (system prompt), which is recomposed every run, and `ctx.state` unless you opt in. Your application owns business state and rebuilds `state` when it constructs the `SessionContext`.

## When checkpoints are written

| Location | Written |
|---|---|
| `AFTER_INPUT` | At turn 0 of a new step, whenever a run starts with the log ending in input, and after any turn boundary that appended inbox mail or task notes or recorded a compaction fold — always before the model responds. |
| `AFTER_TOOL_RESULT` | After each tool round. |
| `AFTER_FINAL_ANSWER` | When the run ends with an answer (own or forced; `stop_reason` says which). |
| `AFTER_RESIDENT_ANSWER` | When a resident answers one inbox message. |
| `ROLLING_BACK`, `ROLLED_BACK` | Only by [rollback](rollback.md). |

Heads are written only at closed boundaries, so a committed prefix never ends in an unanswered tool call. Without a store a "save" still runs (deferred effects run, the in-memory head advances); nothing is persisted.

## Write order of one checkpoint

1. Filesystem snapshot, if due (session writer only).
2. Inside `store.transaction()`, in order:
   1. deferred effects;
   2. session record (session writer only);
   3. message log (append the delta, or rewrite to a new log version);
   4. agent head;
   5. task-record flips;
   6. inbox acks, when the store is `transactional` and holds the mailbox.
3. Cache the new head and retire the effects.
4. On a non-transactional store, ack the inbox messages the persisted turn absorbed.

A crash between steps leaves the filesystem and external artifacts at or after the transcript. Work may be redone; it is never silently missing.

## The message log

- The log is written before the head, so the head's `message_count` only ever undercounts. On load, only the first `message_count` records are trusted; an uncommitted or torn tail is trimmed.
- A full rewrite (after a resume trim, a reset, or a replaced transcript) goes to a new `log_version`. The head is re-pointed only after the rewrite lands, so a crash in between keeps the old head and old log paired.
- Rollback writes the head first, then truncates the log (`truncate_messages`), so the head never points past the log.

## Resume

- `ctx.load_checkpoint()` runs at the start of every processor run but does its work once per `SessionContext`: the outermost run restores the session record (persisted `state`, filesystem snapshot), and nested participants no-op.
- `agent.load_checkpoint()` reloads only on a **cold start**: the instance has never loaded or saved a checkpoint *and* its transcript is empty. A live session, a caller-seeded transcript, or a `reset_transcript()` is never clobbered by a reload.
- The reloaded log is passed through `prepare_messages_for_resume`. A trailing round whose calls lack results is dropped (normally a no-op, since heads sit at closed boundaries); completed rounds always survive.
- Then background tasks are re-spawned or reported, inbox leases dropped, and an interrupted rollback completed.
- On a cold start, a pure resume (`run()` with no input) of a completed step replays its cached output, and re-delivering the same `step=` the reloaded head holds is an idempotent redelivery: a completed step replays, an interrupted one continues. On a live instance, `run(..., step=n)` always starts a new delivery.

## Schema versioning

`CURRENT_SCHEMA_VERSION` is 16 and the oldest loadable version is 13. Every record carries `schema_version`.

- A record newer than the current version raises `CheckpointSchemaError`, and so does one below the floor.
- A record that fails to parse for any other reason is logged and treated as missing.
- Contributors: bump `CURRENT_SCHEMA_VERSION` together with a `SCHEMA_VERSION_SUMMARIES` entry for any persisted-shape change older code could not load.

## Implementing a store

Implement the seven abstract methods: `save`, `load`, `delete`, `list_keys`, `append_messages`, `read_messages`, `rewrite_messages`. `encode_messages` / `decode_message_log` give you JSONL framing; `read_messages` must tolerate a torn final record. Override `truncate_messages` when your backend can delete a tail directly (the default reads and rewrites).

To make checkpoints atomic, override `transaction()` and set `transactional = True`:

```python
import contextvars
from contextlib import asynccontextmanager
from grasp_agents import CheckpointStore

_conn = contextvars.ContextVar("checkpoint_conn", default=None)

class DbCheckpointStore(CheckpointStore):
    transactional = True

    @asynccontextmanager
    async def transaction(self):
        async with self._pool.acquire() as conn, conn.transaction():  # your driver
            token = _conn.set(conn)
            try:
                yield
            finally:
                _conn.reset(token)

    # save / load / ... write through _conn.get() when it is set
```

Keep the open transaction per task (a `ContextVar`), never on the store: one store is shared by agents that checkpoint concurrently, such as a parent and its background sub-agents.

## Deferred effects

A tool whose result the model sees now, but whose durable side effect (a database row, an artifact) must exist exactly when the transcript records the call, defers the effect:

```python
from grasp_agents import function_tool
from grasp_agents.agent import AgentContext

@function_tool
async def save_draft(title: str, body: str, *, agent_ctx: AgentContext) -> str:
    """Save the draft."""
    async def write() -> None:
        await drafts.upsert(key=title, body=body)   # idempotent by key
    agent_ctx.defer_effect(write)
    return f"Draft '{title}' saved."
```

- Effects run first inside the next checkpoint's `transaction()`, in registration order. A tool round's effects run in the `AFTER_TOOL_RESULT` checkpoint that persists the round.
- **Transactional store**, with the effect writing through the same transaction: the effects commit atomically with the checkpoint and are retired only after the whole unit lands.
- **Non-transactional store** (or no store): each effect is retired as soon as it succeeds. A crash after the effects but before the head lands replays the round: **at-least-once**.
- If the commit itself is interrupted, whether it landed is unknown; the effects stay pending and run again. **Effects must tolerate running twice** — upsert by a natural key, or check a version.
- A failing effect aborts the checkpoint (the run fails) and stays pending for the next checkpoint. Drop it with `agent.agent_ctx.effects.discard(call_id)`.
- Effects of calls that leave the transcript are dropped: a failed run's pruned round, calls cut by a rollback or a cancelled task's launch, a reset or replaced transcript. An effect deferred outside any tool call is never dropped.
- Effects must be self-contained closures over the values they write.
- Rollback never undoes an effect that already ran.

### `ctx.state` as a working copy

1. Rebuild `ctx.state` from your source of truth when you construct the `SessionContext` (or use `serialize_state=True` for plain pydantic / dataclass / mapping state).
2. Own what you mutate: tools change in `ctx.state` only data this session owns; read data owned elsewhere live from its source.
3. Write durable external state only through `defer_effect`, never directly from a tool.
4. Make each effect idempotent, or version-check at write time.
5. If a rollback must undo external writes, compensate in your application.

## Contracts

1. Store failures propagate and crash the run; never wrap a save in `try/except` (`LLMAgent.save_checkpoint`). The one tolerated failure is a corrupt record on load, treated as missing (`CheckpointStore.load_json`).
2. One live writer per agent checkpoint key: append offsets are tracked in memory, so two live instances on one key diverge the log. Hosts that cache agents need sticky routing or a lock (`AgentCheckpointPersistMixin`).
3. Build keys only with the `store_keys` helpers.
4. Enable the session-record features (`serialize_state`, `session_metadata`) in one process only; the record has no per-field merge (`SessionContext.save_checkpoint`).
5. Only the session writer writes the session record and snapshots the filesystem (see [Rollback](rollback.md#the-session-writer)).
6. `durability_enabled=False` on a processor opts it and all its descendants out of the store: no log, no heads, no task records (`Processor._inherit_durability`).
7. Input is durable before generation (`AgentLoop.execute_stream`).

## Limitations

- There are no branching or time-travel checkpoints; only linear [step rollback](rollback.md).
- The store does no retention: rollback boundaries grow by one per human turn, and old task records stay until `prune_delivered`.
- Pending tool approvals are not persisted (see [Agent loop](agent-loop.md)).
- `prompt_cache_key` round-trips through checkpoints but nothing populates it.

## See also

[Session](session.md) · [Rollback](rollback.md) · [Background tasks](background-tasks.md) · [Context window](context-window.md) · code: `durability/`, `agent/llm_agent.py` (`save_checkpoint`, `load_checkpoint`), `agent/deferred_effects.py`
