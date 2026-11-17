# Session

A `SessionContext` is the session-scoped container every processor in one session shares: application state, the checkpoint store, the file and exec environment, the approval store, skills, memory, the mailbox, the usage tracker and printer. An `AgentContext` is its per-agent counterpart: the state one agent's tools operate on. This page explains how processors bind to a session, how a session's resources live and die, and how tools reach per-agent state without holding any themselves.

## Key types

| Name | Module | Role |
|---|---|---|
| `SessionContext[CtxT]` | `session_context.py` | Session-scoped DI container (a pydantic model). Root export. |
| `current_session_context()` | `session_context.py` | The ctx a bare-built processor binds to. |
| `reset_default_session_context()` | `session_context.py` | Drop the process-wide default (for test isolation). |
| `Processor.on_adopted` | `processors/processor.py` | Rebinds a child's ctx/path and cascades it down. |
| `Processor.aclose` / `async with` | `processors/processor.py` | Release session-scoped resources. |
| `AgentContext` | `agent/agent_context.py` | Per-agent state passed to every tool call (`agent.agent_ctx`). |

## What a `SessionContext` holds

| Field | Purpose |
|---|---|
| `state: CtxT` | Your application state, shared by every processor and tool in the session. |
| `session_key` | Session identifier; namespaces checkpoint keys and the `SESSION` approval allowlist. Default `"default"`. |
| `checkpoint_store` | Where checkpoints go (`None`: in-memory heads only). See [Durability](durability.md). |
| `serialize_state`, `session_metadata`, `fs_snapshot_policy`, `session_writer` | Session-record features. See [Durability](durability.md) and [Rollback](rollback.md). |
| `environment` | An `ExecutionEnvironment` — the only way to grant command execution. See [Sandbox](sandbox.md). |
| `file_backend` | File I/O for file tools and memory. Taken from `environment` when one is set. See [Files](files.md). |
| `approval_store` | Enables `build_store_approval` gates. See [Agent loop](agent-loop.md). |
| `skills`, `memory` | See [Skills](skills.md) and [Memory](memory.md). |
| `transport` | The session's mailbox (durable over `checkpoint_store` when there is one). |
| `usage_tracker`, `printer`, `responses` | Token accounting, raw debug printing, recent responses per agent. |

`exec_backend` is a read-only property: `ctx.environment.exec_backend`.

## Binding: which session a processor uses

A processor reads its ctx **once, in `__init__`**:

1. An explicit `ctx=` argument.
2. Else the innermost active `with SessionContext(...)` block.
3. Else a lazily created process-wide default.

```python
from grasp_agents import LLMAgent, SessionContext, SequentialWorkflow

with SessionContext(state=MyState(), session_key="user-42") as ctx:
    planner = LLMAgent[str, str, MyState]("planner", llm=llm)
    writer = LLMAgent[str, str, MyState]("writer", llm=llm)
    pipeline = SequentialWorkflow[str, str, MyState](
        name="pipeline", subprocs=[planner, writer]
    )
# All three are bound to ctx; building them later, outside the block, would not.
```

The ambient binding uses a `ContextVar`, so concurrent tasks in one process each see their own block.

Containers (workflows, `ParallelProcessor`, `Runner`, agents wrapping sub-agents as tools) cascade their ctx **top-down** with `child.on_adopted(parent)`, overriding whatever the child was built with. `ProcessorTool` / `AgentTool` clone the wrapped processor per call and bind the clone to the calling session. `SessionContext.__deepcopy__` returns `self`, so copies and replicas never fork the session.

## Session-scoped resources

An agent's shells (`BashSession`), kernels (`RunPython`, `RunCell`), fresh-`Bash` working directory and background tasks live on its `AgentContext` and survive run boundaries. A run is not a resource boundary: no run end, MAX_TURNS or TIMEOUT releases them. Only `aclose()` does.

```python
async with agent:              # __aexit__ calls agent.aclose()
    await agent.run("step 1")
    await agent.run("step 2")  # same shell, same kernel, same background tasks
```

`LLMAgent.aclose()` calls `AgentContext.close()`, in order:

1. Cancel background tasks (their durable records are marked CANCELLED).
2. Close the shell and both kernel holders.
3. Cascade to tools that wrap processors.

Workflows, `ParallelProcessor` and `Runner` cascade `aclose()` to their children. A per-call clone made by `ProcessorTool` / `AgentTool` is closed when its call ends.

## `AgentContext`

One per `LLMAgent`, created at construction, exposed as `agent.agent_ctx`, and passed to every tool call as `agent_ctx`:

| Field | Purpose |
|---|---|
| `cw` | The `ContextWindowManager` that owns the transcript (see [Context window](context-window.md)). `transcript` is its read-only view. |
| `tools` | The agent's tool map (sub-agent tools read it as their parent's). |
| `file_edit_state` | The read-before-write ledger (see [Files](files.md)). |
| `bg_tasks` | The `BackgroundTaskManager` (see [Background tasks](background-tasks.md)). |
| `session_holder`, `shell_state` | `BashSession`'s persistent shell; fresh `Bash`'s working directory. |
| `nb_kernel_holder`, `ipy_kernel_holder` | The `RunCell` and `RunPython` kernels (see [Notebooks](notebooks.md)). |
| `inbox` | A resident agent's inbox (`None` for lone agents). |
| `effects` | The deferred-effects outbox (see [Durability](durability.md#deferred-effects)). |

The loop hands each call `agent_ctx.for_tool_call(call_id)`: a shallow copy sharing all state and differing only in `tool_call_id`. That id scopes `defer_effect` and attributes the call's events. Backgrounded calls get the same per-call copy.

A tool that needs per-agent state reads it from `agent_ctx`, never from `self`:

```python
from grasp_agents import SessionContext, function_tool
from grasp_agents.agent import AgentContext

@function_tool
async def note_progress(text: str, *, ctx: SessionContext[MyState], agent_ctx: AgentContext) -> str:
    """Record progress for this session."""
    ctx.state.notes.append(text)              # session state: in-memory working copy
    turns = len(agent_ctx.transcript)         # per-agent state: from the call context
    return f"Noted after {turns} messages."
```

## State as a working copy

`ctx.state` is the session's in-memory working copy. The application owns business state: rebuild `state` from your database when you construct the `SessionContext` (or opt into `serialize_state=True` for plain containers). Mutate it freely in tools and hooks. Write durable external state only through `agent_ctx.defer_effect`, so the write lands with the checkpoint that records the call. See the [recipe](durability.md#ctxstate-as-a-working-copy).

## Contracts

1. Binding order is explicit `ctx=` → innermost `with SessionContext` → process default, read once at construction (`Processor.__init__`, `current_session_context`).
2. ctx flows strictly top-down. Never pass `ctx=` to a subprocessor: its container's `on_adopted` overrides it. Set the ctx on the outermost processor or with a `with` block.
3. `SessionContext.__deepcopy__` returns `self`; copying a processor never forks its session.
4. Runs never release session-scoped resources; only `aclose()` / `async with` does, cascading through containers and wrapping tools (`LLMAgent.aclose`, `AgentContext.close`).
5. Tools are stateless. All per-agent state arrives as `agent_ctx` per call, so one tool instance is safe to share across agents, sub-agents and replicas (`BaseTool` docstring, `AgentContext.for_tool_call`).
6. Exec is granted only through `ctx.environment`; a `file_backend` that differs from `environment.file_backend` raises at construction (`SessionContext._reconcile_environment`).

## Limitations

- **Multi-tenant servers must scope construction per request.** Two agents built outside any `with` block share the process default: one usage tracker, one `state`, one store binding. Build each request's processors inside `with SessionContext(per_request):`, pass `ctx=` to the top-level processor, or rebind a freshly built agent with `agent.on_adopted(ctx=per_request_ctx)` before its first run.
- Rebinding does not reset an agent's live transcript or `AgentContext`. Don't move an agent that has already run to another session; an agent instance serves one session, and each checkpoint key allows one live writer (see [Durability](durability.md)).
- There is no idle reaper: an agent you never close keeps its shells, kernels and background tasks alive until the process exits.
- `RunContext`, `current_run_context` and `reset_default_run_context` are deprecated aliases of the `SessionContext` names.

## See also

[Concepts](concepts.md) · [Agent loop](agent-loop.md) · [Durability](durability.md) · [Orchestration](orchestration.md) · code: `session_context.py`, `processors/processor.py`, `agent/agent_context.py`
