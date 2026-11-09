# Rollback

Step rollback rewinds an agent's session to the start of an earlier step: the transcript and its durable log, the agent's paired state, its side channels (background tasks, inbox mail) and — when the session snapshots it — the shared filesystem. Use it for "undo my last message" or "go back to before turn N". This page also covers the **session writer**, the one processor that owns session-scoped persistence, and filesystem snapshots.

## Key types

| Name | Module | Role |
|---|---|---|
| `LLMAgent.rollback_to_step(step)` | `agent/llm_agent.py` | Rewind to the start of `step`. |
| `LLMAgent.rollback_steps` | `agent/llm_agent.py` | Steps with a recorded boundary (valid targets). |
| `LLMAgent.read_rollback_anchors()` | `agent/llm_agent.py` | Boundaries read from the persisted head, without rehydrating. |
| `LLMAgent.parked_after_rollback` | `agent/llm_agent.py` | True from a rollback until the step is re-delivered. |
| `StepWatermark` | `durability/checkpoints.py` | One rollback boundary. |
| `SessionContext.session_writer`, `claim_session_writer` | `session_context.py` | The session-persistence owner. |
| `SessionContext.fs_snapshot_policy`, `restore_fs_snapshot` | `session_context.py` | Filesystem snapshots. |
| `SnapshotCapable` | `sandbox/environment.py` | Environment capability: `snapshot()` / `restore(ref)`. |

## Steps and anchors

A **step** is one delivery to an agent, identified by an integer. Before a step's input is appended, the agent archives a boundary: the transcript length, the agent-context state, the last committed turn and filesystem ref.

- `run(..., step=n)` steps a delivery explicitly.
- A chat delivery (`run("text")` with no `step`) auto-mints the next step, so every human turn is an anchor.
- A resident's inbox message from the human sender (`"user"`) is an anchor; peer messages are not.
- Typed-args deliveries (`run(in_args=...)`, sub-agents, workflow nodes) stay unstepped unless the caller passes `step=`.

```python
await agent.run("Draft the intro")      # step 1
await agent.run("Now the conclusion")   # step 2
print(agent.rollback_steps)             # [1, 2]

await agent.rollback_to_step(2)         # discard step 2 and everything after
assert agent.parked_after_rollback
await agent.run("Write a shorter conclusion")  # re-mints step 2 as a fresh delivery
```

## What a rollback does

`rollback_to_step(step)`:

1. Refuses if a run is active (`RuntimeError`) or `step` has no boundary (`KeyError`).
2. Re-marks the persisted head `ROLLING_BACK` before any durable side effect. If the process dies mid-rollback, the next load completes the rollback instead of resuming the pre-rollback transcript.
3. Restores the boundary's filesystem snapshot, if it has one, via `ctx.restore_fs_snapshot(ref, claimant=agent.name)` (the agent must be, or becomes, the session writer).
4. Cuts the transcript to the boundary; folds past the cut are dropped (an earlier fold within the cut applies again) and the usage anchor is dropped when the cut removes the response it came from.
5. Reapplies the boundary's agent-context state: read-before-write ledger, shell cwd, task-record flips. Kernels are re-attached when the filesystem was restored.
6. Reconciles the side channels:
   - background tasks launched after the boundary are cancelled, and pending effects of removed calls are dropped;
   - a delivered task note cut away while its launching call survived is re-injected, so no outcome is lost;
   - inbox mail consumed after the boundary is **voided**, never re-delivered. Human messages are discarded silently; each peer whose message was dropped gets a `<message_dropped>` note so it can resend. Never-consumed mail stays pending.
7. Writes a `ROLLED_BACK` head parked at the start of `step` (head first, then the log is truncated), and drops `step`'s boundary and every later one.

While parked, a run with no input is refused (`ProcInputValidationError`): the step's input was discarded. The next delivery re-enters at `step`.

**The step is a dedup key — re-deliver the input with it.** An orchestrator that, after a crash, redelivers `run(input, step=n)` with the step the persisted head holds gets idempotent behavior on the reloaded instance: a completed step replays its cached output, an interrupted one continues. A rolled-back head is the exception: `run(..., step=n)` there is a fresh delivery. Redelivering the same step to a live instance that never reloaded is not deduplicated.

`await agent.read_rollback_anchors()` lists boundaries from the persisted head without loading the transcript, re-spawning tasks or touching mailbox leases — for a UI picking a rewind point from another process.

## The session writer

Exactly one processor per session writes the `SessionCheckpoint` (serialized state, filesystem ref, metadata), takes filesystem snapshots and may restore one. That is `ctx.session_writer`, a processor name:

- **Claimed automatically** by:
  - a bare (uncontained) agent's first stepped delivery, when a session-record feature is on;
  - a team lead (see [Orchestration](orchestration.md));
  - an agent whose boundary carries a snapshot ref;
  - the rollback that restores a snapshot.
- **Handed over** by a top-level `SequentialWorkflow` / `LoopedWorkflow` to each node as it runs.
- **Declared** explicitly with `SessionContext(session_writer="name")` for topologies the rules above miss.
- While unclaimed, only a bare agent writes. If every agent is contained and none claims, a one-time warning says the session record is never written.
- `claim_session_writer(name)` raises `RuntimeError` if another processor holds the role.

## Filesystem snapshots

The agent's working filesystem inside the environment is session state. Snapshot it opt-in:

```python
from grasp_agents import SessionContext
from grasp_agents.sandbox import e2b_environment

env = e2b_environment(code_interpreter=True)
async with env:
    ctx = SessionContext(environment=env, checkpoint_store=store, fs_snapshot_policy="final")
```

| Policy | Snapshots taken |
|---|---|
| `"off"` (default) | Never. |
| `"final"` | At `AFTER_FINAL_ANSWER` and `AFTER_RESIDENT_ANSWER`. A crash mid-run resumes a transcript ahead of the restored filesystem. |
| `"turn"` | At every checkpoint, including after each tool round. Consistent at every boundary; one provider round-trip per checkpoint. |

- Requires a `SnapshotCapable` `ctx.environment`. Today that is only `E2BEnvironment`, whose `snapshot()` is an E2B snapshot and whose `restore()` spawns a fresh sandbox from it (see [Sandbox](sandbox.md)). Any other environment raises `TypeError` at the first due save.
- Only the opaque ref is stored. The session record keeps the **latest** ref; a boundary that took no snapshot carries it forward. Each step boundary records the ref of the checkpoint before it, so rolling back to step N restores the filesystem as of the end of step N−1.
- On a cold resume of a mid-run head over an older restored snapshot, the agent receives a `<system-reminder subject="filesystem restored">` telling it to re-verify files its recent turns claim to have written.
- A session record carrying a ref resumed without a `SnapshotCapable` environment fails (`RuntimeError`) rather than resuming over divergent files.
- Kernel context ids are kept only alongside a snapshot and re-attached only when the session actually restored one.
- Multi-agent hosts register `ctx.add_environment_restored_callback(cb)`; teams use it to post an `<environment_rewind>` notice to the other members after the lead restores a snapshot. Remove it with `remove_environment_restored_callback` when the host is torn down.

## What rollback never undoes

- External side effects: network calls, writes to systems outside the snapshot, and [deferred effects](durability.md#deferred-effects) that already ran. Compensate in your application.
- Files, when the session takes no snapshots (or the environment cannot): only the conversation and agent state rewind.
- A sub-agent's boundary carries no snapshot, so rolling back a sub-agent never touches the shared filesystem.

## Contracts

1. Rollback runs only between runs (`rollback_to_step` checks `_run_active`) and only on `LLMAgent`s.
2. The `ROLLING_BACK` mark precedes the first durable side effect; a load that finds it completes the rollback (`LLMAgent.load_checkpoint`).
3. Boundaries index the immutable log; a boundary past the live transcript is a framework bug and raises (`_rollback_to_step_unchecked`).
4. Mail consumed past the boundary is voided, never replayed (`mailbox.void_mail_after`).
5. Tasks launched past the boundary are cancelled in memory and their records flipped `CANCELLED`; on resume, records above the restored launch high-water are dead-lettered (`BackgroundTaskManager.cancel_launched_after`, `resume_durable`).
6. One session writer per session; don't reuse its name for another processor (e.g. an `.as_tool()` clone) in the same session (`SessionContext.session_writer`).

## Limitations

- Rollback is linear: there is no branching or forking of sessions, and no rollback of a whole team mid-run.
- Rollback boundaries are never pruned; long-lived sessions accumulate one per human turn.
- Only `E2BEnvironment` is `SnapshotCapable`; local environments have no filesystem rewind.
- A filesystem restore is a provider operation outside any store transaction; crash safety comes from the `ROLLING_BACK` mark and retrying the rollback.

## See also

[Durability](durability.md) · [Background tasks](background-tasks.md) · [Sandbox](sandbox.md) · [Orchestration](orchestration.md) · code: `agent/llm_agent.py` (`rollback_to_step`, `_archive_step_boundary`), `agent/agent_context.py` (`rewind`), `mailbox.py`, `session_context.py`
