---
name: durability-changes
description: Checklist for changing anything that is persisted or replayed in grasp-agents — checkpoints and their schema version, the transcript log and context window, background-task records, deferred effects, mailboxes/inbox, rollback, session checkpoints and filesystem snapshots. Covers the invariants that must hold, when to bump CURRENT_SCHEMA_VERSION, and which crash/resume/rollback tests to run. Use before editing agent/llm_agent.py checkpoint code, agent/context_window.py, agent/background_tasks.py, agent/agent_context.py, durability/, inbox.py, mailbox.py or session_context.py persistence.
---

# Changing durable state

The runtime's hardest guarantees live here, and the tests catch only some
violations. Before editing, read the relevant pages: `docs/durability.md`,
`docs/rollback.md`, `docs/context-window.md`, `docs/background-tasks.md`,
`docs/session.md`. Their **Contracts** sections are the spec.

## Invariants to preserve

1. **The log is append-only and has one writer.** Every transcript write goes through
   `ContextWindowManager` (`agent_ctx.cw`) or `LLMAgent.reset_transcript()` /
   `replace_transcript()`. Never mutate a message in place: in-place edits are not
   re-persisted. Replace the message instead.
2. **Surgery repairs derived state in the same step.** Truncating or replacing
   the log also fixes folds and the usage anchor.
3. **Tool-call pairing.** Every function call gets a result before the next
   input. Closures, resume and rollback all preserve this.
4. **Input is durable before generation.** An `AFTER_INPUT` checkpoint is written
   whenever a boundary appended input: the step input, inbox mail or task notes.
5. **Heads only at closed boundaries.** The message log is written before the head
   (the watermark may only undercount), and the loader trims torn tails.
6. **Write order inside a checkpoint:**
   1. Filesystem snapshot (session writer only).
   2. The unit: deferred effects → session record → message log → agent head →
      task-record flips → inbox acks.
   3. Then `_committed` is set and the effects are retired.

   Don't reorder these without re-reading the crash analysis in
   `docs/durability.md`.
7. **The cold-start rule.** An agent reloads from the store only if it has never
   loaded or saved and its transcript is empty. Don't make a reset or an empty
   transcript trigger a reload.
8. **Deferred effects are at-least-once.**
   - They run first in the unit.
   - A failing effect aborts the checkpoint and stays pending.
   - They are retired only after the unit lands.
   - They are dropped with their round when it is pruned or rolled back.
9. **The background task manager never touches the transcript.**
   `AgentContext.deliver_task_notes` is the single entry point for side-channel
   input. A task is settled at finish, and `pop_completions` is synchronous and
   does no I/O.
10. **Delivery flips track the transcript.** A DELIVERED flip exists if and only if
    its note is in the log.
11. **One session writer per session, one live writer per checkpoint key.**
12. **Store failures propagate.** Never wrap `save` in try/except, and never degrade
    to a warning. The only sanctioned degradation is that a *corrupt record on load*
    logs a warning and is treated as missing.
13. **Keys come from `durability/store_keys.py` helpers only.** `exec_id` never appears in a
    key, and tool-call children use `tc_<call_id>`.
14. **Rollback** runs only between runs and only on agents. The ROLLING_BACK mark
    precedes any side effect. Consumed mail is voided, task outcomes are
    re-injected, and post-boundary launches are cancelled. A parked agent refuses
    no-input runs.

## Schema versions

Any change to a persisted shape needs a bump: a field added, removed, renamed or
re-typed on an `AgentCheckpoint`, `SessionCheckpoint`, `TaskRecord`, a workflow,
parallel or team checkpoint, or a message-log record. To bump:
1. Increment `CURRENT_SCHEMA_VERSION` in `durability/checkpoints.py`.
2. Add a `SCHEMA_VERSION_SUMMARIES` entry. Write it for users: what changed, and
   whether older records still load.
3. If old records can no longer load correctly, raise
   `MIN_SUPPORTED_SCHEMA_VERSION`. Don't write silent migration shims.
4. Update `tests/durability/test_checkpoint_schema_versioning.py`.

## Tests to run

Always run `uv run pytest tests/durability tests/agent -q`. Then add the suites
for the area you touched:

| Touched | Also run |
|---|---|
| Transcript / context window / compaction | `tests/context/`, `tests/agent/` |
| Background tasks | `tests/agent/` (background and task tests), `tests/tui/` if events changed |
| Rollback, mailbox, inbox | `tests/durability/test_rollback.py`, `tests/agent_team/`, `tests/runtime/` |
| Deferred effects / store transaction | `tests/durability/test_deferred_effects.py`, `test_durability_failures.py` |
| Session checkpoint / fs snapshots | `tests/durability/test_session_checkpoint.py`, `test_checkpoint_fs_snapshot.py` |
| Workflows / Runner checkpoints | `tests/durability/test_workflow_durability.py`, `tests/orchestration/` |

New behavior needs a crash test. Simulate the crash at the boundary you changed
(for example, make the store raise after the log write and before the head), then
resume and assert that no work was lost and nothing was duplicated.

## Before you finish

- Update the docs page whose Contracts you touched, in the same change.
- If you changed what an older checkpoint means, say so in the PR description
  (see the `public-writing` skill), together with the schema-version change.
