# Background tasks

Any tool call — a long shell command, a sub-agent, a `@function_tool` — can run in the background while the agent keeps working. The agent gets an immediate "running" note, live progress streams to the caller, and the result is **pushed** into the conversation when the task finishes. There is no polling tool. This page covers how a call is backgrounded, how results are delivered, how tasks are stopped, and what survives a crash.

## Key types

| Name | Module | Role |
|---|---|---|
| `BackgroundTaskManager` | `agent/background_tasks.py` | Per-agent task tracker (`agent.agent_ctx.bg_tasks`). Never touches the transcript. |
| `TaskNote` | `agent/background_tasks.py` | A launch/completion notice ready to enter the transcript. |
| `AgentContext.deliver_task_notes` | `agent/agent_context.py` | The single point where task notes enter the log. |
| `KillTask` | `tools/task_tools.py` | Model-facing tool to stop a task by id. |
| `TaskRecord`, `TaskStatus` | `durability/task_record.py` | Durable per-task record. |
| `BackgroundTaskLaunchedEvent`, `BackgroundTaskCompletedEvent`, `BackgroundTaskInfo` | `types/events.py` | Stream events. |

## The three tool axes

Each is a `BaseTool` constructor argument (also on `@function_tool` and `Processor.as_tool`); they are independent.

| Axis | Default | Meaning |
|---|---|---|
| `auto_background_at` | `None` | When the call moves to the background. `None`: never. `0`: immediately (e.g. a worker sub-agent). `N > 0`: run in the foreground, move to the background if still running after `N` seconds (e.g. a long command). |
| `blocks_final_answer` | `True` | Whether a backgrounded call holds back a lone agent's final answer until its result is delivered. Set `False` for fire-and-forget work ("I started the server" is the answer). |
| `max_inline_result_chars` | `None` | Cap on inlined result text; larger results spill to a `.result` file (see [Context window](context-window.md#tool-output-spill)). |

`has_progress_log=True` (set by `Bash`) additionally mirrors the call's streamed output to a log file.

```python
from grasp_agents.tools import bash_tools

researcher_tool = researcher.as_tool(
    "research", "Research a topic and report back.", auto_background_at=0
)
agent = LLMAgent[str, str, None](
    "lead", llm=llm, tools=[*bash_tools(), researcher_tool], max_background=8
)
```

`bash_tools()` returns `[Bash, KillTask]` with `auto_background_at=120.0`. When any attached tool is backgroundable, the agent's system prompt gets a `background_tasks` section explaining notes, log files and `KillTask`.

## Lifecycle

1. **Launch.** The call becomes task `bg_<n>`. The model's immediate tool result is a `<task_notification>` with `status running`, the `task_id`, `tool_call_id` and, when one is written, `<log_file>`. A `BackgroundTaskLaunchedEvent` is emitted. If `max_background` tasks (default 16) are already tracked, the call instead returns an error result saying it could not be backgrounded.
2. **Running.** The task's events are buffered. They are re-emitted into the parent's stream at every turn boundary and every 0.25 s while the loop is parked waiting. `ToolStreamEvent` output is appended to `<first allowed root>/.grasp/tasks/<call_id>.log` within about a second (capped at 64 MiB).
3. **Settle at finish.** When the task ends, its outcome is settled at once: the log is flushed, the result is serialized and spilled to `<call_id>.result` if over the cap, and the durable record moves to `COMPLETED` or `FAILED`.
4. **Delivery.** At the next turn boundary the loop takes finished tasks (`pop_completions`: synchronous, no I/O). It writes every completion note into the transcript through `deliver_task_notes`, then emits the task's remaining events, a `BackgroundTaskCompletedEvent` and the note's `UserMessageEvent`. The boundary is checkpointed `AFTER_INPUT` before the model responds, and the record flips to `DELIVERED` after that checkpoint lands.

A completion note:

```xml
<task_notification>
<subject> Task completed </subject>
<task_id> bg_2 </task_id>
<tool_name> Bash </tool_name>
<tool_call_id> call_abc </tool_call_id>
<status> completed </status>
<ran_for> 4m12s </ran_for>
<result>
... head ... [N chars omitted — full result in .grasp/tasks/call_abc.result] ... tail ...
</result>
<result_file>
.grasp/tasks/call_abc.result
</result_file>
<log_file>
.grasp/tasks/call_abc.log
</log_file>
</task_notification>
```

`status` is `completed`, `failed` (with `<error>` instead of `<result>`) or `interrupted` (stopped by a restart). `<result_file>` holds the full result; `<log_file>` holds the streamed output, which may be less.

## Waiting and delivery runs

- **Lone agent, blocking task.** A final answer is held back (`NextStepContinue`) while an answer-blocking task is tracked and the run is within budget and deadline. The loop parks until the task finishes, delivers it, and continues.
- **Lone agent, non-blocking task.** The run may end first. The result is delivered at a later run's turn boundary.
- **Delivery run.** Between runs, `agent.agent_ctx.bg_tasks.has_undelivered_completions` tells a host that a finished result is waiting. Calling `agent.run()` with no input then runs a turn that delivers it, with the full turn budget. `has_tracked_tasks` / `has_live_tasks` report outstanding work. A delivery run is refused while the agent is [parked after a rollback](rollback.md).
- **Residents** never wait on blocking tasks: they answer the current message, and a completion wakes the parked loop as a new turn.

## Stopping tasks: kill vs cancel

- **Kill** is model-facing. `KillTask(task_id=...)` stops one task and returns `KillTaskResult` with `status` `cancelled` (stopped while running) or `completed` / `failed` (it had already ended), an excerpt of its output, and its result if it finished. The record becomes `CANCELLED`.
- **Cancel** is framework-initiated:
  - `agent.aclose()` cancels every task and marks its record `CANCELLED`.
  - A failed-run settle or a [rollback](rollback.md) cancels tasks whose launching call was removed from the transcript.
  - Resume dead-letters records whose launching call never reached a checkpoint.
- Running out of turns or time (MAX_TURNS / TIMEOUT) never cancels background work.

## Durability

With a checkpoint store, every backgrounded call gets a `TaskRecord` at `<session_key>/task/<agent path>/tc_<call_id>`. Records go through `RUNNING → COMPLETED | FAILED → DELIVERED`, or `CANCELLED`. They carry `task_id`, `launch_seq`, the serialized call arguments, `result` / `error`, `log_path` and `note_transcript_pos`. `RUNNING` is written before an immediate task starts, or when a deadline-raced call actually backgrounds; a call that finishes in the foreground leaves no record.

On a cold resume (`BackgroundTaskManager.resume_durable`):

- A `RUNNING` record of a resumable tool (`AgentTool`, `ProcessorTool`) is re-spawned silently. It resumes from its own checkpoint, or reruns from its serialized arguments if it never checkpointed.
- Any other `RUNNING` record is reported as `interrupted`, pointing at its log. The guarantee is "recover the partial output", not "resume the command".
- `COMPLETED` / `FAILED` records whose note never became durable are re-injected.
- Records launched after the restored head are dead-lettered (`CANCELLED`).
- Re-injected notes are preceded by a `<system-reminder>` saying the session was resumed from a checkpoint.

Delivered records are kept for inspection; reclaim old ones with `await BackgroundTaskManager.prune_delivered(ctx, older_than=timedelta(days=7))` (import it from `grasp_agents.agent.background_tasks`).

## Events and sub-agents

- Every `ToolStreamEvent`, `ToolOutputEvent` and `ToolErrorEvent` carries `tool_call_id`: the call it belongs to. The first claim wins, so a sub-agent's nested tool events keep their own call ids. Backgrounded calls run with `agent_ctx.for_tool_call(call_id)`, so their events and deferred effects are attributed to the launching call.
- A background sub-agent's whole event stream (its LLM deltas, tool calls, nested tasks) bubbles into the parent's `run_stream` as live progress. Its final answer reaches the parent's transcript only as the completion note. The multi-agent `Runner` event bus is a separate layer (see [Orchestration](orchestration.md)).

## Contracts

1. `BackgroundTaskManager` never writes the transcript; `AgentContext.deliver_task_notes` is the only entry point, and a completion batch is written whole before any of its events is yielded (`AgentLoop._drain_bg_completions`).
2. A task's outcome is settled when it finishes (`_finalize`). Taking it is synchronous and I/O-free. A settle failure (e.g. the record could not be saved) is raised once from the next `pop_completions`, and the batch stays queued.
3. A record flips to `DELIVERED` only after a checkpoint has persisted the transcript holding its note (`flush_flips`).
4. `blocks_final_answer` gates only a bounded run's final answer; bubbling progress is independent of it (`decide_next_step`, `has_blocking_tasks`).
5. "Kill" names the model-facing stop; every framework-initiated stop is "cancel" (`BackgroundTaskManager` section comment).
6. Deterministic keys: task records live under `tc_<call_id>`, so a resumed session finds the same records (`store_keys.make_tool_call_path`).

## Limitations

- Tasks are bounded per agent by `max_background`; there is no queueing beyond it.
- There is no tool for the model to wait on or read a task's output other than `Read`/`Grep` on its log file and the pushed completion note.
- A non-resumable task interrupted by a restart is not rerun automatically; the model decides.
- `RunPython`, `RunCell` and `BashSession` run in the foreground only.

## See also

[Agent loop](agent-loop.md) · [Context window](context-window.md) · [Durability](durability.md) · [Rollback](rollback.md) · [Sandbox](sandbox.md) · code: `agent/background_tasks.py`, `agent/task_progress.py`, `tools/task_tools.py`, `durability/task_record.py`
