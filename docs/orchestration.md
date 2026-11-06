# Orchestration

Ways to compose processors, and when to use each. Every computation unit is a
`Processor[InT, OutT, CtxT]`; composites are processors too (except `Runner`
and `AgentTeam`), so they nest, stream, checkpoint and wrap as tools the same
way a single agent does.

## Choosing a pattern

| Pattern | Use when | Control flow decided by |
|---|---|---|
| `.as_tool()` / `AgentTool` | a manager agent should decide when to call a worker | the manager LLM |
| `ParallelProcessor` | the same processor over many inputs, concurrently | the input list |
| `SequentialWorkflow` | a fixed, typed chain A → B → C | the code |
| `LoopedWorkflow` | repeat a chain until a condition (critique/revise) | a terminator predicate |
| `Runner` | a graph of processors with per-payload routing | each processor's recipient selector |
| `AgentTeam` (experimental) | long-lived peers that message each other | the members, via `SendMessage` |

| Name | Module | Root export |
|---|---|---|
| `ProcessorTool`, `Processor.as_tool` | `tools/processor_tool.py`, `processors/processor.py` | yes |
| `AgentTool` | `tools/agent_tool.py` | yes |
| `ParallelProcessor` | `processors/parallel_processor.py` | yes |
| `SequentialWorkflow`, `LoopedWorkflow`, `WorkflowProcessor` | `workflow/` | yes |
| `Runner`, `EventBus` | `runner/` | `Runner` only |
| `Packet`, `BranchError` | `types/packet.py` | `Packet` only |
| `AgentTeam`, `MemberHost`, `MemberCard`, … | `agent_team/` | no |
| `ActorDriver`, `Transport`, `InProcessTransport` | `runtime/` | no |

## Sessions in compositions

A container resolves its own `SessionContext` (explicit `ctx=`, else the
ambient `with SessionContext(...)` block, else the process default) and
cascades it onto every child through `on_adopted`, overriding whatever the child
was built with. Set `ctx` on the outermost container only; never pass `ctx=` to
a subprocessor. See [Session](session.md).

## Agents and processors as tools

`proc.as_tool(tool_name, tool_description, reset_transcript_on_run=True,
auto_background_at=None, blocks_final_answer=True, max_inline_result_chars=None)`
returns a `ProcessorTool`. The processor's `in_type` must be a `BaseModel`; it
becomes the tool's argument schema. The tool's result is the processor's first
output payload.

```python
reviewer_tool = reviewer.as_tool("review", "Review a draft and list issues.")
manager = LLMAgent[str, str, None](name="manager", llm=llm, tools=[reviewer_tool])
```

The wrapped processor is a template that is never run. Each call clones it and
binds the clone to the calling run's `ctx` and a per-call path, then closes the
clone when the call ends. So one `ProcessorTool` can be shared by several
agents and sessions. With `reset_transcript_on_run=True` (default) an agent
clone starts each call with an empty transcript.

`AgentTool(name=, description=, llm=, tools=, sys_prompt=, ...)` builds a fresh
`LLMAgent` per call from a configuration and takes a single `prompt: str`
argument. `sys_prompt_builder` / `in_prompt_builder` can derive the child's
prompts from the task prompt, the parent's transcript and the `ctx`. With
`inherit_tools=True` the child also gets the parent's explicitly passed tools
(never auto-attached capability tools or other `AgentTool`s).

**Event bubbling.** A foreground tool call runs through `run_stream`, so a
sub-agent's events (its LLM deltas, tool calls, nested sub-agents) are yielded
live in the parent's stream. Each tool event is claimed with its own call's
`tool_call_id`, and stream deltas carry the owning agent as `destination`, so a
UI can attribute nested output. Backgrounded calls deliver progress through the
background-task manager instead; see [Background tasks](background-tasks.md).

## ParallelProcessor

Runs copies of one processor concurrently, one per input payload, and emits the
results in input order.

```python
par = ParallelProcessor(summarizer, on_error="keep")
packet = await par.run(in_args=[doc_a, doc_b, doc_c])
for i, err in packet.failed.items():
    print(i, err.error_type, err.error)
```

- The processor is named `<subproc>_par`; replicas are `<subproc>_<i>`, so
  checkpoint keys and event sources stay distinct. Replicas are closed at the
  end of each run.
- It takes `in_args` (a list), not `chat_inputs`.
- `on_error`: `"raise"` (default) fails the run on any failed branch with a
  `ProcRunError` listing every failure; `"drop"` drops failed branches;
  `"keep"` leaves `None` at each failed slot so payloads align 1:1 with inputs,
  and `Packet.failed` maps each failed index to a `BranchError(error,
  error_type)`. `drop_failed=` is a deprecated alias for `on_error`.
- Completed branches are checkpointed as they finish, so a resumed run re-runs
  only missing or failed branches.
- It mirrors the inner processor's `tracing_enabled` and `durability_enabled`.
  Pass `durability_enabled=False` on the inner agent for many throwaway
  replicas.

## Workflows

`SequentialWorkflow(name, subprocs)` pipes each subprocessor's output packet
into the next. `LoopedWorkflow(name, subprocs, exit_proc, max_iterations=10)`
repeats the chain; after `exit_proc` runs, the registered terminator decides
whether to stop, and the loop always stops after `max_iterations`.

```python
loop = LoopedWorkflow[Draft, Draft, None](
    name="revise", subprocs=[writer, critic], exit_proc=critic, max_iterations=3
)

@loop.add_workflow_loop_terminator
def good_enough(out_packet, **kwargs) -> bool:
    return out_packet.payloads[0].score >= 8
```

Construction checks (`WorkflowConstructionError`): at least two subprocessors;
adjacent output/input types match (for a loop, the last output type must also
match the first input type); subprocessor names are unique and differ from the
workflow's name. A workflow's `recipients` default to its end processor's, and
it delegates recipient selection to that processor. Each completed step is
checkpointed; a resumed workflow continues after the last completed step. A
top-level workflow hands the session-writer role to whichever node is running,
unless a processor outside the workflow already holds it (see [Session](session.md)).
Subclass `WorkflowProcessor` for other shapes.

## Runner

`Runner` executes a graph of processors connected by their `recipients`. It is
not a `Processor`: it cannot be nested in a workflow or wrapped with
`.as_tool()`.

```python
from grasp_agents.runner.runner import END_PROC_NAME

triage = LLMAgent[str, str, None](name="triage", llm=llm, recipients=["billing", "tech"])

@triage.add_recipient_selector
def route(output: str, **kwargs) -> list[str]:
    return ["billing" if output.startswith("BILLING") else "tech"]

# billing and tech: LLMAgents with recipients=["reviewer"]
reviewer = LLMAgent[str, str, None](name="reviewer", llm=llm, recipients=[END_PROC_NAME])
desk = Runner[str, None](entry_proc=triage, procs=[triage, billing, tech, reviewer], name="desk")
result = await desk.run("My invoice is wrong")
```

- The entry processor receives `chat_inputs` from `run` / `run_stream`. Each
  processor's output packet is split by recipient and delivered to those
  processors; routing to `END_PROC_NAME` (`"*END*"`) produces the run's single
  result, returned as a `Packet` (a `RunPacketOutEvent` in the stream).
- Validated at construction (`RunnerError`): the entry is in `procs`; exactly
  one processor lists END; every declared recipient is a processor or END;
  processor names are unique and differ from the runner's name; no tool name
  collides with a processor name.
- Validated at run time: a packet routed both to END and elsewhere, or to no
  one, raises `RunnerError`; with `Runner[OutT, ...]` subscripted, END payloads
  are validated against `OutT`.
- The pending-packet set is checkpointed before any packet is posted, so a
  resumed runner re-delivers unconsumed packets and finished processors are not
  re-run. Resuming a completed session returns its stored result.
- `aclose()` (or `async with runner:`) closes every processor's session.

The node set is fixed at construction; routing between those nodes is fully
dynamic per payload.

**Runtime.** `EventBus` is the actor runtime's `ActorDriver` over an
`InProcessTransport` (bounded per-recipient `asyncio.Queue`s, size
`MAX_QUEUE_SIZE = 1024`, so a fast producer blocks instead of growing memory)
with `terminal` termination: the run ends when a handler finalizes the END
result. Each processor is one actor and handles one packet at a time; different
processors run concurrently.

## Packets and routing

`Packet[T]` (`types/packet.py`) carries `payloads`, the `sender`, an optional
per-payload `routing` (one list of recipient names per payload; a flat list is
broadcast to every payload) and `failed`. `Processor._build_packet` fills
`routing` by calling the recipient selector once per output when the processor
declares `recipients`; with no selector every payload goes to all declared
recipients. Selected names must be among the declared ones
(`PacketRoutingError`). `Packet.split_by_recipient` groups payloads per
destination.

## AgentTeam (experimental)

`grasp_agents.agent_team` is experimental and not exported from the root
package. Its API may change.

An `AgentTeam(members, entry=, cards=, ctx=, max_hops=50, max_tokens=None)`
hosts peers that message each other over one shared mailbox transport
(`ctx.transport`: a `CheckpointMailboxTransport` when the session has a
checkpoint store, else an `InMemoryMailboxTransport`). It runs on the same
`ActorDriver` as `Runner`.

- **Residents and triggered members.** An `LLMAgent` with no static recipients
  runs resident: one long-lived loop consuming its `AgentInbox` between turns,
  so mail arrives mid-task. Plain processors and agents with recipients are
  triggered: one activation per message. `MemberCard(resident=)` overrides the
  inference.
- **Tools.** Residents get `SendMessage(to=, message=, reply_to=)` (`to="*"`
  broadcasts) and, with `MemberCard(wakeups=True)`, `ScheduleWakeup`. A team
  roster section and a `team_sender` attribution attachment are added to the
  members' prompts.
- **Cards.** `MemberCard(name, description, skills, input_type, resident,
  lead, wakeups)` is the roster entry peers see. `input_type` may be a
  `BaseModel` for typed hand-offs; `MemberCard.from_processor` derives it.
- **Lead.** At most one member may be `lead=True`; it must be resident. The
  lead holds the session's environment-rewind right, and its mail drains ahead
  of ordinary peer mail. See [Rollback](rollback.md).
- **Ending.** `run` / `run_stream` end at quiescence (no member running, every
  inbox empty, no background work), at `max_hops` deliveries or at the
  `max_tokens` budget for the run. `run_stream(daemon=True)` never ends on its
  own; cancel the stream to stop it. `submit_message(to, text)` injects human
  input as control-plane mail, which drains ahead of peer mail.
- **Processes.** `MemberHost(member, cards=)` runs one member per process over
  the session's durable mailbox, for a team spread across processes sharing one
  checkpoint store.

## Contracts

1. Containers cascade their `ctx` onto children; a child never keeps its own
   session (`Processor.on_adopted`).
2. `ProcessorTool` / `AgentTool` hold no per-call state; each call clones and
   rebinds to the call's `ctx` (`ProcessorTool._resolve_processor`).
3. `Runner` checkpoints a transition before posting the packets it produces
   (`Runner._checkpoint_transition`).
4. A recipient selector returns only declared recipients.
5. `ParallelProcessor` output order equals input order; in `"keep"` mode slot
   `i` is input `i`.

## Limitations

- `Runner` and `AgentTeam` membership is fixed at construction.
- `AgentTeam` delivery is at-least-once: a message consumed but not yet acked
  when a process dies is delivered again. Structured payloads cross a durable
  mailbox as plain JSON and are revalidated against the recipient's
  `input_type`.
- `MemberHost` shares state through the checkpoint store only; there is no
  network transport between hosts.

## See also

- [Concepts](concepts.md), [Session](session.md), [Durability](durability.md),
  [Background tasks](background-tasks.md), [Hooks](hooks.md).
- Code: `processors/`, `workflow/`, `runner/`, `runtime/`, `agent_team/`,
  `mailbox.py`, `inbox.py`, `tools/processor_tool.py`, `tools/agent_tool.py`.
- Examples: `src/grasp_agents/examples/notebooks/orchestration_durability.ipynb`,
  `src/grasp_agents/examples/tui/team_research.py`.
