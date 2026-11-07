# Hooks and customization

`LLMAgent` is customized by registering callbacks on an instance (`@agent.add_*`).
Hooks are the primary path; subclassing with `*_impl` overrides is the secondary
path for larger, reusable changes. Every hook signature is a `Protocol` in
`grasp_agents/hooks.py`, so a hook is type-checked against the slot it fills.
Processors and workflows have a smaller set of hooks for routing and loop exit.

## The hook table

"Stacks" means several registrations all run, in registration order. "Single"
means a later registration replaces the earlier one.

| Registration (on `LLMAgent` unless noted) | Protocol (`hooks.py`) | Slot | Subclass override |
|---|---|---|---|
| `add_before_llm_hook` | `BeforeLlmHook(*, exec_id, turn, extra_llm_settings)` (agent turns, not compaction summaries) | stacks | `on_before_llm_impl` |
| `add_after_llm_hook` | `AfterLlmHook(response, *, exec_id, turn)` (agent turns, not compaction summaries) | stacks | `on_after_llm_impl` |
| `add_before_tool_hook` | `BeforeToolHook(*, tool_calls, ctx, exec_id) -> Mapping[call_id, ToolCallDecision] \| None` | stacks, most restrictive wins | `on_before_tool_impl` |
| `add_after_tool_hook` | `AfterToolHook(*, tool_calls, tool_messages, exec_id)` | stacks | `on_after_tool_impl` |
| `add_view_projector` | `ViewProjector(messages, *, exec_id, input_tokens) -> Sequence[InputItem]` | stacks (pipeline) | `project_view_impl` |
| `add_compaction(policy=None)` | registers a `CompactionPolicy` and the `compaction` system-prompt section | single | — |
| `add_custom_compactor` | `CustomCompactor(*, agent_ctx, ctx) -> str` (async; writes compaction summaries in place of the agent's model) | single | — |
| `add_initial_context_builder` | `InitialContextBuilder(messages, *, exec_id) -> Sequence[InputItem]` | single | `build_initial_context_impl` |
| `add_input_content_builder` | `InputContentBuilder(in_args, *, exec_id) -> Content` | single | `build_input_content_impl` |
| `add_output_parser` | `OutputParser(final_answer, *, in_args=None, exec_id) -> OutT` | single | `parse_output_impl` |
| `add_final_answer_extractor` | `FinalAnswerExtractor(*, exec_id, **kwargs) -> str \| None` | single | `extract_final_answer_impl` |
| `add_tool_input_converter(tool_name)` | `ToolInputConverter(llm_args, *, exec_id) -> BaseModel` | one per tool name | — |
| `add_tool_output_converter(tool_name)` | `ToolOutputConverter(tool_output, *, exec_id) -> str \| list[ToolOutputPart]` | one per tool name | — |
| `add_system_prompt_section(section)` | `SystemPromptSection` (`context/prompt_builder.py`) | keyed by `name` | — |
| `add_input_attachment(attachment)` | `InputAttachment` (`context/prompt_builder.py`) | keyed by `name` | — |
| `Processor.add_recipient_selector` | `RecipientSelector(output, *, exec_id) -> Sequence[ProcName]` | single | `select_recipients_impl` |
| `LoopedWorkflow.add_workflow_loop_terminator` | `WorkflowLoopTerminator(out_packet, **kwargs) -> bool` | single | `terminate_workflow_loop_impl` |

The `add_*` decorators return the function, so they can be used bare.
`add_tool_input_converter` / `add_tool_output_converter` take the tool name and
return the decorator. `add_system_prompt_section` and `add_input_attachment`
are plain calls that take a dataclass.

```python
from grasp_agents.agent import RejectToolContent

@agent.add_before_llm_hook
async def low_temp_on_retry(*, exec_id, turn, extra_llm_settings):
    if turn > 3:
        extra_llm_settings["temperature"] = 0.0

@agent.add_before_tool_hook
async def guard(*, tool_calls, ctx, exec_id):
    return {
        c.call_id: RejectToolContent(content="Not allowed.")
        for c in tool_calls
        if c.name == "delete_all"
    }

@agent.add_tool_output_converter("search")
async def compact_search(tool_output, *, exec_id):
    return "\n".join(hit.title for hit in tool_output.hits)
```

## How each kind behaves

**Observer and decision hooks** (before/after LLM, before/after tool) stack. A
subclass `*_impl` override is registered first, so it runs before any decorator
hook. Hooks read the session through the agent (`agent.ctx`); only the
before-tool hook receives `ctx` as an argument, because the standalone approval
factories (`build_callback_approval`, `build_store_approval`) have no agent to
read it from.

**Before-LLM** hooks get a per-call copy of the extra settings. Mutating it
overrides settings for that one call. The key `tool_choice` is popped from it
and overrides the loop's tool choice.

**Before-tool** hooks return per-call decisions keyed by `call_id`:
`AllowTool`, `RejectToolContent(content=...)` (skip the tool, the content becomes
its output) or `RaiseToolException(exception=...)` (abort the whole batch).
Decisions from all hooks are merged per call; the most restrictive wins
(raise > reject > allow), ties go to the later hook. A missing key means allow.
`AgentLoop.on_before_tool` enforces this. Approvals are built on this hook; see
[Agent loop](agent-loop.md).

**View projectors** run as a pipeline over the transcript log each turn; each
receives the previous one's output. The result is never persisted. A
`project_view_impl` override runs first. See [Context window](context-window.md).

**Single-slot builders and parsers** replace the default behavior:

- `add_output_parser` replaces `parse_output_impl`. On a tool-free agent whose
  `llm_output_schema` defaulted to `out_type`, registering a parser also clears
  that default schema.
- `add_final_answer_extractor` decides when a response is the final answer
  (`None` = keep going). It is called with `response=`, `exec_id=` and, on the
  normal path, `turn=`.
- `add_input_content_builder` turns typed `in_args` into the user message
  `Content`, bypassing `InputRenderable`, `in_prompt` templating and JSON
  rendering.
- `add_initial_context_builder` transforms the ephemeral initial context (the
  system message built from `sys_prompt` + sections). Return
  `[*messages, extra]` to augment it, or a single system message to replace it.
  `agent.preview_initial_context()` returns what a run entered now would send.

**Tool converters** are keyed by tool name. An input converter maps the
validated LLM arguments to the tool's input. An output converter owns the
result's shape, including spilling of large results. Untrusted-output fencing
still applies after it (see [Security](security.md)).

## Prompt sections and attachments

The system prompt is `sys_prompt` followed by `SystemPromptSection`s in
registration order. The input message gets `InputAttachment`s appended.

| | `SystemPromptSection(name, compute, cache_control=None)` | `InputAttachment(name, compute, wrap_in_system_reminder=True)` |
|---|---|---|
| Where it lands | system message (initial context) | the new user message |
| When `compute` runs | once per run entry | once per delivered input |
| `compute` receives | `ctx`, `exec_id`, `agent_ctx` | `input_message`, `ctx`, `exec_id`, `messages` (the prior conversation as the model sees it: folds applied), `agent_ctx`, `source` |
| Returns | text, or `None` to omit | text (wrapped in `<system-reminder subject="<name>">`), a list of `InputPart`s (appended verbatim), or `None` |
| Use for | stable instructions and catalogs | per-turn signals: relevant memories, current time, sender attribution |

Keep the system prompt cache-stable: put anything that changes turn to turn in
an attachment, not a section. A section's `cache_control` marks a prompt-cache
breakpoint at that block (see [LLM layer](llm.md#prompt-caching)).

Both registries are keyed by name: registering a second section or attachment
with the same name replaces the first in place (`PromptBuilder.add_system_prompt_section`).
This is how you override the auto-attached ones. Built-in names:

| Name | Added when | Source |
|---|---|---|
| `untrusted_content` | MCP clients or any tool with `untrusted_output` at construction | `context/untrusted_content.py` |
| `memory` | `enable_memory=True` | `memory/injection.py` |
| `env_info` | `env_info=True` or a section | `context/env_section.py` |
| `skills` | `enable_skills=True` | `skills/injection.py` |
| `background_tasks` | always (renders nothing without backgroundable tools) | `agent/background_tasks.py` |
| `mcp_instructions` | MCP clients | `mcp/section.py` |
| `compaction` | `add_compaction()` | `context/compaction.py` |
| `relevant_memories` (attachment) | `enable_memory=True` | `memory/injection.py` |
| `current_time` (attachment) | `time_aware=True` or an attachment | `context/env_section.py` |
| `team_sender` (attachment) | member of an `AgentTeam` | `agent_team/prompt.py` |

`<system-reminder>` is the one tag for framework-injected notes
(`context/system_reminder.py` → `wrap_in_system_reminder`). Attachments and skill
invocations use it. A compaction summary has its own `<conversation_summary>`
tag, explained by the `compaction` section, so it does not read as runtime
instructions.

## Routing and workflow hooks

`Processor.select_recipients_impl(output, *, exec_id)` picks the downstream
recipients for each output payload. Override it in a subclass or register a
function with `@proc.add_recipient_selector`. The selector runs only when the
processor declares `recipients=`, and every name it returns must be in that
list (`Processor._validate_recipients` raises `PacketRoutingError`).
`WorkflowProcessor` and `ParallelProcessor` delegate to their end / inner
processor's selector. See [Orchestration](orchestration.md).

`LoopedWorkflow.add_workflow_loop_terminator` registers a predicate called
with the exit processor's output packet after each pass. `True` ends the loop;
`max_iterations` ends it regardless.

## Subclassing

Subclass `LLMAgent` and override the `*_impl` methods in the table.
`LLMAgent._register_overridden_implementations` detects overrides at
construction and installs them in the same slots the decorators use. Custom
`Processor`s override `_process` (return outputs) or `_process_stream` (yield
events); see `Processor._process` for the override rules.

## Run spans

`run` / `run_stream` accept `span_name=` (rename the run span) and
`span_attributes=` (a mapping of `str | float | bool`), applied after the
framework's attributes so the caller wins. For values only known mid-run,
call `grasp_agents.telemetry.set_run_span_attributes(**attrs)` inside a hook
or tool; it targets the currently active span and is a no-op when tracing is
off. See [Observability](observability.md).

```python
await agent.run("hi", span_name="triage", span_attributes={"ticket.id": "T-1"})
```

## Contracts

1. Observer and decision hooks stack in registration order, after any subclass
   override (`LLMAgent._register_overridden_implementations`).
2. A before-tool hook can never downgrade another hook's decision; raise >
   reject > allow, ties to the later hook (`agent/agent_loop.py` →
   `_more_restrictive`).
3. Builders, parsers, extractors, compactor, recipient selector and loop
   terminator are single-slot; the last registration wins.
4. Sections and attachments are unique by name; re-registering replaces in
   place (`PromptBuilder.add_system_prompt_section` / `add_input_attachment`).
5. The initial context is recomposed on every run entry and never stored in
   the transcript (`PromptBuilder.build_initial_context`).
6. A recipient selector may only return names declared in `recipients`.

## Limitations

- The untrusted-content section is added only at construction. A tool with
  `untrusted_output=True` added to `agent.tools` later is still fenced, but the
  model is not told what the fence means unless you add the section yourself
  (`make_untrusted_content_section()`).
- Hooks are instance state. `agent.copy()` copies them; a fresh instance built
  from the same arguments does not have them.

## See also

- [Agent loop](agent-loop.md) — where each hook fires; approvals.
- [Context window](context-window.md) — projectors, compaction, `CompactionPolicy`.
- [Orchestration](orchestration.md) — routing and workflows.
- Code: `hooks.py`, `agent/llm_agent.py` (Decorator API), `context/prompt_builder.py`,
  `processors/processor.py`, `workflow/looped_workflow.py`.
- Example: `src/grasp_agents/examples/notebooks/advanced_patterns.ipynb`.
