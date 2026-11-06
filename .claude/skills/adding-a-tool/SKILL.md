---
name: adding-a-tool
description: Conventions for adding or changing a grasp-agents tool — @function_tool vs a BaseTool subclass, agent-facing description vs developer docstring, stateless tools that read per-agent state from agent_ctx, timeouts and background/spill axes, untrusted output, concurrency conflict keys, and deferring durable side effects with defer_effect. Use when writing a new tool, a tool set, or changing how a built-in tool behaves.
---

# Adding a tool

Read `docs/agent-loop.md` (tool execution), `docs/background-tasks.md` and
`docs/security.md` first if the tool is long-running, returns external content,
or writes durable state.

## Pick the form

- **`@function_tool`** (`tools/function_tool.py`) for a plain async function. The
  schema is inferred from the signature. The **docstring becomes the agent-facing
  description**, so write it in the second person, for the model. Parameters
  named `ctx`, `exec_id` and `agent_ctx` are injected by the framework and hidden
  from the schema. Don't add `**kwargs`.
- **`BaseTool[InT, OutT, CtxT]` subclass** (`tools/base.py`) when you need state in
  the constructor, streaming (`_run_stream`), conflict keys or custom behaviour.
  Implement `async def _run(self, inp, *, ctx=None, exec_id=None,
  progress_callback=None, path=None, agent_ctx=None)`.
  - The **class docstring is developer-facing**: third person, for people reading
    the code. The model never sees it.
  - The **agent-facing text** is `description=` passed to `super().__init__`, the
    input model's `Field(description=...)`s, and the strings the tool returns.

## Rules

1. **Tools are stateless.** One instance may be shared by several agents,
   sub-agents and replicas. Per-agent state (file-edit ledger, kernels, shells,
   background tasks, inbox) comes from the `agent_ctx` argument, which is a per-call
   copy carrying `tool_call_id`. Session state comes from `ctx`. Never store a run's
   state on `self`, and never use module globals or `ContextVar`s for it.
2. **Inputs and outputs are typed.** The input is a Pydantic model; the output is a
   Pydantic model, a string or content parts. Agent-as-tool wrappers
   (`.as_tool()`) keep their processor's types.
3. **Return errors to the model.** Raise ordinary exceptions for tool failures. The
   loop turns them into structured error results the model can act on. Don't
   swallow errors into a success string.
4. **Mark external content `untrusted_output=True`.** This applies to file contents,
   web results, command output and third-party or MCP data. The loop then fences the
   result in `<untrusted_content>` (see `docs/security.md`).
5. **Declare write conflicts.** Override
   `concurrency_conflict_keys(inp) -> Sequence[str]` when two calls can't run in
   parallel. Use POSIX-path-like keys, which conflict when one nests under the
   other; `"/"` claims global exclusivity. The loop serializes a foreground batch
   whose keys overlap.
6. **Bound time and size with the constructor axes** instead of ad-hoc logic:
   - `timeout`: wall-clock bound for the whole call.
   - `auto_background_at`: seconds after which the call moves to the
     background and its result is delivered later as a task note.
   - `blocks_final_answer`: whether a bounded run waits for the backgrounded call
     before answering.
   - `max_inline_result_chars`: larger results spill to a file, and the model
     gets a preview plus the path.
   - `has_progress_log`: the tool streams progress to a log file.
7. **Defer durable side effects.** If the tool writes business state that must
   exist exactly when the transcript records the call, register the write with
   `agent_ctx.defer_effect(async_fn)` instead of performing it inline:

   ```python
   async def _run(self, inp, *, agent_ctx=None, **_):
       draft = build_draft(inp)

       async def persist() -> None:
           await repo.save(draft)          # runs inside the checkpoint unit

       assert agent_ctx is not None
       agent_ctx.defer_effect(persist)
       return "Draft created."
   ```

   Effects are at-least-once. They are atomic with the checkpoint only on a
   transactional store whose transaction the effect writes through, so make them
   idempotent. See `docs/durability.md`.
8. **Execution goes through the environment.** Shell, code and filesystem
   access must use `ctx.environment` and `ctx.file_backend`, never `subprocess` or
   `open()` directly. That keeps the sandbox policy in force (see
   `docs/sandbox.md`).
9. **No thin wrapper dependencies.** For a REST integration, write it on `httpx`
   yourself rather than adding a small SDK-wrapper package.

## Registering and naming

- Tool names must be unique per agent, and none may be named `final_answer`
  (`LLMAgent` raises at construction).
- Group related tools behind a factory or toolkit (`bash_tools()`, `FileToolkit`)
  when they share configuration.
- Export the tool from its subpackage's `__all__`. Add it to the root
  `__init__.py` only if it is headline API.

## Test it

Unit-test `_run` directly, and drive one agent turn with `MockLLM` that calls the
tool by name (see the `writing-tests` skill). For tools that touch the
filesystem, use `tmp_path` and a `LocalFileBackend` scoped to it.
