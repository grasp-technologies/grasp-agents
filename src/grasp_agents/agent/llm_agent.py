import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final, Generic, Literal, cast, final

if TYPE_CHECKING:
    from ..tools.file_edit.session_state import FileEditSessionState
    from .prompt_builder import InputAttachment, SystemPromptSection

from pydantic import BaseModel

from ..durability import AgentCheckpoint
from ..durability.checkpoints import AgentCheckpointLocation, CheckpointKind
from ..durability.context_serialization import (
    ContextKind,
    rehydrate_context,
    serialize_context,
)
from ..durability.resume import prepare_messages_for_resume
from ..env_section import make_current_time_attachment, make_env_info_section
from ..llm.llm import LLM
from ..memory.injection import make_memory_section, relevant_memories_attachment
from ..processors.processor import Processor
from ..run_context import CtxT, RunContext
from ..sandbox.environment import SnapshotCapable
from ..skills.injection import make_skills_section
from ..telemetry import SpanKind
from ..types.content import Content, InputImage, InputText
from ..types.errors import ProcInputValidationError
from ..types.events import (
    Event,
    ProcPayloadOutEvent,
    SystemMessageEvent,
    UserMessageEvent,
)
from ..types.hooks import (
    AfterLlmHook,
    AfterToolHook,
    BeforeLlmHook,
    BeforeToolHook,
    FinalAnswerExtractor,
    InputContentBuilder,
    OutputParser,
    StateBuilder,
    SystemPromptBuilder,
    ToolInputConverter,
    ToolOutputConverter,
    TranscriptBuilder,
)
from ..types.io import InT, LLMPrompt, OutT, ProcName
from ..types.items import FunctionToolCallItem, InputMessageItem
from ..types.response import Response
from ..types.tool import BaseTool
from ..utils.callbacks import is_method_overridden
from ..utils.io import get_prompt
from ..utils.validation import validate_obj_from_json_or_py_string
from .agent_loop import AgentLoop
from .llm_agent_transcript import LLMAgentTranscript
from .prompt_builder import PromptBuilder
from .tool_decision import ToolCallDecision

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ..mcp.client import MCPClient
    from ..mcp.spec import MCPClientSpec

logger = logging.getLogger(__name__)


class LLMAgent(Processor[InT, OutT, CtxT], Generic[InT, OutT, CtxT]):
    _span_kind = SpanKind.AGENT
    _checkpoint_kind = CheckpointKind.AGENT

    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_in_type",
        1: "_out_type",
    }

    reset_transcript_on_run: Final[bool]

    def __init__(
        self,
        name: ProcName,
        *,
        # Session context — bound at construction. The agent reads/writes
        # state on this ``ctx`` for its lifetime; passing a different ``ctx``
        # at ``.run()`` time is not supported.
        ctx: RunContext[CtxT] | None = None,
        # LLM
        llm: LLM,
        # Tools
        tools: list[BaseTool[Any, Any, CtxT]] | None = None,
        # Input prompt template (combines user and received arguments)
        in_prompt: LLMPrompt | None = None,
        in_prompt_path: str | Path | None = None,
        # System prompt template
        sys_prompt: LLMPrompt | None = None,
        sys_prompt_path: str | Path | None = None,
        # LLM response validation
        llm_output_schema: Any | None = None,
        # Agent loop settings
        max_turns: int = 100,
        force_react_mode: bool = False,
        final_answer_as_tool_call: bool = False,
        # Per-run message history (the LLM agent's transcript). Cross-session
        # knowledge memory is separate and lives on ``RunContext.memory``.
        transcript: LLMAgentTranscript | None = None,
        reset_transcript_on_run: bool = False,
        # Agent run retries
        max_retries: int = 0,
        # Multi-agent routing
        recipients: Sequence[ProcName] | None = None,
        # Streaming
        stream_llm: bool = False,
        stream_tools: bool = False,
        # Session persistence
        path: list[str] | None = None,
        session_metadata: dict[str, Any] | None = None,
        # Filesystem-snapshot policy for checkpoints. Requires
        # ``ctx.environment`` to be ``SnapshotCapable`` (e.g. E2B).
        # ``"off"`` (default): never snapshot. ``"final"``: snapshot at
        # run-end boundaries (final answer / max turns). ``"turn"``:
        # snapshot at every checkpoint boundary, including after each
        # tool batch — strongest rewind granularity, but each snapshot
        # costs a provider round-trip. The checkpoint stores only the
        # opaque ref; the bytes live with the snapshot owner.
        fs_snapshot: Literal["off", "final", "turn"] = "off",
        # MCP integration (clients must be ``connect()``-ed before the
        # agent is constructed; pass a ``MCPClientSpec`` to filter tools)
        mcp_clients: "Sequence[MCPClient | MCPClientSpec] | None" = None,
        # Auto-attached environment-info section. ``True`` attaches the
        # default block (date / platform / os / cwd / model); ``False``
        # attaches nothing. Pass a ``SystemPromptSection`` built with
        # ``make_env_info_section(include=..., extra_fields=..., ...)`` to
        # control exactly which facts appear.
        env_info: "bool | SystemPromptSection" = True,
        # Memory feature toggle (opt-in). When True, the agent gets:
        # - the ``memory`` system-prompt section (taxonomy + index)
        # - the ``relevant_memories_attachment`` (per-turn surfacing)
        # - in agentic mode, a :class:`FileToolkit` auto-attached so the
        #   agent can search, author, and maintain memory files with the
        #   generic file tools (CC's model — no specialized memory tools).
        # Default is False — the agent should know it's adding memory
        # to its system prompt before it happens.
        enable_memory: bool = False,
        # Skills feature toggle (opt-in). When True, the agent gets:
        # - the ``skills`` system-prompt section (catalog of available
        #   skills, when ``ctx.skills`` is set)
        # - in agentic mode, the ``load_skill`` tool appended
        # Default is False — same rationale as ``enable_memory``.
        enable_skills: bool = False,
        # Time-awareness toggle (opt-in). When True, each input message gets a
        # ``current_time`` ``InputAttachment`` — a live wall-clock stamp on the
        # *input* (not the cached system prompt, so no per-turn cache churn),
        # giving the agent a clock for deadlines / staleness / "now". Pass an
        # ``InputAttachment`` to customize. Default False.
        time_aware: "bool | InputAttachment" = False,
        # Tracing
        tracing_enabled: bool = True,
        tracing_exclude_input_fields: set[str] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            ctx=ctx,
            recipients=recipients,
            max_retries=max_retries,
            path=path,
            session_metadata=session_metadata,
            tracing_enabled=tracing_enabled,
            tracing_exclude_input_fields=tracing_exclude_input_fields,
        )

        # Distinguish ``tools=None`` (structured-output mode, no agentic
        # loop) from ``tools=[]`` / ``tools=[...]`` (agentic mode). Used
        # below to gate the default llm_output_schema and to scope the
        # memory / skills tool auto-attach.
        agentic_mode = tools is not None
        # Tools are stateless: per-agent state (file-edit ledger, shell session,
        # background tasks, parent transcript / sibling tools) flows through the
        # ``AgentContext`` passed on each call, never stored on the instance. A
        # single tool instance is therefore safe to share across agents. Copy
        # the list (not the tools) only so appends below don't mutate the
        # caller's list.
        tools = list(tools or [])

        if agentic_mode:
            existing_names = {t.name for t in tools}

            # Auto-attach the file toolkit when memory is on — memory
            # authoring (read/edit topic files) and discovery (grep/glob
            # the memdir) both route through it via ``ctx.file_backend``.
            # The toolkit is stateless: backend / allowed_roots / read-state
            # all live on the backend the host wires onto ``RunContext``.
            if enable_memory:
                from ..tools import FileToolkit  # noqa: PLC0415

                for tool in FileToolkit().tools():
                    if tool.name not in existing_names:
                        tools.append(tool)
                        existing_names.add(tool.name)

            # Auto-attach the skill loader when the skills feature is on.
            # ``list_skills`` stays opt-in — the catalog is already in the
            # system prompt.
            if enable_skills:
                from ..skills.tools import load_skill  # noqa: PLC0415

                if load_skill.name not in existing_names:
                    tools.append(load_skill)
                    existing_names.add(load_skill.name)

        # Transcript (per-run message history)
        self._transcript = transcript or LLMAgentTranscript()
        self.reset_transcript_on_run = reset_transcript_on_run
        self._fs_snapshot_mode = fs_snapshot

        # Prompt builder

        sys_prompt = get_prompt(prompt_text=sys_prompt, prompt_path=sys_prompt_path)
        in_prompt = get_prompt(prompt_text=in_prompt, prompt_path=in_prompt_path)

        self._prompt_builder = PromptBuilder[self.in_type, CtxT](
            agent_name=self.name, sys_prompt=sys_prompt, in_prompt=in_prompt
        )

        # MCP clients. The auto-attached ``mcp_instructions`` system-prompt
        # section reads this list at compute time, so adding / removing
        # clients mid-life is reflected automatically. Populated below from
        # the ``mcp_clients`` ctor kwarg and / or :meth:`add_mcp_client`.
        self.mcp_clients: list[MCPClient] = []

        # Auto-attached sections. Each compute returns ``None`` when its
        # input is absent (no ``ctx.memory``, no ``ctx.skills``, no MCP
        # clients) so registering them by feature flag is safe — they
        # just no-op when the relevant data isn't wired. Users override
        # any of them by adding a section with the same name —
        # ``add_system_prompt_section`` dedupes by name.
        #
        # Order mirrors Claude Code's dynamic-tail layout (memory →
        # env_info → mcp_instructions in ``constants/prompts.ts``).
        # Skills slot between env_info and mcp_instructions — both
        # surface "what the agent can do", with MCP last so its
        # ``cache_control`` checkpoint caches the whole system-prompt
        # prefix.
        if enable_memory:
            self._prompt_builder.add_system_prompt_section(make_memory_section())
            self._prompt_builder.add_input_attachment(relevant_memories_attachment)
        if env_info:
            self._prompt_builder.add_system_prompt_section(
                make_env_info_section(model_name=llm.model_name)
                if env_info is True
                else env_info
            )
        if enable_skills:
            self._prompt_builder.add_system_prompt_section(make_skills_section())
        if time_aware:
            self._prompt_builder.add_input_attachment(
                make_current_time_attachment() if time_aware is True else time_aware
            )
        # Local import to dodge the ``mcp`` package's optional-dependency
        # import guard at module load.
        from ..mcp.section import make_mcp_instructions_section  # noqa: PLC0415
        from ..mcp.spec import MCPClientSpec as _MCPClientSpec  # noqa: PLC0415

        self._prompt_builder.add_system_prompt_section(
            make_mcp_instructions_section(lambda: self.mcp_clients)
        )

        for item in mcp_clients or []:
            if isinstance(item, _MCPClientSpec):
                self.add_mcp_client(
                    item.client, include=item.include, exclude=item.exclude
                )
            else:
                self.add_mcp_client(item)

        # Agent loop

        if issubclass(self._out_type, BaseModel):
            final_answer_type = self._out_type
        elif not final_answer_as_tool_call:
            final_answer_type = BaseModel
        else:
            raise TypeError(
                "Final answer type must be a subclass of BaseModel if "
                "final_answer_as_tool_call is True."
            )

        self._used_default_llm_output_schema = False
        if (
            llm_output_schema is None
            and not agentic_mode
            and not is_method_overridden(
                "parse_output_impl", self, LLMAgent[Any, Any, Any]
            )
        ):
            llm_output_schema = self.out_type
            self._used_default_llm_output_schema = True

        self._loop: AgentLoop[CtxT] = AgentLoop[CtxT](
            agent_name=self.name,
            llm=llm,
            tools=tools,
            transcript=self.transcript,
            ctx=self._ctx,
            llm_output_schema=llm_output_schema,
            max_turns=max_turns,
            force_react_mode=force_react_mode,
            final_answer_type=final_answer_type,
            final_answer_as_tool_call=final_answer_as_tool_call,
            stream_llm=stream_llm,
            stream_tools=stream_tools,
            path=path,
            tracing_exclude_input_fields=tracing_exclude_input_fields,
        )

        # Session persistence
        self.step: int = 0  # completed invocation counter (observability)
        self._delivery_step: int | None = None  # caller's step for checkpoint

        # Provider-supplied prompt cache key (OpenAI Responses / Anthropic
        # prompt caching). Populated by provider adapters post-LLM-call;
        # round-tripped through the checkpoint so resume reuses the same
        # key and the model-side cache doesn't get invalidated. ``None``
        # when the provider doesn't support caching (or hasn't set one).
        self.prompt_cache_key: str | None = None

        # Wire the loop's checkpoint callback unconditionally. The
        # callback itself short-circuits when no store is attached to
        # the RunContext at call time.
        self._loop.checkpoint_callback = self.save_checkpoint

        # Subclass hook points (set by decorators or subclass overrides)

        self._register_overridden_implementations()

        # The loop and its tools are built after super().__init__, so the
        # session set there hasn't reached them yet — cascade it now.
        self._propagate_to_children()

    def add_mcp_client(
        self,
        client: "MCPClient",
        *,
        include: "Iterable[str] | None" = None,
        exclude: "Iterable[str] | None" = None,
    ) -> None:
        """
        Register an :class:`MCPClient`'s tools (filtered by ``include`` /
        ``exclude``) and track the client so the auto-attached
        ``mcp_instructions`` system-prompt section can read its
        ``instructions`` at compute time.

        ``include`` (when set) is an allowlist of tool names; only matching
        tools are exposed. ``exclude`` (when set) is a denylist. Both set:
        intersection (allow-listed ∧ not denied). ``include=set()`` blocks
        every tool but still surfaces the server's instructions to the prompt.

        The client must be ``connect()``-ed first — tools are only
        discoverable after the MCP handshake completes. Adding the same
        client twice is a no-op for the client list (tool dict is
        re-overwritten, which is harmless).
        """
        include_set = set(include) if include is not None else None
        exclude_set = set(exclude) if exclude is not None else None
        for tool in client.tools():
            if include_set is not None and tool.name not in include_set:
                continue
            if exclude_set is not None and tool.name in exclude_set:
                continue
            self.tools[tool.name] = cast("BaseTool[BaseModel, Any, CtxT]", tool)
        if client not in self.mcp_clients:
            self.mcp_clients.append(client)

    @property
    def llm(self) -> LLM:
        return self._loop.llm

    @property
    def tools(self) -> dict[str, BaseTool[BaseModel, Any, CtxT]]:
        return self._loop.tools

    @property
    def turn(self) -> int:
        return self._loop.turn

    @property
    def max_turns(self) -> int:
        return self._loop.max_turns

    @property
    def sys_prompt(self) -> LLMPrompt | None:
        return self._prompt_builder.sys_prompt

    @property
    def in_prompt(self) -> LLMPrompt | None:
        return self._prompt_builder.in_prompt

    @property
    def transcript(self) -> LLMAgentTranscript:
        return self._transcript

    @property
    def file_edit_state(self) -> "FileEditSessionState":
        """
        This agent's read-before-write / dotfile-override bookkeeping.

        Activated for the duration of each run via a ContextVar so the
        file-edit tools, file-search tools, and :class:`MemoryProvider`
        share it. Exposed for inspection and for pre-seeding read records.
        """
        return self._loop.file_edit_state

    @property
    def system_prompt_sections(self) -> tuple["SystemPromptSection", ...]:
        """Read-only view of registered system-prompt sections, in order."""
        return tuple(self._prompt_builder.system_prompt_sections)

    async def build_system_prompt(
        self,
        ctx: "RunContext[CtxT] | None" = None,
        exec_id: str = "",
    ) -> str | None:
        """
        Render the agent's full system prompt (base + every section).

        Useful for inspection / debugging — consumers don't normally need to
        call this; the agent invokes it internally on every run. ``ctx`` is
        accepted positionally for debugging convenience (otherwise the
        agent's bound ctx is used).
        """
        return await self._prompt_builder.build_system_prompt(
            ctx=ctx if ctx is not None else self._ctx, exec_id=exec_id
        )

    @property
    def _has_build_transcript_impl(self) -> bool:
        return is_method_overridden(
            "build_transcript_impl", self, LLMAgent[Any, Any, Any]
        )

    @property
    def _has_build_state_impl(self) -> bool:
        return is_method_overridden("build_state_impl", self, LLMAgent[Any, Any, Any])

    # --- Session persistence ---

    def _propagate_to_children(self) -> None:
        # Guarded: ``super().__init__`` runs this hook before ``_loop``
        # exists. The loop is constructed later and picks up ``self._ctx`` /
        # ``self.path`` directly from its ctor args; the final
        # ``_propagate_to_children`` at the end of ``__init__`` then syncs
        # the tools, so the missed early call is harmless.
        loop = getattr(self, "_loop", None)
        if loop is None:
            return
        loop.path = self.path
        # The agent owns its loop; adoption-time rebinding is friend access.
        loop.bg_tasks._path = self.path  # noqa: SLF001
        loop._ctx = self._ctx  # noqa: SLF001
        # Forward adoption onto every tool (no-op for stateless tools;
        # :class:`ProcessorTool` rebinds its wrapped processor).
        for tool in loop.tools.values():
            tool.on_adopted(self)

    async def load_checkpoint(
        self,
        *,
        exec_id: str | None = None,
    ) -> AgentCheckpoint | None:
        """Load session checkpoint from store on first run (if available)."""
        if not self.transcript.is_empty:
            return None  # Already has messages — don't reload

        checkpoint = await self._deserialize_agent_checkpoint(self._ctx)
        if checkpoint is None:
            return None

        resume_state = prepare_messages_for_resume(checkpoint.messages)
        self.transcript.messages = resume_state.messages
        if resume_state.removed_count:
            # Resume stripped an incomplete trailing turn from the committed
            # log; rewrite the log to the cleaned transcript on the next
            # checkpoint rather than appending onto the stale records.
            self._log_dirty = True
        self._loop.turn = checkpoint.turn
        self.prompt_cache_key = checkpoint.prompt_cache_key

        # Restore the filesystem before anything that may touch it
        # (pending background tasks, state_builder). A ref without a
        # capable environment means the session cannot be resumed
        # faithfully — crash rather than continue with divergent files.
        if checkpoint.fs_snapshot_ref is not None:
            environment = self._ctx.environment
            if not isinstance(environment, SnapshotCapable):
                raise RuntimeError(
                    "Checkpoint carries fs_snapshot_ref="
                    f"{checkpoint.fs_snapshot_ref!r} but ctx.environment "
                    f"({type(environment).__name__}) is not SnapshotCapable; "
                    "wire the same kind of environment the session was "
                    "saved with."
                )
            await environment.restore(checkpoint.fs_snapshot_ref)

        # Re-attach the RunPython kernel to its persisted code context (captured
        # with the FS snapshot) so the resumed session keeps its in-memory Python
        # state. Seed-only: the kernel re-opens lazily on the next RunPython,
        # bound to this context inside the restored sandbox.
        if checkpoint.code_context_id is not None:
            code_holder = self._loop.agent_ctx.code_kernel_holder
            if code_holder is not None:
                code_holder.rebind(checkpoint.code_context_id)

        # Restore the read-before-write ledger so the staleness guard
        # resumes where it left off; files changed while suspended still
        # trip it and require a re-Read.
        self._loop.file_edit_state.import_state(
            checkpoint.read_file_state, checkpoint.dotfile_overrides
        )

        # Restore ctx.state for the machine-serializable kinds (mapping /
        # pydantic / dataclass) when it was persisted. A None / OMITTED kind
        # (the default — serialize_state off) leaves state untouched, so
        # state_builder fills it in below.
        self._ctx.state = rehydrate_context(
            checkpoint.context_kind,
            checkpoint.context_data,
            self._ctx.state,
        )

        logger.info(
            "Loaded session %s for agent %s "
            "(checkpoints=%d, messages=%d, interruption=%s, "
            "stripped=%d, step=%d, turn=%d)",
            self._checkpoint_store_key(self._ctx),
            self.name,
            self.checkpoint_number,
            len(resume_state.messages),
            resume_state.interruption.value,
            resume_state.removed_count,
            checkpoint.step,
            checkpoint.turn,
        )

        await self._loop.bg_tasks.resume_durable(
            ctx=self._ctx, exec_id=exec_id, agent_ctx=self._loop.agent_ctx
        )

        # Rebuild business state from external sources (DB, etc.) after
        # conversation restoration is complete. Opt-in; fresh init uses
        # add_transcript_builder instead.
        if self._has_build_state_impl:
            await self.build_state_impl(checkpoint=checkpoint, exec_id=exec_id or "")

        return checkpoint

    def _fs_snapshot_due(self, location: AgentCheckpointLocation) -> bool:
        if self._fs_snapshot_mode == "off":
            return False
        if not isinstance(self._ctx.environment, SnapshotCapable):
            raise TypeError(
                f"fs_snapshot={self._fs_snapshot_mode!r} requires a "
                "SnapshotCapable ctx.environment (e.g. an E2BEnvironment); "
                f"got {type(self._ctx.environment).__name__}."
            )
        if self._fs_snapshot_mode == "turn":
            return True
        return location in {
            AgentCheckpointLocation.AFTER_FINAL_ANSWER,
            AgentCheckpointLocation.AFTER_MAX_TURNS,
        }

    async def save_checkpoint(
        self,
        *,
        turn: int = 0,
        output: str | None = None,
        location: AgentCheckpointLocation = AgentCheckpointLocation.AFTER_INPUT,
    ) -> None:
        """Persist current conversation state to the store."""
        # ``ctx.state`` is persisted only when opted in via
        # ``RunContext.serialize_state``. Off by default: business state is
        # rebuilt on resume through ``@agent.add_state_builder`` (the app's
        # database is the source of truth), keeping the checkpoint small.
        context_kind: ContextKind | None = None
        context_data: Any | None = None
        if self._ctx.serialize_state:
            context_kind, context_data = serialize_context(self._ctx.state)

        # Snapshot the environment filesystem first, so the persisted
        # (messages, files) pair is consistent: the ref always describes
        # the filesystem as of this checkpoint. Snapshot failures crash
        # the save — a checkpoint silently missing its filesystem half
        # is worse than no checkpoint.
        fs_snapshot_ref: str | None = None
        code_context_id: str | None = None
        if self._fs_snapshot_due(location):
            environment = cast("SnapshotCapable", self._ctx.environment)
            fs_snapshot_ref = await environment.snapshot()
            # Capture the RunPython kernel's context with the FS snapshot so the
            # pair stays consistent: resume re-attaches to this context inside the
            # restored sandbox (E2B keeps the kernel running in the snapshot).
            code_holder = self._loop.agent_ctx.code_kernel_holder
            code_context_id = code_holder.context_id if code_holder else None

        read_file_state, dotfile_overrides = self._loop.file_edit_state.export_state()
        checkpoint = AgentCheckpoint(
            session_key=self._ctx.session_key,
            processor_name=self.name,
            messages=list(self.transcript.messages),
            session_metadata=self._session_metadata,
            step=self._delivery_step,
            turn=turn,
            output=output,
            location=location,
            context_kind=context_kind,
            context_data=context_data,
            prompt_cache_key=self.prompt_cache_key,
            read_file_state=read_file_state,
            dotfile_overrides=dotfile_overrides,
            fs_snapshot_ref=fs_snapshot_ref,
            code_context_id=code_context_id,
        )
        await self._serialize_agent_checkpoint(self._ctx, checkpoint)

    async def _prepare_transcript(
        self,
        *,
        chat_inputs: LLMPrompt | Sequence[str | InputImage] | None = None,
        in_args: InT | None = None,
        exec_id: str,
    ) -> list[InputMessageItem]:
        # Build the system prompt as parts so per-section ``cache_control``
        # flows through to providers that honor it (Anthropic prompt
        # caching). The build_transcript_impl hook receives the same parts
        # (the contract mirrors ``transcript.reset``), so a custom builder
        # can preserve the markers too.
        sys_prompt_parts = await self._prompt_builder.build_system_prompt_parts(
            ctx=self._ctx, exec_id=exec_id, agent_ctx=self._loop.agent_ctx
        )
        fresh_init = self.reset_transcript_on_run or self.transcript.is_empty
        if fresh_init:
            # A from-scratch transcript invalidates any persisted message-log;
            # the next checkpoint rewrites it rather than appending a delta onto
            # stale records (e.g. reset_transcript_on_run over a resumed session).
            self._log_dirty = True

        if fresh_init and not self._has_build_transcript_impl:
            self.transcript.reset(sys_prompt_parts)
        elif self._has_build_transcript_impl:
            self.build_transcript_impl(
                instructions=sys_prompt_parts,
                in_args=in_args,
                exec_id=exec_id,
            )

        messages_to_expose: list[InputMessageItem] = []
        if fresh_init:
            for msg in self.transcript.messages:
                if isinstance(msg, InputMessageItem):
                    messages_to_expose.append(msg)

        input_message = self._prompt_builder.build_input_message(
            chat_inputs=chat_inputs,
            in_args=in_args,
            exec_id=exec_id,
        )
        if input_message:
            input_message = await self._prompt_builder.apply_input_attachments(
                input_message,
                ctx=self._ctx,
                exec_id=exec_id,
                messages=list(self.transcript.messages),
                agent_ctx=self._loop.agent_ctx,
            )
            self.transcript.update([input_message])
            messages_to_expose.append(input_message)

        return messages_to_expose

    def parse_output_default(self, final_answer: str) -> OutT:
        return validate_obj_from_json_or_py_string(
            final_answer,
            schema=self._out_type,
            from_substring=False,
            strip_language_markdown=True,
        )

    @final
    def parse_output(
        self,
        final_answer: str,
        *,
        in_args: InT | None = None,
        exec_id: str,
    ) -> OutT:
        if is_method_overridden("parse_output_impl", self, LLMAgent[Any, Any, Any]):
            return self.parse_output_impl(
                final_answer,
                in_args=in_args,
                exec_id=exec_id,
            )

        return self.parse_output_default(final_answer)

    def validate_inputs(
        self,
        exec_id: str,
        chat_inputs: Any = None,
        in_packet: Any = None,
        in_args: Any = None,
    ) -> list[InT] | None:
        # Allow no inputs when the bound ctx has a checkpoint store (resume case)
        has_input = any(x is not None for x in [chat_inputs, in_args, in_packet])
        if not has_input and self.is_resumable:
            return None

        result = super().validate_inputs(
            exec_id=exec_id,
            chat_inputs=chat_inputs,
            in_packet=in_packet,
            in_args=in_args,
        )
        if result is not None and len(result) != 1:
            raise ProcInputValidationError(
                proc_name=self.name,
                exec_id=exec_id,
                message="LLMAgent expects a single input argument.",
            )
        return result

    async def _process_stream(
        self,
        chat_inputs: LLMPrompt | Sequence[str | InputImage] | None = None,
        *,
        in_args: list[InT] | None = None,
        exec_id: str,
        step: int | None = None,
        ctx: RunContext[CtxT] | None = None,  # noqa: ARG002  # deprecated; use self.ctx
    ) -> AsyncIterator[Event[Any]]:
        self._delivery_step = step

        inp = in_args[0] if in_args else None

        # Always load checkpoint (restores memory, background tasks, turn).
        checkpoint = await self.load_checkpoint(exec_id=exec_id)

        is_redelivery = (
            step is not None and checkpoint is not None and checkpoint.step == step
        )

        # Re-delivery with cached output: step already completed, caller
        # re-delivered (e.g. workflow crashed before saving its own checkpoint).
        if is_redelivery and checkpoint is not None and checkpoint.output is not None:
            output = self.parse_output(checkpoint.output, in_args=inp, exec_id=exec_id)
            yield ProcPayloadOutEvent(data=output, source=self.name, exec_id=exec_id)
            return

        # New step (or no checkpoint): memorize inputs and reset turn.
        # Interrupted re-delivery (same step, no output): memory already
        # loaded from checkpoint, skip memorization.
        if not is_redelivery:
            self._loop.turn = 0
            messages_to_expose = await self._prepare_transcript(
                chat_inputs=chat_inputs, in_args=inp, exec_id=exec_id
            )
            self._print_messages(messages_to_expose, exec_id=exec_id)
            for message in messages_to_expose:
                if message.role == "system":
                    yield SystemMessageEvent(
                        data=message, source=self.name, exec_id=exec_id
                    )
                # TODO: set source
                elif message.role == "user":
                    yield UserMessageEvent(
                        data=message,
                        source=None,
                        destination=self.name,
                        exec_id=exec_id,
                    )

        async for event in self._loop.execute_stream(exec_id=exec_id):
            yield event

        assert self._loop.final_answer is not None
        output = self.parse_output(
            self._loop.final_answer, in_args=inp, exec_id=exec_id
        )
        yield ProcPayloadOutEvent(data=output, source=self.name, exec_id=exec_id)

        self.step += 1

    def _print_messages(
        self,
        messages: Sequence[Any],
        exec_id: str,
    ) -> None:
        if self._ctx.printer:
            self._ctx.printer.print_messages(
                messages, agent_name=self.name, exec_id=exec_id
            )

    # --- Subclass hook points ---
    #
    # Override these in subclasses for customization.
    # Alternatively, use the @agent.add_* decorators (preferred).
    # These read the run context off ``self.ctx`` (the single shared
    # session instance); ``on_before_tool_impl`` keeps an explicit ``ctx``
    # for symmetry with the standalone approval-hook factories.

    def build_transcript_impl(
        self,
        *,
        instructions: LLMPrompt | Sequence[InputText] | None = None,
        in_args: InT | None = None,
        exec_id: str,
    ) -> None:
        raise NotImplementedError

    async def build_state_impl(
        self,
        *,
        checkpoint: AgentCheckpoint,
        exec_id: str,
    ) -> None:
        raise NotImplementedError

    def parse_output_impl(
        self,
        final_answer: str,
        *,
        in_args: InT | None = None,
        exec_id: str,
    ) -> OutT:
        raise NotImplementedError

    def build_system_prompt_impl(
        self, *, exec_id: str
    ) -> str | Sequence[InputText] | None:
        raise NotImplementedError

    def build_input_content_impl(self, in_args: InT, *, exec_id: str) -> Content:
        raise NotImplementedError

    def extract_final_answer_impl(
        self,
        *,
        exec_id: str,
        **kwargs: Any,
    ) -> str | None:
        raise NotImplementedError

    async def on_before_llm_impl(
        self,
        *,
        exec_id: str,
        turn: int,
        extra_llm_settings: dict[str, Any],
    ) -> None:
        raise NotImplementedError

    async def on_after_llm_impl(
        self,
        response: Response,
        *,
        exec_id: str,
        turn: int,
    ) -> None:
        raise NotImplementedError

    async def on_before_tool_impl(
        self,
        *,
        tool_calls: Sequence[FunctionToolCallItem],
        ctx: RunContext[CtxT],
        exec_id: str,
    ) -> Mapping[str, ToolCallDecision] | None:
        raise NotImplementedError

    async def on_after_tool_impl(
        self,
        *,
        tool_calls: Sequence[FunctionToolCallItem],
        tool_messages: Sequence[Any],
        exec_id: str,
    ) -> None:
        raise NotImplementedError

    # --- Decorator API ---
    #
    # Preferred over subclassing. Each decorator sets a callback slot
    # on the appropriate component (AgentLoop or PromptBuilder).

    def add_output_parser(
        self, func: OutputParser[InT, OutT]
    ) -> OutputParser[InT, OutT]:
        if self._used_default_llm_output_schema:
            self._loop.llm_output_schema = None
        self.parse_output_impl = func
        return func

    def add_transcript_builder(
        self, func: TranscriptBuilder[InT]
    ) -> TranscriptBuilder[InT]:
        self.build_transcript_impl = func
        return func

    def add_state_builder(self, func: StateBuilder) -> StateBuilder:
        self.build_state_impl = func
        return func

    def add_system_prompt_builder(
        self, func: SystemPromptBuilder
    ) -> SystemPromptBuilder:
        self._prompt_builder.system_prompt_builder = func
        return func

    def add_system_prompt_section(self, section: "SystemPromptSection") -> None:
        """
        Append a :class:`SystemPromptSection` to the agent's prompt builder.

        Sections are rendered after the user's ``sys_prompt`` /
        ``system_prompt_builder`` output, in registration order. Skills, env
        info, MCP instructions, and (later) memory plug in via this method.
        Names are not enforced unique — the same section may appear twice if
        you register it twice; deduplicate on the caller side if needed.
        """
        self._prompt_builder.add_system_prompt_section(section)

    def add_input_content_builder(
        self, func: InputContentBuilder[InT]
    ) -> InputContentBuilder[InT]:
        self._prompt_builder.input_content_builder = func
        return func

    def add_final_answer_extractor(
        self, func: FinalAnswerExtractor
    ) -> FinalAnswerExtractor:
        self._loop.final_answer_extractor = func
        return func

    def add_before_llm_hook(self, func: BeforeLlmHook) -> BeforeLlmHook:
        self._loop.before_llm_hook = func
        return func

    def add_after_llm_hook(self, func: AfterLlmHook) -> AfterLlmHook:
        self._loop.after_llm_hook = func
        return func

    def add_before_tool_hook(self, func: BeforeToolHook[CtxT]) -> BeforeToolHook[CtxT]:
        self._loop.before_tool_hook = func
        return func

    def add_after_tool_hook(self, func: AfterToolHook) -> AfterToolHook:
        self._loop.after_tool_hook = func
        return func

    def add_tool_input_converter(self, tool_name: str) -> Any:
        def decorator(func: ToolInputConverter) -> ToolInputConverter:
            self._loop.tool_input_converters[tool_name] = func
            return func

        return decorator

    def add_tool_output_converter(self, tool_name: str) -> Any:
        def decorator(func: ToolOutputConverter) -> ToolOutputConverter:
            self._loop.tool_output_converters[tool_name] = func
            return func

        return decorator

    # --- Override detection and registration ---

    def _register_overridden_implementations(self) -> None:
        """
        Detect subclass overrides and set them as callback slots
        on the appropriate components (AgentLoop, PromptBuilder).
        """
        base_cls = LLMAgent[Any, Any, Any]

        # Prompt builder
        if is_method_overridden("build_system_prompt_impl", self, base_cls):
            self._prompt_builder.system_prompt_builder = self.build_system_prompt_impl
        if is_method_overridden("build_input_content_impl", self, base_cls):
            self._prompt_builder.input_content_builder = self.build_input_content_impl

        # Agent loop
        if is_method_overridden("extract_final_answer_impl", self, base_cls):
            self._loop.final_answer_extractor = self.extract_final_answer_impl
        if is_method_overridden("on_before_llm_impl", self, base_cls):
            self._loop.before_llm_hook = self.on_before_llm_impl
        if is_method_overridden("on_after_llm_impl", self, base_cls):
            self._loop.after_llm_hook = self.on_after_llm_impl
        if is_method_overridden("on_before_tool_impl", self, base_cls):
            self._loop.before_tool_hook = self.on_before_tool_impl
        if is_method_overridden("on_after_tool_impl", self, base_cls):
            self._loop.after_tool_hook = self.on_after_tool_impl

    def copy(self) -> "LLMAgent[InT, OutT, CtxT]":
        # LLM sharing: handled by LLM.__deepcopy__ (returns self)
        # Tool sharing: handled by BaseTool.__deepcopy__ (_copy_shared_attrs)
        return deepcopy(self)
