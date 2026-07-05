import asyncio
import logging
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Final,
    cast,
    final,
    get_origin,
)

if TYPE_CHECKING:
    from grasp_agents.context.prompt_builder import (
        InputAttachment,
        SystemPromptSection,
    )
    from grasp_agents.tools.file_edit.session_state import FileEditSessionState

from pydantic import BaseModel

from grasp_agents.context.compaction import Compaction
from grasp_agents.context.env_section import (
    make_current_time_attachment,
    make_env_info_section,
)
from grasp_agents.context.prompt_builder import PromptBuilder
from grasp_agents.context.untrusted_content import make_untrusted_content_section
from grasp_agents.durability import AgentCheckpoint, AgentCheckpointPersistMixin
from grasp_agents.durability.checkpoints import (
    AgentCheckpointLocation,
    CheckpointKind,
    StepWatermark,
)
from grasp_agents.durability.resume import prepare_messages_for_resume
from grasp_agents.hooks import (
    AfterLlmHook,
    AfterToolHook,
    BeforeLlmHook,
    BeforeToolHook,
    Compactor,
    FinalAnswerExtractor,
    InitialContextBuilder,
    InputContentBuilder,
    OutputParser,
    ToolInputConverter,
    ToolOutputConverter,
    ViewProjector,
)
from grasp_agents.llm.llm import LLM
from grasp_agents.memory.injection import (
    make_memory_section,
    relevant_memories_attachment,
)
from grasp_agents.processors.processor import Processor
from grasp_agents.sandbox.environment import SnapshotCapable
from grasp_agents.session_context import SessionContext
from grasp_agents.skills.injection import make_skills_section
from grasp_agents.skills.types import SkillFilter
from grasp_agents.telemetry import SpanKind
from grasp_agents.tools.base import BaseTool
from grasp_agents.types.content import Content, InputImage
from grasp_agents.types.errors import ProcInputValidationError
from grasp_agents.types.events import (
    Event,
    ProcPayloadOutEvent,
    SystemMessageEvent,
    UserMessageEvent,
)
from grasp_agents.types.io import LLMPrompt, ProcName
from grasp_agents.types.items import FunctionToolCallItem, InputItem, InputMessageItem
from grasp_agents.types.message import USER_SENDER
from grasp_agents.types.response import Response
from grasp_agents.utils.callbacks import is_method_overridden
from grasp_agents.utils.io import get_prompt
from grasp_agents.utils.validation import validate_obj_from_json_or_py_string

from .agent_loop import AgentLoop
from .context_window import ContextWindowManager
from .llm_agent_transcript import LLMAgentTranscript
from .tool_decision import ToolCallDecision

if TYPE_CHECKING:
    from collections.abc import Iterable

    from grasp_agents.inbox import AgentInbox
    from grasp_agents.mcp.client import MCPClient
    from grasp_agents.mcp.spec import MCPClientSpec
    from grasp_agents.types.message import TeamMessage

    from .background_tasks import BackgroundTaskManager

logger = logging.getLogger(__name__)


class LLMAgent[InT, OutT, CtxT](
    AgentCheckpointPersistMixin, Processor[InT, OutT, CtxT]
):
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
        ctx: SessionContext[CtxT] | None = None,
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
        # Wall-clock budget for one run; on expiry the loop force-generates a
        # final answer and stops with ``StopReason.TIMEOUT`` (checked at turn
        # boundaries). ``None`` = unbounded.
        run_timeout: float | None = None,
        # Max concurrently-running background tasks (auto-backgrounded tool calls
        # / sub-agents). Hitting the cap errors until some finish; raise it for
        # agents that fan out many long-running background commands.
        max_background: int = 16,
        force_react_mode: bool = False,
        final_answer_as_tool_call: bool = False,
        # Per-run message history (the LLM agent's transcript). Cross-session
        # knowledge memory is separate and lives on ``SessionContext.memory``.
        transcript: LLMAgentTranscript | None = None,
        reset_transcript_on_run: bool = False,
        # Agent run retries
        max_retries: int = 0,
        # Multi-agent routing
        recipients: Sequence[ProcName] | None = None,
        # Streaming
        stream_llm: bool = False,
        # Session persistence
        path: list[str] | None = None,
        # MCP integration (clients must be ``connect()``-ed before the
        # agent is constructed; pass a ``MCPClientSpec`` to filter tools)
        mcp_clients: "Sequence[MCPClient | MCPClientSpec] | None" = None,
        # Auto-attached environment-info section. ``True`` attaches the
        # default block (date / platform / os / cwd / model); ``False``
        # attaches nothing. Pass a ``SystemPromptSection`` built with
        # ``make_env_info_section(include=..., extra_fields=..., ...)`` to
        # control exactly which facts appear.
        env_info: "bool | SystemPromptSection" = False,
        # Memory feature toggle (opt-in). When True, the agent gets:
        # - the ``memory`` system-prompt section (taxonomy + index)
        # - the ``relevant_memories_attachment`` (per-turn surfacing)
        # - in agentic mode, a :class:`FileToolkit` auto-attached so the
        #   agent can search, author, and maintain memory files via the
        #   generic file tools.
        # Default is False — the agent should know it's adding memory
        # to its system prompt before it happens.
        enable_memory: bool = False,
        # Skills feature toggle (opt-in). When True, the agent gets:
        # - the ``skills`` system-prompt section (catalog of available
        #   skills, when ``ctx.skills`` is set)
        # - in agentic mode, the ``load_skill`` tool appended
        # Default is False — same rationale as ``enable_memory``.
        enable_skills: bool = False,
        # Per-agent scoping over the session-shared skill catalog (only
        # meaningful with ``enable_skills=True``). ``skill_include`` is an
        # allowlist of skill names, ``skill_exclude`` a blocklist; both set
        # applies the intersection; both default ``None`` (the full
        # ``ctx.skills`` catalog). Mirrors ``MCPClientSpec(include=, exclude=)``.
        skill_include: "Iterable[str] | None" = None,
        skill_exclude: "Iterable[str] | None" = None,
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
            tracing_enabled=tracing_enabled,
            tracing_exclude_input_fields=tracing_exclude_input_fields,
        )

        # Tools are stateless — per-agent state flows through the AgentContext
        # passed on each call — so one tool instance is safe to share across
        # agents. The list is copied so auto-attach appends below don't mutate
        # the caller's.
        tools = list(tools or [])

        existing_names = {t.name for t in tools}
        # Names the agent was *explicitly* given, captured before the
        # capability-tool auto-attach below. A sub-agent (``AgentTool``)
        # inherits only these, never the auto-attached skills/memory/MCP tools.
        explicit_tool_names = frozenset(existing_names)

        # Memory authoring and discovery (read/edit/grep the memdir) route
        # through the file toolkit via ``ctx.file_backend``.
        if enable_memory:
            from grasp_agents.tools import FileToolkit  # noqa: PLC0415

            self._merge_auto_attached(
                tools, existing_names, FileToolkit().tools(), source="memory"
            )

        # Auto-attach the skill loader when the skills feature is on.
        # ``list_skills`` stays opt-in — the catalog is already in the
        # system prompt.
        if enable_skills:
            from grasp_agents.skills.tools import load_skill  # noqa: PLC0415

            self._merge_auto_attached(
                tools, existing_names, [load_skill], source="skills"
            )

        # Transcript (per-run message history)
        self._transcript = transcript or LLMAgentTranscript()
        self.reset_transcript_on_run = reset_transcript_on_run

        # Prompt builder

        sys_prompt = get_prompt(prompt_text=sys_prompt, prompt_path=sys_prompt_path)
        in_prompt = get_prompt(prompt_text=in_prompt, prompt_path=in_prompt_path)

        self._prompt_builder = PromptBuilder[self.in_type, CtxT](
            agent_name=self.name, sys_prompt=sys_prompt, in_prompt=in_prompt
        )

        self.mcp_clients: list[MCPClient] = []

        # Auto-attached sections. Each compute returns ``None`` when its
        # input is absent (no ``ctx.memory``, no ``ctx.skills``, no MCP
        # clients) so registering them by feature flag is safe — they
        # just no-op when the relevant data isn't wired. Users override
        # any of them by adding a section with the same name —
        # ``add_system_prompt_section`` dedupes by name.

        if mcp_clients or any(t.untrusted_output for t in tools):
            self._prompt_builder.add_system_prompt_section(
                make_untrusted_content_section()
            )

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

        if mcp_clients:
            from grasp_agents.mcp.section import (  # noqa: PLC0415
                make_mcp_instructions_section,
            )
            from grasp_agents.mcp.spec import (  # noqa: PLC0415
                MCPClientSpec as _MCPClientSpec,
            )

            self._prompt_builder.add_system_prompt_section(
                make_mcp_instructions_section(lambda: self.mcp_clients)
            )

            mcp_tools: list[BaseTool[BaseModel, Any, CtxT]] = []
            for item in mcp_clients or []:
                if isinstance(item, _MCPClientSpec):
                    mcp_client, include, exclude = (
                        item.client,
                        item.include,
                        item.exclude,
                    )
                else:
                    mcp_client, include, exclude = item, None, None
                mcp_tools.extend(self._filter_mcp_tools(mcp_client, include, exclude))
                # Two specs may partition one client's tools; register the client
                # once so its server instructions aren't rendered twice.
                if not any(mcp_client is c for c in self.mcp_clients):
                    self.mcp_clients.append(mcp_client)

            # MCP tools are auto-sourced (the server names them, not the user), so
            # they yield to explicit tools too — a clash is skipped, not an error.
            self._merge_auto_attached(tools, existing_names, mcp_tools, source="MCP")

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
            and not tools
            and not is_method_overridden(
                "parse_output_impl", self, LLMAgent[Any, Any, Any]
            )
        ):
            llm_output_schema = self.out_type
            self._used_default_llm_output_schema = True

        # Context-window manager (view derivation, token budget, compaction);
        # owned here and shared into the loop so the agent's context operations
        # (checkpoint folds, rollback, hook registration) are single-hop rather
        # than reached through the loop.
        self._cw = ContextWindowManager(
            transcript=self.transcript, model=llm.model_name, source=self.name
        )
        self._skill_filter = SkillFilter.build(skill_include, skill_exclude)
        self._loop: AgentLoop[CtxT] = AgentLoop[CtxT](
            agent_name=self.name,
            llm=llm,
            tools=tools,
            transcript=self.transcript,
            context_window=self._cw,
            ctx=self._ctx,
            llm_output_schema=llm_output_schema,
            max_turns=max_turns,
            run_timeout=run_timeout,
            max_background=max_background,
            force_react_mode=force_react_mode,
            final_answer_type=final_answer_type,
            final_answer_as_tool_call=final_answer_as_tool_call,
            stream_llm=stream_llm,
            path=path,
            tracing_exclude_input_fields=tracing_exclude_input_fields,
            explicit_tool_names=explicit_tool_names,
            skill_filter=self._skill_filter,
        )

        # Session persistence

        # The current/last run's delivery step: the caller's ``step=``, or —
        # for a chat delivery / a resident's human turn — auto-minted, so every
        # human message is a rollback anchor.
        self._step: int | None = None
        # The last human message anchored by ``_anchor_human_turn``: a
        # re-delivery of it (settled turn, rollback re-pend) keeps its step
        # instead of minting a new one.
        self._anchored_message_id: str | None = None

        # Set by ``_prepare_retry`` between ``with_retry`` attempts: the retry
        # continues the settled delivery (see ``_settle_run``) instead of
        # re-memorizing the input. Consumed at the next stream entry.
        self._retry_continuation: bool = False

        # Sender of the current run's input packet, captured in
        # ``validate_inputs`` so an input attachment can attribute the turn;
        # ``None`` for ``chat_inputs`` / no packet.
        self._current_input_source: str | None = None

        # Per-step rollback points and the last-persisted state they're cut from.
        # ``_committed`` mirrors the most recent checkpoint head (set on save and
        # on cold resume); when a new step begins it is archived into
        # ``_step_watermarks`` so the step can later be discarded by
        # :meth:`rollback_to_step`. Persisted in the head, restored on resume.
        self._step_watermarks: list[StepWatermark] = []
        self._committed: StepWatermark | None = None

        # Resume notifications injected into the transcript by load_checkpoint,
        # awaiting emission as stream events (no hidden transcript messages).
        self._resume_notifications: list[InputMessageItem] = []

        # Provider-supplied prompt cache key, set by provider adapters after
        # each LLM call and round-tripped through the checkpoint so resume
        # keeps the model-side cache warm.
        self.prompt_cache_key: str | None = None

        # Always wired; without a store a save still maintains the in-memory
        # head (``_committed``) and just skips persistence.
        self._loop.checkpoint_callback = self.save_checkpoint
        # A resident's human turns are rollback anchors: the boundary is
        # archived at the inbox take, before the message is stamped or
        # appended, so a rollback to it re-pends the message itself.
        self._loop.inbox_take_callback = self._anchor_human_turn

        self._register_overridden_implementations()

        # The loop and its tools are built after super().__init__, so the
        # session set there hasn't reached them yet — cascade it now.
        self._propagate_to_children()

    def _merge_auto_attached(
        self,
        tools: list[BaseTool[Any, Any, CtxT]],
        existing_names: set[str],
        candidates: list[BaseTool[Any, Any, CtxT]],
        *,
        source: str,
    ) -> None:
        """
        Append framework-auto-attached tools, dropping any whose name an
        explicit (or earlier auto-attached) tool already holds.

        Explicit ``tools=`` are authoritative: a memory / skills / MCP tool
        that would shadow one is skipped with a warning rather than colliding.
        Only duplicates *within* the explicit ``tools=`` list reach
        :class:`AgentLoop`'s hard uniqueness guard.
        """
        for tool in candidates:
            if tool.name in existing_names:
                logger.warning(
                    "Agent %r: auto-attached %s tool %r shadows an existing "
                    "tool of the same name and was skipped.",
                    self.name,
                    source,
                    tool.name,
                )
                continue
            tools.append(tool)
            existing_names.add(tool.name)

    def _filter_mcp_tools(
        self,
        client: "MCPClient",
        include: "Iterable[str] | None",
        exclude: "Iterable[str] | None",
    ) -> list[BaseTool[BaseModel, Any, CtxT]]:
        """A connected client's tools, filtered by ``include`` ∧ not ``exclude``."""
        include_set = set(include) if include is not None else None
        exclude_set = set(exclude) if exclude is not None else None
        return [
            cast("BaseTool[BaseModel, Any, CtxT]", tool)
            for tool in client.tools()
            if (include_set is None or tool.name in include_set)
            and (exclude_set is None or tool.name not in exclude_set)
        ]

    async def aclose(self) -> None:
        """
        Tear down this agent's session — delegated to :meth:`AgentContext.close`
        (cancel background tasks, close the shell/kernel holders, cascade to
        tools that wrap sub-processors).

        Execution resources are session-scoped — no run ever releases them —
        so the embedder owns this call (directly, via ``async with agent:``,
        or through the owning workflow/runner's ``aclose``). ``ctx`` is passed
        so ``cancel_all`` can persist CANCELLED task records.
        """
        await self._loop.agent_ctx.close(ctx=self._ctx)

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
    def context_window(self) -> int | None:
        """
        The input-token window context compaction targets, if configured — the
        model window (or explicit limit) of a registered
        :class:`~grasp_agents.context.ContextBudget`, else ``None``. The model's
        own window is available via :func:`~grasp_agents.llm.get_context_window`.
        """
        return self._cw.context_window

    @property
    def step(self) -> int | None:
        """
        The current/last run's delivery step: the caller's ``step=``, or the
        auto-minted step of a chat delivery / resident human turn. ``None``
        for an unstepped run (typed-args delivery, pure resume).
        """
        return self._step

    @property
    def rollback_steps(self) -> list[int]:
        """
        Steps with a recorded rollback boundary, ascending — the valid inputs
        to :meth:`rollback_to_step`. With auto-minted steps, one per human
        turn.
        """
        return sorted(wm.step for wm in self._step_watermarks if wm.step is not None)

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
    def background_tasks(self) -> "BackgroundTaskManager[CtxT]":
        """
        This agent's session-scoped background-task manager.

        Exposed so a driver can read its completion signals
        (:attr:`~BackgroundTaskManager.has_live_tasks` /
        :attr:`~BackgroundTaskManager.has_undelivered_completions`) — e.g. to wake
        an idle agent when a backgrounded task finishes.
        """
        return self._loop.bg_tasks

    @property
    def inbox(self) -> "AgentInbox | None":
        """
        The resident inbox attached to this agent's loop, or ``None``.

        Attaching an inbox makes a run **resident**: the loop consumes peer messages
        from it between turns and runs until its task is cancelled from outside,
        instead of terminating on a final answer (see :class:`AgentLoop`). A
        multi-agent host attaches one keyed inbox per member; a lone agent never sets
        it.
        """
        return self._loop.inbox

    @inbox.setter
    def inbox(self, inbox: "AgentInbox | None") -> None:
        self._loop.inbox = inbox

    def attach_inbox(self) -> None:
        """
        Make this agent run **resident**: bind a per-agent inbox view keyed to
        the agent's own name, so its loop consumes peer messages between turns
        instead of terminating on a final answer.

        Called by a multi-agent host — the agent is built host-agnostic, and
        residency is a hosting role attached per run (and released with
        :meth:`detach_inbox`), like the ``ctx`` / ``path`` cascade rather than
        a constructor argument. The agent supplies its own mailbox address (its
        name); the channel is always the session's mailbox (``ctx.transport``,
        created and installed on first use).
        """
        from grasp_agents.inbox import AgentInbox  # noqa: PLC0415
        from grasp_agents.mailbox import resolve_session_transport  # noqa: PLC0415

        self.inbox = AgentInbox(
            transport=resolve_session_transport(self._ctx), recipient=self.name
        )

    def detach_inbox(self) -> None:
        """Drop the resident inbox, returning the agent to single-answer runs."""
        self.inbox = None

    @property
    def skill_filter(self) -> SkillFilter | None:
        """
        The per-agent allow/deny filter applied to the run's skill catalog, or
        ``None`` when the agent is not scoped to a subset of skills.
        """
        return self._skill_filter

    @property
    def system_prompt_sections(self) -> tuple["SystemPromptSection", ...]:
        """Read-only view of registered system-prompt sections, in order."""
        return tuple(self._prompt_builder.system_prompt_sections)

    async def build_system_prompt(
        self, ctx: "SessionContext[CtxT] | None" = None, exec_id: str = ""
    ) -> str | None:
        """
        Render the agent's full system prompt (base + every section).

        Useful for inspection / debugging — consumers don't normally need to
        call this; the agent invokes it internally on every run. ``ctx`` is
        accepted positionally for debugging convenience (otherwise the
        agent's bound ctx is used).
        """
        return await self._prompt_builder.build_system_prompt(
            ctx=ctx if ctx is not None else self._ctx,
            exec_id=exec_id,
            agent_ctx=self._loop.agent_ctx,
        )

    # --- Session persistence ---

    def _propagate_to_children(self) -> None:
        # ``super().__init__`` runs this hook before ``_loop`` exists; the
        # end-of-``__init__`` call re-syncs, so the missed early call is
        # harmless.
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
        """
        Rehydrate the session from the checkpoint store on a cold start.

        Called at the start of every run but **no-ops unless the transcript is
        empty**. An empty transcript is the signal for a cold start (a fresh
        process / new instance) where the durable log is the only surviving copy
        of the conversation, so it is reloaded here along with turn, kernels,
        file-edit ledger, and background tasks. A non-empty transcript means a
        live in-memory session is already present (a prior run in this process,
        or caller-seeded) and must not be clobbered by a reload.

        Only this agent's working state is restored here; session-scoped state
        (``ctx.state``, the shared filesystem) is restored once per session by
        ``SessionContext.load_checkpoint``.

        Per-turn pruning / compaction belongs in the view projector, which
        never touches this log. A transcript *builder* that reseeds history
        must keep at least one message: an emptied transcript reads as a cold
        start, and the next run reloads the persisted log and silently undoes
        the reseed.
        """
        if not self.transcript.is_empty:
            return None  # Already has messages — don't reload

        checkpoint = await self._deserialize_agent_checkpoint(self._ctx)
        if checkpoint is None:
            return None

        current = checkpoint.current
        self._loop.turn = current.turn
        self.prompt_cache_key = current.prompt_cache_key

        resume_state = prepare_messages_for_resume(checkpoint.messages)
        self.transcript.messages = resume_state.messages
        # Lossy summary folds must be carried so resume doesn't re-summarize;
        # deterministic projectors re-derive from the log for free.
        self._cw.load_folds(checkpoint.folds)
        # Recount the compaction budget for the replaced transcript.
        self._cw.reset_anchor()
        # A stripped trailing turn leaves the transcript a prefix of the log;
        # the next save's prefix check rewrites rather than appends.

        # Kernel-context ids are only meaningful inside the snapshotted
        # filesystem they were captured with — re-attach only when the
        # session restore actually rewound the shared filesystem.
        self._loop.agent_ctx.restore(
            current.agent_ctx_state,
            rebind_kernels=(
                current.fs_snapshot_ref is not None and self._ctx.session_fs_restored
            ),
        )

        self._step_watermarks = list(checkpoint.step_watermarks)
        self._committed = current

        logger.info(
            "Loaded session %s for agent %s "
            "(checkpoints=%d, messages=%d, interruption=%s, "
            "stripped=%d, step=%s, turn=%d)",
            self._checkpoint_store_key(self._ctx),
            self.name,
            self.checkpoint_number,
            len(resume_state.messages),
            resume_state.interruption.value,
            resume_state.removed_count,
            current.step,
            current.turn,
        )

        self._resume_notifications = await self._loop.bg_tasks.resume_durable(
            ctx=self._ctx,
            exec_id=exec_id,
            agent_ctx=self._loop.agent_ctx,
            bg_launch_seq=current.agent_ctx_state.bg_launch_seq,
        )

        return checkpoint

    def _fs_snapshot_due(self, location: AgentCheckpointLocation) -> bool:
        mode = self._ctx.fs_snapshot_policy
        if mode == "off":
            return False
        # Snapshots exist to be rewound to, and only the session's rewinder
        # may rewind — other agents' snapshots would be dead weight (and, at
        # a stepped boundary, a claim conflict).
        rewinder = self._ctx.environment_rewinder
        if rewinder is not None and rewinder != self.name:
            return False
        if not isinstance(self._ctx.environment, SnapshotCapable):
            raise TypeError(
                f"fs_snapshot_policy={mode!r} requires a "
                "SnapshotCapable ctx.environment (e.g. an E2BEnvironment); "
                f"got {type(self._ctx.environment).__name__}."
            )
        if mode == "turn":
            return True
        return location in {
            AgentCheckpointLocation.AFTER_FINAL_ANSWER,
            AgentCheckpointLocation.AFTER_MAX_TURNS,
            # A resident's per-turn boundary is its answer boundary; snapshot so
            # the restored filesystem matches the per-turn transcript checkpoint.
            AgentCheckpointLocation.AFTER_RESIDENT_TURN,
        }

    async def save_checkpoint(
        self,
        *,
        turn: int = 0,
        output: str | None = None,
        location: AgentCheckpointLocation = AgentCheckpointLocation.AFTER_INPUT,
    ) -> None:
        """Persist current conversation state to the store."""
        # Snapshot the filesystem first so the ref describes it as of this
        # checkpoint. Snapshot failures crash the save — a checkpoint silently
        # missing its filesystem half is worse than no checkpoint.
        fs_snapshot_ref: str | None = None
        if self._fs_snapshot_due(location):
            environment = cast("SnapshotCapable", self._ctx.environment)
            fs_snapshot_ref = await environment.snapshot()

        # Session record before this agent's head: a crash in between leaves
        # the filesystem at-or-after the transcript (work gets redone), never
        # the transcript claiming files that don't exist.
        await self._ctx.save_checkpoint(fs_snapshot_ref=fs_snapshot_ref)

        agent_ctx_state = self._loop.agent_ctx.snapshot()
        if fs_snapshot_ref is None:
            agent_ctx_state = agent_ctx_state.model_copy(
                update={"ipy_exec_context_id": None, "nb_exec_context_id": None}
            )

        # The current position's message/log watermark is filled by
        # ``_serialize_agent_checkpoint`` once the log is written.
        current = StepWatermark(
            step=self._step,
            turn=turn,
            prompt_cache_key=self.prompt_cache_key,
            fs_snapshot_ref=fs_snapshot_ref,
            agent_ctx_state=agent_ctx_state,
        )
        checkpoint = AgentCheckpoint(
            processor_name=self.name,
            session_key=self._ctx.session_key,
            messages=list(self.transcript.messages),
            current=current,
            step_watermarks=list(self._step_watermarks),
            folds=list(self._cw.folds),
            output=output,
            location=location,
        )
        await self._serialize_agent_checkpoint(self._ctx, checkpoint)
        # Cache this head as the rewind point a *future* step will be cut from.
        self._committed = current
        # The persisted transcript now contains any delivered background-task
        # notes — only now is it safe to flip their durable records to
        # DELIVERED.
        await self._loop.bg_tasks.flush_delivered(ctx=self._ctx)
        # Same persist covers a resident's drained inbox message: ack it now
        # (the inbox analog of the bg-task flush) so a crash before the
        # persist re-delivers it instead of stranding it.
        if self._loop.inbox is not None:
            await self._loop.inbox.flush_acks()

    def _settle_run(self, *, failed: bool = False) -> None:
        """
        Settle the live run state after an abnormal exit — a mid-turn
        interrupt (Esc / consumer abort) or a genuine failure.

        Drops only the tool round in flight; completed rounds stay (their side
        effects are real and already on the durable log — re-running would
        re-issue them). ``failed`` additionally drops a trailing final answer:
        not a closed turn (e.g. it did not parse), so a retry regenerates it.
        When something is dropped, the paired context state is rewound to the
        last checkpoint boundary, which matches the kept transcript exactly.
        """
        pruned = prepare_messages_for_resume(
            self.transcript.messages, drop_trailing_response=failed
        )
        self.transcript.messages = pruned.messages
        if pruned.removed_count and self._committed is not None:
            state = self._committed.agent_ctx_state
            self._loop.agent_ctx.restore(state)
            # Tasks launched by the pruned round are orphans — their launching
            # calls just left the transcript. Cancel them (sync; the durable
            # CANCELLED flips flush at the next save) so a queued completion
            # can't inject a note for a call the model never made.
            self._loop.bg_tasks.cancel_launched_after(state.bg_launch_seq)
        # The abnormal exit never acked an in-flight inbox message; drop its
        # lease so the resident re-takes it rather than wedging.
        if self._loop.agent_ctx.inbox is not None:
            self._loop.agent_ctx.inbox.rollback()

    def _prepare_retry(self) -> None:
        # The failed attempt settled (``_settle_run``); the retry continues
        # that delivery rather than starting a fresh one.
        self._retry_continuation = True

    def _mint_step(self) -> int:
        """
        The next auto-minted step: one past every recorded boundary. A fresh
        mint never equals a completed head's step (every stepped delivery
        archives its boundary before it first persists), so it cannot read as
        an orchestrator redelivery; after a rollback it lands on the parked
        ``ROLLED_BACK`` step, whose re-delivery is fresh by construction.
        """
        return (
            max(
                (wm.step for wm in self._step_watermarks if wm.step is not None),
                default=0,
            )
            + 1
        )

    def _anchor_human_turn(self, message: "TeamMessage") -> None:
        """
        A resident's rollback anchor: a human message about to be taken from
        the inbox starts a new step. Runs before the message's consumption seq
        is minted and before it is appended, so the boundary's high-waters
        exclude the message — a rollback to it re-pends the message itself.
        Peer messages are not anchors.
        """
        if message.sender != USER_SENDER:
            return
        if message.message_id != self._anchored_message_id:
            self._step = self._mint_step()
            self._anchored_message_id = message.message_id
        # A re-delivery (settled turn / rollback re-pend) keeps its step and
        # re-archives at the settled position — one anchor per human message,
        # not per delivery attempt.
        self._archive_step_boundary()

    def _archive_step_boundary(self) -> None:
        """
        Record the rewind point for the step about to start: the transcript
        length now, the live agent-context state, and the last persisted turn
        / FS ref (``_committed``). :meth:`rollback_to_step` looks boundaries
        up by ``step``, so delivery steps may be sparse. No-op for unstepped
        (chat) deliveries.
        """
        if self._step is None:
            return
        # With snapshots on, the first stepped delivery claims the session's
        # unclaimed rewind right: steps are the rewind points, so the stepper
        # is the rewinder — and claiming before any turn runs gates subagents
        # (which run unstepped) out of snapshotting from the very first
        # delivery. An already-set rewinder wins: a non-rewinder simply steps
        # transcript-only.
        if (
            self._ctx.fs_snapshot_policy != "off"
            and self._ctx.environment_rewinder is None
        ):
            self._ctx.claim_environment_rewind(self.name)
        prior = self._committed
        fs_snapshot_ref = prior.fs_snapshot_ref if prior else None
        if fs_snapshot_ref is not None:
            # A snapshot-carrying boundary (possibly loaded from a persisted
            # head by a cold instance) is a filesystem rewind point — a
            # session-global right held by one agent; fail here at run start,
            # not at a much-later rollback.
            self._ctx.claim_environment_rewind(self.name)
        # At most one boundary per step: a re-run after a rollback re-archives
        # the same step, so drop any prior entry for it before appending.
        self._step_watermarks = [
            wm for wm in self._step_watermarks if wm.step != self._step
        ]
        self._step_watermarks.append(
            StepWatermark(
                step=self._step,
                message_count=len(self.transcript.messages),
                turn=prior.turn if prior else 0,
                prompt_cache_key=self.prompt_cache_key,
                fs_snapshot_ref=fs_snapshot_ref,
                agent_ctx_state=self._loop.agent_ctx.snapshot(),
            )
        )

    async def rollback_to_step(self, step: int) -> None:
        """
        Rewind the session to the start of ``step``, discarding it and every
        later step.

        The inverse of stepped delivery (``run(..., step=...)``): truncates the
        transcript and its durable message-log back to ``step``'s boundary,
        resets the turn counter and cached output, and reapplies the run state
        captured at that boundary.
        Afterwards ``self.step`` is ``step`` (parked at its start) and the
        persisted head is tagged ``ROLLED_BACK``, so ``run(step=step)`` is a
        fresh delivery rather than a cached re-delivery.

        The filesystem (``ctx.environment``) is *session-shared*, so restoring
        it is a global side effect — meaningful only for the agent that owns
        the environment (the coordinator) — and goes through
        :meth:`SessionContext.restore_fs_snapshot`, which re-points the
        session checkpoint at the restored ref so a crash mid-rollback never
        cold-resumes into the pre-rollback filesystem. A subagent's boundary
        carries no ``fs_snapshot_ref``, so rolling one back rewinds its own
        transcript, turn, and agent context but leaves the shared filesystem
        (and, since kernel state lives in it, the kernels) untouched.

        A resident's mailbox rewinds with it: messages absorbed after the
        boundary — the anchoring human message included — return to pending
        for re-delivery (the session mailbox is reached directly when the
        inbox is detached between runs). Roll a hosted resident back *between
        runs*: rewinding under its live loop would race the turn cycle.

        Raises:
            KeyError: ``step`` has no recorded boundary — it never started, or
                an earlier rollback already discarded it.
            RuntimeError: a snapshot ref recorded at the boundary must be
                restored but ``ctx.environment`` is not ``SnapshotCapable``,
                or another agent holds the session's environment-rewind right
                (see :meth:`SessionContext.claim_environment_rewind`).

        """
        boundary = next((wm for wm in self._step_watermarks if wm.step == step), None)
        if boundary is None:
            recorded = sorted(
                wm.step for wm in self._step_watermarks if wm.step is not None
            )
            raise KeyError(
                f"Agent {self.name!r}: no rollback boundary for step {step} "
                f"(recorded steps: {recorded})."
            )
        # Boundary counts index the immutable log (append + suffix-truncate
        # only; view-layer compaction never touches it), so a committed
        # boundary stays a valid prefix. A count past the live transcript
        # means the log was mutated under the boundary map — a framework bug.
        if boundary.message_count > len(self.transcript.messages):
            raise RuntimeError(
                f"Agent {self.name!r}: rollback boundary for step {step} "
                f"({boundary.message_count} messages) exceeds the live transcript "
                f"({len(self.transcript.messages)}) — the log diverged from the "
                "boundary map."
            )

        # Rewind the filesystem before cutting this agent's head: the head
        # still carries this boundary until ``_persist_rollback``, so a crash
        # anywhere in the rewind heals by retrying the rollback. The claim
        # covers boundaries loaded from a persisted head (cold instance).
        fs_ref = boundary.fs_snapshot_ref
        if fs_ref is not None:
            self._ctx.claim_environment_rewind(self.name)
            await self._ctx.restore_fs_snapshot(fs_ref)

        self.transcript.messages = self.transcript.messages[: boundary.message_count]
        self._loop.turn = boundary.turn
        self._loop.agent_ctx.restore(
            boundary.agent_ctx_state, rebind_kernels=fs_ref is not None
        )
        # Tasks launched after this boundary are orphans — the rewound
        # transcript no longer holds their launching calls.
        self._loop.bg_tasks.cancel_launched_after(
            boundary.agent_ctx_state.bg_launch_seq
        )
        # Likewise the inbox messages absorbed after it: their turns left the
        # transcript, so move their acked mailbox records back to pending for
        # re-delivery (a crash mid-move heals by retrying the rollback, like
        # the filesystem rewind above). Between runs the resident inbox is
        # detached, so reach the session mailbox directly — but only when
        # messages were actually absorbed past the boundary, so a lone agent's
        # rollback never resolves a transport.
        mailbox_hw = boundary.agent_ctx_state.mailbox_seq
        if self._loop.inbox is not None:
            await self._loop.inbox.unprocess_after(mailbox_hw)
        elif (
            self._committed is not None
            and self._committed.agent_ctx_state.mailbox_seq > mailbox_hw
        ):
            from grasp_agents.mailbox import resolve_session_transport  # noqa: PLC0415

            await resolve_session_transport(self._ctx).unprocess_after(
                self.name, mailbox_hw
            )
        self._committed = boundary
        self._cw.reset_anchor()

        # Parked at the start of ``step``, ready to (re)deliver it.
        self._step = step
        self.prompt_cache_key = boundary.prompt_cache_key

        # Drop the rewound step's boundary and every later one — the boundary
        # becomes the head. Its ROLLED_BACK marker makes a re-run of ``step``
        # a fresh delivery, not a cached one (see ``_process_stream``).
        self._step_watermarks = [
            wm for wm in self._step_watermarks if wm.step is not None and wm.step < step
        ]

        # Folds index the log; drop any whose span extends past the rewind point
        # (those messages are gone) so the view falls back to the originals.
        self._cw.drop_folds_after(boundary.message_count)

        await self._persist_rollback(boundary, step=step)
        # The rolled-back head no longer references the cancelled tasks'
        # launches — safe to flip their records now rather than waiting for
        # the next loop save.
        await self._loop.bg_tasks.flush_delivered(ctx=self._ctx)
        logger.info(
            "agent '%s' rolled back to step %d (messages=%d, turn=%d)",
            self.name,
            step,
            boundary.message_count,
            boundary.turn,
        )

    async def _persist_rollback(
        self, boundary: StepWatermark, *, step: int | None
    ) -> None:
        """
        Persist the rolled-back session as a ``ROLLED_BACK`` head cut at
        ``boundary`` (store mechanics in
        :meth:`_serialize_rollback_checkpoint`). ``output`` is left unset —
        the rolled-back step's answer is discarded.
        """
        ctx_state = boundary.agent_ctx_state
        fs_ref = boundary.fs_snapshot_ref
        if fs_ref is None:
            ctx_state = ctx_state.model_copy(
                update={"ipy_exec_context_id": None, "nb_exec_context_id": None}
            )
        current = StepWatermark(
            step=step,
            turn=boundary.turn,
            prompt_cache_key=boundary.prompt_cache_key,
            fs_snapshot_ref=fs_ref,
            agent_ctx_state=ctx_state,
        )
        checkpoint = AgentCheckpoint(
            session_key=self._ctx.session_key,
            processor_name=self.name,
            messages=list(self.transcript.messages),
            current=current,
            step_watermarks=list(self._step_watermarks),
            folds=list(self._cw.folds),
            location=AgentCheckpointLocation.ROLLED_BACK,
        )
        await self._serialize_rollback_checkpoint(self._ctx, checkpoint)

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
        # Capture the input packet's sender so an input attachment can
        # attribute the rendered turn (``None`` for a human turn / direct run).
        self._current_input_source = in_packet.sender if in_packet is not None else None

        # No inputs is allowed on resume (a checkpoint store is bound) and for
        # a resident (attached inbox), whose turns arrive from the inbox.
        has_input = any(x is not None for x in [chat_inputs, in_args, in_packet])
        if not has_input and (self.is_resumable or self.inbox is not None):
            return None

        # Fan-in: a multi-payload packet (e.g. a ParallelProcessor's output)
        # feeds a list-typed agent as ONE aggregated input — an LLMAgent
        # processes a single input per run.
        if (
            in_packet is not None
            and len(in_packet.payloads) > 1
            and get_origin(self._in_type) is list
        ):
            in_args = list(in_packet.payloads)
            in_packet = None

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
                message=(
                    f"LLMAgent expects a single input argument; got "
                    f"{len(result)} payloads. Declare the agent's input type "
                    "as list[...] to fan in a multi-payload packet."
                ),
            )
        return result

    async def _process_stream(
        self,
        chat_inputs: LLMPrompt | Sequence[str | InputImage] | None = None,
        *,
        in_args: list[InT] | None = None,
        exec_id: str,
        step: int | None = None,
        ctx: SessionContext[CtxT] | None = None,  # noqa: ARG002  # deprecated; use self.ctx
    ) -> AsyncIterator[Event[Any]]:
        inp = in_args[0] if in_args else None

        # Always load checkpoint (restores memory, background tasks, turn).
        checkpoint = await self.load_checkpoint(exec_id=exec_id)

        # Surface messages resume injected into the transcript (re-delivered
        # background-task notices) — no hidden transcript messages.
        if self._resume_notifications:
            self._print_messages(self._resume_notifications, exec_id=exec_id)
            for message in self._resume_notifications:
                yield UserMessageEvent(data=message, source=self.name, exec_id=exec_id)
            self._resume_notifications = []

        # A direct run with no inputs at all resumes the session — continue
        # the loop on the restored (or live) transcript instead of rendering
        # an input prompt from nothing.
        pure_resume = step is None and chat_inputs is None and inp is None
        if pure_resume and self.transcript.is_empty and self.inbox is None:
            # A resident legitimately starts parked with an empty transcript
            # (its turns come from the inbox); a non-resident has nothing to do.
            raise ProcInputValidationError(
                proc_name=self.name,
                exec_id=exec_id,
                message=(
                    "No inputs were provided and there is no session "
                    "(checkpoint or live transcript) to resume."
                ),
            )

        # A retry after a settled failure continues the same delivery — the
        # input and completed rounds are already in the transcript. An empty
        # transcript means the failure predates the input: deliver fresh.
        retry_continuation = self._retry_continuation and not self.transcript.is_empty
        self._retry_continuation = False

        if retry_continuation:
            # The retry continues the settled delivery under its original step
            # (possibly auto-minted by the failed attempt).
            step = self._step
        elif step is None and chat_inputs is not None:
            # A chat delivery with no explicit step: every human turn is a
            # rollback anchor, so mint the next step. Typed-args deliveries
            # (orchestrators, agents-as-tools) stay unstepped unless the
            # caller steps them.
            step = self._mint_step()
        self._step = step

        # A ``step`` matching the persisted head *resumes* it (an
        # orchestrator's at-least-once redelivery): completed → replay the
        # cached output below; interrupted → continue on the restored
        # transcript. A no-input run resumes the session as a whole (also how
        # a resident always enters). Everything else starts a new step; a
        # rolled-back head is parked at its step's start and delivers fresh.
        resumes_step = (
            retry_continuation
            or pure_resume
            or (
                step is not None
                and checkpoint is not None
                and checkpoint.current.step == step
                and checkpoint.location is not AgentCheckpointLocation.ROLLED_BACK
            )
        )
        starts_step = not resumes_step

        # Resuming an already-completed step: replay the cached output.
        if resumes_step and checkpoint is not None and checkpoint.output is not None:
            output = self.parse_output(checkpoint.output, in_args=inp, exec_id=exec_id)
            yield ProcPayloadOutEvent(data=output, source=self.name, exec_id=exec_id)
            return

        # Compose the ephemeral initial context (system prompt + leading
        # messages) on every entry path. It is never persisted: the loop
        # prepends it to the model-facing view each turn, so resume never sees
        # a stale system prompt and the log stays pure conversation.
        self._cw.initial_context = await self._prompt_builder.build_initial_context(
            ctx=self._ctx, exec_id=exec_id, agent_ctx=self._loop.agent_ctx
        )

        try:
            # Step entry for anything that *starts or (re-)parks* a
            # conversation: new steps and every resident entry. A resumed step
            # skips this — re-archiving would replace its rollback boundary
            # with a mid-step one.
            messages_to_expose: list[InputItem] = []
            if starts_step or self.inbox is not None:
                if self.reset_transcript_on_run:
                    # A reset run starts a new conversation: drop the log (the
                    # system prompt lives in the ephemeral header, not here).
                    self.transcript.clear()
                    self._cw.reset_anchor()
                # Record the rollback point before the step's input, so a
                # rewind lands on the prior steps' conversation with no input
                # yet (the header, being ephemeral, is never lost with it).
                self._archive_step_boundary()
                if self.transcript.is_empty:
                    # Surface the fresh header once, for UI / event parity.
                    messages_to_expose.extend(self._cw.initial_context)

            # New step: reset the turn and memorize the input (a resident's
            # turns arrive from the inbox instead).
            if starts_step:
                self._loop.turn = 0
                input_message = self._prompt_builder.build_input_message(
                    chat_inputs=chat_inputs, in_args=inp, exec_id=exec_id
                )
                if input_message:
                    input_message = await self._prompt_builder.apply_input_attachments(
                        input_message,
                        ctx=self._ctx,
                        exec_id=exec_id,
                        messages=list(self.transcript.messages),
                        agent_ctx=self._loop.agent_ctx,
                        source=self._current_input_source,
                    )
                    self.transcript.update([input_message])
                    messages_to_expose.append(input_message)

            self._print_messages(messages_to_expose, exec_id=exec_id)
            for message in messages_to_expose:
                if isinstance(message, InputMessageItem):
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

            logger.info(
                "agent '%s' run started (model=%s, max_turns=%s)",
                self.name,
                self._loop.llm.model_name,
                self._loop.max_turns,
            )
            run_t0 = time.monotonic()

            async for event in self._loop.execute_stream(exec_id=exec_id):
                yield event

            logger.info(
                "agent '%s' run finished: %d turn(s) in %.2fs",
                self.name,
                self._loop.turn + 1,
                time.monotonic() - run_t0,
            )

            assert self._loop.final_answer is not None
            output = self.parse_output(
                self._loop.final_answer, in_args=inp, exec_id=exec_id
            )

        except (asyncio.CancelledError, GeneratorExit):
            # Interrupted mid-turn (Esc / consumer abort): settle the live
            # session so the transcript and its paired state agree.
            self._settle_run()
            raise

        except BaseException:
            # Genuine failure (LLM error, parser, …): settle to the last
            # closed round, pruning the round in flight and a failed final
            # answer. A retry (``_prepare_retry``) or a later no-input run
            # continues from here instead of re-delivering the whole run.
            self._settle_run(failed=True)
            raise

        yield ProcPayloadOutEvent(data=output, source=self.name, exec_id=exec_id)

    def _print_messages(
        self,
        messages: Sequence[Any],
        exec_id: str,
    ) -> None:
        if self._ctx.printer:
            self._ctx.printer.print_messages(
                messages, agent_name=self.name, exec_id=exec_id
            )

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

    def add_initial_context_builder(
        self, func: InitialContextBuilder
    ) -> InitialContextBuilder:
        # Single-slot (replace): one transform of the ephemeral initial context
        # (system message + sections) prepended to the model-facing view.
        self._prompt_builder.initial_context_builder = func
        return func

    def add_system_prompt_section(self, section: "SystemPromptSection") -> None:
        """
        Append a :class:`SystemPromptSection` to the agent's prompt builder.

        Sections are rendered after the static ``sys_prompt``, in registration
        order. Skills, env info, MCP instructions, and memory plug in via this
        method.
        Names are not enforced unique — the same section may appear twice if
        you register it twice; deduplicate on the caller side if needed.
        """
        self._prompt_builder.add_system_prompt_section(section)

    def add_input_attachment(self, attachment: "InputAttachment") -> None:
        """
        Append an :class:`InputAttachment` to the agent's prompt builder.

        Attachments run when a new user turn is built and append a (usually
        ``<system-reminder>``-wrapped) block to it — per-turn relevance signals such
        as memory topics, the current time, or the teammate a message came from.
        Re-registering the same ``name`` replaces the prior one.
        """
        self._prompt_builder.add_input_attachment(attachment)

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

    def add_view_projector(self, func: ViewProjector) -> ViewProjector:
        # Stacks: registered projectors run as a pipeline in registration order,
        # each transforming the previous one's output (the view is the log when
        # none are registered). A subclass ``project_view_impl`` runs first.
        self._cw.add_view_projector(func)
        return func

    def add_compactor(self, func: Compactor) -> Compactor:
        # Single-slot (replace): the turn-boundary compactor that records a
        # summary fold under context-window pressure.
        self._cw.set_compactor(func)
        return func

    def add_compaction(self, compaction: Compaction | None = None) -> Compaction:
        """
        Enable a :class:`~grasp_agents.context.Compaction` bundle: register its
        recency-gated tool-output collapse projector and, when present, its
        summarizing compactor.

        With no argument, builds the default bundle from this agent's own model and
        LLM (``Compaction(llm=self.llm)``) — the budget is derived from the model,
        so ``agent.add_compaction()`` needs no configuration. Pass a custom
        :class:`Compaction` to override (window, buffer, summarizer model, …).
        """
        if compaction is None:
            compaction = Compaction(llm=self.llm)
        self.add_view_projector(compaction.collapse)
        if compaction.summarize is not None:
            self.add_compactor(compaction.summarize)
        return compaction

    def add_before_llm_hook(self, func: BeforeLlmHook) -> BeforeLlmHook:
        self._loop.before_llm_hooks.append(func)
        return func

    def add_after_llm_hook(self, func: AfterLlmHook) -> AfterLlmHook:
        self._loop.after_llm_hooks.append(func)
        return func

    def add_before_tool_hook(self, func: BeforeToolHook[CtxT]) -> BeforeToolHook[CtxT]:
        self._loop.before_tool_hooks.append(func)
        return func

    def add_after_tool_hook(self, func: AfterToolHook) -> AfterToolHook:
        self._loop.after_tool_hooks.append(func)
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

    # --- Subclass hook points ---
    #
    # Override these in subclasses for customization.
    # Alternatively, use the @agent.add_* decorators (preferred).
    # These read the run context off ``self.ctx`` (the single shared
    # session instance); ``on_before_tool_impl`` keeps an explicit ``ctx``
    # for symmetry with the standalone approval-hook factories.

    async def build_initial_context_impl(
        self, messages: list[InputItem], *, exec_id: str
    ) -> Sequence[InputItem]:
        raise NotImplementedError

    def parse_output_impl(
        self,
        final_answer: str,
        *,
        in_args: InT | None = None,
        exec_id: str,
    ) -> OutT:
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
        ctx: SessionContext[CtxT],
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

    async def project_view_impl(
        self, messages: list[InputItem], *, exec_id: str, input_tokens: int
    ) -> Sequence[InputItem]:
        raise NotImplementedError

    # --- Override detection and registration ---

    def _register_overridden_implementations(self) -> None:
        """
        Detect subclass overrides and set them as callback slots
        on the appropriate components (AgentLoop, PromptBuilder).
        """
        base_cls = LLMAgent[Any, Any, Any]

        # Prompt builder. ``initial_context_builder`` is single-slot (replace).
        if is_method_overridden("build_input_content_impl", self, base_cls):
            self._prompt_builder.input_content_builder = self.build_input_content_impl
        if is_method_overridden("build_initial_context_impl", self, base_cls):
            self._prompt_builder.initial_context_builder = (
                self.build_initial_context_impl
            )

        # Agent loop. ``final_answer_extractor`` / ``view_projector`` are
        # single-slot (replace); the observer / decision hooks stack — a
        # subclass ``*_impl`` override is appended first so it runs before
        # any decorator-registered hook.
        if is_method_overridden("extract_final_answer_impl", self, base_cls):
            self._loop.final_answer_extractor = self.extract_final_answer_impl
        if is_method_overridden("project_view_impl", self, base_cls):
            self._cw.add_view_projector(self.project_view_impl)
        if is_method_overridden("on_before_llm_impl", self, base_cls):
            self._loop.before_llm_hooks.append(self.on_before_llm_impl)
        if is_method_overridden("on_after_llm_impl", self, base_cls):
            self._loop.after_llm_hooks.append(self.on_after_llm_impl)
        if is_method_overridden("on_before_tool_impl", self, base_cls):
            self._loop.before_tool_hooks.append(self.on_before_tool_impl)
        if is_method_overridden("on_after_tool_impl", self, base_cls):
            self._loop.after_tool_hooks.append(self.on_after_tool_impl)

    def copy(self) -> "LLMAgent[InT, OutT, CtxT]":
        # LLM sharing: handled by LLM.__deepcopy__ (returns self)
        # Tool sharing: handled by BaseTool.__deepcopy__ (_copy_shared_attrs)
        return deepcopy(self)
