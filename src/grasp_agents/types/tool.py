import asyncio
import copy as copy_mod
import logging
import posixpath
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from pathlib import PurePosixPath
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Protocol,
    Self,
    runtime_checkable,
)

from pydantic import BaseModel, TypeAdapter

from grasp_agents.run_context import RunContext
from grasp_agents.telemetry import SpanKind, traced
from grasp_agents.utils.generics import AutoInstanceAttributesMixin

from .events import Event, ToolErrorEvent, ToolErrorInfo, ToolStreamEvent

if TYPE_CHECKING:
    from grasp_agents.agent.agent_context import AgentContext
    from grasp_agents.durability.checkpoints import CheckpointKind

logger = logging.getLogger(__name__)


class NamedToolChoice(BaseModel):
    name: str


# TODO: expand to support more options
ToolChoice = Literal["none", "auto", "required"] | NamedToolChoice


@runtime_checkable
class ToolProgressCallback(Protocol):
    """Protocol for reporting tool execution progress."""

    async def __call__(
        self, progress: float, total: float | None, message: str | None
    ) -> None: ...


class BaseTool[InT: BaseModel, OutT, CtxT](AutoInstanceAttributesMixin, ABC):
    """
    Base class for all tools.

    **Tools are stateless and shareable.** A single tool instance may be
    attached to several agents at once (manager/worker graphs, LRU-cached
    agents) and invoked concurrently. A tool must therefore never store
    run-scoped or session-scoped state — a :class:`RunContext`, a bound child
    processor, a live shell/kernel — on ``self``. Everything a call needs is
    passed to :meth:`_run` / :meth:`_run_stream`: the run-scoped ``ctx`` and
    the calling loop's agent-scoped ``agent_ctx`` (transcript, sibling tools,
    file-edit ledger, background tasks, sessions). Resolve per-call state from
    those arguments, never from ``self``. Wrapper tools (:class:`AgentTool`,
    :class:`ProcessorTool`) clone their template and bind the clone to the
    *call's* ``ctx`` — not an adoption-time one — for the same reason.
    """

    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_in_type",
        1: "_out_type",
    }

    name: str = ""
    description: str = ""
    # When True, the tool's output is treated as untrusted external content:
    # the agent loop wraps it in ``<untrusted_content>`` tags and the model is
    # told (via the ``untrusted_content`` system-prompt section) to read tagged
    # text as data, never as instructions. Set it on tools whose result carries
    # content from outside the agent's own reasoning — file contents, web /
    # search results, command output, a third-party or MCP server. Default
    # False (the tool returns the agent's / app's own trusted output).
    untrusted_output: bool = False

    def __init__(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        timeout: float | None = None,
        auto_background_at: float | None = None,
        blocks_final_answer: bool = True,
        max_inline_result_chars: int | None = None,
        has_progress_log: bool = False,
        untrusted_output: bool | None = None,
        tracing_enabled: bool = True,
        tracing_exclude_input_fields: set[str] | None = None,
    ) -> None:
        self._in_type: type[InT]
        self._out_type: type[OutT]

        super().__init__()

        if name is not None:
            self.name = name

        if description is not None:
            self.description = description

        if not self.name:
            raise ValueError(f"{type(self).__name__} must have a non-empty name")

        self.timeout = timeout
        # When (and whether) the agent loop moves this tool call to the
        # background. ``None`` (default): never — the call runs to completion in
        # the foreground. ``0``: immediately — the call launches in the
        # background and the loop is notified on completion (a resumable tool
        # also gets a durable ``TaskRecord``; otherwise ephemeral). ``N`` (> 0):
        # in the foreground until it has run ``N`` seconds, then moved to the
        # background if still running. Orthogonal to ``blocks_final_answer`` —
        # this is *when* it backgrounds, that is *whether the loop waits*.
        self.auto_background_at = auto_background_at
        # Whether a backgrounded call of this tool gates the agent's final
        # answer. ``True`` (default): the loop will not emit a final answer until
        # such a call has finished and its result has been delivered — the result
        # is part of the answer (e.g. a worker sub-agent). ``False``: the call
        # never holds the run back; the agent is still notified on completion but
        # may finish first (e.g. a long shell command). Irrelevant when
        # ``auto_background_at is None`` (the call runs to completion inline).
        self.blocks_final_answer = blocks_final_answer
        # Cap on how many characters of a backgrounded call's result are inlined
        # into its completion notification. ``None`` (default): inline the whole
        # result. When set and the result is larger, the notification carries a
        # head+tail excerpt plus a pointer to the task's on-disk ``.grasp`` log,
        # which holds the full output for ``Read`` / ``Grep``. Cap tools with
        # large mechanical output (shell logs); leave ``None`` where the result
        # *is* the answer (a sub-agent).
        self.max_inline_result_chars = max_inline_result_chars
        # Whether a backgrounded call of this tool mirrors its *incremental*
        # output to an agent-readable ``.grasp`` progress log (it emits
        # ``ToolStreamEvent``s). ``True`` (e.g. a shell command): the launch
        # notification and the launched event point the agent at that log to
        # ``Read`` / ``Grep`` mid-flight. ``False`` (default — e.g. a sub-agent,
        # whose events are structural): no log, so neither cites one. Independent
        # of ``auto_background_at``, which is only *when* the call backgrounds.
        self.has_progress_log = has_progress_log
        # ``None`` keeps the class-level ``untrusted_output``; a bool overrides.
        if untrusted_output is not None:
            self.untrusted_output = untrusted_output
        self.tracing_enabled = tracing_enabled
        self.tracing_exclude_input_fields = tracing_exclude_input_fields
        self._llm_in_type: type[BaseModel] | None = None

    def on_adopted(self, parent: "Any") -> None:
        """
        Lifecycle hook fired when this tool is attached to a parent
        :class:`Processor`.

        Inherits the parent's tracing settings (downward restriction): if the
        parent disables tracing the tool's spans are disabled too, and a field
        the parent masks stays masked in the tool's spans (union). Subclasses
        like :class:`ProcessorTool` override to additionally forward adoption
        onto a wrapped processor.
        """
        if not getattr(parent, "tracing_enabled", True):
            self.tracing_enabled = False
        parent_fields = getattr(parent, "tracing_exclude_input_fields", None)
        if parent_fields:
            self.tracing_exclude_input_fields = (
                self.tracing_exclude_input_fields or set()
            ) | set(parent_fields)

    @property
    def in_type(self) -> type[InT]:
        return self._in_type

    @property
    def out_type(self) -> type[OutT]:
        return self._out_type

    @property
    def llm_in_type(self) -> type[BaseModel]:
        """Schema the LLM sees for tool calls. Defaults to ``in_type``."""
        return self._llm_in_type or self._in_type

    @llm_in_type.setter
    def llm_in_type(self, value: type[BaseModel]) -> None:
        self._llm_in_type = value

    # --- Internal execution (implemented by subclasses) ---

    @abstractmethod
    async def _run(
        self,
        inp: InT,
        *,
        ctx: RunContext[CtxT] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        path: list[str] | None = None,
        agent_ctx: "AgentContext | None" = None,
    ) -> OutT:
        # ``path`` is the per-call tool-call lineage (consumed by resumable
        # ``AgentTool`` / ``ProcessorTool``); ``agent_ctx`` is the calling
        # loop's agent-scope state (file-edit ledger, shell session,
        # background tasks, parent transcript / sibling tools). Both are
        # passed to every tool call; a tool ignores whichever it doesn't use.
        pass

    async def _run_stream(
        self,
        inp: InT,
        *,
        ctx: RunContext[CtxT] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        path: list[str] | None = None,
        agent_ctx: "AgentContext | None" = None,
    ) -> AsyncIterator[Event[Any]]:
        from .events import ToolOutputEvent  # noqa: PLC0415  (circular import)

        out = await self._run(
            inp,
            ctx=ctx,
            exec_id=exec_id,
            progress_callback=progress_callback,
            path=path,
            agent_ctx=agent_ctx,
        )
        yield ToolOutputEvent(data=out, source=self.name, exec_id=exec_id)

    # --- Error handling ---

    def _on_error_impl(self, error: Exception) -> ToolErrorInfo:
        logger.warning("Tool '%s' failed: %s", self.name, error)

        return ToolErrorInfo(tool_name=self.name, error=str(error), timed_out=False)

    def _on_error(self, error: Exception) -> ToolErrorInfo:
        if isinstance(error, asyncio.TimeoutError):
            logger.warning("Tool '%s' timed out after %ss", self.name, self.timeout)
            return ToolErrorInfo(
                tool_name=self.name,
                error=f"Timed out after {self.timeout}s",
                timed_out=True,
            )

        return self._on_error_impl(error)

    # --- Timeout wrappers ---

    async def _stream_with_timeout(
        self,
        stream: AsyncIterator[Event[Any]],
        *,
        exec_id: str | None = None,
    ) -> AsyncIterator[Event[Any]]:
        """
        Yield events from *stream*, applying timeout and error handling.

        ``timeout`` bounds the whole call (a fixed deadline from the first
        event wait), not the gap between events — a stream that trickles
        events forever would otherwise never time out.
        """
        try:
            if self.timeout is not None:
                loop = asyncio.get_running_loop()
                deadline = loop.time() + self.timeout
                while True:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        raise TimeoutError  # noqa: TRY301
                    try:
                        event = await asyncio.wait_for(
                            anext(stream), timeout=remaining
                        )
                    except StopAsyncIteration:
                        break
                    yield event
            else:
                async for event in stream:
                    yield event
        except Exception as e:
            error_data = self._on_error(e)
            yield ToolErrorEvent(data=error_data, source=self.name, exec_id=exec_id)

    async def _run_with_timeout(
        self,
        inp: InT,
        *,
        ctx: RunContext[CtxT] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        path: list[str] | None = None,
        agent_ctx: "AgentContext | None" = None,
    ) -> OutT | ToolErrorInfo:
        try:
            coro = self._run(
                inp,
                ctx=ctx,
                exec_id=exec_id,
                progress_callback=progress_callback,
                path=path,
                agent_ctx=agent_ctx,
            )
            if self.timeout is not None:
                result = await asyncio.wait_for(coro, timeout=self.timeout)
            else:
                result = await coro
            return result  # type: ignore[return-value]
        except Exception as e:
            return self._on_error(e)

    async def _run_stream_with_timeout(
        self,
        inp: InT,
        *,
        ctx: RunContext[CtxT] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        path: list[str] | None = None,
        agent_ctx: "AgentContext | None" = None,
    ) -> AsyncIterator[Event[Any]]:
        stream = self._run_stream(
            inp,
            ctx=ctx,
            exec_id=exec_id,
            progress_callback=progress_callback,
            path=path,
            agent_ctx=agent_ctx,
        )
        async for event in self._stream_with_timeout(stream, exec_id=exec_id):
            yield event

    # --- Public API ---

    @traced(name="tool", span_kind=SpanKind.TOOL)
    async def __call__(
        self,
        *,
        ctx: RunContext[CtxT] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        agent_ctx: "AgentContext | None" = None,
        **kwargs: Any,
    ) -> OutT | ToolErrorInfo:
        inp = TypeAdapter(self.in_type).validate_python(kwargs)
        return await self._run_with_timeout(
            inp,
            ctx=ctx,
            exec_id=exec_id,
            progress_callback=progress_callback,
            agent_ctx=agent_ctx,
        )

    @traced(name="tool", span_kind=SpanKind.TOOL)
    async def run(
        self,
        inp: InT,
        *,
        ctx: RunContext[CtxT] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        path: list[str] | None = None,
        agent_ctx: "AgentContext | None" = None,
    ) -> OutT | ToolErrorInfo:
        return await self._run_with_timeout(
            inp,
            ctx=ctx,
            exec_id=exec_id,
            progress_callback=progress_callback,
            path=path,
            agent_ctx=agent_ctx,
        )

    @traced(name="tool", span_kind=SpanKind.TOOL)
    async def run_stream(
        self,
        inp: InT,
        *,
        ctx: RunContext[CtxT] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        path: list[str] | None = None,
        agent_ctx: "AgentContext | None" = None,
    ) -> AsyncIterator[Event[Any]]:
        # Owning agent for this call; stamped onto the tool's own stream events
        # below so a UI routes their (possibly backgrounded / bubbled) output to
        # the right agent's pane instead of guessing from the most recent one.
        dest = agent_ctx.agent_name if agent_ctx else None
        async for event in self._run_stream_with_timeout(
            inp,
            ctx=ctx,
            exec_id=exec_id,
            progress_callback=progress_callback,
            path=path,
            agent_ctx=agent_ctx,
        ):
            # Only when unset, so a sub-agent's nested tool events keep the inner
            # agent they were already stamped with (this wrapper runs at every
            # level, parent and child).
            if (
                dest
                and isinstance(event, ToolStreamEvent)
                and event.destination is None
            ):
                yield event.model_copy(update={"destination": dest})
            else:
                yield event

    def concurrency_conflict_keys(self, inp: InT) -> list[str] | None:
        """
        Keys this call needs **exclusive** use of while it runs — filesystem
        paths it writes, or any other hierarchical resource identifier it must
        not share with a concurrent call. ``None`` (default) means no
        exclusivity is needed and the call is freely parallelizable. The agent
        loop runs a foreground batch **serially** when two calls declare
        overlapping keys (the same key, or one nesting under another), so
        conflicting operations cannot interleave. File writers return their
        target path; read-only tools declare nothing.
        """
        del inp
        return None

    # --- Session lifecycle ---

    async def aclose(self) -> None:
        """
        Release session-scoped resources this tool owns.

        No-op for plain tools (per-agent state lives on the call's
        ``AgentContext``, not the tool). Tools that wrap a processor
        (``AgentTool`` / ``ProcessorTool``) cascade to it. Called by the
        owning agent's ``aclose()``.
        """
        return

    # --- Session persistence (overridden by resumable tools) ---

    @property
    def resumable(self) -> bool:
        return False

    @property
    def checkpoint_kind(self) -> "CheckpointKind | None":
        """
        :class:`CheckpointKind` of the processor this tool wraps, if any.

        Returned by resumable tools (``AgentTool`` / ``ProcessorTool``)
        so that callers like :class:`BackgroundTaskManager` can compose
        the right lifecycle-store key for a spawned invocation. Plain
        tools that don't wrap a processor return ``None``.
        """
        return None

    async def resume_stream(
        self,
        *,
        ctx: RunContext[CtxT] | None = None,
        exec_id: str | None = None,
        path: list[str] | None = None,
        agent_ctx: "AgentContext | None" = None,
    ) -> AsyncIterator[Event[Any]]:
        """Resume from a session checkpoint. Override in resumable tools."""
        del ctx, exec_id, path, agent_ctx
        raise NotImplementedError(f"{type(self).__name__} does not support resume")
        yield  # type: ignore[unreachable]  # makes this an async generator

    # --- Copy ---

    # Attributes that should be shared (not deepcopied) across copies.
    # Subclasses add entries for shared resources like network sessions.
    _copy_shared_attrs: ClassVar[frozenset[str]] = frozenset()

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        for attr in self._copy_shared_attrs:
            val = getattr(self, attr, None)
            if val is not None:
                memo[id(val)] = val
        cls = type(self)
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in copy_mod.deepcopy(dict(self.__dict__), memo).items():
            object.__setattr__(new, k, v)
        return new

    def copy(self) -> Self:
        """
        Deep copy with shared attributes preserved by reference.

        Attributes listed in ``_copy_shared_attrs`` are kept as-is
        (via ``__deepcopy__``); everything else is deep-copied.
        """
        return copy_mod.deepcopy(self)


def _key_parts(key: str) -> tuple[str, ...]:
    """Lexically-normalized POSIX parts (collapses ``.`` / ``..`` / ``//``)."""
    return PurePosixPath(posixpath.normpath(key)).parts


def _keys_overlap(a: str, b: str) -> bool:
    """
    True if two exclusivity keys collide: the same key, or one nests under the
    other (a prefix of POSIX-path parts — so ``/x`` conflicts with ``/x/a``;
    flat keys conflict only when equal). Keys are normalized lexically, and a
    relative key collides with an absolute one whose tail it matches —
    best-effort without a backend root; a false positive only serializes the
    batch. ``"/"`` claims global exclusivity (mutating exec tools).
    """
    pa = _key_parts(a)
    pb = _key_parts(b)
    if pa == ("/",) or pb == ("/",):
        return True
    n = min(len(pa), len(pb))
    if pa[:n] == pb[:n]:
        return True
    a_abs = bool(pa) and pa[0] == "/"
    b_abs = bool(pb) and pb[0] == "/"
    if a_abs != b_abs:
        rel, absolute = (pb, pa) if a_abs else (pa, pb)
        return bool(rel) and absolute[-len(rel) :] == rel
    return False


def batch_has_concurrency_conflict(
    calls: Sequence[tuple[BaseTool[Any, Any, Any], BaseModel]],
) -> bool:
    """
    True if any two calls in a foreground batch declare overlapping exclusivity
    keys (:meth:`BaseTool.concurrency_conflict_keys`) — the loop then runs the
    batch serially instead of concurrently.
    """
    declared = [
        list(keys)
        for tool, inp in calls
        if (keys := tool.concurrency_conflict_keys(inp))
    ]
    for i in range(len(declared)):
        for j in range(i + 1, len(declared)):
            if any(_keys_overlap(x, y) for x in declared[i] for y in declared[j]):
                return True
    return False
