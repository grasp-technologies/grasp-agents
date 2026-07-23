from __future__ import annotations

import asyncio
import time
from copy import deepcopy
from logging import DEBUG, getLogger
from typing import TYPE_CHECKING, Any, Final, Protocol

from pydantic import BaseModel

from grasp_agents.context.system_reminder import wrap_in_system_reminder
from grasp_agents.context.untrusted_content import wrap_untrusted
from grasp_agents.durability.checkpoints import AgentCheckpointLocation
from grasp_agents.durability.store_keys import make_tool_call_path
from grasp_agents.telemetry import traced
from grasp_agents.tools.base import (
    BaseTool,
    NamedToolChoice,
    ToolChoice,
    batch_has_concurrency_conflict,
)
from grasp_agents.types.errors import AgentFinalAnswerError, LLMToolCallValidationError
from grasp_agents.types.events import (
    Event,
    GenerationEndEvent,
    LLMStreamEvent,
    OutputMessageItemEvent,
    ReasoningItemEvent,
    StopReason,
    ToolCallItemEvent,
    ToolErrorEvent,
    ToolErrorInfo,
    ToolOutputEvent,
    ToolOutputItemEvent,
    TurnEndEvent,
    TurnEndInfo,
    TurnInfo,
    TurnStartEvent,
    UserMessageEvent,
    WebSearchCallItemEvent,
)
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputItem,
    InputMessageItem,
    OutputItem,
    OutputMessageItem,
    ReasoningItem,
    WebSearchCallItem,
)
from grasp_agents.types.llm_errors import LlmContextWindowError
from grasp_agents.types.llm_events import (
    OutputItemDone,
    ResponseCompleted,
    ResponseRetrying,
)
from grasp_agents.utils.errors import format_error_chain
from grasp_agents.utils.streaming import stream_concurrent
from grasp_agents.utils.validation import validate_obj_from_json_or_py_string

from .loop_state import (
    NextStep,
    NextStepContinue,
    NextStepForceFinalAnswer,
    NextStepForceResidentAnswer,
    NextStepResidentAnswer,
    NextStepRunTools,
    NextStepStop,
    decide_next_step,
)
from .task_progress import spill_if_large
from .tool_decision import (
    AllowTool,
    RaiseToolException,
    RejectToolContent,
    ToolCallDecision,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Mapping, Sequence

    from grasp_agents.hooks import (
        AfterLlmHook,
        AfterToolHook,
        BeforeLlmHook,
        BeforeToolHook,
        FinalAnswerExtractor,
        ToolInputConverter,
        ToolOutputConverter,
    )
    from grasp_agents.llm.llm import LLM
    from grasp_agents.session_context import SessionContext
    from grasp_agents.types.response import Response

    from .agent_context import AgentContext
    from .context_window import ContextWindowManager

logger = getLogger(__name__)


class CheckpointCallback(Protocol):
    async def __call__(
        self,
        *,
        turn: int = ...,
        location: AgentCheckpointLocation = ...,
        output: str | None = ...,
        stop_reason: StopReason | None = ...,
    ) -> None: ...


class ResponseCapture:
    """Wraps an event stream, capturing the final Response."""

    def __init__(self, stream: AsyncIterator[Event[Any]]) -> None:
        self._stream = stream
        self.response: Response | None = None

    def __aiter__(self) -> AsyncIterator[Event[Any]]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[Event[Any]]:
        async for event in self._stream:
            if isinstance(event, LLMStreamEvent) and isinstance(
                event.data, ResponseCompleted
            ):
                self.response = event.data.response
            yield event


def _decision_severity(decision: ToolCallDecision) -> int:
    if isinstance(decision, RaiseToolException):
        return 2
    if isinstance(decision, RejectToolContent):
        return 1
    return 0  # AllowTool


def _more_restrictive(
    existing: ToolCallDecision | None, new: ToolCallDecision
) -> ToolCallDecision:
    if existing is None:
        return new
    return new if _decision_severity(new) >= _decision_severity(existing) else existing


class AgentLoop[CtxT]:
    """
    The agentic execution loop: generate → check → tools → repeat.

    Holds the LLM, tools, memory, and all loop mechanics. Hooks are optional
    callback slots set by the owning LLMAgent.
    """

    agent_name: Final[str]
    stream_llm: Final[bool]
    final_answer_type: type[BaseModel]
    final_answer_as_tool_call: Final[bool]
    max_turns: Final[int]
    run_timeout: Final[float | None]
    force_react_mode: Final[bool]
    tracing_exclude_input_fields: Final[set[str] | None]

    # Mutable state

    turn: int
    ctx: SessionContext[CtxT]
    path: list[str] | None
    llm_output_schema: Any | None

    # Hooks and callbacks

    before_llm_hooks: list[BeforeLlmHook]
    after_llm_hooks: list[AfterLlmHook]
    before_tool_hooks: list[BeforeToolHook[CtxT]]
    after_tool_hooks: list[AfterToolHook]
    tool_output_converters: dict[str, ToolOutputConverter]
    tool_input_converters: dict[str, ToolInputConverter]
    final_answer_extractor: FinalAnswerExtractor | None
    checkpoint_callback: CheckpointCallback | None

    # Properties

    _agent_ctx: AgentContext
    _cw: ContextWindowManager
    _llm: LLM
    _final_answer: str | None
    _final_answer_tool: BaseTool[BaseModel, Any, CtxT]

    # Private state

    _deadline: float | None
    _message_start_turn: int
    _skip_call_ids: set[str]
    _inbox_poll_interval: float

    def __init__(
        self,
        *,
        agent_name: str,
        llm: LLM,
        ctx: SessionContext[CtxT],
        agent_ctx: AgentContext,
        context_window: ContextWindowManager,
        path: list[str] | None = None,
        final_answer_type: type[BaseModel] = BaseModel,
        final_answer_as_tool_call: bool = False,
        llm_output_schema: Any | None = None,
        stream_llm: bool = True,
        max_turns: int,
        run_timeout: float | None = None,
        tracing_exclude_input_fields: set[str] | None = None,
        force_react_mode: bool = False,
    ) -> None:
        super().__init__()

        # Mutable

        self.turn = 0
        self.ctx: SessionContext[CtxT] = ctx
        self.path = path
        self.llm_output_schema = llm_output_schema

        # Frozen

        self.agent_name = agent_name
        self.stream_llm = stream_llm
        self.max_turns = max_turns
        self.run_timeout = run_timeout
        self.force_react_mode = force_react_mode
        self.final_answer_type = final_answer_type
        self.final_answer_as_tool_call = final_answer_as_tool_call
        self.tracing_exclude_input_fields = tracing_exclude_input_fields

        # Properties

        self._llm = llm
        self._cw = context_window
        self._agent_ctx = agent_ctx
        self._final_answer = None

        # Private state

        self._deadline = None

        # The lifetime ``turn`` for a resident grows unbounded across inbox
        # messages; this marks where the current message's handling began, so the
        # per-message turn budget is ``turn - _message_start_turn`` (reset on each
        # delivered message — see ``_drain_inbox``).
        self._message_start_turn = 0

        # Tool call_ids whose results were already synthesized at the LLM
        # validation layer (bad arguments → ``LLMToolCallValidationError``
        # caught in :meth:`query_llm`). The dispatcher skips them — their
        # error tool_result is already in the transcript, and re-dispatch
        # would just trigger pydantic validation again.
        self._skip_call_ids = set()

        # Resident operation. The inbox lives on the agent context (the sibling
        # of ``bg_tasks``); when one is attached the loop consumes peer messages
        # from it between turns and runs until its task is cancelled from outside,
        # instead of terminating on a final answer. ``None`` (the default on the
        # context) keeps the original single-answer behavior unchanged.
        self._inbox_poll_interval: float = 0.05

        self.before_llm_hooks = []
        self.after_llm_hooks = []
        self.tool_output_converters = {}
        self.tool_input_converters = {}
        self.before_tool_hooks = []
        self.after_tool_hooks = []
        self.final_answer_extractor = None
        self.checkpoint_callback = None

        self._final_answer_tool = self._make_final_answer_tool()
        if final_answer_as_tool_call:
            if self._final_answer_tool.name in agent_ctx.tools:
                raise ValueError(
                    f"Agent '{agent_name}' has a tool named "
                    f"'{self._final_answer_tool.name}', colliding with the "
                    "final-answer tool; tool names must be unique."
                )
            agent_ctx.tools[self._final_answer_tool.name] = self._final_answer_tool

    @property
    def agent_ctx(self) -> AgentContext:
        return self._agent_ctx

    @property
    def final_answer(self) -> str | None:
        return self._final_answer

    @property
    def llm(self) -> LLM:
        return self._llm

    @property
    def cw(self) -> ContextWindowManager:
        return self._cw

    async def checkpoint(
        self,
        *,
        turn: int,
        location: AgentCheckpointLocation,
        output: str | None = None,
        stop_reason: StopReason | None = None,
    ) -> None:
        """Persist session state if a checkpoint callback is configured."""
        if self.checkpoint_callback:
            await self.checkpoint_callback(
                turn=turn, location=location, output=output, stop_reason=stop_reason
            )

    # --- Hook dispatch ---

    async def on_before_llm(
        self, *, exec_id: str, turn: int, extra_llm_settings: dict[str, Any]
    ) -> None:
        for hook in self.before_llm_hooks:
            await hook(
                exec_id=exec_id, turn=turn, extra_llm_settings=extra_llm_settings
            )

    async def on_after_llm(
        self, response: Response, *, exec_id: str, turn: int
    ) -> None:
        for hook in self.after_llm_hooks:
            await hook(response=response, exec_id=exec_id, turn=turn)

    async def on_before_tool(
        self, *, tool_calls: Sequence[FunctionToolCallItem], exec_id: str
    ) -> Mapping[str, ToolCallDecision] | None:
        # Stacked hooks each return per-call decisions; merge so the most
        # restrictive wins (raise > reject > allow) — no hook can downgrade
        # another's veto, and ties resolve to the later-registered hook.
        merged: dict[str, ToolCallDecision] = {}
        for hook in self.before_tool_hooks:
            decisions = await hook(tool_calls=tool_calls, ctx=self.ctx, exec_id=exec_id)
            for call_id, decision in (decisions or {}).items():
                merged[call_id] = _more_restrictive(merged.get(call_id), decision)

        return merged or None

    async def on_after_tool(
        self,
        *,
        tool_calls: Sequence[FunctionToolCallItem],
        tool_messages: Sequence[FunctionToolOutputItem],
        exec_id: str,
    ) -> None:
        for hook in self.after_tool_hooks:
            await hook(
                tool_calls=tool_calls, tool_messages=tool_messages, exec_id=exec_id
            )

    async def _convert_tool_output(
        self, output: Any, call: FunctionToolCallItem, *, exec_id: str
    ) -> FunctionToolOutputItem:
        converter = self.tool_output_converters.get(call.name)
        tool = self._agent_ctx.tools.get(call.name)
        untrusted = tool is not None and tool.untrusted_output

        # A tool failure surfaces as a ToolErrorInfo result — flag it so the UI
        # can mark the result panel (e.g. a red border).
        is_error = isinstance(output, ToolErrorInfo)

        if converter is not None:
            # A custom converter owns the output's shape, including any spilling.
            parts = await converter(output, exec_id=exec_id)
        else:
            parts = FunctionToolOutputItem.from_tool_result(
                call_id=call.call_id, output=output
            ).output
            if not is_error and isinstance(parts, str):
                parts = await spill_if_large(
                    self.ctx.file_backend,
                    name=call.call_id,
                    text=parts,
                    cap=tool.max_inline_result_chars if tool is not None else None,
                )

        # Fence external content so the model reads it as data, not instructions
        # (paired with the ``untrusted_content`` system-prompt section).
        if untrusted:
            parts = wrap_untrusted(parts, source=call.name)

        return FunctionToolOutputItem(
            call_id=call.call_id, output=parts, is_error=is_error
        )

    async def _convert_tool_input(
        self,
        call: FunctionToolCallItem,
        *,
        exec_id: str,
    ) -> BaseModel:
        tool = self._agent_ctx.tools[call.name]
        llm_args = validate_obj_from_json_or_py_string(
            call.arguments,
            schema=tool.llm_in_type,
            from_substring=False,
            strip_language_markdown=False,
        )
        converter = self.tool_input_converters.get(tool.name)

        if converter is not None:
            return await converter(llm_args, exec_id=exec_id)

        return llm_args

    def _extract_final_answer(
        self,
        *,
        response: Response,
        exec_id: str,
        **kwargs: Any,
    ) -> str | None:
        if self.final_answer_extractor is not None:
            return self.final_answer_extractor(
                exec_id=exec_id, response=response, **kwargs
            )

        if self.final_answer_as_tool_call:
            for tc in response.tool_call_items:
                if tc.name == self._final_answer_tool.name:
                    return tc.arguments
            return None

        if response.tool_call_items:
            return None

        return response.output_text or None

    # --- LLM generation ---

    async def _try_query_llm(
        self,
        *,
        tool_choice: ToolChoice | None = None,
        exec_id: str,
        extra_llm_settings: dict[str, Any],
    ) -> AsyncIterator[Event[Any]]:
        self._agent_ctx.transcript.validate_tool_call_pairing()

        llm_params: dict[str, Any] = {
            "input": await self._cw.project_view(exec_id=exec_id),
            "output_schema": self.llm_output_schema,
            "tools": self._agent_ctx.tools or None,
            "tool_choice": tool_choice,
            **extra_llm_settings,
        }

        response: Response | None = None

        if self.stream_llm:
            # Defer transcript writes until the response is validated.
            # Without this, items added mid-stream leak into the next
            # API call when LLM-layer validation_retries fires — the
            # bad assistant ``tool_calls`` end up in history without
            # matching ``tool_results``, and OpenAI 400s.
            pending: list[OutputItem] = []

            try:
                async for se in self._llm.generate_response_stream(**llm_params):
                    if isinstance(se, ResponseRetrying):
                        # Validation or transient API retry just fired —
                        # the previous attempt's items are about to be
                        # superseded by a fresh attempt. Discard them so
                        # the next attempt's items don't pile on top.
                        pending = []
                    if isinstance(se, OutputItemDone):
                        # Mirror the non-streaming commit: every output item —
                        # including server-tool records (web search) — enters
                        # the transcript, or the histories diverge and
                        # citation round-trips break.
                        pending.append(se.item)
                    elif isinstance(se, ResponseCompleted):
                        response = se.response

                    # Only LLMStream events are yielded immediately. The
                    # grasp-specific item events fire after the transcript
                    # write below (see ``_item_events``), so consumers can
                    # rely on the transcript already containing them.
                    yield LLMStreamEvent(
                        data=se, source=self.agent_name, exec_id=exec_id
                    )

            except LLMToolCallValidationError as exc:
                # Validation retries exhausted at the LLM layer. Commit
                # the bad assistant items + synthesize matching
                # tool_results so the next turn sees the failure and
                # can correct itself. The dispatcher will skip these
                # call_ids via ``_skip_call_ids``.
                if pending:
                    self._agent_ctx.transcript.update(pending)
                    for ev in self._item_events(pending, exec_id=exec_id):
                        yield ev

                async for ev in self._synthesize_validation_tool_results(
                    exc, pending, exec_id=exec_id
                ):
                    yield ev

                response = exc.response
                if response is not None:
                    yield LLMStreamEvent(
                        data=ResponseCompleted(response=response, sequence_number=0),
                        source=self.agent_name,
                        exec_id=exec_id,
                    )
                    self._record_llm_response(response, exec_id=exec_id)
                return

            # Clean completion → commit pending items, then surface their
            # item events (post-write, per the convention above).
            self._agent_ctx.transcript.update(pending)
            for ev in self._item_events(pending, exec_id=exec_id):
                yield ev

        else:
            try:
                response = await self._llm.generate_response(**llm_params)
                self._agent_ctx.transcript.update(response.output)

            except LLMToolCallValidationError as exc:
                response = exc.response
                if response is not None:
                    self._agent_ctx.transcript.update(response.output)
                    for ev in self._item_events(response.output, exec_id=exec_id):
                        yield ev

                async for ev in self._synthesize_validation_tool_results(
                    exc,
                    response.output if response is not None else [],
                    exec_id=exec_id,
                ):
                    yield ev

                if response is not None:
                    yield LLMStreamEvent(
                        data=ResponseCompleted(response=response, sequence_number=0),
                        source=self.agent_name,
                        exec_id=exec_id,
                    )
                    self._record_llm_response(response, exec_id=exec_id)
                return

            # LLMStream event immediate; item events after the write above.
            yield LLMStreamEvent(
                data=ResponseCompleted(response=response, sequence_number=0),
                source=self.agent_name,
                exec_id=exec_id,
            )
            for ev in self._item_events(response.output, exec_id=exec_id):
                yield ev

        if not response:
            return

        self._record_llm_response(response, exec_id=exec_id)

    async def _synthesize_validation_tool_results(
        self,
        exc: LLMToolCallValidationError,
        items: Sequence[OutputItem],
        *,
        exec_id: str,
    ) -> AsyncIterator[Event[Any]]:
        """
        Emit a ``FunctionToolOutputItem`` per failed tool call.

        Used when the LLM layer raised ``LLMToolCallValidationError``
        after exhausting ``validation_retries``. For each offending call
        we add a ``ToolErrorInfo`` tool_result to the transcript, yield
        a ``ToolOutputItemEvent`` so consoles render it, and mark the
        call_id so the dispatcher skips re-execution.
        """
        errors_by_id: dict[str, tuple[str, str]] = {
            call_id: (name, err) for call_id, name, err in exc.failed_calls
        }
        for item in items:
            if not isinstance(item, FunctionToolCallItem):
                continue
            name, error = errors_by_id.get(
                item.call_id,
                (item.name, "Tool call arguments failed validation."),
            )
            err_info = ToolErrorInfo(tool_name=name, error=error, timed_out=False)
            msg = FunctionToolOutputItem.from_tool_result(
                call_id=item.call_id, output=err_info
            )
            self._agent_ctx.transcript.update([msg])
            self._skip_call_ids.add(item.call_id)
            yield ToolOutputItemEvent(
                source=item.name, destination=self.agent_name, exec_id=exec_id, data=msg
            )

    def _item_events(
        self, items: Sequence[OutputItem], *, exec_id: str
    ) -> Iterator[Event[Any]]:
        """
        Yield the grasp-specific item event for each committed output item.

        Called only *after* ``items`` are written to the transcript, so a
        consumer reacting to one of these events can rely on the transcript
        already containing it. (``LLMStreamEvent``s are the raw, immediate
        counterpart — yielded during generation, before the write.)
        """
        for item in items:
            if isinstance(item, FunctionToolCallItem):
                yield ToolCallItemEvent(
                    source=self.agent_name,
                    destination=item.name,
                    exec_id=exec_id,
                    data=item,
                )
            elif isinstance(item, ReasoningItem):
                yield ReasoningItemEvent(
                    source=self.agent_name, exec_id=exec_id, data=item
                )
            elif isinstance(item, OutputMessageItem):
                yield OutputMessageItemEvent(
                    source=self.agent_name, exec_id=exec_id, data=item
                )
            elif isinstance(item, WebSearchCallItem):
                # A server-side web search/fetch call — surface it so consoles
                # show what the model searched, instead of dropping it.
                yield WebSearchCallItemEvent(
                    source=self.agent_name, exec_id=exec_id, data=item
                )
            # UnknownItem: kept in the transcript for round-trip fidelity but
            # has no dedicated event.

    @traced(name="generate")
    async def query_llm(
        self,
        *,
        tool_choice: ToolChoice | None = None,
        exec_id: str,
        extra_llm_settings: dict[str, Any],
    ) -> AsyncIterator[Event[Any]]:
        try:
            async for event in self._try_query_llm(
                tool_choice=tool_choice,
                exec_id=exec_id,
                extra_llm_settings=extra_llm_settings,
            ):
                yield event

        except LlmContextWindowError:
            # The view overflowed the window despite (or without) proactive
            # compaction. Force a fold and retry once; if nothing can be
            # folded, surface the error.
            fold = await self._cw.maybe_compact(exec_id=exec_id, force=True)
            if fold is None:
                raise
            yield self._cw.compaction_event(fold, exec_id=exec_id)
            logger.warning(
                "agent '%s' hit the context window; compacted and retrying",
                self.agent_name,
            )
            async for event in self._try_query_llm(
                tool_choice=tool_choice,
                exec_id=exec_id,
                extra_llm_settings=extra_llm_settings,
            ):
                yield event

    # --- Tool calling ---

    async def execute_tools_stream(
        self, calls: Sequence[FunctionToolCallItem], exec_id: str
    ) -> AsyncIterator[Event[Any]]:
        # Tool call events are now emitted from query_llm via OutputItemDone promotion

        # Resolve inputs and partition the calls by whether they background at
        # all (``auto_background_at is not None``). The manager owns *when* a
        # backgroundable call hands off (immediately for ``0``, after a deadline
        # race for a positive value) and whether it then gates the final answer
        # (the tool's ``blocks_final_answer``); the loop just routes it there.
        # Everything else is a plain foreground call (the immediate batch).
        outputs: list[Any] = [None] * len(calls)
        immediate: list[
            tuple[int, FunctionToolCallItem, BaseTool[BaseModel, Any, CtxT], BaseModel]
        ] = []
        backgroundable: list[
            tuple[int, FunctionToolCallItem, BaseTool[BaseModel, Any, CtxT], BaseModel]
        ] = []
        # Per-call sentinel: True means a synthesized tool_result is
        # already in the transcript (LLM-layer validation failure path)
        # — the post-loop convert step skips it to avoid emitting a
        # second tool_result for the same call_id.
        skipped: list[bool] = [False] * len(calls)

        for i, call in enumerate(calls):
            if call.call_id in self._skip_call_ids:
                # ``query_llm`` already synthesized + recorded a
                # ToolErrorInfo for this call. Don't dispatch and don't
                # re-emit a tool_result below.
                self._skip_call_ids.discard(call.call_id)
                skipped[i] = True
                continue
            tool = self._agent_ctx.tools[call.name]
            try:
                inp = await self._convert_tool_input(call, exec_id=exec_id)
            except Exception as err:
                # A bad input must become a tool_result the model can react
                # to next turn, never an uncaught crash of the whole run.
                outputs[i] = f"Tool '{call.name}' input is invalid: {err}"
                logger.warning(
                    "Tool '%s' (call_id %s) input conversion failed",
                    call.name,
                    call.call_id,
                    exc_info=True,
                )
                continue
            if tool.auto_background_at is not None:
                backgroundable.append((i, call, tool, inp))
            else:
                immediate.append((i, call, tool, inp))

        # Launch backgroundable calls now so each races its own
        # ``auto_background_at`` concurrently with the immediate batch. The
        # manager returns the result if the call finished in the foreground,
        # else a launch note + a ``BackgroundTaskLaunchedEvent`` to bubble.
        bg_tasks_async: dict[int, asyncio.Task[Any]] = {
            i: asyncio.create_task(
                self._agent_ctx.bg_tasks.run_backgroundable(
                    call,
                    tool,
                    inp,
                    ctx=self.ctx,
                    exec_id=exec_id,
                    agent_ctx=self._agent_ctx,
                )
            )
            for i, call, tool, inp in backgroundable
        }

        try:
            if immediate:
                # Foreground tools always run through ``run_stream`` so a
                # streaming tool's incremental events — and a sub-agent /
                # processor tool's nested events — bubble live. A non-streaming
                # tool's default ``run_stream`` just yields its single terminal
                # event, so nothing is lost; ``run`` stays the direct path for
                # tests / debugging.
                streams = [
                    tool.run_stream(
                        inp=inp,
                        ctx=self.ctx,
                        exec_id=exec_id,
                        path=make_tool_call_path(self.path, call.call_id),
                        agent_ctx=self._agent_ctx,
                    )
                    for _, call, tool, inp in immediate
                ]
                # Serialize the batch (each stream drained fully before the
                # next) when two calls need exclusive access to overlapping keys
                # so their writes can't interleave; otherwise run concurrently.
                # Per-stream failure isolation comes from ``merged.errors`` in
                # both modes.
                conflict = batch_has_concurrency_conflict(
                    [(tool, inp) for _, _, tool, inp in immediate]
                )
                merged = stream_concurrent(
                    streams, max_concurrency=1 if conflict else None
                )
                async for stream_idx, event in merged:
                    # Capture the tool's terminal event — its result
                    # (ToolOutputEvent) or failure (ToolErrorEvent). Nested
                    # sub-agent tool events bubble through too; the tool's OWN
                    # terminal is always emitted last.
                    if isinstance(event, (ToolOutputEvent, ToolErrorEvent)):
                        outputs[immediate[stream_idx][0]] = event.data
                    yield event

                for err in merged.errors:
                    i = immediate[err.index][0]
                    tool_name = immediate[err.index][2].name
                    outputs[i] = (
                        f"Tool '{tool_name}' failed: "
                        f"{format_error_chain(err.exception)}"
                    )
                    logger.warning(
                        "Tool '%s' (call index %d) failed",
                        tool_name,
                        i,
                        exc_info=err.exception,
                    )

            # Collect backgroundable results — each already raced its own
            # deadline concurrently with the immediate batch (finished in the
            # foreground, or backgrounded into ``bg_tasks`` and returned a launch
            # note + a ``BackgroundTaskLaunchedEvent`` to bubble).
            for i, bg_task in bg_tasks_async.items():
                outputs[i], launched = await bg_task
                if launched is not None:
                    yield launched
        finally:
            # On cancellation (turn abort), don't leak a call still racing in the
            # foreground (an already-backgrounded one is done here and lives on
            # under ``bg_tasks``).
            for bg_task in bg_tasks_async.values():
                if not bg_task.done():
                    bg_task.cancel()

        tool_messages: list[FunctionToolOutputItem] = []

        for idx, (output, call) in enumerate(zip(outputs, calls, strict=True)):
            if skipped[idx]:
                # Tool result already synthesized in ``query_llm`` for
                # this call_id (LLM validation failure path).
                continue
            msg = await self._convert_tool_output(output, call, exec_id=exec_id)
            tool_messages.append(msg)
            self._agent_ctx.transcript.update([msg])
            yield ToolOutputItemEvent(
                source=call.name, destination=self.agent_name, exec_id=exec_id, data=msg
            )

        if self.ctx.printer:
            self.ctx.printer.print_messages(
                tool_messages,
                agent_name=self.agent_name,
                exec_id=exec_id,
            )

    # --- Final answer ---

    @traced(name="force_generate_final_answer")
    async def _force_generate_final_answer_stream(
        self,
        exec_id: str,
        extra_llm_settings: dict[str, Any],
    ) -> AsyncIterator[Event[Any]]:
        user_message = InputMessageItem.from_text(
            wrap_in_system_reminder(
                "Exceeded the maximum number of turns: provide a final answer now!"
            ),
            role="user",
        )
        self._agent_ctx.transcript.update([user_message])
        # TODO: set source
        yield UserMessageEvent(
            source=None,
            destination=self.agent_name,
            exec_id=exec_id,
            data=user_message,
        )

        if self.ctx.printer:
            self.ctx.printer.print_messages(
                [user_message],
                agent_name=self.agent_name,
                exec_id=exec_id,
            )

        settings = deepcopy(extra_llm_settings)
        await self.on_before_llm(
            extra_llm_settings=settings, turn=self.turn, exec_id=exec_id
        )

        hook_tool_choice = settings.pop("tool_choice", None)
        tool_choice = (
            NamedToolChoice(name=self._final_answer_tool.name)
            if self.final_answer_as_tool_call
            else hook_tool_choice
        )
        stream = ResponseCapture(
            self.query_llm(
                tool_choice=tool_choice,
                extra_llm_settings=settings,
                exec_id=exec_id,
            ),
        )
        async for event in stream:
            yield event

        assert stream.response is not None

        await self.on_after_llm(stream.response, turn=self.turn, exec_id=exec_id)

        self._final_answer = self._extract_final_answer(
            response=stream.response, exec_id=exec_id
        )
        if self._final_answer is None:
            raise AgentFinalAnswerError(proc_name=self.agent_name, exec_id=exec_id)

        closures = self._close_stop_tool_calls(stream.response)
        for closure_event in self._closure_events(closures, exec_id=exec_id):
            yield closure_event

    # --- Tool call utils ---

    def _compute_tool_choice(self, had_tool_calls: bool) -> ToolChoice:
        """Compute tool_choice for the current turn."""
        if not self.force_react_mode:
            return "auto"
        # force_react_mode alternates: reason (no tools) → act (must use tools) → …
        if self.turn == 0 or had_tool_calls:
            return "none"  # first turn or just acted → reason
        return "required"  # just reasoned → must act

    def _unanswered_tool_calls(self, response: Response) -> list[FunctionToolCallItem]:
        """
        The response's tool calls that have no tool_result in the transcript.

        A call whose result was already synthesized (LLM-layer validation
        failure) must not be paired a second time — providers reject
        duplicate tool_results, and they would persist into checkpoints.
        """
        tool_calls = response.tool_call_items
        if not tool_calls:
            return []
        answered = {
            m.call_id
            for m in self._agent_ctx.transcript.messages
            if isinstance(m, FunctionToolOutputItem)
        }
        for tc in tool_calls:
            self._skip_call_ids.discard(tc.call_id)
        return [tc for tc in tool_calls if tc.call_id not in answered]

    def _close_dangling_tool_calls(
        self, response: Response
    ) -> list[tuple[FunctionToolCallItem, FunctionToolOutputItem]]:
        """
        Inject synthetic tool outputs for tool calls that will never execute.

        When the loop stops (e.g. max_turns) after ACT generated tool calls,
        those calls are already in the transcript but have no outputs. Most LLM APIs
        reject requests with unmatched tool_use/tool_result pairs. This method
        adds cancellation outputs so the conversation stays valid.
        """
        closures = [
            (
                tc,
                FunctionToolOutputItem.from_tool_result(
                    call_id=tc.call_id,
                    output="[Tool call cancelled: agent reached turn limit]",
                ),
            )
            for tc in self._unanswered_tool_calls(response)
        ]
        if closures:
            self._agent_ctx.transcript.update([msg for _, msg in closures])

        return closures

    def _close_stop_tool_calls(
        self, response: Response
    ) -> list[tuple[FunctionToolCallItem, FunctionToolOutputItem]]:
        """
        Pair every unanswered tool call in a stopping response with a
        synthetic output.

        A stop extracted from a ``final_answer`` tool call (forced or
        voluntary) — or from a custom extractor that stops despite pending
        calls — leaves the calls in the transcript with no results. The next
        run on the same transcript (or a durable resume) would then violate
        the tool_call → tool_result pairing invariant.
        """
        closures = [
            (
                tc,
                FunctionToolOutputItem.from_tool_result(
                    call_id=tc.call_id,
                    output=(
                        "Final answer recorded."
                        if tc.name == self._final_answer_tool.name
                        else "[Tool call cancelled: agent stopped with a final answer]"
                    ),
                ),
            )
            for tc in self._unanswered_tool_calls(response)
        ]
        if closures:
            self._agent_ctx.transcript.update([msg for _, msg in closures])

        return closures

    def _closure_events(
        self,
        closures: list[tuple[FunctionToolCallItem, FunctionToolOutputItem]],
        *,
        exec_id: str,
    ) -> Iterator[Event[Any]]:
        """
        Announce synthetic closure outputs on the event stream (+ printer).

        Parity with every other synthetic tool_result (validation synthesis,
        rejections): the transcript write is surfaced as a
        :class:`ToolOutputItemEvent`, so event-driven consumers (UIs,
        embedders mirroring history) never diverge from the persisted
        transcript. Deliberately NOT fed to ``on_after_tool`` — closures are
        bookkeeping, not tool executions.
        """
        if not closures:
            return
        for call, msg in closures:
            yield ToolOutputItemEvent(
                source=call.name,
                destination=self.agent_name,
                exec_id=exec_id,
                data=msg,
            )
        if self.ctx.printer:
            self.ctx.printer.print_messages(
                [msg for _, msg in closures],
                agent_name=self.agent_name,
                exec_id=exec_id,
            )

    # --- Next-step classification ---

    def _decide_next_step(
        self,
        response: Response,
        *,
        exec_id: str,
    ) -> NextStep:
        """
        Classify the post-ACT loop transition.

        Delegates to the pure :func:`decide_next_step` so the JUDGE-phase
        state machine can be tested without mocking the loop.
        """
        final = self._extract_final_answer(
            response=response,
            exec_id=exec_id,
            turn=self.turn,
        )
        return decide_next_step(
            final_answer=final,
            tool_calls=response.tool_call_items,
            turn=self.turn,
            max_turns=self.max_turns,
            blocking_bg_tasks=self._agent_ctx.bg_tasks.has_blocking_tasks,
            deadline_exceeded=self._deadline is not None
            and time.monotonic() >= self._deadline,
            inbox_open=self._agent_ctx.inbox is not None,
            turns_on_message=self.turn - self._message_start_turn,
        )

    # --- Per-state handlers (dispatched from execute_stream) ---

    async def _handle_run_tools(
        self,
        step: NextStepRunTools,
        *,
        exec_id: str,
    ) -> AsyncIterator[Event[Any]]:
        r"""
        ``NextStepRunTools``: execute tools, checkpoint, emit TurnEnd.

        Non-terminal: caller continues the loop after this handler completes.

        Before any tool runs, the ``BeforeToolHook`` is consulted for
        per-call :class:`ToolCallDecision`\ s. A :class:`RaiseToolException`
        anywhere aborts the batch; :class:`RejectToolContent` synthesizes
        a tool output and skips execution; otherwise the call runs
        normally.
        """
        # The approval gate (or any before-tool hook) can park indefinitely;
        # bound it by the run deadline so ``run_timeout`` holds here too. On
        # expiry the calls are rejected and the next JUDGE forces a final
        # answer with ``StopReason.TIMEOUT``.
        try:
            if self._deadline is not None:
                async with asyncio.timeout_at(
                    asyncio.get_running_loop().time()
                    + max(0.0, self._deadline - time.monotonic())
                ):
                    decisions = await self.on_before_tool(
                        tool_calls=step.tool_calls, exec_id=exec_id
                    )
            else:
                decisions = await self.on_before_tool(
                    tool_calls=step.tool_calls, exec_id=exec_id
                )
        except TimeoutError:
            logger.warning(
                "Before-tool hook exceeded the run deadline (%ss); "
                "rejecting the batch.",
                self.run_timeout,
            )
            decisions = {
                call.call_id: RejectToolContent(
                    content="[Tool call cancelled: run deadline exceeded]"
                )
                for call in step.tool_calls
            }

        if decisions:
            for decision in decisions.values():
                if isinstance(decision, RaiseToolException):
                    raise decision.exception

        tool_msgs: list[FunctionToolOutputItem] = []
        allowed_calls: list[FunctionToolCallItem] = []
        rejection_msgs: list[FunctionToolOutputItem] = []

        for call in step.tool_calls:
            if call.call_id in self._skip_call_ids:
                # ``query_llm`` already synthesized a tool_result for this
                # call (argument-validation failure). A Reject decision here
                # would append a second output for the same call_id — route
                # it to the dispatcher, which consumes and skips it.
                allowed_calls.append(call)
                continue
            decision = (decisions or {}).get(call.call_id, AllowTool())
            if isinstance(decision, RejectToolContent):
                msg = FunctionToolOutputItem.from_tool_result(
                    call_id=call.call_id, output=decision.content
                )
                tool_msgs.append(msg)
                rejection_msgs.append(msg)
                self._agent_ctx.transcript.update([msg])
                yield ToolOutputItemEvent(
                    source=call.name,
                    destination=self.agent_name,
                    exec_id=exec_id,
                    data=msg,
                )
            else:
                allowed_calls.append(call)

        if rejection_msgs and self.ctx.printer:
            self.ctx.printer.print_messages(
                rejection_msgs,
                agent_name=self.agent_name,
                exec_id=exec_id,
            )

        if allowed_calls:
            async for event in self.execute_tools_stream(
                allowed_calls, exec_id=exec_id
            ):
                # A foreground sub-agent's internal tool_results bubble
                # through with the sub-agent as destination — collecting
                # them would hand user hooks results that don't pair with
                # this agent's calls.
                if (
                    isinstance(event, ToolOutputItemEvent)
                    and event.destination == self.agent_name
                ):
                    tool_msgs.append(event.data)
                yield event

        await self.on_after_tool(
            tool_calls=step.tool_calls, tool_messages=tool_msgs, exec_id=exec_id
        )

        # Resume point is the NEXT turn: this turn's ACT (the tool calls) and
        # its results are already in the transcript, so resuming must continue
        # to turn+1's generation — not re-run this turn's ACT (which would
        # re-issue the same tool calls, e.g. re-launching background workers).
        # The loop increments ``turn`` right after this handler returns.
        await self.checkpoint(
            turn=self.turn + 1,
            location=AgentCheckpointLocation.AFTER_TOOL_RESULT,
        )

        yield TurnEndEvent(
            source=self.agent_name,
            exec_id=exec_id,
            data=TurnEndInfo(
                turn=self.turn, had_tool_calls=True, tool_outputs=tool_msgs
            ),
        )

    async def _handle_continue(
        self,
        *,
        exec_id: str,
    ) -> AsyncIterator[Event[Any]]:
        """
        ``NextStepContinue``: no tools, no final answer — just emit TurnEnd.

        Non-terminal: reached on reason-turns under ``force_react_mode`` or
        when a final answer is suppressed by pending background tasks.
        """
        yield TurnEndEvent(
            source=self.agent_name,
            exec_id=exec_id,
            data=TurnEndInfo(turn=self.turn, had_tool_calls=False),
        )

    async def _handle_stop(
        self,
        step: NextStepStop,
        response: Response,
        *,
        exec_id: str,
    ) -> AsyncIterator[Event[Any]]:
        """``NextStepStop``: final answer extracted, end loop cleanly."""
        self._final_answer = step.final_answer
        closures = self._close_stop_tool_calls(response)
        for closure_event in self._closure_events(closures, exec_id=exec_id):
            yield closure_event

        await self.checkpoint(
            turn=self.turn,
            location=AgentCheckpointLocation.AFTER_FINAL_ANSWER,
            output=self._final_answer,
            stop_reason=step.stop_reason,
        )

        yield TurnEndEvent(
            source=self.agent_name,
            exec_id=exec_id,
            data=TurnEndInfo(
                turn=self.turn,
                had_tool_calls=False,
                stop_reason=step.stop_reason,
                tool_outputs=[msg for _, msg in closures],
            ),
        )

    async def _handle_force_final_answer(
        self,
        response: Response,
        *,
        exec_id: str,
        stop_reason: StopReason,
        extra_llm_settings: dict[str, Any],
    ) -> AsyncIterator[Event[Any]]:
        """
        ``NextStepForceFinalAnswer``: budget exhausted (turn count or the run's
        wall-clock deadline).

        Closes dangling tool calls, force-generates a final answer, and ends
        the loop with the given ``stop_reason`` (``MAX_TURNS`` or ``TIMEOUT``).
        Background tasks are NOT cancelled — exhausting a turn budget must not
        kill deliberately backgrounded work; completion notes land at a later
        run's drain.
        """
        closures = self._close_dangling_tool_calls(response)
        for closure_event in self._closure_events(closures, exec_id=exec_id):
            yield closure_event

        async for event in self._force_generate_final_answer_stream(
            exec_id=exec_id,
            extra_llm_settings=extra_llm_settings,
        ):
            yield event

        await self.checkpoint(
            turn=self.turn,
            location=AgentCheckpointLocation.AFTER_FINAL_ANSWER,
            output=self._final_answer,
            stop_reason=stop_reason,
        )

        yield TurnEndEvent(
            source=self.agent_name,
            exec_id=exec_id,
            data=TurnEndInfo(
                turn=self.turn,
                had_tool_calls=False,
                stop_reason=stop_reason,
                tool_outputs=[msg for _, msg in closures],
            ),
        )

        if stop_reason is StopReason.TIMEOUT:
            logger.info(
                "Run timeout reached: %ss. Forcing a final answer.",
                self.run_timeout,
            )
        else:
            logger.info(
                "Max turns reached: %s. Exiting the tool call loop.",
                self.max_turns,
            )

    async def _handle_resident_answer(
        self, step: NextStepResidentAnswer, *, exec_id: str
    ) -> AsyncIterator[Event[Any]]:
        """
        ``NextStepResidentAnswer``: a resident produced its answer to the current
        message and the open inbox recycles the loop instead of stopping. Persist
        the turn — a resident answer otherwise never checkpoints, so the answer
        would be re-generated on resume — which also releases any message this turn
        absorbed (the ack flush in :meth:`checkpoint`). Non-terminal: the loop
        continues and parks for the next message (the tail is now an answer, so it
        no longer owes a response).
        """
        del step

        await self.checkpoint(
            turn=self.turn + 1,
            location=AgentCheckpointLocation.AFTER_RESIDENT_ANSWER,
        )

        yield TurnEndEvent(
            source=self.agent_name,
            exec_id=exec_id,
            data=TurnEndInfo(turn=self.turn, had_tool_calls=False),
        )

    async def _handle_force_resident_answer(
        self,
        response: Response,
        *,
        exec_id: str,
        extra_llm_settings: dict[str, Any],
    ) -> AsyncIterator[Event[Any]]:
        """
        ``NextStepForceResidentAnswer``: a resident burned its per-message turn
        budget without an answer it can return. The resident analog of
        :meth:`_handle_force_final_answer`: close dangling tool calls, force-generate
        a final answer for this message, persist it (``AFTER_RESIDENT_ANSWER``,
        releasing the message), then park for the next — the run continues. Caps one
        runaway message; background tasks are left running (same as the lone-agent
        force path).
        """
        logger.warning(
            "Resident '%s' hit its per-message turn budget (%s) — forcing an answer "
            "and moving on to the next message.",
            self.agent_name,
            self.max_turns,
        )
        closures = self._close_dangling_tool_calls(response)
        for closure_event in self._closure_events(closures, exec_id=exec_id):
            yield closure_event

        async for event in self._force_generate_final_answer_stream(
            exec_id=exec_id,
            extra_llm_settings=extra_llm_settings,
        ):
            yield event

        await self.checkpoint(
            turn=self.turn + 1,
            location=AgentCheckpointLocation.AFTER_RESIDENT_ANSWER,
        )

        yield TurnEndEvent(
            source=self.agent_name,
            exec_id=exec_id,
            data=TurnEndInfo(
                turn=self.turn,
                had_tool_calls=False,
                tool_outputs=[msg for _, msg in closures],
            ),
        )

    # --- ACT phase ---

    async def _run_act_stream(
        self,
        *,
        exec_id: str,
        extra_llm_settings: dict[str, Any],
        had_tool_calls: bool,
    ) -> AsyncIterator[Event[Any]]:
        """
        ACT phase: prepare settings, query the LLM, observe the response.

        Dispatches ``on_before_llm`` / ``on_after_llm`` hooks around the LLM
        call, yields all streaming events from :meth:`query_llm`, and
        concludes with a :class:`GenerationEndEvent` carrying the response.
        Symmetric with :meth:`_force_generate_final_answer_stream`, which
        wraps ``query_llm`` with the same :class:`ResponseCapture` idiom.

        Callers receive the response via :class:`ResponseCapture` around
        this stream.
        """
        settings = deepcopy(extra_llm_settings)
        await self.on_before_llm(
            extra_llm_settings=settings, turn=self.turn, exec_id=exec_id
        )

        tool_choice: ToolChoice | None = None
        if self._agent_ctx.tools:
            tool_choice = self._compute_tool_choice(had_tool_calls)
        tool_choice = settings.pop("tool_choice", tool_choice)

        stream = ResponseCapture(
            self.query_llm(
                tool_choice=tool_choice, extra_llm_settings=settings, exec_id=exec_id
            ),
        )
        async for event in stream:
            yield event

        response = stream.response
        assert response is not None

        await self.on_after_llm(response, turn=self.turn, exec_id=exec_id)

        yield GenerationEndEvent(source=self.agent_name, exec_id=exec_id, data=response)

    # --- Inbox / background-task message queues ---

    async def _await_bg_tasks(self) -> None:
        idle_timeout = (
            max(0.0, self._deadline - time.monotonic())
            if self._deadline is not None
            else None
        )
        await self._agent_ctx.bg_tasks.wait_idle(timeout=idle_timeout)

    async def _await_inbox(self) -> None:
        """
        Resident PRE-ACT wait — the inbox counterpart to
        :meth:`BackgroundTaskManager.wait_idle`. Block until there is something to
        deliver: a peer message queued in the inbox, or a pending background-task
        completion (both surfaced by the drains below). Ended from outside by
        cancelling the resident run's task.
        """
        inbox = self._agent_ctx.inbox
        assert inbox is not None

        # Mark the inbox parked ("waiting") so a team supervisor can read this loop
        # as idle while it blocks here (the per-actor quiescence signal).
        with inbox.waiting():
            while not (
                await inbox.has_pending()
                or self._agent_ctx.bg_tasks.has_undelivered_completions
            ):
                await inbox.wait(timeout=self._inbox_poll_interval)

    async def _drain_inbox(self, *, exec_id: str) -> AsyncIterator[Event[Any]]:
        """
        Deliver the next peer / human message as a user turn at the turn boundary —
        the resident counterpart to :meth:`BackgroundTaskManager.drain`. The inbox
        leases the message (released on this turn's checkpoint), and holds the next
        until then, so one message lands per turn yet new mail enters *between*
        turns — the agent need not reach a final answer first. A no-op with no inbox
        attached (non-resident) or none queued (the wait was woken by a
        background-task completion instead).
        """
        inbox = self._agent_ctx.inbox
        if inbox is None:
            return

        message = await inbox.take()
        if message is None:
            return

        # A new message resets the per-message turn budget (see decide_next_step).
        self._message_start_turn = self.turn
        item = message.to_input_message()
        self._agent_ctx.transcript.update([item])

        # ``source`` names the mailbox sender (a peer, or "user" for human
        # input), so a UI can tell queued human turns from peer hand-offs.
        yield UserMessageEvent(
            data=item,
            source=message.sender,
            destination=self.agent_name,
            exec_id=exec_id,
        )

    def _turn_input_messages(self) -> list[InputItem]:
        """
        The input messages the upcoming turn responds to: the transcript's
        trailing run of input items (the step's input message, a resident's
        drained inbox message, injected background-task notes). Empty when the
        turn follows a tool round or an answer.
        """
        inputs: list[InputItem] = []
        for item in reversed(self._agent_ctx.transcript.messages):
            if not isinstance(item, InputMessageItem):
                break
            inputs.append(item)
        inputs.reverse()
        return inputs

    # --- Main execution loop ---

    async def execute_stream(
        self,
        exec_id: str,
        extra_llm_settings: dict[str, Any] | None = None,
    ) -> AsyncIterator[Event[Any]]:
        had_tool_calls = False
        self._final_answer = None

        # Start (or, on resume, restart) the current message's per-message turn
        # budget from here — a resumed resident gets a fresh budget for whatever
        # message it was mid-handling, which only ever loosens the cap.
        self._message_start_turn = self.turn

        self._deadline = (
            time.monotonic() + self.run_timeout
            if self.run_timeout is not None
            else None
        )

        # Checkpoint after input memorization (new step only)
        if self.turn == 0:
            await self.checkpoint(
                turn=self.turn,
                location=AgentCheckpointLocation.AFTER_INPUT,
            )

        extra_llm_settings = deepcopy(extra_llm_settings or {})

        # A resident agent (inbox attached) loops until its task is cancelled from
        # outside — it must NOT exit on the lifetime turn budget, which bounds a lone
        # agent's whole run. A resident is instead bounded *per inbox message*
        # (``turns_on_message`` in ``decide_next_step``): a runaway message force-
        # finalizes and the loop moves on, but the run itself only ends on cancel.
        while self._agent_ctx.inbox is not None or self.turn <= self.max_turns:
            # ── PRE-ACT: prepare for generation ──

            # Wait for new inbox messages or background-task completions only if
            # the agent has nothing to respond to (user messages or tool outputs)
            # at this turn.
            if not self._agent_ctx.transcript.owes_response:
                if self._agent_ctx.inbox is not None:
                    # Resident: park for the next inbox message (or a pending
                    # background completion). Indefinite — ended only when the
                    # run's task is cancelled from outside.
                    await self._await_inbox()
                else:
                    # Lone agent: a bounded idle-wait for outstanding
                    # background work (returns at once when there is none).
                    # Capped by the remaining run-timeout budget so a
                    # non-blocking task that never completes can't block past
                    # the deadline (the next JUDGE check then stops the run).
                    await self._await_bg_tasks()

            # Turn-boundary delivery: the next resident inbox message as a user
            # turn, then any background-task completions (bubbling their events,
            # mirroring stream output to the .grasp logs, etc.).
            async for event in self._drain_inbox(exec_id=exec_id):
                yield event

            bg_tasks = self._agent_ctx.bg_tasks
            async for event in bg_tasks.drain(exec_id=exec_id, ctx=self.ctx):
                yield event

            # Compaction: fold an old span before generating if the view
            # approaches the budget (no-op without a compactor / under budget).
            fold = await self._cw.maybe_compact(exec_id=exec_id)
            if fold is not None:
                yield self._cw.compaction_event(fold, exec_id=exec_id)

            yield TurnStartEvent(
                source=self.agent_name,
                exec_id=exec_id,
                data=TurnInfo(
                    turn=self.turn, input_messages=self._turn_input_messages()
                ),
            )

            # ── ACT: LLM generates response ──

            act = ResponseCapture(
                self._run_act_stream(
                    exec_id=exec_id,
                    extra_llm_settings=extra_llm_settings,
                    had_tool_calls=had_tool_calls,
                ),
            )
            async for event in act:
                yield event

            assert act.response is not None
            response = act.response

            # ── JUDGE: classify next transition ──

            step = self._decide_next_step(response, exec_id=exec_id)

            if logger.isEnabledFor(DEBUG):
                n_calls = (
                    len(step.tool_calls) if isinstance(step, NextStepRunTools) else 0
                )
                logger.debug(
                    "agent '%s' turn %d → %s%s",
                    self.agent_name,
                    self.turn,
                    type(step).__name__,
                    f" ({n_calls} tool calls)" if n_calls else "",
                )

            # ── Dispatch to per-state handler ──

            if isinstance(step, NextStepRunTools):
                async for event in self._handle_run_tools(step, exec_id=exec_id):
                    yield event

            if isinstance(step, NextStepContinue):
                async for event in self._handle_continue(exec_id=exec_id):
                    yield event

            if isinstance(step, NextStepStop):
                async for event in self._handle_stop(step, response, exec_id=exec_id):
                    yield event
                return

            if isinstance(step, NextStepForceFinalAnswer):
                async for event in self._handle_force_final_answer(
                    response,
                    exec_id=exec_id,
                    stop_reason=step.stop_reason,
                    extra_llm_settings=extra_llm_settings,
                ):
                    yield event
                return

            if isinstance(step, NextStepResidentAnswer):
                async for event in self._handle_resident_answer(step, exec_id=exec_id):
                    yield event

            if isinstance(step, NextStepForceResidentAnswer):
                async for event in self._handle_force_resident_answer(
                    response,
                    exec_id=exec_id,
                    extra_llm_settings=extra_llm_settings,
                ):
                    yield event

            had_tool_calls = isinstance(step, NextStepRunTools)
            self.turn += 1

    def _make_final_answer_tool(self) -> BaseTool[BaseModel, None, Any]:
        class FinalAnswerTool(BaseTool[self.final_answer_type, None, Any]):
            name = "final_answer"
            description = (
                "You must call this tool to provide the final answer. "
                "DO NOT output your answer before calling the tool. "
            )

            async def _run(
                self,
                inp: BaseModel,
                *,
                ctx: SessionContext[Any] | None = None,
                exec_id: str | None = None,
                progress_callback: Any = None,
                path: Sequence[str] | None = None,
                agent_ctx: AgentContext | None = None,
            ) -> None:
                del inp, ctx, exec_id, progress_callback, path, agent_ctx

        return FinalAnswerTool()

    def _record_llm_response(self, response: Response, *, exec_id: str) -> None:
        self.ctx.record_response(self.agent_name, response)
        usage = response.usage

        if usage is not None and usage.input_tokens:
            # Anchor the compaction budget on the provider's exact reported count
            # for the view just sent.
            self._cw.note_response_usage(usage.input_tokens)

        self.ctx.usage_tracker.update(agent_name=self.agent_name, responses=[response])

        # Mirror generated output (reasoning, text, tool calls) to the raw debug
        # printer (``ctx.printer``), if attached — the counterpart to the input /
        # tool-result mirroring elsewhere in the loop, so a non-streaming run
        # with a :class:`Printer` shows the full raw conversation, model output
        # included. (For live token-by-token raw output, stream through
        # ``print_events`` instead; don't also set ``ctx.printer``.)
        if self.ctx.printer:
            self.ctx.printer.print_messages(
                response.output, agent_name=self.agent_name, exec_id=exec_id
            )
