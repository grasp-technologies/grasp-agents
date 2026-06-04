from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from logging import getLogger
from typing import TYPE_CHECKING, Any, Final, Generic, Protocol

from pydantic import BaseModel, TypeAdapter

from grasp_agents.durability.checkpoints import AgentCheckpointLocation
from grasp_agents.durability.store_keys import make_tool_call_path
from grasp_agents.run_context import CtxT, RunContext
from grasp_agents.telemetry import traced
from grasp_agents.tools.bash import (
    BashProcessRegistry,
    BashSessionHolder,
    reset_current_bash_registry,
    reset_current_session_holder,
    set_current_bash_registry,
    set_current_session_holder,
)
from grasp_agents.tools.file_edit.agent_state import (
    reset_current_file_edit_state,
    set_current_file_edit_state,
)
from grasp_agents.tools.file_edit.session_state import FileEditSessionState
from grasp_agents.types.errors import AgentFinalAnswerError, LLMToolCallValidationError
from grasp_agents.types.events import (
    Event,
    GenerationEndEvent,
    LLMStreamEvent,
    OutputMessageItemEvent,
    ReasoningItemEvent,
    StopReason,
    ToolCallItemEvent,
    ToolErrorInfo,
    ToolOutputEvent,
    ToolOutputItemEvent,
    TurnEndEvent,
    TurnEndInfo,
    TurnInfo,
    TurnStartEvent,
    UserMessageEvent,
)
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
    OutputItem,
    OutputMessageItem,
    ReasoningItem,
)
from grasp_agents.types.llm_events import (
    OutputItemDone,
    ResponseCompleted,
    ResponseRetrying,
)
from grasp_agents.types.tool import BaseTool, NamedToolChoice, ToolChoice
from grasp_agents.utils.streaming import stream_concurrent

from .background_tasks import BackgroundTaskManager
from .loop_state import (
    NextStep,
    NextStepContinue,
    NextStepForceFinalAnswer,
    NextStepRunTools,
    NextStepStop,
    decide_next_step,
)
from .tool_decision import (
    AllowTool,
    RaiseToolException,
    RejectToolContent,
    ToolCallDecision,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Mapping, Sequence

    from grasp_agents.llm.llm import LLM
    from grasp_agents.types.hooks import (
        AfterLlmHook,
        AfterToolHook,
        BeforeLlmHook,
        BeforeToolHook,
        FinalAnswerExtractor,
        ToolInputConverter,
        ToolOutputConverter,
    )
    from grasp_agents.types.response import Response

    from .llm_agent_transcript import LLMAgentTranscript

logger = getLogger(__name__)


class CheckpointCallback(Protocol):
    async def __call__(
        self,
        *,
        turn: int = ...,
        location: AgentCheckpointLocation = ...,
        output: str | None = ...,
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


class AgentLoop(Generic[CtxT]):
    """
    The agentic execution loop: generate → check → tools → repeat.

    Holds the LLM, tools, memory, and all loop mechanics. Hooks are optional
    callback slots set by the owning LLMAgent.
    """

    agent_name: Final[str]
    llm: Final[LLM]
    final_answer_type: type[BaseModel]
    max_turns: Final[int]
    force_react_mode: Final[bool]
    final_answer_as_tool_call: Final[bool]
    tracing_exclude_input_fields: Final[set[str] | None]
    stream_llm: Final[bool]
    stream_tools: Final[bool]

    # Mutable state
    transcript: LLMAgentTranscript
    tools: dict[str, BaseTool[BaseModel, Any, CtxT]]
    final_answer: str | None  # extracted by _check_stop / _force_generate
    turn: int  # current LLM cycle within a step
    bg_tasks: BackgroundTaskManager[CtxT]
    path: list[str] | None
    # Per-agent file-edit bookkeeping (read records + dotfile overrides).
    # Activated for the duration of ``execute_stream`` via a ContextVar so
    # sub-agents (ProcessorTool, parallel replicas) each get their own
    # slot — preventing a parent's record_write from letting a child's
    # EditTool skip its own Read.
    file_edit_state: FileEditSessionState

    # Hook callback slots — set by LLMAgent, None = no-op
    final_answer_extractor: FinalAnswerExtractor | None
    before_llm_hook: BeforeLlmHook | None
    after_llm_hook: AfterLlmHook | None
    before_tool_hook: BeforeToolHook[CtxT] | None
    after_tool_hook: AfterToolHook | None
    tool_output_converters: dict[str, ToolOutputConverter]
    tool_input_converters: dict[str, ToolInputConverter]

    # Session persistence — wired by LLMAgent.setup_session()
    checkpoint_callback: CheckpointCallback | None

    _llm_output_schema: Any | None
    _final_answer_tool: BaseTool[BaseModel, Any, CtxT]

    def __init__(
        self,
        *,
        agent_name: str,
        llm: LLM,
        transcript: LLMAgentTranscript,
        tools: list[BaseTool[BaseModel, Any, CtxT]] | None,
        ctx: RunContext[CtxT],
        llm_output_schema: Any | None = None,
        max_turns: int,
        force_react_mode: bool = False,
        final_answer_type: type[BaseModel] = BaseModel,
        final_answer_as_tool_call: bool = False,
        stream_llm: bool = True,
        stream_tools: bool = False,
        path: list[str] | None = None,
        tracing_exclude_input_fields: set[str] | None = None,
    ) -> None:
        super().__init__()

        self._ctx: RunContext[CtxT] = ctx

        self.final_answer = None
        self.turn = 0
        self.transcript = transcript

        self.agent_name = agent_name
        self.llm = llm
        self.max_turns = max_turns

        self.stream_llm = stream_llm
        self.stream_tools = stream_tools

        self.force_react_mode = force_react_mode
        self.final_answer_type = final_answer_type
        self.final_answer_as_tool_call = final_answer_as_tool_call

        self.tracing_exclude_input_fields = tracing_exclude_input_fields

        self._llm_output_schema = llm_output_schema
        self._final_answer_tool = self._make_final_answer_tool()

        tools_list = (tools or [])[:]
        if tools and final_answer_as_tool_call:
            tools_list = tools + [self._final_answer_tool]
        self.tools = {t.name: t for t in tools_list}

        # Tool call_ids whose results were already synthesized at the LLM
        # validation layer (bad arguments → ``LLMToolCallValidationError``
        # caught in :meth:`query_llm`). The dispatcher skips them — their
        # error tool_result is already in the transcript, and re-dispatch
        # would just trigger pydantic validation again.
        self._skip_call_ids: set[str] = set()

        self.final_answer_extractor = None
        self.before_llm_hook = None
        self.after_llm_hook = None
        self.tool_output_converters = {}
        self.tool_input_converters = {}
        self.before_tool_hook = None
        self.after_tool_hook = None

        self.checkpoint_callback = None
        self.path = path

        self.file_edit_state = FileEditSessionState()

        self.bg_tasks = BackgroundTaskManager[CtxT](
            agent_name=agent_name,
            transcript=transcript,
            tools=self.tools,
            path=path,
        )

        # Auto-backgrounded Bash sessions, one registry per loop — the same
        # isolation rule as ``file_edit_state``: sub-agents and parallel
        # replicas must not see (or announce) each other's sessions. The bash
        # tools resolve it through a ContextVar set around each run; the loop
        # injects completion notes and waits on running sessions in PRE-ACT.
        self.bash_registry = BashProcessRegistry()
        # Persistent (stateful) Bash shell — one per loop, same isolation rule.
        # Opened lazily on first use, closed when the run ends.
        self.bash_session_holder = BashSessionHolder()

    @property
    def llm_output_schema(self) -> Any | None:
        return self._llm_output_schema

    @llm_output_schema.setter
    def llm_output_schema(self, value: Any | None) -> None:
        self._llm_output_schema = value

    @property
    def ctx(self) -> RunContext[CtxT]:
        return self._ctx

    async def checkpoint(
        self,
        *,
        turn: int,
        location: AgentCheckpointLocation,
        output: str | None = None,
    ) -> None:
        """Persist session state if a checkpoint callback is configured."""
        if self.checkpoint_callback:
            await self.checkpoint_callback(turn=turn, location=location, output=output)

    # --- Hook dispatch ---

    async def on_before_llm(
        self,
        *,
        exec_id: str,
        turn: int,
        extra_llm_settings: dict[str, Any],
    ) -> None:
        if self.before_llm_hook is not None:
            await self.before_llm_hook(
                exec_id=exec_id,
                turn=turn,
                extra_llm_settings=extra_llm_settings,
            )

    async def on_after_llm(
        self,
        response: Response,
        *,
        exec_id: str,
        turn: int,
    ) -> None:
        if self.after_llm_hook is not None:
            await self.after_llm_hook(
                response=response,
                exec_id=exec_id,
                turn=turn,
            )

    async def on_before_tool(
        self,
        *,
        tool_calls: Sequence[FunctionToolCallItem],
        exec_id: str,
    ) -> Mapping[str, ToolCallDecision] | None:
        if self.before_tool_hook is not None:
            return await self.before_tool_hook(
                tool_calls=tool_calls, ctx=self._ctx, exec_id=exec_id
            )
        return None

    async def on_after_tool(
        self,
        *,
        tool_calls: Sequence[FunctionToolCallItem],
        tool_messages: Sequence[FunctionToolOutputItem],
        exec_id: str,
    ) -> None:
        if self.after_tool_hook is not None:
            await self.after_tool_hook(
                tool_calls=tool_calls,
                tool_messages=tool_messages,
                exec_id=exec_id,
            )

    async def _convert_tool_output(
        self,
        output: Any,
        call: FunctionToolCallItem,
        *,
        exec_id: str,
    ) -> FunctionToolOutputItem:
        converter = self.tool_output_converters.get(call.name)
        if converter is not None:
            parts = await converter(output, exec_id=exec_id)

            return FunctionToolOutputItem(call_id=call.call_id, output_parts=parts)

        return FunctionToolOutputItem.from_tool_result(
            call_id=call.call_id, output=output
        )

    async def _convert_tool_input(
        self,
        call: FunctionToolCallItem,
        *,
        exec_id: str,
    ) -> BaseModel:
        tool = self.tools[call.name]
        args = json.loads(call.arguments)
        llm_args = TypeAdapter(tool.llm_in_type).validate_python(args)
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

    @traced(name="generate")
    async def query_llm(
        self,
        *,
        tool_choice: ToolChoice | None = None,
        exec_id: str,
        extra_llm_settings: dict[str, Any],
    ) -> AsyncIterator[Event[Any]]:
        # Enforce the tool_call → tool_result pairing invariant before every
        # provider call (where a violation would otherwise 400).
        self.transcript.validate_tool_call_pairing()

        llm_params: dict[str, Any] = {
            "input": self.transcript.messages,
            "output_schema": self.llm_output_schema,
            "tools": self.tools or None,
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
                async for se in self.llm.generate_response_stream(**llm_params):
                    if isinstance(se, ResponseRetrying):
                        # Validation or transient API retry just fired —
                        # the previous attempt's items are about to be
                        # superseded by a fresh attempt. Discard them so
                        # the next attempt's items don't pile on top.
                        pending = []
                    if isinstance(se, OutputItemDone):
                        item = se.item
                        if isinstance(
                            item,
                            (OutputMessageItem, FunctionToolCallItem, ReasoningItem),
                        ):
                            pending.append(item)
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
                response = exc.response
                if pending:
                    self.transcript.update(pending, ctx=self._ctx)
                    for ev in self._item_events(pending, exec_id=exec_id):
                        yield ev
                async for ev in self._synthesize_validation_tool_results(
                    exc, pending, exec_id=exec_id
                ):
                    yield ev
                if response is not None:
                    yield LLMStreamEvent(
                        data=ResponseCompleted(response=response, sequence_number=0),
                        source=self.agent_name,
                        exec_id=exec_id,
                    )
                    self._process_response(response)
                return
            # Clean completion → commit pending items, then surface their
            # item events (post-write, per the convention above).
            self.transcript.update(pending, ctx=self._ctx)
            for ev in self._item_events(pending, exec_id=exec_id):
                yield ev

        else:
            try:
                response = await self.llm.generate_response(**llm_params)
                self.transcript.update(response.output_items, ctx=self._ctx)
            except LLMToolCallValidationError as exc:
                response = exc.response
                if response is not None:
                    self.transcript.update(response.output_items, ctx=self._ctx)
                    for ev in self._item_events(response.output_items, exec_id=exec_id):
                        yield ev
                async for ev in self._synthesize_validation_tool_results(
                    exc,
                    response.output_items if response is not None else [],
                    exec_id=exec_id,
                ):
                    yield ev
                if response is not None:
                    yield LLMStreamEvent(
                        data=ResponseCompleted(response=response, sequence_number=0),
                        source=self.agent_name,
                        exec_id=exec_id,
                    )
                    self._process_response(response)
                return

            # LLMStream event immediate; item events after the write above.
            yield LLMStreamEvent(
                data=ResponseCompleted(response=response, sequence_number=0),
                source=self.agent_name,
                exec_id=exec_id,
            )
            for ev in self._item_events(response.output_items, exec_id=exec_id):
                yield ev

        if not response:
            return

        self._process_response(response)

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
            self.transcript.update([msg], ctx=self._ctx)
            self._skip_call_ids.add(item.call_id)
            yield ToolOutputItemEvent(
                source=item.name,
                destination=self.agent_name,
                exec_id=exec_id,
                data=msg,
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

    # --- Tool calling ---

    async def execute_tools_stream(
        self,
        calls: Sequence[FunctionToolCallItem],
        exec_id: str,
    ) -> AsyncIterator[Event[Any]]:
        # Tool call events are now emitted from query_llm via OutputItemDone promotion

        # Resolve inputs, partition into background vs immediate

        outputs: list[Any] = [None] * len(calls)
        immediate: list[
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
            tool = self.tools[call.name]
            inp = await self._convert_tool_input(call, exec_id=exec_id)
            if tool.background:
                task_id, event = await self.bg_tasks.spawn(
                    call,
                    tool,
                    inp,
                    ctx=self._ctx,
                    exec_id=exec_id,
                )
                yield event
                outputs[i] = f"Task launched in background (id: {task_id})"
            else:
                immediate.append((i, call, tool, inp))

        if immediate and self.stream_tools:
            streams = [
                tool.run_stream(
                    inp=inp,
                    ctx=self._ctx,
                    exec_id=exec_id,
                    path=make_tool_call_path(self.path, call.call_id),
                )
                for _, call, tool, inp in immediate
            ]
            merged = stream_concurrent(streams)
            async for stream_idx, event in merged:
                if isinstance(event, ToolOutputEvent):
                    outputs[immediate[stream_idx][0]] = event.data
                yield event

            for err in merged.errors:
                i = immediate[err.index][0]
                tool_name = immediate[err.index][2].name
                outputs[i] = f"Tool '{tool_name}' failed: {err.exception}"
                logger.warning(
                    "Tool '%s' (call index %d) failed: %r",
                    tool_name,
                    i,
                    err.exception,
                )

        elif immediate:
            results = await asyncio.gather(
                *[
                    tool.run(
                        inp,
                        ctx=self._ctx,
                        exec_id=exec_id,
                        path=make_tool_call_path(self.path, call.call_id),
                    )
                    for _, call, tool, inp in immediate
                ],
                return_exceptions=True,
            )
            for (i, _call, t, _inp), result in zip(immediate, results, strict=True):
                if isinstance(result, BaseException):
                    outputs[i] = f"Tool '{t.name}' failed: {result}"
                    logger.warning(
                        "Tool '%s' (call index %d) failed: %r", t.name, i, result
                    )
                else:
                    outputs[i] = result

        tool_messages: list[FunctionToolOutputItem] = []

        for idx, (output, call) in enumerate(zip(outputs, calls, strict=True)):
            if skipped[idx]:
                # Tool result already synthesized in ``query_llm`` for
                # this call_id (LLM validation failure path).
                continue
            msg = await self._convert_tool_output(output, call, exec_id=exec_id)
            tool_messages.append(msg)
            self.transcript.update([msg], ctx=self._ctx)
            yield ToolOutputItemEvent(
                source=call.name, destination=self.agent_name, exec_id=exec_id, data=msg
            )

        if self._ctx.printer:
            self._ctx.printer.print_messages(
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
            "Exceeded the maximum number of turns: provide a final answer now!",
            role="user",
        )
        self.transcript.update([user_message])
        # TODO: set source
        yield UserMessageEvent(
            source=None,
            destination=self.agent_name,
            exec_id=exec_id,
            data=user_message,
        )

        if self._ctx.printer:
            self._ctx.printer.print_messages(
                [user_message],
                agent_name=self.agent_name,
                exec_id=exec_id,
            )

        tool_choice = (
            NamedToolChoice(name=self._final_answer_tool.name)
            if self.final_answer_as_tool_call
            else None
        )
        stream = ResponseCapture(
            self.query_llm(
                tool_choice=tool_choice,
                extra_llm_settings=extra_llm_settings,
                exec_id=exec_id,
            ),
        )
        async for event in stream:
            yield event

        assert stream.response is not None

        self.transcript.update(stream.response.output_items)
        self._process_response(stream.response)

        self.final_answer = self._extract_final_answer(
            response=stream.response,
            exec_id=exec_id,
        )
        if self.final_answer is None:
            raise AgentFinalAnswerError(proc_name=self.agent_name, exec_id=exec_id)

    # --- Main execution loop ---

    def _compute_tool_choice(self, had_tool_calls: bool) -> ToolChoice:
        """Compute tool_choice for the current turn."""
        if not self.force_react_mode:
            return "auto"
        # force_react_mode alternates: reason (no tools) → act (must use tools) → …
        if self.turn == 0 or had_tool_calls:
            return "none"  # first turn or just acted → reason
        return "required"  # just reasoned → must act

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
            bg_tasks_pending=self.bg_tasks.has_pending,
        )

    async def _drain_bash_notes(self, *, exec_id: str) -> AsyncIterator[Event[Any]]:
        """
        Inject a user-role message for each backgrounded Bash command that has
        finished since the last turn (ephemeral — no ``TaskRecord``).
        """
        for note in self.bash_registry.collect_notes():
            message = InputMessageItem.from_text(note, role="user")
            self.transcript.update([message])
            yield UserMessageEvent(
                source=self.agent_name,
                destination=self.agent_name,
                exec_id=exec_id,
                data=message,
            )

    def _close_dangling_tool_calls(
        self, response: Response
    ) -> list[FunctionToolOutputItem]:
        """
        Inject synthetic tool outputs for tool calls that will never execute.

        When the loop stops (e.g. max_turns) after ACT generated tool calls,
        those calls are already in the transcript but have no outputs. Most LLM APIs
        reject requests with unmatched tool_use/tool_result pairs. This method
        adds cancellation outputs so the conversation stays valid.
        """
        tool_calls = response.tool_call_items
        if not tool_calls:
            return []

        cancellations = [
            FunctionToolOutputItem.from_tool_result(
                call_id=tc.call_id,
                output="[Tool call cancelled: agent reached turn limit]",
            )
            for tc in tool_calls
        ]
        self.transcript.update(cancellations)

        return cancellations

    # --- Per-state handlers (dispatched from execute_stream) ---

    async def _handle_stop(
        self,
        step: NextStepStop,
        *,
        exec_id: str,
    ) -> AsyncIterator[Event[Any]]:
        """``NextStepStop``: final answer extracted, end loop cleanly."""
        self.final_answer = step.final_answer
        await self.checkpoint(
            turn=self.turn,
            location=AgentCheckpointLocation.AFTER_FINAL_ANSWER,
            output=self.final_answer,
        )
        yield TurnEndEvent(
            source=self.agent_name,
            exec_id=exec_id,
            data=TurnEndInfo(
                turn=self.turn,
                had_tool_calls=False,
                stop_reason=step.stop_reason,
            ),
        )

    async def _handle_force_final_answer(
        self,
        response: Response,
        *,
        exec_id: str,
        extra_llm_settings: dict[str, Any],
    ) -> AsyncIterator[Event[Any]]:
        """
        ``NextStepForceFinalAnswer``: turn budget exhausted.

        Cancels background tasks, closes dangling tool calls, force-generates
        a final answer, and ends the loop with ``stop_reason=MAX_TURNS``.
        """
        await self.bg_tasks.cancel_all(ctx=self._ctx)
        self._close_dangling_tool_calls(response)

        async for event in self._force_generate_final_answer_stream(
            exec_id=exec_id,
            extra_llm_settings=extra_llm_settings,
        ):
            yield event

        await self.checkpoint(
            turn=self.turn,
            location=AgentCheckpointLocation.AFTER_MAX_TURNS,
            output=self.final_answer,
        )

        yield TurnEndEvent(
            source=self.agent_name,
            exec_id=exec_id,
            data=TurnEndInfo(
                turn=self.turn,
                had_tool_calls=False,
                stop_reason=StopReason.MAX_TURNS,
            ),
        )

        logger.info(
            "Max turns reached: %s. Exiting the tool call loop.",
            self.max_turns,
        )

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
        decisions = await self.on_before_tool(
            tool_calls=step.tool_calls, exec_id=exec_id
        )

        if decisions:
            for decision in decisions.values():
                if isinstance(decision, RaiseToolException):
                    raise decision.exception

        tool_msgs: list[FunctionToolOutputItem] = []
        allowed_calls: list[FunctionToolCallItem] = []
        rejection_msgs: list[FunctionToolOutputItem] = []

        for call in step.tool_calls:
            decision = (decisions or {}).get(call.call_id, AllowTool())
            if isinstance(decision, RejectToolContent):
                msg = FunctionToolOutputItem.from_tool_result(
                    call_id=call.call_id, output=decision.content
                )
                tool_msgs.append(msg)
                rejection_msgs.append(msg)
                self.transcript.update([msg], ctx=self._ctx)
                yield ToolOutputItemEvent(
                    source=call.name,
                    destination=self.agent_name,
                    exec_id=exec_id,
                    data=msg,
                )
            else:
                allowed_calls.append(call)

        if rejection_msgs and self._ctx.printer:
            self._ctx.printer.print_messages(
                rejection_msgs,
                agent_name=self.agent_name,
                exec_id=exec_id,
            )

        if allowed_calls:
            async for event in self.execute_tools_stream(
                allowed_calls, exec_id=exec_id
            ):
                if isinstance(event, ToolOutputItemEvent):
                    tool_msgs.append(event.data)
                yield event

        await self.on_after_tool(
            tool_calls=step.tool_calls,
            tool_messages=tool_msgs,
            exec_id=exec_id,
        )

        await self.checkpoint(
            turn=self.turn,
            location=AgentCheckpointLocation.AFTER_TOOL_RESULT,
        )

        yield TurnEndEvent(
            source=self.agent_name,
            exec_id=exec_id,
            data=TurnEndInfo(turn=self.turn, had_tool_calls=True),
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
            extra_llm_settings=settings,
            turn=self.turn,
            exec_id=exec_id,
        )

        tool_choice: ToolChoice | None = None
        if self.tools:
            tool_choice = self._compute_tool_choice(had_tool_calls)
        tool_choice = settings.pop("tool_choice", tool_choice)

        stream = ResponseCapture(
            self.query_llm(
                tool_choice=tool_choice,
                extra_llm_settings=settings,
                exec_id=exec_id,
            ),
        )
        async for event in stream:
            yield event

        response = stream.response
        assert response is not None

        await self.on_after_llm(
            response,
            turn=self.turn,
            exec_id=exec_id,
        )

        yield GenerationEndEvent(source=self.agent_name, exec_id=exec_id, data=response)

    # --- Main execution loop ---

    async def execute_stream(
        self,
        exec_id: str,
        extra_llm_settings: dict[str, Any] | None = None,
    ) -> AsyncIterator[Event[Any]]:
        had_tool_calls = False
        self.final_answer = None

        # Activate this agent's :class:`FileEditSessionState` for the
        # entire run. File-edit tools, file-search tools, and
        # :class:`MemoryProvider` consult the ContextVar set here so
        # sub-agents (ProcessorTool, parallel replicas) get their own
        # read-before-write slot. Asyncio tasks spawned downstream copy
        # the current context, so the value propagates into
        # ``stream_concurrent`` / ``asyncio.gather`` children.
        state_token = set_current_file_edit_state(self.file_edit_state)
        bash_token = set_current_bash_registry(self.bash_registry)
        holder_token = set_current_session_holder(self.bash_session_holder)
        try:
            # Checkpoint after input memorization (new step only)
            if self.turn == 0:
                await self.checkpoint(
                    turn=self.turn,
                    location=AgentCheckpointLocation.AFTER_INPUT,
                )

            while self.turn <= self.max_turns:
                # ── PRE-ACT: prepare for generation ──

                # When the model has nothing to do but wait on background work,
                # block on the next completion — durable tasks or running
                # backgrounded Bash commands — instead of spinning poll turns.
                # (Only tasks can delay a final answer; see JUDGE / has_pending.)
                if self.turn > 0 and not had_tool_calls:
                    waitables = [
                        *self.bg_tasks.pending_futures(),
                        *self.bash_registry.pending_drains(),
                    ]
                    if waitables:
                        await asyncio.wait(
                            waitables, return_when=asyncio.FIRST_COMPLETED
                        )

                async for event in self.bg_tasks.drain(exec_id=exec_id, ctx=self._ctx):
                    yield event

                async for event in self._drain_bash_notes(exec_id=exec_id):
                    yield event

                yield TurnStartEvent(
                    source=self.agent_name,
                    exec_id=exec_id,
                    data=TurnInfo(turn=self.turn),
                )

                # ── ACT: LLM generates response ──

                act = ResponseCapture(
                    self._run_act_stream(
                        exec_id=exec_id,
                        extra_llm_settings=extra_llm_settings or {},
                        had_tool_calls=had_tool_calls,
                    ),
                )
                async for event in act:
                    yield event

                assert act.response is not None
                response = act.response

                # ── JUDGE: classify next transition ──

                step = self._decide_next_step(response, exec_id=exec_id)

                # ── Dispatch to per-state handler ──

                if isinstance(step, NextStepStop):
                    async for event in self._handle_stop(step, exec_id=exec_id):
                        yield event
                    return

                if isinstance(step, NextStepForceFinalAnswer):
                    async for event in self._handle_force_final_answer(
                        response,
                        exec_id=exec_id,
                        extra_llm_settings=deepcopy(extra_llm_settings or {}),
                    ):
                        yield event
                    return

                if isinstance(step, NextStepRunTools):
                    async for event in self._handle_run_tools(step, exec_id=exec_id):
                        yield event
                else:
                    assert isinstance(step, NextStepContinue)
                    async for event in self._handle_continue(exec_id=exec_id):
                        yield event

                had_tool_calls = isinstance(step, NextStepRunTools)
                self.turn += 1

        finally:
            await self.bg_tasks.cancel_all(ctx=self._ctx)
            await self.bash_session_holder.close()
            reset_current_session_holder(holder_token)
            reset_current_bash_registry(bash_token)
            reset_current_file_edit_state(state_token)

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
                ctx: RunContext[Any] | None = None,
                exec_id: str | None = None,
                progress_callback: Any = None,
                path: Sequence[str] | None = None,
            ) -> None:
                del inp, ctx, exec_id, progress_callback, path

        return FinalAnswerTool()

    def _process_response(self, response: Response) -> None:
        self._ctx.responses[self.agent_name].append(response)
        self._ctx.usage_tracker.update(
            agent_name=self.agent_name,
            responses=[response],
            model_name=self.llm.model_name,
            litellm_provider=self.llm.litellm_provider,
        )
