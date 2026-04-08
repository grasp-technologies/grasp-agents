import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from copy import deepcopy
from logging import getLogger
from typing import TYPE_CHECKING, Any, Generic

from pydantic import BaseModel, TypeAdapter

from grasp_agents.tracing_decorators import task

from .background_tasks import BackgroundTaskManager
from .errors import AgentFinalAnswerError
from .llm import LLM
from .llm_agent_memory import LLMAgentMemory
from .run_context import CtxT, RunContext
from .types.events import (
    Event,
    GenerationEndEvent,
    LLMStreamEvent,
    OutputMessageItemEvent,
    ReasoningItemEvent,
    StopReason,
    ToolCallItemEvent,
    ToolOutputEvent,
    ToolResultEvent,
    TurnEndEvent,
    TurnEndInfo,
    TurnInfo,
    TurnStartEvent,
    UserMessageEvent,
)

if TYPE_CHECKING:
    from .types.hooks import (
        AfterLlmHook,
        AfterToolHook,
        BeforeLlmHook,
        BeforeToolHook,
        FinalAnswerExtractor,
        ToolInputConverter,
        ToolOutputConverter,
    )
from .types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
    OutputMessageItem,
    ReasoningItem,
)
from .types.llm_events import OutputItemDone, ResponseCompleted
from .types.response import Response
from .types.tool import BaseTool, NamedToolChoice, ToolChoice
from .utils.streaming import stream_concurrent

logger = getLogger(__name__)


CheckpointCallback = Callable[[], Awaitable[None]]


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

    def __init__(
        self,
        *,
        agent_name: str,
        llm: LLM,
        memory: LLMAgentMemory,
        tools: list[BaseTool[BaseModel, Any, CtxT]] | None,
        response_schema: Any | None = None,
        response_schema_by_xml_tag: Mapping[str, Any] | None = None,
        max_turns: int,
        force_react_mode: bool = False,
        final_answer_type: type[BaseModel] = BaseModel,
        final_answer_as_tool_call: bool = False,
        stream_llm_responses: bool = True,
        stream_tools: bool = False,
        tracing_exclude_input_fields: set[str] | None = None,
    ) -> None:
        super().__init__()

        self._agent_name = agent_name
        self._max_turns = max_turns
        self._force_react_mode = force_react_mode
        self._tracing_exclude_input_fields = tracing_exclude_input_fields

        self._llm = llm
        self._response_schema = response_schema
        self._response_schema_by_xml_tag = response_schema_by_xml_tag

        self.memory = memory

        self._final_answer_type = final_answer_type
        self._final_answer_as_tool_call = final_answer_as_tool_call
        self._final_answer_tool = self._make_final_answer_tool()

        self._stream_llm_responses = stream_llm_responses
        self._stream_tools = stream_tools

        tools_list: list[BaseTool[BaseModel, Any, CtxT]] | None = tools
        if tools and final_answer_as_tool_call:
            tools_list = tools + [self._final_answer_tool]
        self._tools = {t.name: t for t in tools_list} if tools_list else None

        # Hook callback slots — set by LLMAgent, None = no-op
        self.final_answer_extractor: FinalAnswerExtractor[CtxT] | None = None
        self.before_llm_hook: BeforeLlmHook[CtxT] | None = None
        self.after_llm_hook: AfterLlmHook[CtxT] | None = None
        self.tool_output_converters: dict[str, ToolOutputConverter[CtxT]] = {}
        self.tool_input_converters: dict[str, ToolInputConverter[CtxT]] = {}
        self.before_tool_hook: BeforeToolHook[CtxT] | None = None
        self.after_tool_hook: AfterToolHook[CtxT] | None = None

        # Background task manager
        self.bg_tasks = BackgroundTaskManager[CtxT](
            agent_name=agent_name,
            memory=memory,
            tools=self._tools,
        )

        # Extracted final answer (set by _check_stop / _force_generate)
        self.final_answer: str | None = None

        # Session persistence — set by LLMAgent
        self.checkpoint_callback: CheckpointCallback | None = None

    @property
    def agent_name(self) -> str:
        return self._agent_name

    @property
    def max_turns(self) -> int:
        return self._max_turns

    @property
    def force_react_mode(self) -> bool:
        return self._force_react_mode

    @property
    def llm(self) -> LLM:
        return self._llm

    @property
    def response_schema(self) -> Any | None:
        return self._response_schema

    @response_schema.setter
    def response_schema(self, value: Any | None) -> None:
        self._response_schema = value

    @property
    def response_schema_by_xml_tag(self) -> Mapping[str, Any] | None:
        return self._response_schema_by_xml_tag

    @property
    def final_answer_as_tool_call(self) -> bool:
        return self._final_answer_as_tool_call

    @property
    def tools(self) -> dict[str, BaseTool[BaseModel, Any, CtxT]]:
        return self._tools or {}

    @property
    def tracing_exclude_input_fields(self) -> set[str] | None:
        return self._tracing_exclude_input_fields

    async def checkpoint(self) -> None:
        """Persist session state if a checkpoint callback is configured."""
        if self.checkpoint_callback:
            await self.checkpoint_callback()

    # --- Hook dispatch ---

    async def on_before_llm(
        self,
        *,
        ctx: RunContext[CtxT],
        exec_id: str,
        num_turns: int,
        extra_llm_settings: dict[str, Any],
    ) -> None:
        if self.before_llm_hook is not None:
            await self.before_llm_hook(
                ctx=ctx,
                exec_id=exec_id,
                num_turns=num_turns,
                extra_llm_settings=extra_llm_settings,
            )

    async def on_after_llm(
        self,
        response: Response,
        *,
        ctx: RunContext[CtxT],
        exec_id: str,
        num_turns: int,
    ) -> None:
        if self.after_llm_hook is not None:
            await self.after_llm_hook(
                response=response,
                ctx=ctx,
                exec_id=exec_id,
                num_turns=num_turns,
            )

    async def on_before_tool(
        self,
        *,
        tool_calls: Sequence[FunctionToolCallItem],
        ctx: RunContext[CtxT],
        exec_id: str,
    ) -> None:
        if self.before_tool_hook is not None:
            await self.before_tool_hook(tool_calls=tool_calls, ctx=ctx, exec_id=exec_id)

    async def on_after_tool(
        self,
        *,
        tool_calls: Sequence[FunctionToolCallItem],
        tool_messages: Sequence[FunctionToolOutputItem],
        ctx: RunContext[CtxT],
        exec_id: str,
    ) -> None:
        if self.after_tool_hook is not None:
            await self.after_tool_hook(
                tool_calls=tool_calls,
                tool_messages=tool_messages,
                ctx=ctx,
                exec_id=exec_id,
            )

    async def _convert_tool_output(
        self,
        output: Any,
        call: FunctionToolCallItem,
        *,
        ctx: RunContext[CtxT],
        exec_id: str,
    ) -> FunctionToolOutputItem:
        converter = self.tool_output_converters.get(call.name)
        if converter is not None:
            parts = await converter(output, ctx=ctx, exec_id=exec_id)

            return FunctionToolOutputItem(call_id=call.call_id, output_parts=parts)

        return FunctionToolOutputItem.from_tool_result(
            call_id=call.call_id, output=output
        )

    async def _convert_tool_input(
        self,
        call: FunctionToolCallItem,
        *,
        ctx: RunContext[CtxT],
        exec_id: str,
    ) -> BaseModel:
        tool = self.tools[call.name]
        args = json.loads(call.arguments)
        llm_args = TypeAdapter(tool.in_type).validate_python(args)
        converter = self.tool_input_converters.get(tool.name)

        if converter is not None:
            return await converter(llm_args, ctx=ctx, exec_id=exec_id)

        return llm_args

    def _extract_final_answer(
        self,
        *,
        response: Response,
        ctx: RunContext[CtxT],
        exec_id: str,
        **kwargs: Any,
    ) -> str | None:
        if self.final_answer_extractor is not None:
            return self.final_answer_extractor(
                ctx=ctx, exec_id=exec_id, response=response, **kwargs
            )

        if self._final_answer_as_tool_call:
            for tc in response.tool_call_items:
                if tc.name == self._final_answer_tool.name:
                    return tc.arguments
            return None

        if response.tool_call_items:
            return None

        return response.output_text or None

    # --- LLM generation ---

    @task(name="generate")  # type: ignore
    async def query_llm(
        self,
        *,
        tool_choice: ToolChoice | None = None,
        ctx: RunContext[CtxT],
        exec_id: str,
        extra_llm_settings: dict[str, Any],
    ) -> AsyncIterator[Event[Any]]:
        llm_params: dict[str, Any] = {
            "input": self.memory.messages,
            "response_schema": self.response_schema,
            "response_schema_by_xml_tag": self.response_schema_by_xml_tag,
            "tools": self.tools or None,
            "tool_choice": tool_choice,
            **extra_llm_settings,
        }

        response: Response | None = None

        if self._stream_llm_responses:
            async for se in self.llm.generate_response_stream(**llm_params):
                item = getattr(se, "item", None)
                if isinstance(
                    item, (OutputMessageItem, FunctionToolCallItem, ReasoningItem)
                ):
                    self.memory.update([item], ctx=ctx)

                if isinstance(se, OutputItemDone):
                    if isinstance(item, FunctionToolCallItem):
                        yield ToolCallItemEvent(
                            source=self.agent_name,
                            exec_id=exec_id,
                            data=item,
                        )
                    elif isinstance(item, ReasoningItem):
                        yield ReasoningItemEvent(
                            source=self.agent_name,
                            exec_id=exec_id,
                            data=item,
                        )
                    elif isinstance(item, OutputMessageItem):
                        yield OutputMessageItemEvent(
                            source=self.agent_name,
                            exec_id=exec_id,
                            data=item,
                        )

                elif isinstance(se, ResponseCompleted):
                    response = se.response

                yield LLMStreamEvent(
                    data=se,
                    source=self.agent_name,
                    exec_id=exec_id,
                )

        else:
            response = await self.llm.generate_response(**llm_params)
            self.memory.update(response.output_items, ctx=ctx)

            for item in response.output_items:
                if isinstance(item, FunctionToolCallItem):
                    yield ToolCallItemEvent(
                        source=self.agent_name,
                        exec_id=exec_id,
                        data=item,
                    )
                elif isinstance(item, ReasoningItem):
                    yield ReasoningItemEvent(
                        source=self.agent_name,
                        exec_id=exec_id,
                        data=item,
                    )
                elif isinstance(item, OutputMessageItem):
                    yield OutputMessageItemEvent(
                        source=self.agent_name,
                        exec_id=exec_id,
                        data=item,
                    )

            yield LLMStreamEvent(
                data=ResponseCompleted(response=response, sequence_number=0),
                source=self.agent_name,
                exec_id=exec_id,
            )

        if not response:
            return

        self._process_response(response, ctx=ctx, exec_id=exec_id)

    # --- Tool calling ---

    async def execute_tools_stream(
        self,
        calls: Sequence[FunctionToolCallItem],
        ctx: RunContext[CtxT],
        exec_id: str,
    ) -> AsyncIterator[Event[Any]]:
        # Tool call events are now emitted from query_llm via OutputItemDone promotion

        # Resolve inputs, partition into background vs immediate

        outputs: list[Any] = [None] * len(calls)
        immediate: list[tuple[int, BaseTool[BaseModel, Any, CtxT], BaseModel]] = []

        for i, call in enumerate(calls):
            tool = self.tools[call.name]
            inp = await self._convert_tool_input(call, ctx=ctx, exec_id=exec_id)
            if tool.background:
                task_id, event = await self.bg_tasks.spawn(
                    call, tool, inp, ctx=ctx, exec_id=exec_id
                )
                yield event
                outputs[i] = f"Task launched in background (id: {task_id})"
            else:
                immediate.append((i, tool, inp))

        # Execute immediate tools concurrently

        if immediate and self._stream_tools:
            streams = [
                t.run_stream(
                    inp=inp,
                    ctx=ctx,
                    exec_id=exec_id,
                    _validated=True,
                )
                for _, t, inp in immediate
            ]
            async for stream_idx, event in stream_concurrent(streams):
                if isinstance(event, ToolOutputEvent):
                    outputs[immediate[stream_idx][0]] = event.data
                else:
                    yield event

        elif immediate:
            results = await asyncio.gather(
                *[
                    t.run(
                        inp,
                        ctx=ctx,
                        exec_id=exec_id,
                        _validated=True,
                    )
                    for _, t, inp in immediate
                ],
            )
            for (i, _, _), result in zip(immediate, results, strict=True):
                outputs[i] = result

        tool_messages: list[FunctionToolOutputItem] = []
        for output, call in zip(outputs, calls, strict=True):
            msg = await self._convert_tool_output(
                output, call, ctx=ctx, exec_id=exec_id
            )
            tool_messages.append(msg)
            self.memory.update([msg], ctx=ctx)
            yield ToolResultEvent(source=call.name, exec_id=exec_id, data=msg)

        if ctx.printer:
            ctx.printer.print_messages(
                tool_messages,
                agent_name=self.agent_name,
                exec_id=exec_id,
            )

    # --- Final answer ---

    @task(name="force_generate_final_answer")  # type: ignore
    async def _force_generate_final_answer_stream(
        self,
        ctx: RunContext[CtxT],
        exec_id: str,
        extra_llm_settings: dict[str, Any],
    ) -> AsyncIterator[Event[Any]]:
        user_message = InputMessageItem.from_text(
            "Exceeded the maximum number of turns: provide a final answer now!",
            role="user",
        )
        self.memory.update([user_message])
        yield UserMessageEvent(
            source=self.agent_name,
            exec_id=exec_id,
            data=user_message,
        )

        if ctx.printer:
            ctx.printer.print_messages(
                [user_message],
                agent_name=self.agent_name,
                exec_id=exec_id,
            )

        tool_choice = (
            NamedToolChoice(name=self._final_answer_tool.name)
            if self._final_answer_as_tool_call
            else None
        )
        stream = ResponseCapture(
            self.query_llm(
                tool_choice=tool_choice,
                extra_llm_settings=extra_llm_settings,
                exec_id=exec_id,
                ctx=ctx,
            ),
        )
        async for event in stream:
            yield event

        assert stream.response is not None

        self.memory.update(stream.response.output_items)
        self._process_response(stream.response, ctx=ctx, exec_id=exec_id)

        self.final_answer = self._extract_final_answer(
            response=stream.response,
            ctx=ctx,
            exec_id=exec_id,
        )
        if self.final_answer is None:
            raise AgentFinalAnswerError(proc_name=self.agent_name, exec_id=exec_id)

    # --- Main execution loop ---

    def _compute_tool_choice(
        self,
        turn: int,
        had_tool_calls: bool,
    ) -> ToolChoice:
        """Compute tool_choice for the current turn."""
        if not self.force_react_mode:
            return "auto"
        # force_react_mode alternates: reason (no tools) → act (must use tools) → …
        if turn == 0 or had_tool_calls:
            return "none"  # first turn or just acted → reason
        return "required"  # just reasoned → must act

    def _check_stop(
        self,
        response: Response,
        turn: int,
        *,
        ctx: RunContext[CtxT],
        exec_id: str,
    ) -> StopReason | None:
        """Single decision point for all loop termination conditions."""
        final = self._extract_final_answer(
            response=response,
            ctx=ctx,
            exec_id=exec_id,
            num_turns=turn,
        )
        if final is not None:
            if self.bg_tasks.has_pending and turn < self.max_turns:
                return None  # suppress, we have turns left to wait
            self.final_answer = final
            return StopReason.FINAL_ANSWER

        if turn >= self.max_turns:
            return StopReason.MAX_TURNS

        return None

    def _close_dangling_tool_calls(
        self,
        response: Response,
    ) -> list[FunctionToolOutputItem]:
        """
        Inject synthetic tool outputs for tool calls that will never execute.

        When the loop stops (e.g. max_turns) after ACT generated tool calls,
        those calls are already in memory but have no outputs. Most LLM APIs
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
        self.memory.update(cancellations)

        return cancellations

    async def execute_stream(
        self,
        ctx: RunContext[CtxT],
        exec_id: str,
        extra_llm_settings: dict[str, Any] | None = None,
    ) -> AsyncIterator[Event[Any]]:
        had_tool_calls = False
        # Don't clear _background_tasks — may have pre-registered tasks
        # from session resume. The finally block cancels any remaining.
        self.final_answer = None

        try:
            for turn in range(self.max_turns + 1):
                # ── PRE-ACT: prepare for generation ──

                # Drain completed background tasks; wait if LLM has nothing to do
                should_wait = (
                    turn > 0 and not had_tool_calls and self.bg_tasks.has_pending
                )
                async for event in self.bg_tasks.drain(
                    wait=should_wait, exec_id=exec_id
                ):
                    yield event

                yield TurnStartEvent(
                    source=self.agent_name, exec_id=exec_id, data=TurnInfo(turn=turn)
                )

                # ── ACT: LLM generates response ──

                settings = deepcopy(extra_llm_settings or {})
                await self.on_before_llm(
                    extra_llm_settings=settings,
                    num_turns=turn,
                    ctx=ctx,
                    exec_id=exec_id,
                )

                tool_choice: ToolChoice | None = None
                if self._tools:
                    tool_choice = self._compute_tool_choice(turn, had_tool_calls)
                tool_choice = settings.pop("tool_choice", tool_choice)

                stream = ResponseCapture(
                    self.query_llm(
                        tool_choice=tool_choice,
                        extra_llm_settings=settings,
                        exec_id=exec_id,
                        ctx=ctx,
                    ),
                )
                async for event in stream:
                    yield event

                assert stream.response is not None
                response = stream.response

                await self.on_after_llm(
                    response, num_turns=turn, ctx=ctx, exec_id=exec_id
                )

                yield GenerationEndEvent(
                    source=self.agent_name, exec_id=exec_id, data=response
                )

                # ── JUDGE: should the loop stop? ──

                stop = self._check_stop(response, turn, ctx=ctx, exec_id=exec_id)

                if stop == StopReason.FINAL_ANSWER:
                    await self.checkpoint()

                    yield TurnEndEvent(
                        source=self.agent_name,
                        exec_id=exec_id,
                        data=TurnEndInfo(
                            turn=turn,
                            had_tool_calls=False,
                            stop_reason=StopReason.FINAL_ANSWER,
                        ),
                    )
                    return

                if stop == StopReason.MAX_TURNS:
                    await self.bg_tasks.cancel_all()
                    self._close_dangling_tool_calls(response)

                    async for event in self._force_generate_final_answer_stream(
                        ctx=ctx,
                        exec_id=exec_id,
                        extra_llm_settings=deepcopy(extra_llm_settings or {}),
                    ):
                        yield event

                    await self.checkpoint()

                    yield TurnEndEvent(
                        source=self.agent_name,
                        exec_id=exec_id,
                        data=TurnEndInfo(
                            turn=turn,
                            had_tool_calls=False,
                            stop_reason=StopReason.MAX_TURNS,
                        ),
                    )

                    logger.info(
                        "Max turns reached: %s. Exiting the tool call loop.",
                        self.max_turns,
                    )

                    return

                # ── OBSERVE: execute tools ──

                tool_calls = response.tool_call_items
                if tool_calls:
                    await self.on_before_tool(
                        tool_calls=tool_calls, ctx=ctx, exec_id=exec_id
                    )

                    tool_messages: list[FunctionToolOutputItem] = []
                    async for event in self.execute_tools_stream(
                        tool_calls, ctx=ctx, exec_id=exec_id
                    ):
                        if isinstance(event, ToolResultEvent):
                            tool_messages.append(event.data)
                        yield event

                    await self.on_after_tool(
                        tool_calls=tool_calls,
                        tool_messages=tool_messages,
                        ctx=ctx,
                        exec_id=exec_id,
                    )

                    await self.checkpoint()

                yield TurnEndEvent(
                    source=self.agent_name,
                    exec_id=exec_id,
                    data=TurnEndInfo(turn=turn, had_tool_calls=bool(tool_calls)),
                )

                had_tool_calls = bool(tool_calls)

        finally:
            await self.bg_tasks.cancel_all()

    def _make_final_answer_tool(self) -> BaseTool[BaseModel, None, Any]:
        class FinalAnswerTool(BaseTool[self._final_answer_type, None, Any]):
            def __init__(self) -> None:
                super().__init__(
                    name="final_answer",
                    description=(
                        "You must call this tool to provide the final answer. "
                        "DO NOT output your answer before calling the tool. "
                    ),
                )

            async def _run(
                self,
                inp: BaseModel,
                *,
                ctx: RunContext[Any] | None = None,
                exec_id: str | None = None,
                progress_callback: Any = None,
            ) -> None:
                return None

        return FinalAnswerTool()

    def _process_response(
        self,
        response: Response,
        *,
        ctx: RunContext[CtxT],
        exec_id: str,
    ) -> None:
        ctx.responses[self.agent_name].append(response)
        ctx.usage_tracker.update(
            agent_name=self.agent_name,
            responses=[response],
            model_name=self.llm.model_name,
            litellm_provider=self.llm.litellm_provider,
        )
