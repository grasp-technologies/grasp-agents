import asyncio
import json
from collections.abc import AsyncIterator, Coroutine, Mapping, Sequence
from copy import deepcopy
from logging import getLogger
from typing import Any, Generic, Protocol, final

from pydantic import BaseModel
from typing_extensions import TypedDict

from grasp_agents.tracing_decorators import task

from .errors import AgentFinalAnswerError
from .llm import LLM
from .llm_agent_memory import LLMAgentMemory
from .run_context import CtxT, RunContext
from .types.events import (
    Event,
    LLMStreamEvent,
    ToolCallEvent,
    ToolMessageEvent,
    ToolOutputEvent,
    UserMessageEvent,
)
from .types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
)
from .types.llm_events import (
    OutputItemAdded,
    OutputItemDone,
    ResponseCompleted,
    ResponseCreated,
)
from .types.response import Response
from .types.tool import BaseTool, NamedToolChoice, ToolChoice
from .utils.callbacks import is_method_overridden
from .utils.streaming import stream_concurrent

logger = getLogger(__name__)


class FinalAnswerChecker(Protocol[CtxT]):
    def __call__(
        self,
        *,
        ctx: RunContext[CtxT],
        call_id: str,
        **kwargs: Any,
    ) -> str | None: ...


class BeforeGenerateHook(Protocol[CtxT]):
    async def __call__(
        self,
        *,
        ctx: RunContext[CtxT],
        call_id: str,
        num_turns: int,
        extra_llm_settings: dict[str, Any],
    ) -> None: ...


class AfterGenerateHook(Protocol[CtxT]):
    async def __call__(
        self,
        response: Response,
        *,
        ctx: RunContext[CtxT],
        call_id: str,
        num_turns: int,
    ) -> None: ...


class ToolOutputConverter(Protocol[CtxT]):
    async def __call__(
        self,
        tool_outputs: Sequence[Any],
        tool_calls: Sequence[FunctionToolCallItem],
        *,
        ctx: RunContext[CtxT],
        call_id: str,
        **kwargs: Any,
    ) -> Sequence[FunctionToolOutputItem | InputMessageItem]: ...


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


class CallArgs(TypedDict, total=False):
    ctx: RunContext[Any]
    call_id: str


class LLMPolicyExecutor(Generic[CtxT]):
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
        react_mode: bool = False,
        final_answer_type: type[BaseModel] = BaseModel,
        final_answer_as_tool_call: bool = False,
        stream_llm_responses: bool = True,
        stream_tools: bool = False,
        tracing_exclude_input_fields: set[str] | None = None,
    ) -> None:
        super().__init__()

        self._agent_name = agent_name
        self._max_turns = max_turns
        self._react_mode = react_mode

        self._llm = llm
        self._response_schema = response_schema
        self._response_schema_by_xml_tag = response_schema_by_xml_tag

        self.memory = memory

        self._final_answer_type = final_answer_type
        self._final_answer_as_tool_call = final_answer_as_tool_call
        self._final_answer_tool = self.get_final_answer_tool()

        self._stream_llm_responses = stream_llm_responses
        self._stream_tools = stream_tools

        tools_list: list[BaseTool[BaseModel, Any, CtxT]] | None = tools
        if tools and final_answer_as_tool_call:
            tools_list = tools + [self._final_answer_tool]
        self._tools = {t.name: t for t in tools_list} if tools_list else None

        self._tracing_exclude_input_fields = tracing_exclude_input_fields

    @property
    def agent_name(self) -> str:
        return self._agent_name

    @property
    def max_turns(self) -> int:
        return self._max_turns

    @property
    def react_mode(self) -> bool:
        return self._react_mode

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

    def check_for_final_answer_impl(
        self,
        *,
        ctx: RunContext[CtxT],
        call_id: str,
        **kwargs: Any,
    ) -> str | None:
        raise NotImplementedError

    @final
    def check_for_final_answer(
        self,
        *,
        response: Response,
        ctx: RunContext[CtxT],
        call_id: str,
        **kwargs: Any,
    ) -> str | None:
        if is_method_overridden("check_for_final_answer_impl", self):
            return self.check_for_final_answer_impl(
                ctx=ctx, call_id=call_id, response=response, **kwargs
            )

        if self._final_answer_as_tool_call:
            return self.get_final_answer(response)

        return None

    async def on_before_generate_impl(
        self,
        *,
        ctx: RunContext[CtxT],
        call_id: str,
        num_turns: int,
        extra_llm_settings: dict[str, Any],
    ) -> None:
        raise NotImplementedError

    @final
    async def on_before_generate(
        self,
        *,
        ctx: RunContext[CtxT],
        call_id: str,
        num_turns: int,
        extra_llm_settings: dict[str, Any],
    ) -> None:
        if is_method_overridden("on_before_generate_impl", self):
            await self.on_before_generate_impl(
                ctx=ctx,
                call_id=call_id,
                num_turns=num_turns,
                extra_llm_settings=extra_llm_settings,
            )

    async def on_after_generate_impl(
        self,
        response: Response,
        *,
        ctx: RunContext[CtxT],
        call_id: str,
        num_turns: int,
    ) -> None:
        raise NotImplementedError

    @final
    async def on_after_generate(
        self,
        response: Response,
        *,
        ctx: RunContext[CtxT],
        call_id: str,
        num_turns: int,
    ) -> None:
        if is_method_overridden("on_after_generate_impl", self):
            await self.on_after_generate_impl(
                response=response,
                ctx=ctx,
                call_id=call_id,
                num_turns=num_turns,
            )

    # --- LLM generation ---

    @task(name="generate")  # type: ignore
    async def generate_message_stream(
        self,
        *,
        tool_choice: ToolChoice | None = None,
        ctx: RunContext[CtxT],
        call_id: str,
        extra_llm_settings: dict[str, Any],
    ) -> AsyncIterator[Event[Any]]:
        response: Response | None = None

        llm_params: dict[str, Any] = {
            "input": self.memory.messages,
            "response_schema": self.response_schema,
            "response_schema_by_xml_tag": self.response_schema_by_xml_tag,
            "tools": self.tools or None,
            "tool_choice": tool_choice,
            **extra_llm_settings,
        }

        if self._stream_llm_responses:
            async for se in self.llm.generate_response_stream(**llm_params):
                yield LLMStreamEvent(data=se, src_name=self.agent_name, call_id=call_id)
                if isinstance(se, ResponseCompleted):
                    response = se.response
        else:
            response = await self.llm.generate_response(**llm_params)
            # Synthesize stream events for uniform consumer interface
            seq = 0
            seq += 1
            yield LLMStreamEvent(
                data=ResponseCreated(
                    response=response,
                    sequence_number=seq,  # type: ignore[arg-type]
                ),
                src_name=self.agent_name,
                call_id=call_id,
            )
            for idx, item in enumerate(response.output_items):
                seq += 1
                yield LLMStreamEvent(
                    data=OutputItemAdded(
                        item=item, output_index=idx, sequence_number=seq
                    ),
                    src_name=self.agent_name,
                    call_id=call_id,
                )
                seq += 1
                yield LLMStreamEvent(
                    data=OutputItemDone(
                        item=item, output_index=idx, sequence_number=seq
                    ),
                    src_name=self.agent_name,
                    call_id=call_id,
                )
            seq += 1
            yield LLMStreamEvent(
                data=ResponseCompleted(
                    response=response,
                    sequence_number=seq,  # type: ignore[arg-type]
                ),
                src_name=self.agent_name,
                call_id=call_id,
            )

        assert response is not None

        # Update memory with response output items
        self.memory.update(response.output_items)

        # Track response
        self._process_response(response, ctx=ctx, call_id=call_id)

    async def generate_message(
        self,
        *,
        tool_choice: ToolChoice | None = None,
        ctx: RunContext[CtxT],
        call_id: str,
        extra_llm_settings: dict[str, Any],
    ) -> Response:
        stream = ResponseCapture(
            self.generate_message_stream(
                tool_choice=tool_choice,
                extra_llm_settings=extra_llm_settings,
                ctx=ctx,
                call_id=call_id,
            )
        )
        async for _ in stream:
            pass
        assert stream.response is not None
        return stream.response

    # --- Tool calling ---

    async def tool_outputs_to_messages_impl(
        self,
        tool_outputs: Sequence[Any],
        tool_calls: Sequence[FunctionToolCallItem],
        *,
        ctx: RunContext[CtxT],
        call_id: str,
    ) -> Sequence[FunctionToolOutputItem | InputMessageItem]:
        raise NotImplementedError

    def tool_outputs_to_messages_default(
        self,
        tool_outputs: Sequence[Any],
        tool_calls: Sequence[FunctionToolCallItem],
    ) -> Sequence[FunctionToolOutputItem | InputMessageItem]:
        return [
            FunctionToolOutputItem.from_tool_result(call_id=call.call_id, output=output)
            for output, call in zip(tool_outputs, tool_calls, strict=True)
        ]

    @final
    async def tool_outputs_to_messages(
        self,
        tool_outputs: Sequence[Any],
        tool_calls: Sequence[FunctionToolCallItem],
        *,
        ctx: RunContext[CtxT],
        call_id: str,
    ) -> Sequence[FunctionToolOutputItem | InputMessageItem]:
        if is_method_overridden("tool_outputs_to_messages_impl", self):
            return await self.tool_outputs_to_messages_impl(
                tool_outputs, tool_calls, ctx=ctx, call_id=call_id
            )
        return self.tool_outputs_to_messages_default(tool_outputs, tool_calls)

    async def _get_tool_outputs(
        self,
        calls: Sequence[FunctionToolCallItem],
        ctx: RunContext[CtxT],
        call_id: str,
    ) -> Sequence[Any]:
        corouts: list[Coroutine[Any, Any, BaseModel]] = []
        for call in calls:
            tool = self.tools[call.name]
            args = json.loads(call.arguments)
            corouts.append(tool(ctx=ctx, call_id=call_id, **args))

        return await asyncio.gather(*corouts)

    async def call_tools_stream(
        self,
        calls: Sequence[FunctionToolCallItem],
        ctx: RunContext[CtxT],
        call_id: str,
    ) -> AsyncIterator[Event[Any]]:
        for call in calls:
            yield ToolCallEvent(src_name=self.agent_name, call_id=call_id, data=call)

        if self._stream_tools:
            streams: list[AsyncIterator[Event[Any]]] = []
            for call in calls:
                tool = self.tools[call.name]
                args = json.loads(call.arguments)
                streams.append(
                    tool.run_stream(inp=tool.in_type(**args), ctx=ctx, call_id=call_id)
                )

            # TODO: treat None outputs on stream failure

            outputs_map: dict[int, Any] = {}
            async for idx, event in stream_concurrent(streams):
                if isinstance(event, ToolOutputEvent):
                    outputs_map[idx] = event.data
                else:
                    yield event
            outputs = [outputs_map[idx] for idx in sorted(outputs_map)]

        else:
            outputs = await self._get_tool_outputs(calls, ctx=ctx, call_id=call_id)

        tool_messages = await self.tool_outputs_to_messages(
            outputs, calls, ctx=ctx, call_id=call_id
        )

        call_name_by_id = {c.call_id: c.name for c in calls}
        for tool_message in tool_messages:
            if isinstance(tool_message, FunctionToolOutputItem):
                src = call_name_by_id.get(tool_message.call_id, self.agent_name)
                yield ToolMessageEvent(src_name=src, call_id=call_id, data=tool_message)
            else:
                yield UserMessageEvent(
                    src_name=self.agent_name,
                    call_id=call_id,
                    data=tool_message,
                )

        self.memory.update(tool_messages)

        if ctx.printer:
            ctx.printer.print_messages(
                tool_messages, agent_name=self.agent_name, call_id=call_id
            )

    async def call_tools(
        self,
        calls: Sequence[FunctionToolCallItem],
        ctx: RunContext[CtxT],
        call_id: str,
    ) -> Sequence[FunctionToolOutputItem | InputMessageItem]:
        tool_messages: list[FunctionToolOutputItem | InputMessageItem] = []
        async for event in self.call_tools_stream(calls, ctx=ctx, call_id=call_id):
            if isinstance(event, ToolMessageEvent | UserMessageEvent):
                tool_messages.append(event.data)

        return tool_messages

    # --- Final answer ---

    def get_final_answer(self, response: Response) -> str | None:
        if self._final_answer_as_tool_call:
            for tc in response.tool_call_items:
                if tc.name == self._final_answer_tool.name:
                    return tc.arguments
            return None
        text = response.output_text
        return text or None

    @task(name="force_generate_final_answer")  # type: ignore
    async def _force_generate_final_answer_stream(
        self,
        ctx: RunContext[CtxT],
        call_id: str,
        extra_llm_settings: dict[str, Any],
    ) -> AsyncIterator[Event[Any]]:
        user_message = InputMessageItem.from_text(
            "Exceeded the maximum number of turns: provide a final answer now!",
            role="user",
        )
        self.memory.update([user_message])
        yield UserMessageEvent(
            src_name=self.agent_name, call_id=call_id, data=user_message
        )
        if ctx.printer:
            ctx.printer.print_messages(
                [user_message], agent_name=self.agent_name, call_id=call_id
            )

        tool_choice = (
            NamedToolChoice(name=self._final_answer_tool.name)
            if self._final_answer_as_tool_call
            else None
        )
        stream = ResponseCapture(
            self.generate_message_stream(
                tool_choice=tool_choice,
                ctx=ctx,
                call_id=call_id,
                extra_llm_settings=extra_llm_settings,
            )
        )
        async for event in stream:
            yield event

        assert stream.response is not None
        final_answer = self.get_final_answer(stream.response)
        if final_answer is None:
            raise AgentFinalAnswerError(proc_name=self.agent_name, call_id=call_id)

    # --- Main execution loop ---

    async def execute_stream(
        self,
        ctx: RunContext[CtxT],
        call_id: str,
        extra_llm_settings: dict[str, Any] | None = None,
    ) -> AsyncIterator[Event[Any]]:
        call_kwargs: CallArgs = CallArgs(ctx=ctx, call_id=call_id)

        turns = 0

        # 1. Generate the first message and update memory

        _extra_llm_settings = deepcopy(extra_llm_settings or {})
        await self.on_before_generate(
            extra_llm_settings=_extra_llm_settings,
            num_turns=turns,
            **call_kwargs,
        )

        tool_choice: ToolChoice | None = None
        if self.tools:
            tool_choice = "none" if self.react_mode else "auto"
        tool_choice = _extra_llm_settings.pop("tool_choice", tool_choice)

        stream = ResponseCapture(
            self.generate_message_stream(
                tool_choice=tool_choice,
                extra_llm_settings=_extra_llm_settings,
                **call_kwargs,
            )
        )
        async for event in stream:
            yield event

        assert stream.response is not None
        response = stream.response
        await self.on_after_generate(response, num_turns=turns, **call_kwargs)

        if not self.tools:
            return

        while True:
            # 2. Check if we have a final answer

            final_answer = self.check_for_final_answer(
                response=response,
                ctx=ctx,
                call_id=call_id,
                num_turns=turns,
            )
            if final_answer is not None:
                return

            if turns >= self.max_turns:
                async for event in self._force_generate_final_answer_stream(
                    extra_llm_settings=_extra_llm_settings, **call_kwargs
                ):
                    yield event
                logger.info(
                    f"Max turns reached: {self.max_turns}. Exiting the tool call loop."
                )
                return

            # 3. Call tools and update memory

            tool_calls = response.tool_call_items
            if tool_calls:
                async for event in self.call_tools_stream(tool_calls, **call_kwargs):
                    yield event

            # 4. Generate the next message and update memory

            _extra_llm_settings = deepcopy(extra_llm_settings or {})
            await self.on_before_generate(
                extra_llm_settings=_extra_llm_settings,
                num_turns=turns,
                **call_kwargs,
            )

            if self.react_mode and tool_calls:
                tool_choice = "none"
            elif self.react_mode:
                tool_choice = "required"
            else:
                tool_choice = "auto"
            tool_choice = _extra_llm_settings.pop("tool_choice", tool_choice)

            stream = ResponseCapture(
                self.generate_message_stream(
                    tool_choice=tool_choice,
                    extra_llm_settings=_extra_llm_settings,
                    **call_kwargs,
                )
            )
            async for event in stream:
                yield event

            assert stream.response is not None
            response = stream.response
            await self.on_after_generate(response, num_turns=turns, **call_kwargs)

            turns += 1

    async def execute(
        self,
        ctx: RunContext[CtxT],
        call_id: str,
        extra_llm_settings: dict[str, Any] | None = None,
    ) -> str:
        stream = ResponseCapture(
            self.execute_stream(
                ctx=ctx,
                call_id=call_id,
                extra_llm_settings=extra_llm_settings,
            )
        )
        async for _ in stream:
            pass
        assert stream.response is not None
        return self.get_final_answer(stream.response) or ""

    def get_final_answer_tool(self) -> BaseTool[BaseModel, None, Any]:
        class FinalAnswerTool(BaseTool[self._final_answer_type, None, Any]):
            name: str = "final_answer"
            description: str = (
                "You must call this tool to provide the final answer. "
                "DO NOT output your answer before calling the tool. "
            )

            async def run(
                self,
                inp: BaseModel,
                *,
                ctx: RunContext[Any] | None = None,
                call_id: str | None = None,
            ) -> None:
                return None

        return FinalAnswerTool()

    def _process_response(
        self, response: Response, *, ctx: RunContext[CtxT], call_id: str
    ) -> None:
        ctx.responses[self.agent_name].append(response)
        ctx.usage_tracker.update(
            agent_name=self.agent_name,
            responses=[response],
            model_name=self.llm.model_name,
            litellm_provider=self.llm.litellm_provider,
        )
