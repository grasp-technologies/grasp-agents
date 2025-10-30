import asyncio
import json
from collections.abc import AsyncIterator, Coroutine, Mapping, Sequence
from copy import deepcopy
from itertools import starmap
from logging import getLogger
from typing import Any, Generic, Protocol, TypedDict, cast, final

from pydantic import BaseModel

from grasp_agents.tracing_decorators import task

from .errors import AgentFinalAnswerError
from .llm import LLM
from .llm_agent_memory import LLMAgentMemory
from .run_context import CtxT, RunContext
from .typing.completion import Completion
from .typing.events import (
    Event,
    GenMessageEvent,
    ToolCallEvent,
    ToolMessageEvent,
    ToolOutputEvent,
    UserMessageEvent,
)
from .typing.message import AssistantMessage, ToolMessage, UserMessage
from .typing.tool import BaseTool, NamedToolChoice, ToolCall, ToolChoice
from .utils.callbacks import is_method_overridden
from .utils.streaming import EventStream, stream_concurrent

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
        gen_message: AssistantMessage,
        *,
        ctx: RunContext[CtxT],
        call_id: str,
        num_turns: int,
    ) -> None: ...


class ToolOutputConverter(Protocol[CtxT]):
    def __call__(
        self,
        tool_outputs: Sequence[Any],
        tool_calls: Sequence[ToolCall],
        *,
        ctx: RunContext[CtxT],
        call_id: str,
        **kwargs: Any,
    ) -> Sequence[ToolMessage | UserMessage]: ...


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
        ctx: RunContext[CtxT],
        call_id: str,
        **kwargs: Any,
    ) -> str | None:
        if is_method_overridden("check_for_final_answer_impl", self):
            return self.check_for_final_answer_impl(ctx=ctx, call_id=call_id, **kwargs)

        if self._final_answer_as_tool_call:
            return self._get_final_answer_from_tool_call()

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
        gen_message: AssistantMessage,
        *,
        ctx: RunContext[CtxT],
        call_id: str,
        num_turns: int,
    ) -> None:
        raise NotImplementedError

    @final
    async def on_after_generate(
        self,
        gen_message: AssistantMessage,
        *,
        ctx: RunContext[CtxT],
        call_id: str,
        num_turns: int,
    ) -> None:
        if is_method_overridden("on_after_generate_impl", self):
            await self.on_after_generate_impl(
                gen_message=gen_message,
                ctx=ctx,
                call_id=call_id,
                num_turns=num_turns,
            )

    @task(name="generate")  # type: ignore
    async def generate_message_stream(
        self,
        *,
        tool_choice: ToolChoice | None = None,
        ctx: RunContext[CtxT],
        call_id: str,
        extra_llm_settings: dict[str, Any],
    ) -> AsyncIterator[Event[Any]]:
        completion: Completion | None = None

        if self._stream_llm_responses:
            llm_stream = self.llm.generate_completion_stream(
                self.memory.messages,
                response_schema=self.response_schema,
                response_schema_by_xml_tag=self.response_schema_by_xml_tag,
                tools=self.tools,
                tool_choice=tool_choice,
                proc_name=self.agent_name,
                call_id=call_id,
                **extra_llm_settings,
            )
            llm_stream_post = self.llm.postprocess_event_stream(llm_stream)
            llm_stream_wrapped = EventStream[Completion](llm_stream_post, Completion)
            async for event in llm_stream_wrapped:
                yield event
            completion = await llm_stream_wrapped.final_data()

        else:
            completion = await self.llm.generate_completion(
                self.memory.messages,
                response_schema=self.response_schema,
                response_schema_by_xml_tag=self.response_schema_by_xml_tag,
                tools=self.tools,
                tool_choice=tool_choice,
                proc_name=self.agent_name,
                call_id=call_id,
                **extra_llm_settings,
            )

        yield GenMessageEvent(
            src_name=self.agent_name, call_id=call_id, data=completion.message
        )
        self.memory.update([completion.message])
        self._process_completion(completion, ctx=ctx, call_id=call_id)

    async def generate_message(
        self,
        *,
        tool_choice: ToolChoice | None = None,
        ctx: RunContext[CtxT],
        call_id: str,
        extra_llm_settings: dict[str, Any],
    ) -> AssistantMessage:
        gen_message: AssistantMessage | None = None
        async for event in self.generate_message_stream(
            tool_choice=tool_choice,
            extra_llm_settings=extra_llm_settings,
            ctx=ctx,
            call_id=call_id,
        ):
            if isinstance(event, GenMessageEvent):
                gen_message = event.data

        return cast("AssistantMessage", gen_message)

    def tool_outputs_to_messages_impl(
        self,
        tool_outputs: Sequence[Any],
        tool_calls: Sequence[ToolCall],
        *,
        ctx: RunContext[CtxT],
        call_id: str,
    ) -> Sequence[ToolMessage | UserMessage]:
        raise NotImplementedError

    def tool_outputs_to_messages_default(
        self, tool_outputs: Sequence[Any], tool_calls: Sequence[ToolCall]
    ) -> Sequence[ToolMessage | UserMessage]:
        return list(
            starmap(
                ToolMessage.from_tool_output, zip(tool_outputs, tool_calls, strict=True)
            )
        )

    @final
    def tool_outputs_to_messages(
        self,
        tool_outputs: Sequence[Any],
        tool_calls: Sequence[ToolCall],
        *,
        ctx: RunContext[CtxT],
        call_id: str,
    ) -> Sequence[ToolMessage | UserMessage]:
        if is_method_overridden("tool_outputs_to_messages_impl", self):
            return self.tool_outputs_to_messages_impl(
                tool_outputs, tool_calls, ctx=ctx, call_id=call_id
            )
        return self.tool_outputs_to_messages_default(tool_outputs, tool_calls)

    async def _get_tool_outputs(
        self, calls: Sequence[ToolCall], ctx: RunContext[CtxT], call_id: str
    ) -> Sequence[Any]:
        corouts: list[Coroutine[Any, Any, BaseModel]] = []
        for call in calls:
            tool = self.tools[call.tool_name]
            args = json.loads(call.tool_arguments)
            corouts.append(tool(ctx=ctx, call_id=call_id, **args))

        return await asyncio.gather(*corouts)

    async def call_tools_stream(
        self, calls: Sequence[ToolCall], ctx: RunContext[CtxT], call_id: str
    ) -> AsyncIterator[Event[Any]]:
        for call in calls:
            yield ToolCallEvent(src_name=self.agent_name, call_id=call_id, data=call)

        if self._stream_tools:
            streams: list[AsyncIterator[Event[Any]]] = []
            for call in calls:
                tool = self.tools[call.tool_name]
                args = json.loads(call.tool_arguments)
                streams.append(
                    tool.run_stream(inp=tool.in_type(**args), ctx=ctx, call_id=call_id)
                )

            outputs_map: dict[int, Any] = {}
            async for idx, event in stream_concurrent(streams):
                if isinstance(event, ToolOutputEvent):
                    outputs_map[idx] = event.data
                else:
                    yield event
            outputs = [outputs_map[idx] for idx in sorted(outputs_map)]

        else:
            outputs = await self._get_tool_outputs(calls, ctx=ctx, call_id=call_id)

        tool_messages = self.tool_outputs_to_messages(
            outputs, calls, ctx=ctx, call_id=call_id
        )

        for tool_message, call in zip(tool_messages, calls, strict=True):
            if isinstance(tool_message, UserMessage):
                yield UserMessageEvent(
                    src_name=call.tool_name, call_id=call_id, data=tool_message
                )
            else:
                yield ToolMessageEvent(
                    src_name=call.tool_name, call_id=call_id, data=tool_message
                )

        self.memory.update(tool_messages)

        if ctx.printer:
            ctx.printer.print_messages(
                tool_messages, agent_name=self.agent_name, call_id=call_id
            )

    async def call_tools(
        self, calls: Sequence[ToolCall], ctx: RunContext[CtxT], call_id: str
    ) -> Sequence[ToolMessage | UserMessage]:
        tool_messages: list[ToolMessage | UserMessage] = []
        async for event in self.call_tools_stream(calls, ctx=ctx, call_id=call_id):
            if isinstance(event, ToolMessageEvent | UserMessageEvent):
                tool_messages.append(event.data)

        return tool_messages

    def _get_final_answer_from_tool_call(self) -> str | None:
        msgs = self.memory.messages
        if (
            msgs
            and isinstance(msgs[-1], AssistantMessage)
            and msgs[-1].tool_calls
            and msgs[-1].tool_calls[0].tool_name == self._final_answer_tool.name
        ):
            return msgs[-1].tool_calls[0].tool_arguments
        return None

    def _get_final_answer_from_message(self) -> str | None:
        msgs = self.memory.messages
        if msgs and isinstance(msgs[-1], AssistantMessage) and msgs[-1].content:
            return msgs[-1].content
        return None

    def get_final_answer(self) -> str | None:
        if self._final_answer_as_tool_call:
            return self._get_final_answer_from_tool_call()
        return self._get_final_answer_from_message()

    @task(name="force_generate_final_answer")  # type: ignore
    async def _force_generate_final_answer_stream(
        self, ctx: RunContext[CtxT], call_id: str, extra_llm_settings: dict[str, Any]
    ) -> AsyncIterator[Event[Any]]:
        # NOTE: Might not need the user message when forcing the tool call
        user_message = UserMessage.from_text(
            "Exceeded the maximum number of turns: provide a final answer now!",
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
        async for event in self.generate_message_stream(
            tool_choice=tool_choice,
            ctx=ctx,
            call_id=call_id,
            extra_llm_settings=extra_llm_settings,
        ):
            yield event

        final_answer = self.get_final_answer()
        if final_answer is None:
            raise AgentFinalAnswerError(proc_name=self.agent_name, call_id=call_id)

    async def _force_generate_final_answer(
        self, ctx: RunContext[CtxT], call_id: str, extra_llm_settings: dict[str, Any]
    ) -> str:
        async for _ in self._force_generate_final_answer_stream(
            ctx=ctx, call_id=call_id, extra_llm_settings=extra_llm_settings
        ):
            pass
        return cast("str", self.get_final_answer())

    async def execute_stream(
        self,
        ctx: RunContext[CtxT],
        call_id: str,
        extra_llm_settings: dict[str, Any] | None = None,
    ) -> AsyncIterator[Event[Any]]:
        call_kwargs: CallArgs = CallArgs(ctx=ctx, call_id=call_id)

        turns = 0
        gen_message: AssistantMessage | None = None

        # 1. Generate the first message and update memory

        _extra_llm_settings = deepcopy(extra_llm_settings or {})
        await self.on_before_generate(
            extra_llm_settings=_extra_llm_settings, num_turns=turns, **call_kwargs
        )

        tool_choice: ToolChoice | None = None
        if self.tools:
            tool_choice = "none" if self.react_mode else "auto"
        tool_choice = _extra_llm_settings.pop("tool_choice", tool_choice)

        async for event in self.generate_message_stream(
            tool_choice=tool_choice,
            extra_llm_settings=_extra_llm_settings,
            **call_kwargs,
        ):
            if isinstance(event, GenMessageEvent):
                gen_message = event.data
            yield event
        gen_message = cast("AssistantMessage", gen_message)

        await self.on_after_generate(gen_message, num_turns=turns, **call_kwargs)

        if not self.tools:
            # No tools to call, return the content of the generated message
            return

        while True:
            # 2. Check if we have a final answer

            final_answer = self.check_for_final_answer(
                ctx=ctx, call_id=call_id, num_turns=turns
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

            if gen_message.tool_calls:
                async for event in self.call_tools_stream(
                    gen_message.tool_calls, **call_kwargs
                ):
                    yield event

            # 4. Generate the next message and update memory

            _extra_llm_settings = deepcopy(extra_llm_settings or {})
            await self.on_before_generate(
                extra_llm_settings=_extra_llm_settings, num_turns=turns, **call_kwargs
            )

            if self.react_mode and gen_message.tool_calls:
                tool_choice = "none"
            elif self.react_mode:
                tool_choice = "required"
            else:
                tool_choice = "auto"
            tool_choice = _extra_llm_settings.pop("tool_choice", tool_choice)

            async for event in self.generate_message_stream(
                tool_choice=tool_choice,
                extra_llm_settings=_extra_llm_settings,
                **call_kwargs,
            ):
                if isinstance(event, GenMessageEvent):
                    gen_message = event.data
                yield event

            await self.on_after_generate(gen_message, num_turns=turns, **call_kwargs)

            turns += 1

    async def execute(
        self,
        ctx: RunContext[CtxT],
        call_id: str,
        extra_llm_settings: dict[str, Any] | None = None,
    ) -> str:
        async for _ in self.execute_stream(
            ctx=ctx, call_id=call_id, extra_llm_settings=extra_llm_settings
        ):
            pass
        return cast("str", self.get_final_answer())

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

    def _process_completion(
        self, completion: Completion, *, ctx: RunContext[CtxT], call_id: str
    ) -> None:
        ctx.completions[self.agent_name].append(completion)
        ctx.usage_tracker.update(
            agent_name=self.agent_name,
            completions=[completion],
            model_name=self.llm.model_name,
        )
        if ctx.printer:
            ctx.printer.print_messages(
                [completion.message],
                usages=[completion.usage],
                agent_name=self.agent_name,
                call_id=call_id,
            )

    # def wrapped_generate_message_stream(
    #     self,
    #     *,
    #     tool_choice: ToolChoice | None = None,
    #     ctx: RunContext[CtxT],
    #     call_id: str,
    #     extra_llm_settings: dict[str, Any],
    # ) -> EventStream[AssistantMessage]:
    #     stream = self.generate_message_stream(
    #         tool_choice=tool_choice,
    #         ctx=ctx,
    #         call_id=call_id,
    #         extra_llm_settings=extra_llm_settings,
    #     )
    #     return EventStream[AssistantMessage](stream, AssistantMessage)
