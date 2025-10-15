import asyncio
import json
from collections.abc import AsyncIterator, Coroutine, Mapping, Sequence
from copy import deepcopy
from itertools import starmap
from logging import getLogger
from typing import Any, Generic, Protocol, TypedDict, final

from pydantic import BaseModel

from grasp_agents.tracing_decorators import task
from grasp_agents.typing.completion_chunk import CompletionChunk

from .errors import AgentFinalAnswerError
from .llm import LLM
from .llm_agent_memory import LLMAgentMemory
from .run_context import CtxT, RunContext
from .typing.completion import Completion
from .typing.events import (
    CompletionChunkEvent,
    CompletionEvent,
    Event,
    GenMessageEvent,
    LLMStreamingErrorEvent,
    ToolCallEvent,
    ToolMessageEvent,
    UserMessageEvent,
)
from .typing.message import AssistantMessage, ToolMessage, UserMessage
from .typing.tool import BaseTool, NamedToolChoice, ToolCall, ToolChoice
from .utils import is_method_overridden

logger = getLogger(__name__)


class FinalAnswerChecker(Protocol[CtxT]):
    def __call__(
        self,
        memory: LLMAgentMemory,
        *,
        ctx: RunContext[CtxT],
        call_id: str,
        **kwargs: Any,
    ) -> str | None: ...


class BeforeGenerateHook(Protocol[CtxT]):
    async def __call__(
        self,
        memory: LLMAgentMemory,
        *,
        ctx: RunContext[CtxT],
        call_id: str,
        num_turns: int,
        llm_settings: dict[str, Any],
    ) -> None: ...


class AfterGenerateHook(Protocol[CtxT]):
    async def __call__(
        self,
        memory: LLMAgentMemory,
        *,
        ctx: RunContext[CtxT],
        call_id: str,
        num_turns: int,
        llm_settings: dict[str, Any],
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


class HookArgs(TypedDict, total=False):
    memory: LLMAgentMemory
    ctx: RunContext[Any]
    call_id: str
    llm_settings: dict[str, Any]


class LLMPolicyExecutor(Generic[CtxT]):
    def __init__(
        self,
        *,
        agent_name: str,
        llm: LLM,
        tools: list[BaseTool[BaseModel, Any, CtxT]] | None,
        response_schema: Any | None = None,
        response_schema_by_xml_tag: Mapping[str, Any] | None = None,
        max_turns: int,
        react_mode: bool = False,
        final_answer_type: type[BaseModel] = BaseModel,
        final_answer_as_tool_call: bool = False,
    ) -> None:
        super().__init__()

        self._agent_name = agent_name
        self._max_turns = max_turns
        self._react_mode = react_mode

        self._llm = llm
        self._response_schema = response_schema
        self._response_schema_by_xml_tag = response_schema_by_xml_tag

        self._final_answer_type = final_answer_type
        self._final_answer_as_tool_call = final_answer_as_tool_call
        self._final_answer_tool = self.get_final_answer_tool()

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
    def tools(self) -> dict[str, BaseTool[BaseModel, Any, CtxT]]:
        return self._tools or {}

    def check_for_final_answer_impl(
        self,
        memory: LLMAgentMemory,
        *,
        ctx: RunContext[CtxT],
        call_id: str,
        **kwargs: Any,
    ) -> str | None:
        raise NotImplementedError

    @final
    def check_for_final_answer(
        self,
        memory: LLMAgentMemory,
        *,
        ctx: RunContext[CtxT],
        call_id: str,
        **kwargs: Any,
    ) -> str | None:
        if is_method_overridden("check_for_final_answer_impl", self):
            return self.check_for_final_answer_impl(
                memory, ctx=ctx, call_id=call_id, **kwargs
            )

        if self._final_answer_as_tool_call:
            return self._get_final_answer_from_tool_call(memory)

        return None

    async def on_before_generate_impl(
        self,
        memory: LLMAgentMemory,
        *,
        ctx: RunContext[CtxT],
        call_id: str,
        num_turns: int,
        llm_settings: dict[str, Any],
    ) -> None:
        raise NotImplementedError

    @final
    async def on_before_generate(
        self,
        memory: LLMAgentMemory,
        *,
        ctx: RunContext[CtxT],
        call_id: str,
        num_turns: int,
        llm_settings: dict[str, Any],
    ) -> None:
        if is_method_overridden("on_before_generate_impl", self):
            await self.on_before_generate_impl(
                memory=memory,
                ctx=ctx,
                call_id=call_id,
                num_turns=num_turns,
                llm_settings=llm_settings,
            )

    async def on_after_generate_impl(
        self,
        memory: LLMAgentMemory,
        *,
        ctx: RunContext[CtxT],
        call_id: str,
        num_turns: int,
        llm_settings: dict[str, Any],
    ) -> None:
        raise NotImplementedError

    @final
    async def on_after_generate(
        self,
        memory: LLMAgentMemory,
        *,
        ctx: RunContext[CtxT],
        call_id: str,
        num_turns: int,
        llm_settings: dict[str, Any],
    ) -> None:
        if is_method_overridden("on_after_generate_impl", self):
            await self.on_after_generate_impl(
                memory=memory,
                ctx=ctx,
                call_id=call_id,
                num_turns=num_turns,
                llm_settings=llm_settings,
            )

    @task(name="generate")  # type: ignore
    async def generate_message(
        self,
        memory: LLMAgentMemory,
        *,
        tool_choice: ToolChoice | None = None,
        ctx: RunContext[CtxT],
        call_id: str,
        extra_llm_settings: dict[str, Any],
    ) -> AssistantMessage:
        completion = await self.llm.generate_completion(
            memory.messages,
            response_schema=self.response_schema,
            response_schema_by_xml_tag=self.response_schema_by_xml_tag,
            tools=self.tools,
            tool_choice=tool_choice,
            proc_name=self.agent_name,
            call_id=call_id,
            **extra_llm_settings,
        )
        memory.update([completion.message])
        self._process_completion(completion, ctx=ctx, call_id=call_id)

        return completion.message

    @task(name="generate")  # type: ignore
    async def generate_message_stream(
        self,
        memory: LLMAgentMemory,
        *,
        tool_choice: ToolChoice | None = None,
        ctx: RunContext[CtxT],
        call_id: str,
        extra_llm_settings: dict[str, Any],
    ) -> AsyncIterator[
        CompletionChunkEvent[CompletionChunk]
        | CompletionEvent
        | GenMessageEvent
        | LLMStreamingErrorEvent
    ]:
        completion: Completion | None = None

        llm_event_stream = self.llm.generate_completion_stream(
            memory.messages,
            response_schema=self.response_schema,
            response_schema_by_xml_tag=self.response_schema_by_xml_tag,
            tools=self.tools,
            tool_choice=tool_choice,
            proc_name=self.agent_name,
            call_id=call_id,
            **extra_llm_settings,
        )
        llm_event_stream_post = self.llm.postprocess_event_stream(llm_event_stream)

        async for event in llm_event_stream_post:
            if isinstance(event, CompletionEvent):
                completion = event.data
            yield event
        if completion is None:
            return
        yield GenMessageEvent(
            proc_name=self.agent_name, call_id=call_id, data=completion.message
        )

        memory.update([completion.message])
        self._process_completion(completion, ctx=ctx, call_id=call_id)

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

    # @task(name="call_tools")  # type: ignore
    async def call_tools(
        self,
        calls: Sequence[ToolCall],
        memory: LLMAgentMemory,
        ctx: RunContext[CtxT],
        call_id: str,
    ) -> Sequence[ToolMessage | UserMessage]:
        corouts: list[Coroutine[Any, Any, BaseModel]] = []
        for call in calls:
            tool = self.tools[call.tool_name]
            args = json.loads(call.tool_arguments)
            corouts.append(tool(ctx=ctx, call_id=call_id, **args))

        outs = await asyncio.gather(*corouts)
        tool_messages = self.tool_outputs_to_messages(
            outs, calls, ctx=ctx, call_id=call_id
        )
        memory.update(tool_messages)

        if ctx.printer:
            ctx.printer.print_messages(
                tool_messages, agent_name=self.agent_name, call_id=call_id
            )

        return tool_messages

    # @task(name="call_tools")  # type: ignore
    async def call_tools_stream(
        self,
        calls: Sequence[ToolCall],
        memory: LLMAgentMemory,
        ctx: RunContext[CtxT],
        call_id: str,
    ) -> AsyncIterator[ToolCallEvent | ToolMessageEvent | UserMessageEvent]:
        for tool_call in calls:
            yield ToolCallEvent(
                proc_name=self.agent_name, call_id=call_id, data=tool_call
            )

        tool_messages = await self.call_tools(
            calls, memory=memory, ctx=ctx, call_id=call_id
        )

        for tool_message, call in zip(tool_messages, calls, strict=True):
            if isinstance(tool_message, UserMessage):
                yield UserMessageEvent(
                    proc_name=call.tool_name, call_id=call_id, data=tool_message
                )
            else:
                yield ToolMessageEvent(
                    proc_name=call.tool_name, call_id=call_id, data=tool_message
                )

    def _get_final_answer_from_tool_call(self, memory: LLMAgentMemory) -> str | None:
        msgs = memory.messages
        if (
            msgs
            and isinstance(msgs[-1], AssistantMessage)
            and msgs[-1].tool_calls
            and msgs[-1].tool_calls[0].tool_name == self._final_answer_tool.name
        ):
            return msgs[-1].tool_calls[0].tool_arguments
        return None

    def _get_final_answer_from_message(self, memory: LLMAgentMemory) -> str | None:
        msgs = memory.messages
        if msgs and isinstance(msgs[-1], AssistantMessage) and msgs[-1].content:
            return msgs[-1].content
        return None

    def get_final_answer(self, memory: LLMAgentMemory) -> str | None:
        if self._final_answer_as_tool_call:
            return self._get_final_answer_from_tool_call(memory)
        return self._get_final_answer_from_message(memory)

    @task(name="force_generate_final_answer")  # type: ignore
    async def _force_generate_final_answer(
        self,
        memory: LLMAgentMemory,
        ctx: RunContext[CtxT],
        call_id: str,
        extra_llm_settings: dict[str, Any],
    ) -> str:
        # NOTE: Might not need the user message when forcing the tool call
        user_message = UserMessage.from_text(
            "Exceeded the maximum number of turns: provide a final answer now!"
        )
        memory.update([user_message])

        if ctx.printer:
            ctx.printer.print_messages(
                [user_message], agent_name=self.agent_name, call_id=call_id
            )

        tool_choice = (
            NamedToolChoice(name=self._final_answer_tool.name)
            if self._final_answer_as_tool_call
            else None
        )
        _ = await self.generate_message(
            memory,
            tool_choice=tool_choice,
            ctx=ctx,
            call_id=call_id,
            extra_llm_settings=extra_llm_settings,
        )

        final_answer = self.get_final_answer(memory)
        if final_answer is None:
            raise AgentFinalAnswerError(proc_name=self.agent_name, call_id=call_id)

        return final_answer

    @task(name="force_generate_final_answer")  # type: ignore
    async def _force_generate_final_answer_stream(
        self,
        memory: LLMAgentMemory,
        ctx: RunContext[CtxT],
        call_id: str,
        extra_llm_settings: dict[str, Any],
    ) -> AsyncIterator[Event[Any]]:
        # NOTE: Might not need the user message when forcing the tool call
        user_message = UserMessage.from_text(
            "Exceeded the maximum number of turns: provide a final answer now!",
        )
        memory.update([user_message])
        yield UserMessageEvent(
            proc_name=self.agent_name, call_id=call_id, data=user_message
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
            memory,
            tool_choice=tool_choice,
            ctx=ctx,
            call_id=call_id,
            extra_llm_settings=extra_llm_settings,
        ):
            yield event

        final_answer = self.get_final_answer(memory)
        if final_answer is None:
            raise AgentFinalAnswerError(proc_name=self.agent_name, call_id=call_id)

    async def execute(
        self,
        memory: LLMAgentMemory,
        ctx: RunContext[CtxT],
        call_id: str,
        extra_llm_settings: dict[str, Any] | None = None,
    ) -> str:
        """
        Some LLMs do not output tool calls and message content in the same response.
        To enable planning/observation before/after tool calls for such models,
        we might want to force the agent to output a message without
        tool calls (planning) first, then force tool calls in the next message, etc.
        For this, we use the `react_mode` flag.
        """
        _extra_llm_settings = deepcopy(extra_llm_settings or {})
        hooks_kwargs: HookArgs = HookArgs(
            memory=memory,
            ctx=ctx,
            call_id=call_id,
            llm_settings=_extra_llm_settings,
        )

        turns = 0

        # 1. Generate the first message and update memory

        # In ReAct mode, we generate the first message without tool calls
        # to enforce planning.

        tool_choice: ToolChoice | None = None
        if self.tools:
            tool_choice = "none" if self.react_mode else "auto"

        # LLM settings can be modified in-place
        await self.on_before_generate(**hooks_kwargs, num_turns=turns)

        gen_message = await self.generate_message(
            memory,
            tool_choice=tool_choice,
            ctx=ctx,
            call_id=call_id,
            extra_llm_settings=_extra_llm_settings,
        )

        # LLM settings can be modified in-place
        await self.on_after_generate(**hooks_kwargs, num_turns=turns)

        if not self.tools:
            return gen_message.content or ""

        while True:
            # 2. Check if we have a final answer

            final_answer = self.check_for_final_answer(
                memory, ctx=ctx, call_id=call_id, num_turns=turns
            )
            if final_answer is not None:
                return final_answer

            if turns >= self.max_turns:
                final_answer = await self._force_generate_final_answer(
                    memory,
                    ctx=ctx,
                    call_id=call_id,
                    extra_llm_settings=_extra_llm_settings,
                )
                logger.info(
                    f"Max turns reached: {self.max_turns}. Exiting the tool call loop."
                )
                return final_answer

            # 3. Call tools and update memory

            if gen_message.tool_calls:
                await self.call_tools(
                    gen_message.tool_calls, memory=memory, ctx=ctx, call_id=call_id
                )

            # 4. Generate the next message and update memory

            if self.react_mode and gen_message.tool_calls:
                # ReAct mode: used tools in the last message -> avoid tool calls in the next message
                tool_choice = "none"
            elif self.react_mode:
                # ReAct mode: no tool calls in the last message -> force tool calls in the next message
                tool_choice = "required"
            else:
                # No ReAct mode: let the model decide
                tool_choice = "auto"

            await self.on_before_generate(**hooks_kwargs, num_turns=turns)

            gen_message = await self.generate_message(
                memory,
                tool_choice=tool_choice,
                ctx=ctx,
                call_id=call_id,
                extra_llm_settings=_extra_llm_settings,
            )

            await self.on_after_generate(**hooks_kwargs, num_turns=turns)

            turns += 1

    async def execute_stream(
        self,
        memory: LLMAgentMemory,
        ctx: RunContext[CtxT],
        call_id: str,
        extra_llm_settings: dict[str, Any] | None = None,
    ) -> AsyncIterator[Event[Any]]:
        _extra_llm_settings = deepcopy(extra_llm_settings or {})
        hooks_kwargs: HookArgs = HookArgs(
            memory=memory,
            ctx=ctx,
            call_id=call_id,
            llm_settings=_extra_llm_settings,
        )

        turns = 0

        # 1. Generate the first message and update memory

        tool_choice: ToolChoice | None = None
        if self.tools:
            tool_choice = "none" if self.react_mode else "auto"

        await self.on_before_generate(**hooks_kwargs, num_turns=turns)

        gen_message: AssistantMessage | None = None
        async for event in self.generate_message_stream(
            memory,
            tool_choice=tool_choice,
            ctx=ctx,
            call_id=call_id,
            extra_llm_settings=_extra_llm_settings,
        ):
            if isinstance(event, GenMessageEvent):
                gen_message = event.data
            yield event

        if gen_message is None:
            # Exit if generation failed
            return

        await self.on_after_generate(**hooks_kwargs, num_turns=turns)

        if not self.tools:
            # No tools to call, return the content of the generated message
            return

        while True:
            # 2. Check if we have a final answer

            final_answer = self.check_for_final_answer(
                memory, ctx=ctx, call_id=call_id, num_turns=turns
            )
            if final_answer is not None:
                return

            if turns >= self.max_turns:
                async for event in self._force_generate_final_answer_stream(
                    memory,
                    ctx=ctx,
                    call_id=call_id,
                    extra_llm_settings=_extra_llm_settings,
                ):
                    yield event
                logger.info(
                    f"Max turns reached: {self.max_turns}. Exiting the tool call loop."
                )
                return

            # 3. Call tools and update memory

            if gen_message.tool_calls:
                async for event in self.call_tools_stream(
                    gen_message.tool_calls, memory=memory, ctx=ctx, call_id=call_id
                ):
                    yield event

            # 4. Generate the next message and update memory

            if self.react_mode and gen_message.tool_calls:
                tool_choice = "none"
            elif self.react_mode:
                tool_choice = "required"
            else:
                tool_choice = "auto"

            await self.on_before_generate(**hooks_kwargs, num_turns=turns)

            async for event in self.generate_message_stream(
                memory,
                tool_choice=tool_choice,
                ctx=ctx,
                call_id=call_id,
                extra_llm_settings=_extra_llm_settings,
            ):
                yield event
                if isinstance(event, GenMessageEvent):
                    gen_message = event.data

            await self.on_after_generate(**hooks_kwargs, num_turns=turns)

            turns += 1

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
