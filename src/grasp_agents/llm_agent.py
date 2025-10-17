from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Generic, Protocol, TypeVar, cast, final

from pydantic import BaseModel

from .llm import LLM
from .llm_agent_memory import LLMAgentMemory
from .llm_policy_executor import (
    AfterGenerateHook,
    BeforeGenerateHook,
    FinalAnswerChecker,
    LLMPolicyExecutor,
    ToolOutputConverter,
)
from .processors.parallel_processor import ParallelProcessor
from .prompt_builder import (
    InputContentBuilder,
    PromptBuilder,
    SystemPromptBuilder,
)
from .run_context import CtxT, RunContext
from .typing.content import Content, ImageData
from .typing.events import (
    Event,
    ProcPayloadOutputEvent,
    SystemMessageEvent,
    UserMessageEvent,
)
from .typing.io import InT, LLMPrompt, OutT, ProcName
from .typing.message import AssistantMessage, Message, SystemMessage, UserMessage
from .typing.tool import BaseTool, ToolCall
from .utils import get_prompt, is_method_overridden
from .validation import validate_obj_from_json_or_py_string

_InT_contra = TypeVar("_InT_contra", contravariant=True)
_OutT_co = TypeVar("_OutT_co", covariant=True)


class MemoryPreparator(Protocol):
    def __call__(
        self,
        memory: "LLMAgentMemory",
        *,
        instructions: LLMPrompt | None = None,
        in_args: Any | None,
        ctx: RunContext[Any],
        call_id: str,
    ) -> None: ...


class OutputParser(Protocol[_InT_contra, _OutT_co, CtxT]):
    def __call__(
        self,
        final_answer: str,
        *,
        memory: LLMAgentMemory,
        in_args: _InT_contra | None = None,
        ctx: RunContext[CtxT],
        call_id: str,
    ) -> _OutT_co: ...


class LLMAgent(
    ParallelProcessor[InT, OutT, LLMAgentMemory, CtxT],
    Generic[InT, OutT, CtxT],
):
    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_in_type",
        1: "_out_type",
    }

    def __init__(
        self,
        name: ProcName,
        *,
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
        response_schema: Any | None = None,
        response_schema_by_xml_tag: Mapping[str, Any] | None = None,
        # Agent loop settings
        max_turns: int = 100,
        react_mode: bool = False,
        final_answer_as_tool_call: bool = False,
        # Agent memory management
        memory: LLMAgentMemory | None = None,
        reset_memory_on_run: bool = False,
        # Agent run retries
        max_retries: int = 0,
        # Multi-agent routing
        recipients: Sequence[ProcName] | None = None,
    ) -> None:
        super().__init__(name=name, recipients=recipients, max_retries=max_retries)

        # Agent memory

        self._memory: LLMAgentMemory = memory or LLMAgentMemory()
        self._reset_memory_on_run = reset_memory_on_run

        # LLM policy executor

        if issubclass(self._out_type, BaseModel):
            final_answer_type = self._out_type
        elif not final_answer_as_tool_call:
            final_answer_type = BaseModel
        else:
            raise TypeError(
                "Final answer type must be a subclass of BaseModel if "
                "final_answer_as_tool_call is True."
            )

        self._used_default_llm_response_schema: bool = False
        if (
            response_schema is None
            and tools is None
            and not is_method_overridden(
                "parse_output_impl", self, LLMAgent[Any, Any, Any]
            )
        ):
            response_schema = self.out_type
            self._used_default_llm_response_schema = True

        self._policy_executor: LLMPolicyExecutor[CtxT] = LLMPolicyExecutor[CtxT](
            agent_name=self.name,
            llm=llm,
            tools=tools,
            response_schema=response_schema,
            response_schema_by_xml_tag=response_schema_by_xml_tag,
            max_turns=max_turns,
            react_mode=react_mode,
            final_answer_type=final_answer_type,
            final_answer_as_tool_call=final_answer_as_tool_call,
        )

        # Prompt builder

        sys_prompt = get_prompt(prompt_text=sys_prompt, prompt_path=sys_prompt_path)
        in_prompt = get_prompt(prompt_text=in_prompt, prompt_path=in_prompt_path)

        self._prompt_builder = PromptBuilder[self.in_type, CtxT](
            agent_name=self._name, sys_prompt=sys_prompt, in_prompt=in_prompt
        )

        self._register_overridden_implementations()

    @property
    def llm(self) -> LLM:
        return self._policy_executor.llm

    @property
    def tools(self) -> dict[str, BaseTool[BaseModel, Any, CtxT]]:
        return self._policy_executor.tools

    @property
    def max_turns(self) -> int:
        return self._policy_executor.max_turns

    @property
    def sys_prompt(self) -> LLMPrompt | None:
        return self._prompt_builder.sys_prompt

    @property
    def in_prompt(self) -> LLMPrompt | None:
        return self._prompt_builder.in_prompt

    @final
    def prepare_memory(
        self,
        memory: LLMAgentMemory,
        *,
        instructions: LLMPrompt | None = None,
        in_args: InT | None = None,
        ctx: RunContext[Any],
        call_id: str,
    ) -> None:
        if is_method_overridden("prepare_memory_impl", self, LLMAgent[Any, Any, Any]):
            return self.prepare_memory_impl(
                memory=memory,
                instructions=instructions,
                in_args=in_args,
                ctx=ctx,
                call_id=call_id,
            )

    def _memorize_inputs(
        self,
        memory: LLMAgentMemory,
        *,
        chat_inputs: LLMPrompt | Sequence[str | ImageData] | None = None,
        in_args: InT | None = None,
        ctx: RunContext[CtxT],
        call_id: str,
    ) -> tuple[SystemMessage | None, UserMessage | None]:
        formatted_sys_prompt = self._prompt_builder.build_system_prompt(
            ctx=ctx, call_id=call_id
        )

        system_message: SystemMessage | None = None
        if self._reset_memory_on_run or memory.is_empty:
            memory.reset(formatted_sys_prompt)
            if formatted_sys_prompt is not None:
                system_message = cast("SystemMessage", memory.messages[0])
        else:
            self.prepare_memory(
                memory=memory,
                instructions=formatted_sys_prompt,
                in_args=in_args,
                ctx=ctx,
                call_id=call_id,
            )

        input_message = self._prompt_builder.build_input_message(
            chat_inputs=chat_inputs, in_args=in_args, ctx=ctx, call_id=call_id
        )
        if input_message:
            memory.update([input_message])

        return system_message, input_message

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
        memory: LLMAgentMemory,
        in_args: InT | None = None,
        ctx: RunContext[CtxT],
        call_id: str,
    ) -> OutT:
        if is_method_overridden("parse_output_impl", self, LLMAgent[Any, Any, Any]):
            return self.parse_output_impl(
                final_answer,
                memory=memory,
                in_args=in_args,
                ctx=ctx,
                call_id=call_id,
            )

        return self.parse_output_default(final_answer)

    async def _process(
        self,
        chat_inputs: LLMPrompt | Sequence[str | ImageData] | None = None,
        *,
        in_args: InT | None = None,
        memory: LLMAgentMemory,
        ctx: RunContext[CtxT],
        call_id: str,
    ) -> OutT:
        system_message, input_message = self._memorize_inputs(
            memory=memory,
            chat_inputs=chat_inputs,
            in_args=in_args,
            ctx=ctx,
            call_id=call_id,
        )
        if system_message:
            self._print_messages([system_message], ctx=ctx, call_id=call_id)
        if input_message:
            self._print_messages([input_message], ctx=ctx, call_id=call_id)

        final_answer = await self._policy_executor.execute(
            memory, ctx=ctx, call_id=call_id
        )

        return self.parse_output(
            final_answer,
            memory=memory,
            in_args=in_args,
            ctx=ctx,
            call_id=call_id,
        )

    async def _process_stream(
        self,
        chat_inputs: LLMPrompt | Sequence[str | ImageData] | None = None,
        *,
        in_args: InT | None = None,
        memory: LLMAgentMemory,
        ctx: RunContext[CtxT],
        call_id: str,
    ) -> AsyncIterator[Event[Any]]:
        system_message, input_message = self._memorize_inputs(
            memory=memory,
            chat_inputs=chat_inputs,
            in_args=in_args,
            ctx=ctx,
            call_id=call_id,
        )
        if system_message:
            self._print_messages([system_message], ctx=ctx, call_id=call_id)
            yield SystemMessageEvent(
                data=system_message, proc_name=self.name, call_id=call_id
            )
        if input_message:
            self._print_messages([input_message], ctx=ctx, call_id=call_id)
            yield UserMessageEvent(
                data=input_message, proc_name=self.name, call_id=call_id
            )

        event: Event[Any] | None = None
        async for event in self._policy_executor.execute_stream(
            memory, ctx=ctx, call_id=call_id
        ):
            yield event

        final_answer = self._policy_executor.get_final_answer(memory)

        output = self.parse_output(
            final_answer or "",
            memory=memory,
            in_args=in_args,
            ctx=ctx,
            call_id=call_id,
        )
        yield ProcPayloadOutputEvent(data=output, proc_name=self.name, call_id=call_id)

    def _print_messages(
        self, messages: Sequence[Message], ctx: RunContext[CtxT], call_id: str
    ) -> None:
        if ctx.printer:
            ctx.printer.print_messages(messages, agent_name=self.name, call_id=call_id)

    # Methods that can be overridden in subclasses

    def prepare_memory_impl(
        self,
        memory: LLMAgentMemory,
        *,
        instructions: LLMPrompt | None = None,
        in_args: InT | None = None,
        ctx: RunContext[Any],
        call_id: str,
    ) -> None:
        raise NotImplementedError

    def parse_output_impl(
        self,
        final_answer: str,
        *,
        memory: LLMAgentMemory,
        in_args: InT | None = None,
        ctx: RunContext[CtxT],
        call_id: str,
    ) -> OutT:
        raise NotImplementedError

    def build_system_prompt_impl(
        self, *, ctx: RunContext[CtxT], call_id: str
    ) -> str | None:
        return self._prompt_builder.build_system_prompt_impl(ctx=ctx, call_id=call_id)

    def build_input_content_impl(
        self, in_args: InT, *, ctx: RunContext[CtxT], call_id: str
    ) -> Content:
        return self._prompt_builder.build_input_content_impl(
            in_args=in_args, ctx=ctx, call_id=call_id
        )

    def check_for_final_answer_impl(
        self,
        memory: LLMAgentMemory,
        *,
        ctx: RunContext[CtxT],
        call_id: str,
        **kwargs: Any,
    ) -> str | None:
        return self._policy_executor.check_for_final_answer_impl(
            memory, ctx=ctx, call_id=call_id, **kwargs
        )

    async def on_before_generate_impl(
        self,
        memory: LLMAgentMemory,
        *,
        ctx: RunContext[CtxT],
        call_id: str,
        num_turns: int,
        extra_llm_settings: dict[str, Any],
    ) -> None:
        return await self._policy_executor.on_before_generate_impl(
            memory,
            ctx=ctx,
            call_id=call_id,
            num_turns=num_turns,
            extra_llm_settings=extra_llm_settings,
        )

    async def on_after_generate_impl(
        self,
        gen_message: AssistantMessage,
        *,
        memory: LLMAgentMemory,
        ctx: RunContext[CtxT],
        call_id: str,
        num_turns: int,
    ) -> None:
        return await self._policy_executor.on_after_generate_impl(
            gen_message=gen_message,
            memory=memory,
            ctx=ctx,
            call_id=call_id,
            num_turns=num_turns,
        )

    def tool_outputs_to_messages_impl(
        self,
        tool_outputs: Sequence[Any],
        tool_calls: Sequence[ToolCall],
        *,
        ctx: RunContext[CtxT],
        call_id: str,
    ):
        return self._policy_executor.tool_outputs_to_messages_impl(
            tool_outputs=tool_outputs,
            tool_calls=tool_calls,
            ctx=ctx,
            call_id=call_id,
        )

    # Decorators as an alternative to overriding methods

    def add_output_parser(
        self, func: OutputParser[InT, OutT, CtxT]
    ) -> OutputParser[InT, OutT, CtxT]:
        if self._used_default_llm_response_schema:
            self._policy_executor.response_schema = None
        self.parse_output_impl = func
        return func

    def add_memory_preparator(self, func: MemoryPreparator) -> MemoryPreparator:
        self.memory_preparator_impl = func
        return func

    def add_system_prompt_builder(
        self, func: SystemPromptBuilder[CtxT]
    ) -> SystemPromptBuilder[CtxT]:
        self._prompt_builder.build_system_prompt_impl = func
        return func

    def add_input_content_builder(
        self, func: InputContentBuilder[InT, CtxT]
    ) -> InputContentBuilder[InT, CtxT]:
        self._prompt_builder.build_input_content_impl = func
        return func

    def add_final_answer_checker(
        self, func: FinalAnswerChecker[CtxT]
    ) -> FinalAnswerChecker[CtxT]:
        self._policy_executor.check_for_final_answer_impl = func
        return func

    def add_before_generate_hook(
        self, func: BeforeGenerateHook[CtxT]
    ) -> BeforeGenerateHook[CtxT]:
        self._policy_executor.on_before_generate_impl = func
        return func

    def add_after_generate_hook(
        self, func: AfterGenerateHook[CtxT]
    ) -> AfterGenerateHook[CtxT]:
        self._policy_executor.on_after_generate_impl = func
        return func

    def add_tool_output_converter(
        self, func: ToolOutputConverter[CtxT]
    ) -> ToolOutputConverter[CtxT]:
        self._policy_executor.tool_outputs_to_messages_impl = func
        return func

    # When methods are overridden in subclasses, pass them to the components

    def _register_overridden_implementations(self) -> None:
        base_cls = LLMAgent[Any, Any, Any]

        # Prompt builder

        if is_method_overridden("build_system_prompt_impl", self, base_cls):
            self._prompt_builder.build_system_prompt_impl = (
                self.build_system_prompt_impl
            )

        if is_method_overridden("build_input_content_impl", self, base_cls):
            self._prompt_builder.build_input_content_impl = (
                self.build_input_content_impl
            )

        # Policy executor

        if is_method_overridden("check_for_final_answer_impl", self, base_cls):
            self._policy_executor.check_for_final_answer_impl = (
                self.check_for_final_answer_impl
            )

        if is_method_overridden("on_before_generate_impl", self, base_cls):
            self._policy_executor.on_before_generate_impl = self.on_before_generate_impl

        if is_method_overridden("on_after_generate_impl", self, base_cls):
            self._policy_executor.on_after_generate_impl = self.on_after_generate_impl

        if is_method_overridden("tool_outputs_to_messages_impl", self, base_cls):
            self._policy_executor.tool_outputs_to_messages_impl = (
                self.tool_outputs_to_messages_impl
            )
