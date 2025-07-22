from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any, ClassVar, Generic, Protocol, TypeVar, cast

from pydantic import BaseModel

from .llm import LLM, LLMSettings
from .llm_agent_memory import LLMAgentMemory, PrepareMemoryHandler
from .llm_policy_executor import (
    ExitToolCallLoopHandler,
    LLMPolicyExecutor,
    ManageMemoryHandler,
)
from .processor import Processor
from .prompt_builder import (
    MakeInputContentHandler,
    MakeSystemPromptHandler,
    PromptBuilder,
)
from .run_context import CtxT, RunContext
from .typing.content import Content, ImageData
from .typing.converters import Converters
from .typing.events import (
    Event,
    ProcPayloadOutputEvent,
    SystemMessageEvent,
    UserMessageEvent,
)
from .typing.io import InT, LLMPrompt, LLMPromptArgs, OutT, ProcName
from .typing.message import Message, Messages, SystemMessage, UserMessage
from .typing.tool import BaseTool
from .utils import get_prompt, validate_obj_from_json_or_py_string

_InT_contra = TypeVar("_InT_contra", contravariant=True)
_OutT_co = TypeVar("_OutT_co", covariant=True)


class ParseOutputHandler(Protocol[_InT_contra, _OutT_co, CtxT]):
    def __call__(
        self,
        conversation: Messages,
        *,
        in_args: _InT_contra | None,
        ctx: RunContext[CtxT] | None,
    ) -> _OutT_co: ...


class LLMAgent(
    Processor[InT, OutT, LLMAgentMemory, CtxT],
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
        llm: LLM[LLMSettings, Converters],
        # Tools
        tools: list[BaseTool[Any, Any, CtxT]] | None = None,
        # Input prompt template (combines user and received arguments)
        in_prompt: LLMPrompt | None = None,
        in_prompt_path: str | Path | None = None,
        # System prompt template
        sys_prompt: LLMPrompt | None = None,
        sys_prompt_path: str | Path | None = None,
        # System args (static args provided via RunContext)
        sys_args_schema: type[LLMPromptArgs] | None = None,
        # Agent loop settings
        max_turns: int = 100,
        react_mode: bool = False,
        final_answer_as_tool_call: bool = False,
        # Agent memory management
        reset_memory_on_run: bool = False,
        # Retries
        max_retries: int = 0,
        # Multi-agent routing
        recipients: list[ProcName] | None = None,
    ) -> None:
        super().__init__(name=name, recipients=recipients, max_retries=max_retries)

        # Agent memory

        self._memory: LLMAgentMemory = LLMAgentMemory()
        self._reset_memory_on_run = reset_memory_on_run

        # LLM policy executor

        self._used_default_llm_response_schema: bool = False
        if llm.response_schema is None and tools is None:
            llm.response_schema = self.out_type
            self._used_default_llm_response_schema = True

        if issubclass(self._out_type, BaseModel):
            final_answer_type = self._out_type
        elif not final_answer_as_tool_call:
            final_answer_type = BaseModel
        else:
            raise TypeError(
                "Final answer type must be a subclass of BaseModel if "
                "final_answer_as_tool_call is True."
            )

        self._policy_executor: LLMPolicyExecutor[CtxT] = LLMPolicyExecutor[CtxT](
            agent_name=self.name,
            llm=llm,
            tools=tools,
            max_turns=max_turns,
            react_mode=react_mode,
            final_answer_type=final_answer_type,
            final_answer_as_tool_call=final_answer_as_tool_call,
        )

        # Prompt builder

        sys_prompt = get_prompt(prompt_text=sys_prompt, prompt_path=sys_prompt_path)
        in_prompt = get_prompt(prompt_text=in_prompt, prompt_path=in_prompt_path)
        self._prompt_builder: PromptBuilder[InT, CtxT] = PromptBuilder[
            self.in_type, CtxT
        ](
            agent_name=self._name,
            sys_prompt=sys_prompt,
            in_prompt=in_prompt,
            sys_args_schema=sys_args_schema,
        )

        self._prepare_memory_impl: PrepareMemoryHandler | None = None
        self._parse_output_impl: ParseOutputHandler[InT, OutT, CtxT] | None = None
        self._register_overridden_handlers()

    @property
    def llm(self) -> LLM[LLMSettings, Converters]:
        return self._policy_executor.llm

    @property
    def tools(self) -> dict[str, BaseTool[BaseModel, Any, CtxT]]:
        return self._policy_executor.tools

    @property
    def max_turns(self) -> int:
        return self._policy_executor.max_turns

    @property
    def sys_args_schema(self) -> type[LLMPromptArgs] | None:
        return self._prompt_builder.sys_args_schema

    @property
    def sys_prompt(self) -> LLMPrompt | None:
        return self._prompt_builder.sys_prompt

    @property
    def in_prompt(self) -> LLMPrompt | None:
        return self._prompt_builder.in_prompt

    def _prepare_memory(
        self,
        memory: LLMAgentMemory,
        in_args: InT | None = None,
        sys_prompt: LLMPrompt | None = None,
        ctx: RunContext[Any] | None = None,
    ) -> None:
        if self._prepare_memory_impl:
            return self._prepare_memory_impl(
                memory=memory, in_args=in_args, sys_prompt=sys_prompt, ctx=ctx
            )

    def _memorize_inputs(
        self,
        memory: LLMAgentMemory,
        chat_inputs: LLMPrompt | Sequence[str | ImageData] | None = None,
        in_args: InT | None = None,
        ctx: RunContext[CtxT] | None = None,
    ) -> tuple[SystemMessage | None, UserMessage | None]:
        # 1. Get system arguments
        sys_args = ctx.sys_args.get(self.name) if ctx and ctx.sys_args else None

        # 2. Make system prompt (can be None)

        formatted_sys_prompt = self._prompt_builder.make_system_prompt(
            sys_args=sys_args, ctx=ctx
        )

        # 3. Set agent memory

        system_message: SystemMessage | None = None
        if self._reset_memory_on_run or memory.is_empty:
            memory.reset(formatted_sys_prompt)
            if formatted_sys_prompt is not None:
                system_message = cast("SystemMessage", memory.message_history[0])
        else:
            self._prepare_memory(
                memory=memory, in_args=in_args, sys_prompt=formatted_sys_prompt, ctx=ctx
            )

        # 3. Make and add input messages

        input_message = self._prompt_builder.make_input_message(
            chat_inputs=chat_inputs, in_args=in_args, ctx=ctx
        )
        if input_message:
            memory.update([input_message])

        return system_message, input_message

    def _parse_output(
        self,
        conversation: Messages,
        *,
        in_args: InT | None = None,
        ctx: RunContext[CtxT] | None = None,
    ) -> OutT:
        if self._parse_output_impl:
            return self._parse_output_impl(
                conversation=conversation, in_args=in_args, ctx=ctx
            )

        return validate_obj_from_json_or_py_string(
            str(conversation[-1].content or ""),
            schema=self._out_type,
            from_substring=False,
            strip_language_markdown=True,
        )

    async def _process(
        self,
        chat_inputs: LLMPrompt | Sequence[str | ImageData] | None = None,
        *,
        in_args: InT | None = None,
        memory: LLMAgentMemory,
        call_id: str,
        ctx: RunContext[CtxT] | None = None,
    ) -> OutT:
        system_message, input_message = self._memorize_inputs(
            memory=memory,
            chat_inputs=chat_inputs,
            in_args=in_args,
            ctx=ctx,
        )
        if system_message:
            self._print_messages([system_message], call_id=call_id, ctx=ctx)
        if input_message:
            self._print_messages([input_message], call_id=call_id, ctx=ctx)

        await self._policy_executor.execute(memory, call_id=call_id, ctx=ctx)

        return self._parse_output(
            conversation=memory.message_history, in_args=in_args, ctx=ctx
        )

    async def _process_stream(
        self,
        chat_inputs: LLMPrompt | Sequence[str | ImageData] | None = None,
        *,
        in_args: InT | None = None,
        memory: LLMAgentMemory,
        call_id: str,
        ctx: RunContext[CtxT] | None = None,
    ) -> AsyncIterator[Event[Any]]:
        system_message, input_message = self._memorize_inputs(
            memory=memory,
            chat_inputs=chat_inputs,
            in_args=in_args,
            ctx=ctx,
        )
        if system_message:
            self._print_messages([system_message], call_id=call_id, ctx=ctx)
            yield SystemMessageEvent(
                data=system_message, proc_name=self.name, call_id=call_id
            )
        if input_message:
            self._print_messages([input_message], call_id=call_id, ctx=ctx)
            yield UserMessageEvent(
                data=input_message, proc_name=self.name, call_id=call_id
            )

        async for event in self._policy_executor.execute_stream(
            memory, call_id=call_id, ctx=ctx
        ):
            yield event

        output = self._parse_output(
            conversation=memory.message_history, in_args=in_args, ctx=ctx
        )
        yield ProcPayloadOutputEvent(data=output, proc_name=self.name, call_id=call_id)

    def _print_messages(
        self,
        messages: Sequence[Message],
        call_id: str,
        ctx: RunContext[CtxT] | None = None,
    ) -> None:
        if ctx and ctx.printer:
            ctx.printer.print_messages(messages, agent_name=self.name, call_id=call_id)

    # -- Override these methods in subclasses if needed --

    def _register_overridden_handlers(self) -> None:
        cur_cls = type(self)
        base_cls = LLMAgent[Any, Any, Any]

        # Packet routing
        if cur_cls._select_recipients is not base_cls._select_recipients:  # noqa: SLF001
            self.select_recipients_impl = self._select_recipients

        # Prompt builder

        if cur_cls._make_system_prompt is not base_cls._make_system_prompt:  # noqa: SLF001
            self._prompt_builder.make_system_prompt_impl = self._make_system_prompt

        if cur_cls._make_input_content is not base_cls._make_input_content:  # noqa: SLF001
            self._prompt_builder.make_input_content_impl = self._make_input_content

        # Policy executor

        if (
            cur_cls._exit_tool_call_loop is not base_cls._exit_tool_call_loop  # noqa: SLF001
        ):
            self._policy_executor.exit_tool_call_loop_impl = self._exit_tool_call_loop

        if cur_cls._manage_memory is not base_cls._manage_memory:  # noqa: SLF001
            self._policy_executor.manage_memory_impl = self._manage_memory

        # Make sure default LLM response schema is not used when custom output
        # parsing is provided
        if (
            cur_cls._parse_output is not base_cls._parse_output  # noqa: SLF001
            and self._used_default_llm_response_schema
        ):
            self._policy_executor.llm.response_schema = None

    def _make_system_prompt(
        self, sys_args: LLMPromptArgs | None, *, ctx: RunContext[CtxT] | None = None
    ) -> str | None:
        return self._prompt_builder.make_system_prompt(sys_args=sys_args, ctx=ctx)

    def _make_input_content(
        self, in_args: InT | None = None, *, ctx: RunContext[CtxT] | None = None
    ) -> Content:
        return self._prompt_builder.make_input_content(in_args=in_args, ctx=ctx)

    def _exit_tool_call_loop(
        self,
        conversation: Messages,
        *,
        ctx: RunContext[CtxT] | None = None,
        **kwargs: Any,
    ) -> bool:
        return self._policy_executor._exit_tool_call_loop(  # type: ignore[return-value]
            conversation=conversation, ctx=ctx, **kwargs
        )

    def _manage_memory(
        self,
        memory: LLMAgentMemory,
        *,
        ctx: RunContext[CtxT] | None = None,
        **kwargs: Any,
    ) -> None:
        return self._policy_executor._manage_memory(  # type: ignore[return-value]
            memory=memory, ctx=ctx, **kwargs
        )

    # Decorators for custom implementations as an alternative to overriding methods

    def make_system_prompt(
        self, func: MakeSystemPromptHandler[CtxT]
    ) -> MakeSystemPromptHandler[CtxT]:
        self._prompt_builder.make_system_prompt_impl = func

        return func

    def make_input_content(
        self, func: MakeInputContentHandler[InT, CtxT]
    ) -> MakeInputContentHandler[InT, CtxT]:
        self._prompt_builder.make_input_content_impl = func

        return func

    def parse_output(
        self, func: ParseOutputHandler[InT, OutT, CtxT]
    ) -> ParseOutputHandler[InT, OutT, CtxT]:
        if self._used_default_llm_response_schema:
            self._policy_executor.llm.response_schema = None
        self._parse_output_impl = func

        return func

    def prepare_memory(self, func: PrepareMemoryHandler) -> PrepareMemoryHandler:
        self._prepare_memory_impl = func

        return func

    def manage_memory(
        self, func: ManageMemoryHandler[CtxT]
    ) -> ManageMemoryHandler[CtxT]:
        self._policy_executor.manage_memory_impl = func

        return func

    def exit_tool_call_loop(
        self, func: ExitToolCallLoopHandler[CtxT]
    ) -> ExitToolCallLoopHandler[CtxT]:
        self._policy_executor.exit_tool_call_loop_impl = func

        return func
