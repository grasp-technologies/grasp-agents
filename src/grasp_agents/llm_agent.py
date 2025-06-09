from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Generic, Protocol, cast, final

from pydantic import BaseModel

from .comm_processor import CommProcessor
from .llm import LLM, LLMSettings
from .llm_agent_memory import LLMAgentMemory, SetMemoryHandler
from .packet_pool import PacketPool
from .prompt_builder import (
    FormatInputArgsHandler,
    FormatSystemArgsHandler,
    PromptBuilder,
)
from .run_context import CtxT, RunContext
from .tool_orchestrator import (
    ExitToolCallLoopHandler,
    ManageMemoryHandler,
    ToolOrchestrator,
)
from .typing.content import ImageData
from .typing.converters import Converters
from .typing.io import (
    InT_contra,
    LLMFormattedArgs,
    LLMFormattedSystemArgs,
    LLMPrompt,
    LLMPromptArgs,
    OutT_co,
    ProcessorName,
)
from .typing.message import Message, Messages, SystemMessage
from .typing.tool import BaseTool
from .utils import get_prompt, validate_obj_from_json_or_py_string


class ParseOutputHandler(Protocol[InT_contra, OutT_co, CtxT]):
    def __call__(
        self,
        conversation: Messages,
        *,
        in_args: InT_contra | None,
        batch_idx: int,
        ctx: RunContext[CtxT] | None,
    ) -> OutT_co: ...


class LLMAgent(
    CommProcessor[InT_contra, OutT_co, LLMAgentMemory, CtxT],
    Generic[InT_contra, OutT_co, CtxT],
):
    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_in_type",
        1: "_out_type",
    }

    def __init__(
        self,
        name: ProcessorName,
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
        sys_args_schema: type[LLMPromptArgs] = LLMPromptArgs,
        # User args (static args provided via RunContext)
        usr_args_schema: type[LLMPromptArgs] = LLMPromptArgs,
        # Agent loop settings
        max_turns: int = 1000,
        react_mode: bool = False,
        # Agent memory management
        reset_memory_on_run: bool = False,
        # Multi-agent routing
        packet_pool: PacketPool[CtxT] | None = None,
        recipients: list[ProcessorName] | None = None,
    ) -> None:
        super().__init__(name=name, packet_pool=packet_pool, recipients=recipients)

        # Agent memory

        self._memory: LLMAgentMemory = LLMAgentMemory()
        self._reset_memory_on_run = reset_memory_on_run
        self._set_memory_impl: SetMemoryHandler | None = None

        # Tool orchestrator

        self._using_default_llm_response_format: bool = False
        if llm.response_format is None and tools is None:
            llm.response_format = self.out_type
            self._using_default_llm_response_format = True

        self._tool_orchestrator: ToolOrchestrator[CtxT] = ToolOrchestrator[CtxT](
            agent_name=self.name,
            llm=llm,
            tools=tools,
            max_turns=max_turns,
            react_mode=react_mode,
        )

        # Prompt builder

        sys_prompt = get_prompt(prompt_text=sys_prompt, prompt_path=sys_prompt_path)
        in_prompt = get_prompt(prompt_text=in_prompt, prompt_path=in_prompt_path)
        self._prompt_builder: PromptBuilder[InT_contra, CtxT] = PromptBuilder[
            self.in_type, CtxT
        ](
            agent_name=self._name,
            sys_prompt=sys_prompt,
            in_prompt=in_prompt,
            sys_args_schema=sys_args_schema,
            usr_args_schema=usr_args_schema,
        )

        self.no_tqdm = getattr(llm, "no_tqdm", False)

        self._register_overridden_handlers()

    @property
    def llm(self) -> LLM[LLMSettings, Converters]:
        return self._tool_orchestrator.llm

    @property
    def tools(self) -> dict[str, BaseTool[BaseModel, Any, CtxT]]:
        return self._tool_orchestrator.tools

    @property
    def max_turns(self) -> int:
        return self._tool_orchestrator.max_turns

    @property
    def sys_args_schema(self) -> type[LLMPromptArgs]:
        return self._prompt_builder.sys_args_schema

    @property
    def usr_args_schema(self) -> type[LLMPromptArgs]:
        return self._prompt_builder.usr_args_schema

    @property
    def sys_prompt(self) -> LLMPrompt | None:
        return self._prompt_builder.sys_prompt

    @property
    def in_prompt(self) -> LLMPrompt | None:
        return self._prompt_builder.in_prompt

    def _parse_output(
        self,
        conversation: Messages,
        *,
        in_args: InT_contra | None = None,
        batch_idx: int = 0,
        ctx: RunContext[CtxT] | None = None,
    ) -> OutT_co:
        if self._parse_output_impl:
            if self._using_default_llm_response_format:
                # When using custom output parsing, the required LLM response format
                # can differ from the final agent output type ->
                # set it back to None unless it was specified explicitly at init.
                self._tool_orchestrator.llm.response_format = None
                # self._using_default_llm_response_format = False

            return self._parse_output_impl(
                conversation=conversation, in_args=in_args, batch_idx=batch_idx, ctx=ctx
            )

        return validate_obj_from_json_or_py_string(
            str(conversation[-1].content or ""),
            adapter=self._out_type_adapter,
            from_substring=True,
        )

    @final
    async def _process(
        self,
        chat_inputs: LLMPrompt | Sequence[str | ImageData] | None = None,
        *,
        in_args: InT_contra | Sequence[InT_contra] | None = None,
        entry_point: bool = False,
        forgetful: bool = False,
        ctx: RunContext[CtxT] | None = None,
    ) -> Sequence[OutT_co]:
        # Get run arguments
        sys_args: LLMPromptArgs = LLMPromptArgs()
        usr_args: LLMPromptArgs | Sequence[LLMPromptArgs] = LLMPromptArgs()
        if ctx is not None:
            run_args = ctx.run_args.get(self.name)
            if run_args is not None:
                sys_args = run_args.sys
                usr_args = run_args.usr

        # 1. Make system prompt (can be None)

        formatted_sys_prompt = self._prompt_builder.make_sys_prompt(
            sys_args=sys_args, ctx=ctx
        )

        # 2. Set agent state

        _memory = self.memory.model_copy(deep=True)
        prev_message_hist_length = len(_memory.message_history)

        if self._reset_memory_on_run or _memory.is_empty:
            _memory.reset(formatted_sys_prompt)
        elif self._set_memory_impl:
            _memory = self._set_memory_impl(
                prev_memory=_memory,
                in_args=in_args,
                sys_prompt=formatted_sys_prompt,
                ctx=ctx,
            )

        self._print_sys_msg(
            memory=_memory, prev_message_hist_length=prev_message_hist_length, ctx=ctx
        )

        # 3. Make and add user messages (can be empty)

        user_message_batch = self._prompt_builder.make_user_messages(
            chat_inputs=chat_inputs,
            in_args=in_args,
            usr_args=usr_args,
            entry_point=entry_point,
            ctx=ctx,
        )
        if user_message_batch:
            _memory.update(message_batch=user_message_batch)
            self._print_msgs(user_message_batch, ctx=ctx)

        if not self.tools:
            # 4. Generate messages without tools
            await self._tool_orchestrator.generate_once(_memory, ctx=ctx)
        else:
            # 4. Run tool call loop (new messages are added to the message
            #    history inside the loop)
            await self._tool_orchestrator.run_loop(_memory, ctx=ctx)

        if not forgetful:
            self._memory = _memory

        # 5. Parse outputs

        outputs: list[OutT_co] = []
        for i, _conv in enumerate(_memory.message_history.conversations):
            if isinstance(in_args, Sequence):
                _in_args_list = cast("Sequence[InT_contra]", in_args)
                _in_args = _in_args_list[min(i, len(_in_args_list) - 1)]
            else:
                _in_args = cast("InT_contra | None", in_args)

            outputs.append(
                self._parse_output(
                    conversation=_conv, in_args=_in_args, batch_idx=i, ctx=ctx
                )
            )

        return outputs

    def _print_msgs(
        self, messages: Sequence[Message], ctx: RunContext[CtxT] | None = None
    ) -> None:
        if ctx:
            ctx.printer.print_llm_messages(messages, agent_name=self.name)

    def _print_sys_msg(
        self,
        memory: LLMAgentMemory,
        prev_message_hist_length: int,
        ctx: RunContext[CtxT] | None = None,
    ) -> None:
        added_sys_message = (
            len(memory.message_history) == 1
            and prev_message_hist_length == 0
            and isinstance(memory.message_history[0][0], SystemMessage)
        )
        if added_sys_message:
            self._print_msgs([memory.message_history[0][0]], ctx=ctx)

    # -- Decorators for custom implementations --

    def format_sys_args(
        self, func: FormatSystemArgsHandler[CtxT]
    ) -> FormatSystemArgsHandler[CtxT]:
        self._prompt_builder.format_sys_args_impl = func

        return func

    def format_in_args(
        self, func: FormatInputArgsHandler[InT_contra, CtxT]
    ) -> FormatInputArgsHandler[InT_contra, CtxT]:
        self._prompt_builder.format_in_args_impl = func

        return func

    def parse_output(
        self, func: ParseOutputHandler[InT_contra, OutT_co, CtxT]
    ) -> ParseOutputHandler[InT_contra, OutT_co, CtxT]:
        self._parse_output_impl = func

        return func

    def set_memory(self, func: SetMemoryHandler) -> SetMemoryHandler:
        self._set_memory_impl = func

        return func

    def manage_memory(
        self, func: ManageMemoryHandler[CtxT]
    ) -> ManageMemoryHandler[CtxT]:
        self._tool_orchestrator.manage_memory_impl = func

        return func

    def exit_tool_call_loop(
        self, func: ExitToolCallLoopHandler[CtxT]
    ) -> ExitToolCallLoopHandler[CtxT]:
        self._tool_orchestrator.exit_tool_call_loop_impl = func

        return func

    # -- Override these methods in subclasses if needed --

    def _register_overridden_handlers(self) -> None:
        cur_cls = type(self)
        base_cls = LLMAgent[Any, Any, Any]

        if cur_cls._format_sys_args_fn is not base_cls._format_sys_args_fn:  # noqa: SLF001
            self._prompt_builder.format_sys_args_impl = self._format_sys_args_fn

        if cur_cls._format_in_args_fn is not base_cls._format_in_args_fn:  # noqa: SLF001
            self._prompt_builder.format_in_args_impl = self._format_in_args_fn

        if cur_cls._set_memory_fn is not base_cls._set_memory_fn:  # noqa: SLF001
            self._set_memory_impl = self._set_memory_fn

        if cur_cls._manage_memory_fn is not base_cls._manage_memory_fn:  # noqa: SLF001
            self._tool_orchestrator.manage_memory_impl = self._manage_memory_fn

        if (
            cur_cls._exit_tool_call_loop_fn is not base_cls._exit_tool_call_loop_fn  # noqa: SLF001
        ):
            self._tool_orchestrator.exit_tool_call_loop_impl = (
                self._exit_tool_call_loop_fn
            )

        self._parse_output_impl: (
            ParseOutputHandler[InT_contra, OutT_co, CtxT] | None
        ) = None

    def _format_sys_args_fn(
        self, sys_args: LLMPromptArgs, *, ctx: RunContext[CtxT] | None = None
    ) -> LLMFormattedSystemArgs:
        raise NotImplementedError(
            "LLMAgent._format_sys_args must be overridden by a subclass "
            "if it's intended to be used as the system arguments formatter."
        )

    def _format_in_args_fn(
        self,
        *,
        in_args: InT_contra,
        usr_args: LLMPromptArgs,
        batch_idx: int = 0,
        ctx: RunContext[CtxT] | None = None,
    ) -> LLMFormattedArgs:
        raise NotImplementedError(
            "LLMAgent._format_in_args must be overridden by a subclass"
        )

    def _set_memory_fn(
        self,
        prev_memory: LLMAgentMemory,
        in_args: InT_contra | Sequence[InT_contra] | None = None,
        sys_prompt: LLMPrompt | None = None,
        ctx: RunContext[Any] | None = None,
    ) -> LLMAgentMemory:
        raise NotImplementedError(
            "LLMAgent._set_memory must be overridden by a subclass"
        )

    def _exit_tool_call_loop_fn(
        self,
        conversation: Messages,
        *,
        ctx: RunContext[CtxT] | None = None,
        **kwargs: Any,
    ) -> bool:
        raise NotImplementedError(
            "LLMAgent._exit_tool_call_loop must be overridden by a subclass"
        )

    def _manage_memory_fn(
        self,
        memory: LLMAgentMemory,
        *,
        ctx: RunContext[CtxT] | None = None,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError(
            "LLMAgent._manage_memory must be overridden by a subclass"
        )
