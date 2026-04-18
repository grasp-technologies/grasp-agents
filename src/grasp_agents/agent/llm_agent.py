import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar, Generic, cast, final

from pydantic import BaseModel
from typing_extensions import TypedDict

from ..durability import AgentCheckpoint
from ..durability.checkpoints import AgentCheckpointLocation
from ..durability.resume import prepare_messages_for_resume
from ..llm.llm import LLM
from ..processors.processor import Processor
from ..run_context import CtxT, RunContext
from ..telemetry import SpanKind
from ..types.content import Content, InputImage
from ..types.errors import ProcInputValidationError
from ..types.events import (
    Event,
    ProcPayloadOutEvent,
    SystemMessageEvent,
    UserMessageEvent,
)
from ..types.hooks import (
    AfterLlmHook,
    AfterToolHook,
    BeforeLlmHook,
    BeforeToolHook,
    FinalAnswerExtractor,
    InputContentBuilder,
    MemoryBuilder,
    OutputParser,
    SystemPromptBuilder,
    ToolInputConverter,
    ToolOutputConverter,
)
from ..types.io import InT, LLMPrompt, OutT, ProcName
from ..types.items import FunctionToolCallItem, InputMessageItem
from ..types.response import Response
from ..types.tool import BaseTool
from ..utils.callbacks import is_method_overridden
from ..utils.io import get_prompt
from ..utils.validation import validate_obj_from_json_or_py_string
from .agent_loop import AgentLoop
from .llm_agent_memory import LLMAgentMemory
from .prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class CallArgs(TypedDict):
    ctx: RunContext[Any]
    exec_id: str


class LLMAgent(Processor[InT, OutT, CtxT], Generic[InT, OutT, CtxT]):
    _span_kind = SpanKind.AGENT

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
        force_react_mode: bool = False,
        final_answer_as_tool_call: bool = False,
        # Agent memory management
        memory: LLMAgentMemory | None = None,
        reset_memory_on_run: bool = False,
        # Agent run retries
        max_retries: int = 0,
        # Multi-agent routing
        recipients: Sequence[ProcName] | None = None,
        # Streaming
        stream_llm_responses: bool = False,
        stream_tools: bool = False,
        # Tracing
        tracing_enabled: bool = True,
        tracing_exclude_input_fields: set[str] | None = None,
        # Session persistence (opt-in)
        session_id: str | None = None,
        session_metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            memory=memory,
            recipients=recipients,
            max_retries=max_retries,
            tracing_enabled=tracing_enabled,
            tracing_exclude_input_fields=tracing_exclude_input_fields,
            session_id=session_id,
            session_metadata=session_metadata,
        )

        if tracing_exclude_input_fields:
            for tool in tools or []:
                tool.tracing_exclude_input_fields = tracing_exclude_input_fields

        # Memory

        # Don't narrow the base '_memory' type (Memory in Processor)
        self._memory = memory or LLMAgentMemory()
        self._reset_memory_on_run = reset_memory_on_run

        # Wire parent context for AgentTool instances (after memory init)
        from .agent_tool import AgentTool  # local: avoid circular

        for tool in tools or []:
            if isinstance(tool, AgentTool):
                tool.set_parent_memory(self._memory)
                if tool.inherit_tools:
                    tool.set_parent_tools(tools or [])

        # Prompt builder

        sys_prompt = get_prompt(prompt_text=sys_prompt, prompt_path=sys_prompt_path)
        in_prompt = get_prompt(prompt_text=in_prompt, prompt_path=in_prompt_path)

        self._prompt_builder = PromptBuilder[self.in_type, CtxT](
            agent_name=self._name, sys_prompt=sys_prompt, in_prompt=in_prompt
        )

        # Agent loop

        if issubclass(self._out_type, BaseModel):
            final_answer_type = self._out_type
        elif not final_answer_as_tool_call:
            final_answer_type = BaseModel
        else:
            raise TypeError(
                "Final answer type must be a subclass of BaseModel if "
                "final_answer_as_tool_call is True."
            )

        self._used_default_llm_response_schema = False
        if (
            response_schema is None
            and tools is None
            and not is_method_overridden(
                "parse_output_impl", self, LLMAgent[Any, Any, Any]
            )
        ):
            response_schema = self.out_type
            self._used_default_llm_response_schema = True

        self._loop: AgentLoop[CtxT] = AgentLoop[CtxT](
            agent_name=self.name,
            llm=llm,
            tools=tools,
            memory=self.memory,
            response_schema=response_schema,
            response_schema_by_xml_tag=response_schema_by_xml_tag,
            max_turns=max_turns,
            force_react_mode=force_react_mode,
            final_answer_type=final_answer_type,
            final_answer_as_tool_call=final_answer_as_tool_call,
            stream_llm_responses=stream_llm_responses,
            stream_tools=stream_tools,
            tracing_exclude_input_fields=tracing_exclude_input_fields,
        )

        # Session persistence
        self._step: int = 0  # completed invocation counter (observability)
        self._delivery_step: int | None = None  # caller's step for checkpoint

        if self._session_id:
            self.setup_session(self._session_id)

        # Subclass hook points (set by decorators or subclass overrides)

        self._register_overridden_implementations()

    @property
    def _checkpoint_store_key(self) -> str | None:
        return f"agent/{self._session_id}" if self._session_id else None

    @property
    def llm(self) -> LLM:
        return self._loop.llm

    @property
    def tools(self) -> dict[str, BaseTool[BaseModel, Any, CtxT]]:
        return self._loop.tools

    @property
    def max_turns(self) -> int:
        return self._loop.max_turns

    @property
    def step(self) -> int:
        return self._step

    @property
    def turn(self) -> int:
        return self._loop.turn

    @property
    def sys_prompt(self) -> LLMPrompt | None:
        return self._prompt_builder.sys_prompt

    @property
    def in_prompt(self) -> LLMPrompt | None:
        return self._prompt_builder.in_prompt

    @property
    def memory(self) -> LLMAgentMemory:
        return cast("LLMAgentMemory", self._memory)

    @property
    def reset_memory_on_run(self) -> bool:
        return self._reset_memory_on_run

    @property
    def _has_build_memory_impl(self) -> bool:
        return is_method_overridden("build_memory_impl", self, LLMAgent[Any, Any, Any])

    # --- Session persistence ---

    def setup_session(self, session_id: str) -> None:
        """
        Dynamically set up session persistence for this agent.

        Store is read from ``ctx.store`` at runtime.
        """
        super().setup_session(session_id)
        self._loop.checkpoint_callback = self.save_checkpoint
        self._loop.bg_tasks.session_id = session_id

    async def load_checkpoint(
        self,
        ctx: RunContext[CtxT],
        *,
        exec_id: str | None = None,
    ) -> AgentCheckpoint | None:
        """Load session checkpoint from store on first run (if available)."""
        if not self.memory.is_empty:
            return None  # Already has messages — don't reload

        checkpoint = await self._deserialize_checkpoint(ctx, AgentCheckpoint)
        if checkpoint is None:
            return None

        resume_state = prepare_messages_for_resume(checkpoint.messages)
        self.memory.messages = resume_state.messages
        self._loop.turn = checkpoint.turn

        logger.info(
            "Loaded session %s for agent %s "
            "(checkpoints=%d, messages=%d, interruption=%s, "
            "stripped=%d, step=%d, turn=%d)",
            self._session_id,
            self.name,
            self._checkpoint_number,
            len(resume_state.messages),
            resume_state.interruption.value,
            resume_state.removed_count,
            checkpoint.step,
            checkpoint.turn,
        )

        await self._loop.bg_tasks.handle_pending(ctx=ctx, exec_id=exec_id)

        return checkpoint

    async def save_checkpoint(
        self,
        ctx: RunContext[CtxT],
        *,
        turn: int = 0,
        output: str | None = None,
        location: AgentCheckpointLocation = AgentCheckpointLocation.AFTER_INPUT,
    ) -> None:
        """Persist current conversation state to the store."""
        checkpoint = AgentCheckpoint(
            session_id=self._session_id or "",
            processor_name=self.name,
            messages=list(self.memory.messages),
            session_metadata=self._session_metadata,
            step=self._delivery_step,
            turn=turn,
            output=output,
            location=location,
        )
        await self._serialize_checkpoint(ctx, checkpoint)

    def _memorize_inputs(
        self,
        *,
        chat_inputs: LLMPrompt | Sequence[str | InputImage] | None = None,
        in_args: InT | None = None,
        ctx: RunContext[CtxT],
        exec_id: str,
    ) -> list[InputMessageItem]:
        call_kwargs = CallArgs(ctx=ctx, exec_id=exec_id)

        formatted_sys_prompt = self._prompt_builder.build_system_prompt(
            ctx=ctx, exec_id=exec_id
        )
        fresh_init = self._reset_memory_on_run or self.memory.is_empty

        if fresh_init and not self._has_build_memory_impl:
            self.memory.reset(formatted_sys_prompt)
        elif self._has_build_memory_impl:
            self.build_memory_impl(
                instructions=formatted_sys_prompt, in_args=in_args, **call_kwargs
            )

        messages_to_expose: list[InputMessageItem] = []
        if fresh_init:
            for msg in self.memory.messages:
                if isinstance(msg, InputMessageItem):
                    messages_to_expose.append(msg)

        input_message = self._prompt_builder.build_input_message(
            chat_inputs=chat_inputs, in_args=in_args, **call_kwargs
        )
        if input_message:
            self.memory.update([input_message])
            messages_to_expose.append(input_message)

        return messages_to_expose

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
        in_args: InT | None = None,
        ctx: RunContext[CtxT],
        exec_id: str,
    ) -> OutT:
        if is_method_overridden("parse_output_impl", self, LLMAgent[Any, Any, Any]):
            return self.parse_output_impl(
                final_answer,
                in_args=in_args,
                ctx=ctx,
                exec_id=exec_id,
            )

        return self.parse_output_default(final_answer)

    def validate_inputs(
        self,
        exec_id: str,
        chat_inputs: Any = None,
        in_packet: Any = None,
        in_args: Any = None,
    ) -> list[InT] | None:
        # Allow no inputs when a session is configured (resume case)
        has_input = any(x is not None for x in [chat_inputs, in_args, in_packet])
        if not has_input and self._session_id is not None:
            return None

        result = super().validate_inputs(
            exec_id=exec_id,
            chat_inputs=chat_inputs,
            in_packet=in_packet,
            in_args=in_args,
        )
        if result is not None and len(result) != 1:
            raise ProcInputValidationError(
                proc_name=self.name,
                exec_id=exec_id,
                message="LLMAgent expects a single input argument.",
            )
        return result

    async def _process_stream(
        self,
        chat_inputs: LLMPrompt | Sequence[str | InputImage] | None = None,
        *,
        in_args: list[InT] | None = None,
        ctx: RunContext[CtxT],
        exec_id: str,
        step: int | None = None,
    ) -> AsyncIterator[Event[Any]]:
        call_kwargs = CallArgs(ctx=ctx, exec_id=exec_id)
        self._delivery_step = step

        inp = in_args[0] if in_args else None

        # Always load checkpoint (restores memory, background tasks, turn).
        checkpoint = await self.load_checkpoint(ctx, exec_id=exec_id)

        is_redelivery = (
            step is not None and checkpoint is not None and checkpoint.step == step
        )

        # Re-delivery with cached output: step already completed, caller
        # re-delivered (e.g. workflow crashed before saving its own checkpoint).
        if is_redelivery and checkpoint is not None and checkpoint.output is not None:
            output = self.parse_output(checkpoint.output, in_args=inp, **call_kwargs)
            yield ProcPayloadOutEvent(data=output, source=self.name, exec_id=exec_id)
            return

        # New step (or no checkpoint): memorize inputs and reset turn.
        # Interrupted re-delivery (same step, no output): memory already
        # loaded from checkpoint, skip memorization.
        if not is_redelivery:
            self._loop.turn = 0
            messages_to_expose = self._memorize_inputs(
                chat_inputs=chat_inputs, in_args=inp, **call_kwargs
            )
            self._print_messages(messages_to_expose, **call_kwargs)
            for message in messages_to_expose:
                if message.role == "system":
                    yield SystemMessageEvent(
                        data=message, source=self.name, exec_id=exec_id
                    )
                # TODO: set source
                elif message.role == "user":
                    yield UserMessageEvent(
                        data=message,
                        source=None,
                        destination=self.name,
                        exec_id=exec_id,
                    )

        async for event in self._loop.execute_stream(**call_kwargs):
            yield event

        assert self._loop.final_answer is not None
        output = self.parse_output(self._loop.final_answer, in_args=inp, **call_kwargs)
        yield ProcPayloadOutEvent(data=output, source=self.name, exec_id=exec_id)

        self._step += 1

    def _print_messages(
        self,
        messages: Sequence[Any],
        ctx: RunContext[CtxT],
        exec_id: str,
    ) -> None:
        if ctx.printer:
            ctx.printer.print_messages(messages, agent_name=self.name, exec_id=exec_id)

    # --- Subclass hook points ---
    #
    # Override these in subclasses for customization.
    # Alternatively, use the @agent.add_* decorators (preferred).

    def build_memory_impl(
        self,
        *,
        instructions: LLMPrompt | None = None,
        in_args: InT | None = None,
        ctx: RunContext[Any],
        exec_id: str,
    ) -> None:
        raise NotImplementedError

    def parse_output_impl(
        self,
        final_answer: str,
        *,
        in_args: InT | None = None,
        ctx: RunContext[CtxT],
        exec_id: str,
    ) -> OutT:
        raise NotImplementedError

    def build_system_prompt_impl(
        self, *, ctx: RunContext[CtxT], exec_id: str
    ) -> str | None:
        raise NotImplementedError

    def build_input_content_impl(
        self, in_args: InT, *, ctx: RunContext[CtxT], exec_id: str
    ) -> Content:
        raise NotImplementedError

    def extract_final_answer_impl(
        self,
        *,
        ctx: RunContext[CtxT],
        exec_id: str,
        **kwargs: Any,
    ) -> str | None:
        raise NotImplementedError

    async def on_before_llm_impl(
        self,
        *,
        ctx: RunContext[CtxT],
        exec_id: str,
        num_turns: int,
        extra_llm_settings: dict[str, Any],
    ) -> None:
        raise NotImplementedError

    async def on_after_llm_impl(
        self,
        response: Response,
        *,
        ctx: RunContext[CtxT],
        exec_id: str,
        num_turns: int,
    ) -> None:
        raise NotImplementedError

    async def on_before_tool_impl(
        self,
        *,
        tool_calls: Sequence[FunctionToolCallItem],
        ctx: RunContext[CtxT],
        exec_id: str,
    ) -> None:
        raise NotImplementedError

    async def on_after_tool_impl(
        self,
        *,
        tool_calls: Sequence[FunctionToolCallItem],
        tool_messages: Sequence[Any],
        ctx: RunContext[CtxT],
        exec_id: str,
    ) -> None:
        raise NotImplementedError

    # --- Decorator API ---
    #
    # Preferred over subclassing. Each decorator sets a callback slot
    # on the appropriate component (AgentLoop or PromptBuilder).

    def add_output_parser(
        self, func: OutputParser[InT, OutT, CtxT]
    ) -> OutputParser[InT, OutT, CtxT]:
        if self._used_default_llm_response_schema:
            self._loop.response_schema = None
        self.parse_output_impl = func
        return func

    def add_memory_builder(self, func: MemoryBuilder[InT]) -> MemoryBuilder[InT]:
        self.build_memory_impl = func
        return func

    def add_system_prompt_builder(
        self, func: SystemPromptBuilder[CtxT]
    ) -> SystemPromptBuilder[CtxT]:
        self._prompt_builder.system_prompt_builder = func
        return func

    def add_input_content_builder(
        self, func: InputContentBuilder[InT, CtxT]
    ) -> InputContentBuilder[InT, CtxT]:
        self._prompt_builder.input_content_builder = func
        return func

    def add_final_answer_extractor(
        self, func: FinalAnswerExtractor[CtxT]
    ) -> FinalAnswerExtractor[CtxT]:
        self._loop.final_answer_extractor = func
        return func

    def add_before_llm_hook(self, func: BeforeLlmHook[CtxT]) -> BeforeLlmHook[CtxT]:
        self._loop.before_llm_hook = func
        return func

    def add_after_llm_hook(self, func: AfterLlmHook[CtxT]) -> AfterLlmHook[CtxT]:
        self._loop.after_llm_hook = func
        return func

    def add_before_tool_hook(self, func: BeforeToolHook[CtxT]) -> BeforeToolHook[CtxT]:
        self._loop.before_tool_hook = func
        return func

    def add_after_tool_hook(self, func: AfterToolHook[CtxT]) -> AfterToolHook[CtxT]:
        self._loop.after_tool_hook = func
        return func

    def add_tool_input_converter(self, tool_name: str) -> Any:
        def decorator(func: ToolInputConverter[CtxT]) -> ToolInputConverter[CtxT]:
            self._loop.tool_input_converters[tool_name] = func
            return func

        return decorator

    def add_tool_output_converter(self, tool_name: str) -> Any:
        def decorator(func: ToolOutputConverter[CtxT]) -> ToolOutputConverter[CtxT]:
            self._loop.tool_output_converters[tool_name] = func
            return func

        return decorator

    # --- Override detection and registration ---

    def _register_overridden_implementations(self) -> None:
        """
        Detect subclass overrides and set them as callback slots
        on the appropriate components (AgentLoop, PromptBuilder).
        """
        base_cls = LLMAgent[Any, Any, Any]

        # Prompt builder
        if is_method_overridden("build_system_prompt_impl", self, base_cls):
            self._prompt_builder.system_prompt_builder = self.build_system_prompt_impl
        if is_method_overridden("build_input_content_impl", self, base_cls):
            self._prompt_builder.input_content_builder = self.build_input_content_impl

        # Agent loop
        if is_method_overridden("extract_final_answer_impl", self, base_cls):
            self._loop.final_answer_extractor = self.extract_final_answer_impl
        if is_method_overridden("on_before_llm_impl", self, base_cls):
            self._loop.before_llm_hook = self.on_before_llm_impl
        if is_method_overridden("on_after_llm_impl", self, base_cls):
            self._loop.after_llm_hook = self.on_after_llm_impl
        if is_method_overridden("on_before_tool_impl", self, base_cls):
            self._loop.before_tool_hook = self.on_before_tool_impl
        if is_method_overridden("on_after_tool_impl", self, base_cls):
            self._loop.after_tool_hook = self.on_after_tool_impl

    def copy(self) -> "LLMAgent[InT, OutT, CtxT]":
        # LLM sharing: handled by LLM.__deepcopy__ (returns self)
        # Tool sharing: handled by BaseTool.__deepcopy__ (_copy_shared_attrs)
        return deepcopy(self)
