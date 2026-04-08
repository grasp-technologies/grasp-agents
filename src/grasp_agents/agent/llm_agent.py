import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Generic, cast, final

from pydantic import BaseModel
from typing_extensions import TypedDict

from ..durability import AgentCheckpoint
from ..durability.resume import prepare_messages_for_resume
from ..llm.llm import LLM
from ..processors.processor import Processor
from ..run_context import CtxT, RunContext
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
from .agent_loop import AgentLoop, ResponseCapture
from .llm_agent_memory import LLMAgentMemory
from .prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class CallArgs(TypedDict):
    ctx: RunContext[Any]
    exec_id: str


class LLMAgent(Processor[InT, OutT, CtxT], Generic[InT, OutT, CtxT]):
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
        )

        if tracing_exclude_input_fields:
            for tool in tools or []:
                tool.tracing_exclude_input_fields = tracing_exclude_input_fields

        # Session persistence

        self._session_id = session_id
        self._session_metadata = session_metadata or {}
        self._session_checkpoint: AgentCheckpoint | None = None
        self._session_turn_number: int = 0
        self._session_loaded: bool = False

        # Memory

        # Don't narrow the base '_memory' type (Memory in BaseProcessor)
        self._memory = memory or LLMAgentMemory()
        self._reset_memory_on_run = reset_memory_on_run

        # Wire parent context for AgentTool instances (after memory init)
        from .agent_tool import AgentTool  # local: avoid circular

        for tool in tools or []:
            if isinstance(tool, AgentTool):
                tool.set_parent_memory(self._memory)  # type: ignore[arg-type]
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

        # Wire persistence callbacks if session_id provided
        if self._session_id:
            self._loop.checkpoint_callback = self._save_checkpoint
            self._loop.bg_tasks.session_id = self._session_id

        self._register_overridden_implementations()

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def _session_store_key(self) -> str | None:
        return f"session/{self._session_id}" if self._session_id else None

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

    @property
    def resumable(self) -> bool:
        return True

    def reset_session(self, session_id: str) -> None:
        """
        Dynamically set up session persistence for this agent.

        Store is read from ``ctx.store`` at runtime.
        """
        self._session_id = session_id
        self._session_loaded = False
        self._session_checkpoint = None
        self._session_turn_number = 0
        self._loop.checkpoint_callback = self._save_checkpoint
        self._loop.bg_tasks.session_id = session_id

    async def _maybe_load_session(
        self,
        *,
        ctx: RunContext[CtxT],
        exec_id: str | None = None,
    ) -> None:
        """Load session checkpoint from store on first run (if available)."""
        store = ctx.store
        if store is None or self._session_id is None:
            return
        if not self.memory.is_empty:
            return  # Already has messages — don't reload

        assert self._session_store_key is not None
        data = await store.load(self._session_store_key)
        if data is None:
            return  # Fresh session

        checkpoint = AgentCheckpoint.model_validate_json(data)
        resume_state = prepare_messages_for_resume(checkpoint.messages)
        self.memory.messages = resume_state.messages
        self._session_checkpoint = checkpoint
        self._session_turn_number = checkpoint.turn_number
        self._session_loaded = True

        logger.info(
            "Loaded session %s for agent %s "
            "(turns=%d, messages=%d, interruption=%s, stripped=%d)",
            self._session_id,
            self.name,
            self._session_turn_number,
            len(resume_state.messages),
            resume_state.interruption.value,
            resume_state.removed_count,
        )

        await self._loop.bg_tasks.handle_pending(ctx=ctx, exec_id=exec_id)

    async def _save_checkpoint(self, ctx: RunContext[CtxT]) -> None:
        """Persist current conversation state to the store."""
        store = ctx.store
        if store is None or self._session_id is None:
            return

        now = datetime.now(UTC)
        self._session_turn_number += 1

        checkpoint = AgentCheckpoint(
            session_id=self._session_id,
            processor_name=self.name,
            messages=list(self.memory.messages),
            turn_number=self._session_turn_number,
            created_at=(
                self._session_checkpoint.created_at if self._session_checkpoint else now
            ),
            updated_at=now,
            metadata=self._session_metadata,
        )
        self._session_checkpoint = checkpoint

        assert self._session_store_key is not None
        await store.save(
            self._session_store_key,
            checkpoint.model_dump_json().encode("utf-8"),
        )

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
        # If store is set, don't reset — the store manages memory lifetime.
        # If a session was just loaded, also skip reset (one-time flag).
        has_store = ctx.store is not None and self._session_id is not None
        fresh_init = (
            (self._reset_memory_on_run and not has_store) or self.memory.is_empty
        ) and not self._session_loaded
        self._session_loaded = False  # Consumed

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
        return super().validate_inputs(
            exec_id=exec_id,
            chat_inputs=chat_inputs,
            in_packet=in_packet,
            in_args=in_args,
        )

    def _extract_input_args(
        self, in_args: list[InT] | None, exec_id: str
    ) -> InT | None:
        if in_args and len(in_args) != 1:
            raise ProcInputValidationError(
                proc_name=self.name,
                exec_id=exec_id,
                message="LLMAgent expects a single input argument.",
            )

        return in_args[0] if in_args else None

    async def _process_stream(
        self,
        chat_inputs: LLMPrompt | Sequence[str | InputImage] | None = None,
        *,
        in_args: list[InT] | None = None,
        ctx: RunContext[CtxT],
        exec_id: str,
    ) -> AsyncIterator[Event[Any]]:
        call_kwargs = CallArgs(ctx=ctx, exec_id=exec_id)

        inp = self._extract_input_args(in_args, exec_id)

        # Auto-load session from store on first run
        await self._maybe_load_session(ctx=ctx, exec_id=exec_id)

        # Resume detection: no new inputs + session was loaded → skip memorization
        is_resume = inp is None and chat_inputs is None and self._session_loaded
        if is_resume:
            self._session_loaded = False  # Consumed

        if not is_resume:
            messages_to_expose = self._memorize_inputs(
                chat_inputs=chat_inputs, in_args=inp, **call_kwargs
            )
            self._print_messages(messages_to_expose, **call_kwargs)
            for message in messages_to_expose:
                if message.role == "system":
                    yield SystemMessageEvent(
                        data=message,
                        source=self.name,
                        exec_id=exec_id,
                    )
                elif message.role == "user":
                    yield UserMessageEvent(
                        data=message,
                        source=self.name,
                        exec_id=exec_id,
                    )

            # Checkpoint after user message — survives crash before LLM responds
            await self._loop.checkpoint(ctx)

        stream = ResponseCapture(self._loop.execute_stream(**call_kwargs))
        async for event in stream:
            yield event

        assert self._loop.final_answer is not None
        output = self.parse_output(
            self._loop.final_answer,
            in_args=inp,
            **call_kwargs,
        )
        yield ProcPayloadOutEvent(data=output, source=self.name, exec_id=exec_id)

    async def _process(
        self,
        chat_inputs: LLMPrompt | Sequence[str | InputImage] | None = None,
        *,
        in_args: list[InT] | None = None,
        ctx: RunContext[CtxT],
        exec_id: str,
    ) -> list[OutT]:
        async for event in self._process_stream(
            chat_inputs=chat_inputs,
            in_args=in_args,
            ctx=ctx,
            exec_id=exec_id,
        ):
            if isinstance(event, ProcPayloadOutEvent):
                return [event.data]
        return []

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
