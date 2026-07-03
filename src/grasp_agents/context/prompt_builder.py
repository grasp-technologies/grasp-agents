from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Protocol,
    cast,
    final,
    runtime_checkable,
)

from pydantic import BaseModel, TypeAdapter

from grasp_agents.types.content import (
    CacheControl,
    Content,
    InputFile,
    InputImage,
    InputPart,
    InputRenderable,
    InputText,
)
from grasp_agents.types.errors import InputPromptBuilderError
from grasp_agents.utils.generics import AutoInstanceAttributesMixin

from .system_reminder import wrap_in_system_reminder

if TYPE_CHECKING:
    from collections.abc import Awaitable, Sequence

    from grasp_agents.agent.agent_context import AgentContext
    from grasp_agents.hooks import InitialContextBuilder, InputContentBuilder
    from grasp_agents.session_context import SessionContext
    from grasp_agents.types.io import LLMPrompt
    from grasp_agents.types.items import InputItem
from grasp_agents.types.items import InputMessageItem

type PromptArgumentType = str | bool | int


@runtime_checkable
class SectionCompute(Protocol):
    """
    Lazy compute for a :class:`SystemPromptSection`.

    Sections receive ``ctx`` and ``exec_id`` and return rendered text or
    ``None`` (omit). Sync or async. Implementations that don't need a
    particular kwarg can absorb the rest with ``**_: Any``::

        def compute(*, ctx=None, exec_id=None, **_: Any) -> str | None:
            ...

    Per-turn relevance signals (e.g. the running transcript) live on the
    :class:`InputAttachment` seam — system-prompt sections stay
    cache-stable across turns.
    """

    def __call__(
        self,
        *,
        ctx: SessionContext[Any] | None = None,
        exec_id: str | None = None,
        agent_ctx: AgentContext | None = None,
    ) -> str | Awaitable[str | None] | None: ...


@dataclass(frozen=True)
class SystemPromptSection:
    """
    A named, lazy block appended to the system prompt at run time.

    ``compute`` is invoked with ``(ctx, exec_id)`` per agent run, may be sync
    or async, and returns the rendered text or ``None`` (omit).
    ``cache_control``, when set, marks a prompt-cache checkpoint on this
    section's block — providers with prompt caching (e.g. Anthropic) cache
    the prefix up to and including it; the rest ignore it.
    """

    name: str
    compute: SectionCompute
    cache_control: CacheControl | None = None


@runtime_checkable
class InputAttachmentCompute(Protocol):
    """
    Lazy compute for a :class:`InputAttachment`.

    Receives the just-built user message, the run context, and the
    pre-existing transcript (``messages``). Returns either rendered text
    (wrapped in ``<system-reminder>`` by default), a sequence of
    :class:`InputPart` to append verbatim, or ``None`` to skip.

    This is the per-turn relevance seam — selectors that need to look at
    the transcript to pick what's relevant *now* belong here, not on the
    system prompt sections (which stay cache-stable across turns).
    """

    def __call__(
        self,
        *,
        input_message: InputMessageItem,
        ctx: SessionContext[Any] | None = None,
        exec_id: str | None = None,
        messages: Sequence[InputItem] | None = None,
        agent_ctx: AgentContext | None = None,
        source: str | None = None,
    ) -> (
        str | Sequence[InputPart] | Awaitable[str | Sequence[InputPart] | None] | None
    ): ...


@dataclass(frozen=True)
class InputAttachment:
    """
    A named, lazy block appended to the user message at run time.

    ``compute`` runs once per agent invocation (when the new user message
    is built) and may surface per-turn relevance signals — e.g. memory
    topics chosen by a relevance selector against the running transcript.
    Text returns are wrapped in ``<system-reminder>...</system-reminder>``
    when ``wrap_in_system_reminder`` is true (default), to tell the model
    "this is a system-injected note, not the user speaking".
    """

    name: str
    compute: InputAttachmentCompute
    wrap_in_system_reminder: bool = True


class PromptBuilder[InT, CtxT](AutoInstanceAttributesMixin):
    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {0: "_in_type"}

    def __init__(
        self, agent_name: str, sys_prompt: LLMPrompt | None, in_prompt: LLMPrompt | None
    ):
        self._in_type: type[InT]
        super().__init__()

        self._agent_name = agent_name
        self._sys_prompt = sys_prompt
        self._in_prompt = in_prompt
        self._in_args_type_adapter: TypeAdapter[InT] = TypeAdapter(self._in_type)

        # Hook callback slots — set by LLMAgent, None = use defaults
        self.input_content_builder: InputContentBuilder[InT] | None = None
        # Transforms the ephemeral initial context (system message + sections);
        # see ``build_initial_context``. None = use the default unchanged.
        self.initial_context_builder: InitialContextBuilder | None = None

        # Sections appended to the system prompt at run time. Each section's
        # compute receives (ctx, exec_id) and may be sync or async. Order is
        # preserved; ``None`` outputs are dropped. Skills (and later memory)
        # plug in here.
        self.system_prompt_sections: list[SystemPromptSection] = []

        # Attachments appended to the user message at run time. Each
        # attachment's compute receives the just-built input_message plus
        # ``ctx``, ``exec_id``, and the pre-existing transcript
        # (``messages``), and may return text (wrapped in
        # ``<system-reminder>``) or extra content parts. Order is preserved;
        # ``None`` outputs are dropped.
        self.input_attachments: list[InputAttachment] = []

    @property
    def sys_prompt(self) -> LLMPrompt | None:
        return self._sys_prompt

    @property
    def in_prompt(self) -> LLMPrompt | None:
        return self._in_prompt

    def add_system_prompt_section(self, section: SystemPromptSection) -> None:
        """
        Register ``section``. If a section with the same name is already
        registered, replace it in place — the later definition wins.

        This makes auto-attached sections (e.g. ``env_info``, ``skills``,
        ``memory``, ``mcp_instructions``) trivially overridable: pass a
        custom-configured section with the same name and it supersedes the
        default without leaving a duplicate behind.
        """
        for i, existing in enumerate(self.system_prompt_sections):
            if existing.name == section.name:
                self.system_prompt_sections[i] = section
                return
        self.system_prompt_sections.append(section)

    def add_input_attachment(self, attachment: InputAttachment) -> None:
        """
        Register ``attachment`` for the user message. If an attachment with
        the same name is already registered, replace it in place — the later
        definition wins.
        """
        for i, existing in enumerate(self.input_attachments):
            if existing.name == attachment.name:
                self.input_attachments[i] = attachment
                return
        self.input_attachments.append(attachment)

    @final
    async def build_system_prompt(
        self,
        *,
        ctx: SessionContext[CtxT],
        exec_id: str,
        agent_ctx: AgentContext | None = None,
    ) -> str | None:
        """
        Render the full system prompt as a single joined string.

        Use :meth:`build_system_prompt_parts` instead when you need
        per-section :class:`CacheControl` checkpoints preserved (Anthropic
        prompt caching, etc.) — that method returns the same content as a
        list of :class:`InputText` parts carrying their ``cache_control``.
        """
        parts = await self.build_system_prompt_parts(
            ctx=ctx, exec_id=exec_id, agent_ctx=agent_ctx
        )
        if parts is None:
            return None
        return "\n\n".join(p.text for p in parts)

    @final
    async def build_system_prompt_parts(
        self,
        *,
        ctx: SessionContext[CtxT],
        exec_id: str,
        agent_ctx: AgentContext | None = None,
    ) -> list[InputText] | None:
        """
        Render the full system prompt as a list of :class:`InputText`
        parts — one per section, preceded by the base prompt (when set).

        The static ``sys_prompt`` is the base. Each section's ``cache_control``
        rides on its part; providers with prompt caching honor it and the rest
        ignore it. For a dynamic prompt use a section (dynamic ``compute``) or an
        :class:`InitialContextBuilder` returning a system message.

        Returns ``None`` when there's no base prompt and every section
        renders empty.
        """
        base = self.sys_prompt

        parts: list[InputText] = []
        if base:
            parts.append(InputText(text=base))

        for section in self.system_prompt_sections:
            result = section.compute(ctx=ctx, exec_id=exec_id, agent_ctx=agent_ctx)
            if inspect.isawaitable(result):
                result = await result
            if not result:
                continue
            parts.append(InputText(text=result, cache_control=section.cache_control))

        if not parts:
            return None
        return parts

    @final
    async def build_initial_context(
        self,
        *,
        ctx: SessionContext[CtxT],
        exec_id: str,
        agent_ctx: AgentContext | None = None,
    ) -> list[InputItem]:
        """
        Compose the ephemeral initial context for one step.

        The default is the system-prompt message (``sys_prompt`` + sections,
        carrying each section's :class:`CacheControl`). A registered
        :class:`InitialContextBuilder` then transforms it — augment, replace, or
        reorder. The loop prepends the result to the model-facing view each turn
        and never stores it in the transcript log, so it reflects current state
        and the log stays pure conversation. Empty when there is no
        system prompt and no builder adds anything.
        """
        messages: list[InputItem] = []
        parts = await self.build_system_prompt_parts(
            ctx=ctx, exec_id=exec_id, agent_ctx=agent_ctx
        )
        if parts is not None:
            content_parts: list[InputPart] = list(parts)
            messages.append(InputMessageItem(content=content_parts, role="system"))
        if self.initial_context_builder is not None:
            return list(await self.initial_context_builder(messages, exec_id=exec_id))
        return messages

    @final
    async def apply_input_attachments(
        self,
        input_message: InputMessageItem,
        *,
        ctx: SessionContext[CtxT],
        exec_id: str,
        messages: Sequence[InputItem] | None = None,
        agent_ctx: AgentContext | None = None,
        source: str | None = None,
    ) -> InputMessageItem:
        """
        Run every registered :class:`InputAttachment` and append the
        non-empty results as additional content parts on ``input_message``.

        Text returns are wrapped in ``<system-reminder>...</system-reminder>``
        unless ``wrap_in_system_reminder`` is false on the attachment.
        :class:`InputPart` sequences are appended verbatim. ``source`` is the
        sender of the run's input packet (an upstream processor / teammate), passed
        through so an attachment can attribute the turn.
        """
        if not self.input_attachments:
            return input_message

        extra_parts: list[InputPart] = []
        for attachment in self.input_attachments:
            result = attachment.compute(
                input_message=input_message,
                ctx=ctx,
                exec_id=exec_id,
                messages=messages,
                agent_ctx=agent_ctx,
                source=source,
            )
            if inspect.isawaitable(result):
                result = await result
            if result is None:
                continue
            if isinstance(result, str):
                text = result
                if attachment.wrap_in_system_reminder:
                    text = wrap_in_system_reminder(text)
                extra_parts.append(InputText(text=text))
            else:
                extra_parts.extend(result)

        if not extra_parts:
            return input_message

        return InputMessageItem(
            content=[*input_message.content, *extra_parts],
            role=input_message.role,
        )

    @final
    def build_input_message(
        self,
        chat_inputs: LLMPrompt | Sequence[str | InputImage] | None = None,
        *,
        in_args: InT | None = None,
        exec_id: str,
    ) -> InputMessageItem | None:
        if chat_inputs is not None:
            if in_args is not None:
                raise InputPromptBuilderError(
                    proc_name=self._agent_name,
                    message="Cannot use both chat inputs and input arguments "
                    f"at the same time [agent_name={self._agent_name}]",
                )

            if isinstance(chat_inputs, str):
                return InputMessageItem.from_text(chat_inputs, role="user")

            input_parts: list[InputPart] = [
                InputText(text=part) if isinstance(part, str) else part
                for part in chat_inputs
            ]
            return InputMessageItem(content=input_parts, role="user")

        if (
            in_args is None
            and self._in_type is type(None)
            and self.input_content_builder is None
        ):
            # An ``InT=None`` agent has no input payload to render — emit no
            # user message rather than a JSON-dumped ``"null"``.
            return None

        content = self.build_input_content(in_args=in_args, exec_id=exec_id)

        return InputMessageItem(content=content.parts, role="user")

    @final
    def build_input_content(self, in_args: InT | None, *, exec_id: str) -> Content:
        if in_args is None and self._in_type is not type(None):
            raise InputPromptBuilderError(
                proc_name=self._agent_name,
                message="Either chat inputs or input arguments must be provided "
                f"when input type is not None [agent_name={self._agent_name}]",
            )

        in_args = cast("InT", in_args)
        val_in_args = self._in_args_type_adapter.validate_python(in_args)

        # 1. InputContentBuilder hook (full custom control)
        if self.input_content_builder is not None:
            return self.input_content_builder(in_args=val_in_args, exec_id=exec_id)

        # 2. Model implements InputRenderable.to_input_parts()
        if isinstance(val_in_args, InputRenderable):
            parts = val_in_args.to_input_parts()
            if isinstance(parts, str):
                return Content.from_text(parts)
            return Content(parts=parts)

        # 2b. A content part (text / image / file) renders as-is — a message routed
        # as a typed packet may carry one (a peer's text / multimodal send); it is
        # input content, not a value to JSON-dump.
        if isinstance(val_in_args, (InputText, InputImage, InputFile)):
            return Content(parts=[val_in_args])

        # 3. BaseModel with in_prompt template (text-only)
        if issubclass(self._in_type, BaseModel) and isinstance(val_in_args, BaseModel):
            val_in_args_map = self._format_pydantic_prompt_args(val_in_args)
            if self.in_prompt is not None:
                return Content.from_formatted_prompt(self.in_prompt, **val_in_args_map)
            return Content.from_text(json.dumps(val_in_args_map, indent=2))

        # 4. Plain string payload → render verbatim. A JSON dump would wrap it
        # in redundant quotes; this matches the chat_inputs path, so a string
        # routed between processors reads the same as a top-level string input.
        if isinstance(val_in_args, str):
            return Content.from_text(val_in_args)

        # 5. JSON fallback
        fmt_in_args = self._in_args_type_adapter.dump_json(
            val_in_args, indent=2, warnings="error"
        ).decode("utf-8")

        return Content.from_text(fmt_in_args)

    @staticmethod
    def _format_pydantic_prompt_args(inp: BaseModel) -> dict[str, PromptArgumentType]:
        formatted_args: dict[str, PromptArgumentType] = {}
        for field_name, field_info in type(inp).model_fields.items():
            if field_info.exclude:
                continue

            val = getattr(inp, field_name)
            if isinstance(val, (int, str, bool)):
                formatted_args[field_name] = val
            else:
                formatted_args[field_name] = (
                    TypeAdapter(type(val))  # type: ignore[return-value]
                    .dump_json(val, indent=2, warnings="error")
                    .decode("utf-8")
                )

        return formatted_args
