from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Protocol,
    TypeAlias,
    cast,
    final,
    runtime_checkable,
)

from pydantic import BaseModel, TypeAdapter

from grasp_agents.run_context import CtxT
from grasp_agents.types.content import (
    Content,
    InputImage,
    InputPart,
    InputRenderable,
    InputText,
)
from grasp_agents.types.errors import InputPromptBuilderError
from grasp_agents.utils.generics import AutoInstanceAttributesMixin

if TYPE_CHECKING:
    from collections.abc import Awaitable, Sequence

    from grasp_agents.run_context import RunContext
    from grasp_agents.types.hooks import InputContentBuilder, SystemPromptBuilder
    from grasp_agents.types.items import InputItem
from grasp_agents.types.io import InT, LLMPrompt
from grasp_agents.types.items import InputMessageItem

PromptArgumentType: TypeAlias = str | bool | int


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
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
    ) -> str | Awaitable[str | None] | None: ...


@dataclass(frozen=True)
class SystemPromptSection:
    """
    A named, lazy block appended to the system prompt at run time.

    ``compute`` is invoked with ``(ctx, exec_id)`` per agent run, may be sync
    or async, and returns the rendered text or ``None`` (omit). ``cache_break``
    is a forward-compat placeholder for provider-level cache control; it is
    not honored by any provider today.
    """

    name: str
    compute: SectionCompute
    cache_break: bool = False


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
        user_message: InputMessageItem,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        messages: Sequence[InputItem] | None = None,
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


class PromptBuilder(AutoInstanceAttributesMixin, Generic[InT, CtxT]):
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
        self.system_prompt_builder: SystemPromptBuilder[CtxT] | None = None
        self.input_content_builder: InputContentBuilder[InT, CtxT] | None = None

        # Sections appended to the system prompt at run time. Each section's
        # compute receives (ctx, exec_id) and may be sync or async. Order is
        # preserved; ``None`` outputs are dropped. Skills (and later memory)
        # plug in here.
        self.system_prompt_sections: list[SystemPromptSection] = []

        # Attachments appended to the user message at run time. Each
        # attachment's compute receives the just-built user_message plus
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
        ctx: RunContext[CtxT],
        exec_id: str,
    ) -> str | None:
        if self.system_prompt_builder is not None:
            base = self.system_prompt_builder(ctx=ctx, exec_id=exec_id)
        else:
            base = self.sys_prompt

        if not self.system_prompt_sections:
            return base

        rendered: list[str] = []
        for section in self.system_prompt_sections:
            result = section.compute(ctx=ctx, exec_id=exec_id)
            if inspect.isawaitable(result):
                result = await result
            if result:
                rendered.append(result)

        if not rendered:
            return base
        if not base:
            return "\n\n".join(rendered)
        return "\n\n".join([base, *rendered])

    @final
    async def apply_input_attachments(
        self,
        user_message: InputMessageItem,
        *,
        ctx: RunContext[CtxT],
        exec_id: str,
        messages: Sequence[InputItem] | None = None,
    ) -> InputMessageItem:
        """
        Run every registered :class:`InputAttachment` and append the
        non-empty results as additional content parts on ``user_message``.

        Text returns are wrapped in ``<system-reminder>...</system-reminder>``
        unless ``wrap_in_system_reminder`` is false on the attachment.
        :class:`InputPart` sequences are appended verbatim.
        """
        if not self.input_attachments:
            return user_message

        extra_parts: list[InputPart] = []
        for attachment in self.input_attachments:
            result = attachment.compute(
                user_message=user_message,
                ctx=ctx,
                exec_id=exec_id,
                messages=messages,
            )
            if inspect.isawaitable(result):
                result = await result
            if result is None:
                continue
            if isinstance(result, str):
                text = result
                if attachment.wrap_in_system_reminder:
                    text = f"<system-reminder>\n{text}\n</system-reminder>"
                extra_parts.append(InputText(text=text))
            else:
                extra_parts.extend(result)

        if not extra_parts:
            return user_message

        return InputMessageItem(
            content_parts=[*user_message.content_parts, *extra_parts],
            role=user_message.role,
        )

    @final
    def build_input_message(
        self,
        chat_inputs: LLMPrompt | Sequence[str | InputImage] | None = None,
        *,
        in_args: InT | None = None,
        exec_id: str,
        ctx: RunContext[CtxT],
    ) -> InputMessageItem | None:
        if chat_inputs is not None:
            if in_args is not None:
                raise InputPromptBuilderError(
                    proc_name=self._agent_name,
                    message="Cannot use both chat inputs and input arguments "
                    f"at the same time [agent_name={self._agent_name}]",
                )

            if isinstance(chat_inputs, LLMPrompt):
                return InputMessageItem.from_text(chat_inputs, role="user")

            input_parts: list[InputPart] = [
                InputText(text=part) if isinstance(part, str) else part
                for part in chat_inputs
            ]
            return InputMessageItem(content_parts=input_parts, role="user")

        content = self.build_input_content(in_args=in_args, ctx=ctx, exec_id=exec_id)

        return InputMessageItem(content_parts=content.parts, role="user")

    @final
    def build_input_content(
        self, in_args: InT | None, *, ctx: RunContext[CtxT], exec_id: str
    ) -> Content:
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
            return self.input_content_builder(
                in_args=val_in_args, ctx=ctx, exec_id=exec_id
            )

        # 2. Model implements InputRenderable.to_input_parts()
        if isinstance(val_in_args, InputRenderable):
            parts = val_in_args.to_input_parts()
            if isinstance(parts, str):
                return Content.from_text(parts)
            return Content(parts=parts)

        # 3. BaseModel with in_prompt template (text-only)
        if issubclass(self._in_type, BaseModel) and isinstance(val_in_args, BaseModel):
            val_in_args_map = self._format_pydantic_prompt_args(val_in_args)
            if self.in_prompt is not None:
                return Content.from_formatted_prompt(self.in_prompt, **val_in_args_map)
            return Content.from_text(json.dumps(val_in_args_map, indent=2))

        # 4. JSON fallback
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
