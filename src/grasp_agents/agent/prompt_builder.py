import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Generic, TypeAlias, cast, final

from pydantic import BaseModel, TypeAdapter

from ..run_context import CtxT, RunContext
from ..types.content import (
    Content,
    InputImage,
    InputPart,
    InputRenderable,
    InputText,
)
from ..types.errors import InputPromptBuilderError
from ..utils.generics import AutoInstanceAttributesMixin

if TYPE_CHECKING:
    from ..types.hooks import InputContentBuilder, SystemPromptBuilder
from ..types.io import InT, LLMPrompt
from ..types.items import InputMessageItem

PromptArgumentType: TypeAlias = str | bool | int


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

    @property
    def sys_prompt(self) -> LLMPrompt | None:
        return self._sys_prompt

    @property
    def in_prompt(self) -> LLMPrompt | None:
        return self._in_prompt

    @final
    def build_system_prompt(self, *, ctx: RunContext[CtxT], exec_id: str) -> str | None:
        if self.system_prompt_builder is not None:
            return self.system_prompt_builder(ctx=ctx, exec_id=exec_id)

        return self.sys_prompt

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
