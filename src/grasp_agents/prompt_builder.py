import json
from collections.abc import Sequence
from typing import ClassVar, Generic, Protocol, TypeAlias, TypeVar

from pydantic import BaseModel, TypeAdapter

from .errors import InputPromptBuilderError, SystemPromptBuilderError
from .generics_utils import AutoInstanceAttributesMixin
from .run_context import CtxT, RunContext
from .typing.content import Content, ImageData
from .typing.io import InT, LLMPrompt, LLMPromptArgs
from .typing.message import UserMessage

_InT_contra = TypeVar("_InT_contra", contravariant=True)


class MakeSystemPromptHandler(Protocol[CtxT]):
    def __call__(
        self,
        sys_args: LLMPromptArgs | None,
        *,
        ctx: RunContext[CtxT] | None,
    ) -> str | None: ...


class MakeInputContentHandler(Protocol[_InT_contra, CtxT]):
    def __call__(
        self,
        in_args: _InT_contra | None,
        *,
        ctx: RunContext[CtxT] | None,
    ) -> Content: ...


PromptArgumentType: TypeAlias = str | bool | int | ImageData


class PromptBuilder(AutoInstanceAttributesMixin, Generic[InT, CtxT]):
    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {0: "_in_type"}

    def __init__(
        self,
        agent_name: str,
        sys_prompt: LLMPrompt | None,
        in_prompt: LLMPrompt | None,
        sys_args_schema: type[LLMPromptArgs] | None = None,
    ):
        self._in_type: type[InT]
        super().__init__()

        self._agent_name = agent_name
        self.sys_prompt = sys_prompt
        self.in_prompt = in_prompt
        self.sys_args_schema = sys_args_schema
        self.make_system_prompt_impl: MakeSystemPromptHandler[CtxT] | None = None
        self.make_input_content_impl: MakeInputContentHandler[InT, CtxT] | None = None

        self._in_args_type_adapter: TypeAdapter[InT] = TypeAdapter(self._in_type)

    def _validate_sys_args(
        self, sys_args: LLMPromptArgs | None
    ) -> LLMPromptArgs | None:
        val_sys_args = sys_args
        if sys_args is not None:
            if self.sys_args_schema is not None:
                val_sys_args = self.sys_args_schema.model_validate(sys_args)
            else:
                raise SystemPromptBuilderError(
                    proc_name=self._agent_name,
                    message="System prompt template and arguments is set, "
                    "but system arguments schema is not provided "
                    f"[agent_name={self._agent_name}]",
                )

        return val_sys_args

    def make_system_prompt(
        self,
        sys_args: LLMPromptArgs | None = None,
        ctx: RunContext[CtxT] | None = None,
    ) -> str | None:
        if self.sys_prompt is None:
            return None

        val_sys_args = self._validate_sys_args(sys_args=sys_args)

        if self.make_system_prompt_impl:
            return self.make_system_prompt_impl(sys_args=val_sys_args, ctx=ctx)

        sys_args_dict = val_sys_args.model_dump() if val_sys_args else {}

        return self.sys_prompt.format(**sys_args_dict)

    def _validate_input_args(self, in_args: InT) -> InT:
        val_in_args = self._in_args_type_adapter.validate_python(in_args)
        if isinstance(val_in_args, BaseModel):
            has_image = self._has_image_data(val_in_args)
            if has_image and self.in_prompt is None:
                raise InputPromptBuilderError(
                    proc_name=self._agent_name,
                    message="BaseModel input arguments contain ImageData, "
                    "but input prompt template is not set "
                    f"[agent_name={self._agent_name}]. Cannot format input arguments.",
                )
        elif self.in_prompt is not None:
            raise InputPromptBuilderError(
                proc_name=self._agent_name,
                message="Cannot use the input prompt template with "
                f"non-BaseModel input arguments [agent_name={self._agent_name}]",
            )

        return val_in_args

    def make_input_content(
        self,
        in_args: InT | None = None,
        ctx: RunContext[CtxT] | None = None,
    ) -> Content:
        val_in_args = in_args
        if in_args is not None:
            val_in_args = self._validate_input_args(in_args=in_args)

        if self.make_input_content_impl:
            return self.make_input_content_impl(in_args=val_in_args, ctx=ctx)

        if val_in_args is None:
            raise InputPromptBuilderError(
                proc_name=self._agent_name,
                message="Input arguments are not provided, "
                f"but input content is required [agent_name={self._agent_name}]",
            )

        if issubclass(self._in_type, BaseModel) and isinstance(val_in_args, BaseModel):
            val_in_args_map = self._format_pydantic_prompt_args(val_in_args)
            if self.in_prompt is not None:
                return Content.from_formatted_prompt(self.in_prompt, **val_in_args_map)
            return Content.from_text(json.dumps(val_in_args_map, indent=2))

        fmt_in_args = self._in_args_type_adapter.dump_json(
            val_in_args, indent=2, warnings="error"
        ).decode("utf-8")
        return Content.from_text(fmt_in_args)

    def make_input_message(
        self,
        chat_inputs: LLMPrompt | Sequence[str | ImageData] | None = None,
        in_args: InT | None = None,
        ctx: RunContext[CtxT] | None = None,
    ) -> UserMessage | None:
        if chat_inputs is not None and in_args is not None:
            raise InputPromptBuilderError(
                proc_name=self._agent_name,
                message="Cannot use both chat inputs and input arguments "
                f"at the same time [agent_name={self._agent_name}]",
            )

        if chat_inputs:
            if isinstance(chat_inputs, LLMPrompt):
                return UserMessage.from_text(chat_inputs, name=self._agent_name)
            return UserMessage.from_content_parts(chat_inputs, name=self._agent_name)

        return UserMessage(
            content=self.make_input_content(in_args=in_args, ctx=ctx),
            name=self._agent_name,
        )

    @staticmethod
    def _has_image_data(inp: BaseModel) -> bool:
        contains_image_data = False
        for field in type(inp).model_fields:
            if isinstance(getattr(inp, field), ImageData):
                contains_image_data = True

        return contains_image_data

    @staticmethod
    def _format_pydantic_prompt_args(inp: BaseModel) -> dict[str, PromptArgumentType]:
        formatted_args: dict[str, PromptArgumentType] = {}
        for field_name, field_info in type(inp).model_fields.items():
            if field_info.exclude:
                continue

            val = getattr(inp, field_name)
            if isinstance(val, (int, str, bool, ImageData)):
                formatted_args[field_name] = val
            else:
                formatted_args[field_name] = (
                    TypeAdapter(type(val))  # type: ignore[return-value]
                    .dump_json(val, indent=2, warnings="error")
                    .decode("utf-8")
                )

        return formatted_args
