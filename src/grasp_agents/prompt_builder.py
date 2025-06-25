import json
from collections.abc import Mapping, Sequence
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
    ) -> str: ...


class MakeInputContentHandler(Protocol[_InT_contra, CtxT]):
    def __call__(
        self,
        *,
        in_args: _InT_contra | None,
        usr_args: LLMPromptArgs | None,
        ctx: RunContext[CtxT] | None,
    ) -> Content: ...


PromptArgumentType: TypeAlias = str | bool | int | ImageData


class PromptBuilder(AutoInstanceAttributesMixin, Generic[InT, CtxT]):
    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {0: "_in_type"}

    def __init__(
        self,
        agent_name: str,
        sys_prompt_template: LLMPrompt | None,
        in_prompt_template: LLMPrompt | None,
        sys_args_schema: type[LLMPromptArgs] | None = None,
        usr_args_schema: type[LLMPromptArgs] | None = None,
    ):
        self._in_type: type[InT]
        super().__init__()

        self._agent_name = agent_name
        self.sys_prompt_template = sys_prompt_template
        self.in_prompt_template = in_prompt_template
        self.sys_args_schema = sys_args_schema
        self.usr_args_schema = usr_args_schema
        self.make_system_prompt_impl: MakeSystemPromptHandler[CtxT] | None = None
        self.make_input_content_impl: MakeInputContentHandler[InT, CtxT] | None = None

        self._in_args_type_adapter: TypeAdapter[InT] = TypeAdapter(self._in_type)

    def make_system_prompt(
        self, sys_args: LLMPromptArgs | None = None, ctx: RunContext[CtxT] | None = None
    ) -> str | None:
        if self.sys_prompt_template is None:
            return None

        val_sys_args = sys_args
        if sys_args is not None:
            if self.sys_args_schema is not None:
                val_sys_args = self.sys_args_schema.model_validate(sys_args)
            else:
                raise SystemPromptBuilderError(
                    "System prompt template and arguments is set, but system arguments "
                    "schema is not provided."
                )

        if self.make_system_prompt_impl:
            return self.make_system_prompt_impl(sys_args=val_sys_args, ctx=ctx)

        sys_args_dict = val_sys_args.model_dump() if val_sys_args else {}

        return self.sys_prompt_template.format(**sys_args_dict)

    def make_input_content(
        self,
        *,
        in_args: InT | None,
        usr_args: LLMPromptArgs | None,
        ctx: RunContext[CtxT] | None = None,
    ) -> Content:
        val_in_args, val_usr_args = self._validate_prompt_args(
            in_args=in_args, usr_args=usr_args
        )

        if self.make_input_content_impl:
            return self.make_input_content_impl(
                in_args=val_in_args, usr_args=val_usr_args, ctx=ctx
            )

        combined_args = self._combine_args(in_args=val_in_args, usr_args=val_usr_args)
        if isinstance(combined_args, str):
            return Content.from_text(combined_args)

        if self.in_prompt_template is not None:
            return Content.from_formatted_prompt(
                self.in_prompt_template, **combined_args
            )

        return Content.from_text(json.dumps(combined_args, indent=2))

    def make_user_message(
        self,
        chat_inputs: LLMPrompt | Sequence[str | ImageData] | None = None,
        in_args: InT | None = None,
        usr_args: LLMPromptArgs | None = None,
        ctx: RunContext[CtxT] | None = None,
    ) -> UserMessage | None:
        if chat_inputs is None and in_args is None and usr_args is None:
            return None

        if chat_inputs:
            if isinstance(chat_inputs, LLMPrompt):
                return UserMessage.from_text(chat_inputs, name=self._agent_name)
            return UserMessage.from_content_parts(chat_inputs, name=self._agent_name)

        in_content = self.make_input_content(
            in_args=in_args, usr_args=usr_args, ctx=ctx
        )

        return UserMessage(content=in_content, name=self._agent_name)

    def _validate_prompt_args(
        self,
        *,
        in_args: InT | None,
        usr_args: LLMPromptArgs | None,
    ) -> tuple[InT | None, LLMPromptArgs | None]:
        val_usr_args = usr_args
        if usr_args is not None:
            if self.in_prompt_template is None:
                raise InputPromptBuilderError(
                    "Input prompt template is not set, but user arguments are provided."
                )
            if self.usr_args_schema is None:
                raise InputPromptBuilderError(
                    "User arguments schema is not provided, but user arguments are "
                    "given."
                )
            val_usr_args = self.usr_args_schema.model_validate(usr_args)

        val_in_args = in_args
        if in_args is not None:
            val_in_args = self._in_args_type_adapter.validate_python(in_args)
            if isinstance(val_in_args, BaseModel):
                has_image = self._has_image_data(val_in_args)
                if has_image and self.in_prompt_template is None:
                    raise InputPromptBuilderError(
                        "BaseModel input arguments contain ImageData, but input prompt "
                        "template is not set. Cannot format input arguments."
                    )
            elif self.in_prompt_template is not None:
                raise InputPromptBuilderError(
                    "Cannot use the input prompt template with "
                    "non-BaseModel input arguments."
                )

        return val_in_args, val_usr_args

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
        for field in type(inp).model_fields:
            if field == "selected_recipients":
                continue

            val = getattr(inp, field)
            if isinstance(val, (int, str, bool, ImageData)):
                formatted_args[field] = val
            else:
                formatted_args[field] = (
                    TypeAdapter(type(val))  # type: ignore[return-value]
                    .dump_json(val, indent=2, warnings="error")
                    .decode("utf-8")
                )

        return formatted_args

    def _combine_args(
        self, *, in_args: InT | None, usr_args: LLMPromptArgs | None
    ) -> Mapping[str, PromptArgumentType] | str:
        fmt_usr_args = self._format_pydantic_prompt_args(usr_args) if usr_args else {}

        if in_args is None:
            return fmt_usr_args

        if isinstance(in_args, BaseModel):
            fmt_in_args = self._format_pydantic_prompt_args(in_args)
            return fmt_in_args | fmt_usr_args

        combined_args_str = self._in_args_type_adapter.dump_json(
            in_args, indent=2, warnings="error"
        ).decode("utf-8")
        if usr_args is not None:
            fmt_usr_args_str = usr_args.model_dump_json(indent=2, warnings="error")
            combined_args_str += "\n" + fmt_usr_args_str

        return combined_args_str
