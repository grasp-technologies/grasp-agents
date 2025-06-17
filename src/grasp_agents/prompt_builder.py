import json
from collections.abc import Mapping, Sequence
from typing import ClassVar, Generic, Protocol, TypeAlias

from pydantic import BaseModel, TypeAdapter

from .generics_utils import AutoInstanceAttributesMixin
from .run_context import CtxT, RunContext
from .typing.content import Content, ImageData
from .typing.io import InT_contra, LLMPrompt, LLMPromptArgs
from .typing.message import UserMessage


class MakeSystemPromptHandler(Protocol[CtxT]):
    def __call__(
        self,
        sys_args: LLMPromptArgs | None,
        *,
        ctx: RunContext[CtxT] | None,
    ) -> str: ...


class MakeInputContentHandler(Protocol[InT_contra, CtxT]):
    def __call__(
        self,
        *,
        in_args: InT_contra | None,
        usr_args: LLMPromptArgs | None,
        batch_idx: int,
        ctx: RunContext[CtxT] | None,
    ) -> Content: ...


PromptArgumentType: TypeAlias = str | bool | int | ImageData


class PromptBuilder(AutoInstanceAttributesMixin, Generic[InT_contra, CtxT]):
    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {0: "_in_type"}

    def __init__(
        self,
        agent_name: str,
        sys_prompt_template: LLMPrompt | None,
        in_prompt_template: LLMPrompt | None,
        sys_args_schema: type[LLMPromptArgs] | None = None,
        usr_args_schema: type[LLMPromptArgs] | None = None,
    ):
        self._in_type: type[InT_contra]
        super().__init__()

        self._agent_name = agent_name
        self.sys_prompt_template = sys_prompt_template
        self.in_prompt_template = in_prompt_template
        self.sys_args_schema = sys_args_schema
        self.usr_args_schema = usr_args_schema
        self.make_sys_prompt_impl: MakeSystemPromptHandler[CtxT] | None = None
        self.make_in_content_impl: MakeInputContentHandler[InT_contra, CtxT] | None = (
            None
        )

        self._in_args_type_adapter: TypeAdapter[InT_contra] = TypeAdapter(self._in_type)

    def make_sys_prompt(
        self, sys_args: LLMPromptArgs | None = None, ctx: RunContext[CtxT] | None = None
    ) -> str | None:
        if self.sys_prompt_template is None:
            return None

        val_sys_args = sys_args
        if sys_args is not None:
            if self.sys_args_schema is not None:
                val_sys_args = self.sys_args_schema.model_validate(sys_args)
            else:
                raise TypeError(
                    "System prompt template is set, but system arguments schema is not "
                    "provided."
                )

        if self.make_sys_prompt_impl:
            return self.make_sys_prompt_impl(sys_args=val_sys_args, ctx=ctx)

        sys_args_dict = (
            val_sys_args.model_dump(exclude_unset=True) if val_sys_args else {}
        )

        return self.sys_prompt_template.format(**sys_args_dict)

    def make_in_content(
        self,
        *,
        in_args: InT_contra | None,
        usr_args: LLMPromptArgs | None,
        batch_idx: int = 0,
        ctx: RunContext[CtxT] | None = None,
    ) -> Content:
        val_in_args, val_usr_args = self._validate_prompt_args(
            in_args=in_args, usr_args=usr_args
        )

        if self.make_in_content_impl:
            return self.make_in_content_impl(
                in_args=val_in_args, usr_args=val_usr_args, batch_idx=batch_idx, ctx=ctx
            )

        combined_args = self._combine_args(in_args=val_in_args, usr_args=val_usr_args)
        if isinstance(combined_args, str):
            return Content.from_text(combined_args)

        if self.in_prompt_template is not None:
            return Content.from_formatted_prompt(
                self.in_prompt_template, prompt_args=combined_args
            )

        return Content.from_text(json.dumps(combined_args, indent=2))

    def make_user_messages(
        self,
        chat_inputs: LLMPrompt | Sequence[str | ImageData] | None = None,
        in_args_batch: Sequence[InT_contra] | None = None,
        usr_args: LLMPromptArgs | None = None,
        ctx: RunContext[CtxT] | None = None,
    ) -> Sequence[UserMessage]:
        if chat_inputs:
            if isinstance(chat_inputs, LLMPrompt):
                return self._usr_messages_from_text(chat_inputs)
            return self._usr_messages_from_content_parts(chat_inputs)

        in_content_batch = [
            self.make_in_content(
                in_args=in_args, usr_args=usr_args, batch_idx=i, ctx=ctx
            )
            for i, in_args in enumerate(in_args_batch or [None])
        ]
        return [
            UserMessage(content=in_content, name=self._agent_name)
            for in_content in in_content_batch
        ]

    def _usr_messages_from_text(self, text: str) -> list[UserMessage]:
        return [UserMessage.from_text(text, name=self._agent_name)]

    def _usr_messages_from_content_parts(
        self, content_parts: Sequence[str | ImageData]
    ) -> list[UserMessage]:
        return [UserMessage.from_content_parts(content_parts, name=self._agent_name)]

    def _validate_prompt_args(
        self,
        *,
        in_args: InT_contra | None,
        usr_args: LLMPromptArgs | None,
    ) -> tuple[InT_contra | None, LLMPromptArgs | None]:
        val_usr_args = usr_args
        if usr_args is not None:
            if self.in_prompt_template is None:
                raise TypeError(
                    "Input prompt template is not set, but user arguments are provided."
                )
            if self.usr_args_schema is None:
                raise TypeError(
                    "User arguments schema is not provided, but user arguments are "
                    "given."
                )
            val_usr_args = self.usr_args_schema.model_validate(usr_args)

        val_in_args = in_args
        if in_args is not None:
            val_in_args = self._in_args_type_adapter.validate_python(in_args)
            if isinstance(val_in_args, BaseModel):
                _, has_image = self._format_pydantic_prompt_args(val_in_args)
                if has_image and self.in_prompt_template is None:
                    raise TypeError(
                        "BaseModel input arguments contain ImageData, but input prompt "
                        "template is not set. Cannot format input arguments."
                    )
            elif self.in_prompt_template is not None:
                raise TypeError(
                    "Cannot use the input prompt template with "
                    "non-BaseModel input arguments."
                )

        return val_in_args, val_usr_args

    @staticmethod
    def _format_pydantic_prompt_args(
        inp: BaseModel,
    ) -> tuple[dict[str, PromptArgumentType], bool]:
        formatted_args: dict[str, PromptArgumentType] = {}
        contains_image_data = False
        for field in type(inp).model_fields:
            if field == "selected_recipients":
                continue

            val = getattr(inp, field)
            if isinstance(val, (int, str, bool)):
                formatted_args[field] = val
            elif isinstance(val, ImageData):
                formatted_args[field] = val
                contains_image_data = True
            elif isinstance(val, BaseModel):
                formatted_args[field] = val.model_dump_json(indent=2, warnings="error")
            else:
                raise TypeError(
                    f"Field '{field}' in prompt arguments must be of type "
                    "int, str, bool, BaseModel, or ImageData."
                )

        return formatted_args, contains_image_data

    def _combine_args(
        self,
        *,
        in_args: InT_contra | None,
        usr_args: LLMPromptArgs | None,
    ) -> Mapping[str, PromptArgumentType] | str:
        fmt_usr_args, _ = (
            self._format_pydantic_prompt_args(usr_args) if usr_args else ({}, False)
        )

        if in_args is None:
            return fmt_usr_args

        if isinstance(in_args, BaseModel):
            fmt_in_args, _ = self._format_pydantic_prompt_args(in_args)
            return fmt_in_args | fmt_usr_args

        combined_args_str = self._in_args_type_adapter.dump_json(
            in_args, indent=2, warnings="error"
        ).decode("utf-8")
        if usr_args is not None:
            fmt_usr_args_str = usr_args.model_dump_json(indent=2, warnings="error")
            combined_args_str += "\n" + fmt_usr_args_str

        return combined_args_str
