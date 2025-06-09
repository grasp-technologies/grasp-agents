from collections.abc import Sequence
from copy import deepcopy
from typing import ClassVar, Generic, Protocol, cast

from pydantic import BaseModel, TypeAdapter

from .generics_utils import AutoInstanceAttributesMixin
from .run_context import CtxT, RunContext
from .typing.content import ImageData
from .typing.io import (
    InT_contra,
    LLMFormattedArgs,
    LLMFormattedSystemArgs,
    LLMPrompt,
    LLMPromptArgs,
)
from .typing.message import UserMessage


class DummySchema(BaseModel):
    pass


class FormatSystemArgsHandler(Protocol[CtxT]):
    def __call__(
        self,
        sys_args: LLMPromptArgs,
        *,
        ctx: RunContext[CtxT] | None,
    ) -> LLMFormattedSystemArgs: ...


class FormatInputArgsHandler(Protocol[InT_contra, CtxT]):
    def __call__(
        self,
        *,
        in_args: InT_contra,
        usr_args: LLMPromptArgs,
        batch_idx: int,
        ctx: RunContext[CtxT] | None,
    ) -> LLMFormattedArgs: ...


class PromptBuilder(AutoInstanceAttributesMixin, Generic[InT_contra, CtxT]):
    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {0: "_in_type"}

    def __init__(
        self,
        agent_name: str,
        sys_prompt: LLMPrompt | None,
        in_prompt: LLMPrompt | None,
        sys_args_schema: type[LLMPromptArgs],
        usr_args_schema: type[LLMPromptArgs],
    ):
        self._in_type: type[InT_contra]
        super().__init__()

        self._agent_name = agent_name
        self.sys_prompt = sys_prompt
        self.in_prompt = in_prompt
        self.sys_args_schema = sys_args_schema
        self.usr_args_schema = usr_args_schema
        self.format_sys_args_impl: FormatSystemArgsHandler[CtxT] | None = None
        self.format_in_args_impl: FormatInputArgsHandler[InT_contra, CtxT] | None = None

        self._in_args_type_adapter: TypeAdapter[InT_contra] = TypeAdapter(self._in_type)

    def _format_sys_args_fn(
        self, sys_args: LLMPromptArgs, ctx: RunContext[CtxT] | None = None
    ) -> LLMFormattedSystemArgs:
        if self.format_sys_args_impl:
            return self.format_sys_args_impl(sys_args=sys_args, ctx=ctx)

        return sys_args.model_dump(exclude_unset=True)

    def _format_in_args_fn(
        self,
        *,
        in_args: InT_contra,
        usr_args: LLMPromptArgs,
        batch_idx: int = 0,
        ctx: RunContext[CtxT] | None = None,
    ) -> LLMFormattedArgs:
        if self.format_in_args_impl:
            return self.format_in_args_impl(
                in_args=in_args, usr_args=usr_args, batch_idx=batch_idx, ctx=ctx
            )

        if not isinstance(in_args, BaseModel) and in_args is not None:
            raise TypeError(
                "Cannot apply default formatting to non-BaseModel received arguments."
            )

        in_args_ = in_args or DummySchema()
        usr_args_ = usr_args

        in_args_dump = in_args_.model_dump(exclude={"selected_recipients"})
        usr_args_dump = usr_args_.model_dump(exclude_unset=True)

        return usr_args_dump | in_args_dump

    def make_sys_prompt(
        self, sys_args: LLMPromptArgs, *, ctx: RunContext[CtxT] | None
    ) -> LLMPrompt | None:
        if self.sys_prompt is None:
            return None
        val_sys_args = self.sys_args_schema.model_validate(sys_args)
        fmt_sys_args = self._format_sys_args_fn(val_sys_args, ctx=ctx)

        return self.sys_prompt.format(**fmt_sys_args)

    def _usr_messages_from_text(self, text: str) -> list[UserMessage]:
        return [UserMessage.from_text(text, model_id=self._agent_name)]

    def _usr_messages_from_content_parts(
        self, content_parts: Sequence[str | ImageData]
    ) -> Sequence[UserMessage]:
        return [
            UserMessage.from_content_parts(content_parts, model_id=self._agent_name)
        ]

    def _usr_messages_from_in_args(
        self, in_args_batch: Sequence[InT_contra]
    ) -> Sequence[UserMessage]:
        return [
            UserMessage.from_text(
                self._in_args_type_adapter.dump_json(
                    inp,
                    exclude_unset=True,
                    indent=2,
                    exclude={"selected_recipients"},
                    warnings="error",
                ).decode("utf-8"),
                model_id=self._agent_name,
            )
            for inp in in_args_batch
        ]

    def _usr_messages_from_prompt_template(
        self,
        in_prompt: LLMPrompt,
        in_args_batch: Sequence[InT_contra] | None = None,
        usr_args_batch: Sequence[LLMPromptArgs] | None = None,
        ctx: RunContext[CtxT] | None = None,
    ) -> Sequence[UserMessage]:
        in_args_batch_, usr_args_batch_ = self._align_input_and_user_batches(
            in_args_batch, usr_args_batch
        )

        val_in_args_batch_ = [
            self._in_args_type_adapter.validate_python(inp) for inp in in_args_batch_
        ]
        val_usr_args_batch_ = [
            self.usr_args_schema.model_validate(u) for u in usr_args_batch_
        ]

        formatted_in_args_batch = [
            self._format_in_args_fn(
                in_args=val_in_args, usr_args=val_usr_args, batch_idx=i, ctx=ctx
            )
            for i, (val_usr_args, val_in_args) in enumerate(
                zip(val_usr_args_batch_, val_in_args_batch_, strict=False)
            )
        ]

        return [
            UserMessage.from_formatted_prompt(
                prompt_template=in_prompt, prompt_args=in_args
            )
            for in_args in formatted_in_args_batch
        ]

    def make_user_messages(
        self,
        chat_inputs: LLMPrompt | Sequence[str | ImageData] | None = None,
        in_args: InT_contra | Sequence[InT_contra] | None = None,
        usr_args: LLMPromptArgs | Sequence[LLMPromptArgs] | None = None,
        entry_point: bool = False,
        ctx: RunContext[CtxT] | None = None,
    ) -> Sequence[UserMessage]:
        # 1) Chat inputs
        if chat_inputs is not None or entry_point:
            """
            * If chat inputs are provided, they override the input prompt template
            * In a multi-agent system, the input prompt template is used to
                construct agent inputs using the combination of input and
                user arguments.
                However, the initial agent (entry point) has no input
                messages, so we use the chat inputs directly, if provided.
            """
            if isinstance(chat_inputs, LLMPrompt):
                return self._usr_messages_from_text(chat_inputs)

            if isinstance(chat_inputs, Sequence) and chat_inputs:
                return self._usr_messages_from_content_parts(chat_inputs)

        # 2) No input prompt template + input args → raw JSON messages
        in_args_batch = cast(
            "Sequence[InT_contra] | None",
            in_args if (isinstance(in_args, Sequence) or not in_args) else [in_args],
        )
        if self.in_prompt is None and in_args_batch:
            return self._usr_messages_from_in_args(in_args_batch)

        # 3) Input prompt template + any args → batch & format
        usr_args_batch = cast(
            "Sequence[LLMPromptArgs] | None",
            (
                usr_args
                if (isinstance(usr_args, Sequence) or not usr_args)
                else [usr_args]
            ),
        )
        if self.in_prompt is not None:
            if in_args_batch and not isinstance(in_args_batch[0], BaseModel):
                raise TypeError(
                    "Cannot use the input prompt template with "
                    "non-BaseModel input arguments."
                )
            return self._usr_messages_from_prompt_template(
                in_prompt=self.in_prompt,
                in_args_batch=in_args_batch,
                usr_args_batch=usr_args_batch,
                ctx=ctx,
            )

        return []

    def _align_input_and_user_batches(
        self,
        in_args_batch: Sequence[InT_contra] | None = None,
        usr_args_batch: Sequence[LLMPromptArgs] | None = None,
    ) -> tuple[Sequence[InT_contra | None], Sequence[LLMPromptArgs | DummySchema]]:
        in_args_batch_ = in_args_batch or [None]
        usr_args_batch_ = usr_args_batch or [DummySchema()]

        # Broadcast singleton → match lengths
        if len(in_args_batch_) == 1 and len(usr_args_batch_) > 1:
            in_args_batch_ = [deepcopy(in_args_batch_[0]) for _ in usr_args_batch_]

        if len(usr_args_batch_) == 1 and len(in_args_batch_) > 1:
            usr_args_batch_ = [deepcopy(usr_args_batch_[0]) for _ in in_args_batch_]

        if len(usr_args_batch_) != len(in_args_batch_):
            raise ValueError("User args and input args must have the same length")

        return in_args_batch_, usr_args_batch_
