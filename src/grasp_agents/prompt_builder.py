from collections.abc import Sequence
from copy import deepcopy
from typing import ClassVar, Generic, Protocol

from pydantic import BaseModel, TypeAdapter

from .generics_utils import AutoInstanceAttributesMixin
from .run_context import CtxT, RunContextWrapper, UserRunArgs
from .typing.content import ImageData
from .typing.io import (
    InT,
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
        ctx: RunContextWrapper[CtxT] | None,
    ) -> LLMFormattedSystemArgs: ...


class FormatInputArgsHandler(Protocol[InT, CtxT]):
    def __call__(
        self,
        *,
        usr_args: LLMPromptArgs,
        in_args: InT,
        batch_idx: int,
        ctx: RunContextWrapper[CtxT] | None,
    ) -> LLMFormattedArgs: ...


class PromptBuilder(AutoInstanceAttributesMixin, Generic[InT, CtxT]):
    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {0: "_in_type"}

    def __init__(
        self,
        agent_id: str,
        sys_prompt: LLMPrompt | None,
        in_prompt: LLMPrompt | None,
        sys_args_schema: type[LLMPromptArgs],
        usr_args_schema: type[LLMPromptArgs],
    ):
        self._in_type: type[InT]
        super().__init__()

        self._agent_id = agent_id
        self.sys_prompt = sys_prompt
        self.in_prompt = in_prompt
        self.sys_args_schema = sys_args_schema
        self.usr_args_schema = usr_args_schema
        self.format_sys_args_impl: FormatSystemArgsHandler[CtxT] | None = None
        self.format_in_args_impl: FormatInputArgsHandler[InT, CtxT] | None = None

        self._in_args_type_adapter: TypeAdapter[InT] = TypeAdapter(self._in_type)

    def _format_sys_args(
        self,
        sys_args: LLMPromptArgs,
        ctx: RunContextWrapper[CtxT] | None = None,
    ) -> LLMFormattedSystemArgs:
        if self.format_sys_args_impl:
            return self.format_sys_args_impl(sys_args=sys_args, ctx=ctx)

        return sys_args.model_dump(exclude_unset=True)

    def _format_in_args(
        self,
        *,
        usr_args: LLMPromptArgs,
        in_args: InT,
        batch_idx: int = 0,
        ctx: RunContextWrapper[CtxT] | None = None,
    ) -> LLMFormattedArgs:
        if self.format_in_args_impl:
            return self.format_in_args_impl(
                usr_args=usr_args, in_args=in_args, batch_idx=batch_idx, ctx=ctx
            )

        if not isinstance(in_args, BaseModel) and in_args is not None:
            raise TypeError(
                "Cannot apply default formatting to non-BaseModel received arguments."
            )

        usr_args_ = usr_args
        in_args_ = DummySchema() if in_args is None else in_args

        usr_args_dump = usr_args_.model_dump(exclude_unset=True)
        in_args_dump = in_args_.model_dump(exclude={"selected_recipient_ids"})

        return usr_args_dump | in_args_dump

    def make_sys_prompt(
        self,
        sys_args: LLMPromptArgs,
        *,
        ctx: RunContextWrapper[CtxT] | None,
    ) -> LLMPrompt | None:
        if self.sys_prompt is None:
            return None
        val_sys_args = self.sys_args_schema.model_validate(sys_args)
        fmt_sys_args = self._format_sys_args(val_sys_args, ctx=ctx)

        return self.sys_prompt.format(**fmt_sys_args)

    def _usr_messages_from_text(self, text: str) -> list[UserMessage]:
        return [UserMessage.from_text(text, model_id=self._agent_id)]

    def _usr_messages_from_content_parts(
        self, content_parts: Sequence[str | ImageData]
    ) -> Sequence[UserMessage]:
        return [UserMessage.from_content_parts(content_parts, model_id=self._agent_id)]

    def _usr_messages_from_in_args(
        self, in_args_batch: Sequence[InT]
    ) -> Sequence[UserMessage]:
        return [
            UserMessage.from_text(
                self._in_args_type_adapter.dump_json(
                    inp,
                    exclude_unset=True,
                    indent=2,
                    exclude={"selected_recipient_ids"},
                    warnings="error",
                ).decode("utf-8"),
                model_id=self._agent_id,
            )
            for inp in in_args_batch
        ]

    def _usr_messages_from_prompt_template(
        self,
        in_prompt: LLMPrompt,
        usr_args: UserRunArgs | None = None,
        in_args_batch: Sequence[InT] | None = None,
        ctx: RunContextWrapper[CtxT] | None = None,
    ) -> Sequence[UserMessage]:
        usr_args_batch_, in_args_batch_ = self._make_batched(usr_args, in_args_batch)

        val_usr_args_batch_ = [
            self.usr_args_schema.model_validate(u) for u in usr_args_batch_
        ]
        val_in_args_batch_ = [
            self._in_args_type_adapter.validate_python(inp) for inp in in_args_batch_
        ]

        formatted_in_args_batch = [
            self._format_in_args(
                usr_args=val_usr_args, in_args=val_in_args, batch_idx=i, ctx=ctx
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
        usr_args: UserRunArgs | None = None,
        in_args_batch: Sequence[InT] | None = None,
        entry_point: bool = False,
        ctx: RunContextWrapper[CtxT] | None = None,
    ) -> Sequence[UserMessage]:
        # 1) Direct user input (e.g. chat input)
        if chat_inputs is not None or entry_point:
            """
            * If chat inputs are provided, use them instead of the predefined
                input prompt template
            * In a multi-agent system, the predefined input prompt is used to
                construct agent inputs using the combination of received
                and user arguments.
                However, the first agent run (entry point) has no received
                messages, so we use the chat inputs directly, if provided.
            """
            if isinstance(chat_inputs, LLMPrompt):
                return self._usr_messages_from_text(chat_inputs)

            if isinstance(chat_inputs, Sequence) and chat_inputs:
                return self._usr_messages_from_content_parts(chat_inputs)

        # 2) No input prompt template + received args → raw JSON messages
        if self.in_prompt is None and in_args_batch:
            return self._usr_messages_from_in_args(in_args_batch)

        # 3) Input prompt template + any args → batch & format
        if self.in_prompt is not None:
            if in_args_batch and not isinstance(in_args_batch[0], BaseModel):
                raise TypeError(
                    "Cannot use the input prompt template with "
                    "non-BaseModel received arguments."
                )
            return self._usr_messages_from_prompt_template(
                in_prompt=self.in_prompt,
                usr_args=usr_args,
                in_args_batch=in_args_batch,
                ctx=ctx,
            )

        return []

    def _make_batched(
        self,
        usr_args: UserRunArgs | None = None,
        in_args_batch: Sequence[InT] | None = None,
    ) -> tuple[Sequence[LLMPromptArgs | DummySchema], Sequence[InT | DummySchema]]:
        usr_args_batch_ = (
            usr_args if isinstance(usr_args, list) else [usr_args or DummySchema()]
        )
        in_args_batch_ = in_args_batch or [DummySchema()]

        # Broadcast singleton → match lengths
        if len(usr_args_batch_) == 1 and len(in_args_batch_) > 1:
            usr_args_batch_ = [deepcopy(usr_args_batch_[0]) for _ in in_args_batch_]
        if len(in_args_batch_) == 1 and len(usr_args_batch_) > 1:
            in_args_batch_ = [deepcopy(in_args_batch_[0]) for _ in usr_args_batch_]
        if len(usr_args_batch_) != len(in_args_batch_):
            raise ValueError("User args and received args must have the same length")

        return usr_args_batch_, in_args_batch_
