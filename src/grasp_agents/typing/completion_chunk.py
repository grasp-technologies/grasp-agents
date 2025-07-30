import time
from collections import defaultdict
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from litellm import ChatCompletionAnnotation as LiteLLMAnnotation
from litellm.types.utils import ChoiceLogprobs as LiteLLMChoiceLogprobs
from openai.types.chat.chat_completion import (
    ChoiceLogprobs as OpenAIChoiceLogprobs,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceLogprobs as OpenAIChunkChoiceLogprobs,
)
from openai.types.chat.chat_completion_token_logprob import (
    ChatCompletionTokenLogprob as OpenAITokenLogprob,
)
from pydantic import BaseModel, Field, ValidationError, field_validator

from ..errors import CombineCompletionChunksError
from .completion import Completion, CompletionChoice, FinishReason, Usage
from .message import (
    AssistantMessage,
    RedactedThinkingBlock,
    Role,
    ThinkingBlock,
    ToolCall,
)


class CompletionChunkDeltaToolCall(BaseModel):
    id: str | None
    index: int
    tool_name: str | None
    tool_arguments: str | None


class CompletionChunkChoiceDelta(BaseModel):
    content: str | None = None
    refusal: str | None = None
    role: Role | None = None
    tool_calls: list[CompletionChunkDeltaToolCall] | None = None
    reasoning_content: str | None = None
    thinking_blocks: list[ThinkingBlock | RedactedThinkingBlock] | None = None
    annotations: list[LiteLLMAnnotation] | None = None
    provider_specific_fields: dict[str, Any] | None = None

    @property
    def thinking_delta(self) -> "CompletionChunkChoiceDelta | None":
        return (
            CompletionChunkChoiceDelta(
                reasoning_content=self.reasoning_content,
                thinking_blocks=self.thinking_blocks,
                role=self.role,
                provider_specific_fields=self.provider_specific_fields,
            )
            if self.reasoning_content or self.thinking_blocks
            else None
        )

    @property
    def tool_call_deltas(self) -> "list[CompletionChunkChoiceDelta] | None":
        return (
            [
                CompletionChunkChoiceDelta(
                    tool_calls=[tool_call],
                    role=self.role,
                    provider_specific_fields=self.provider_specific_fields,
                )
                for tool_call in self.tool_calls
            ]
            if self.tool_calls
            else None
        )

    @property
    def response_delta(self) -> "CompletionChunkChoiceDelta | None":
        return (
            CompletionChunkChoiceDelta(
                content=self.content,
                role=self.role,
                provider_specific_fields=self.provider_specific_fields,
            )
            if self.content
            else None
        )

    @property
    def annotations_delta(self) -> "CompletionChunkChoiceDelta | None":
        return (
            CompletionChunkChoiceDelta(
                annotations=self.annotations,
                role=self.role,
                provider_specific_fields=self.provider_specific_fields,
            )
            if self.annotations
            else None
        )

    @property
    def refusal_delta(self) -> "CompletionChunkChoiceDelta | None":
        return (
            CompletionChunkChoiceDelta(
                refusal=self.refusal,
                role=self.role,
                provider_specific_fields=self.provider_specific_fields,
            )
            if self.refusal
            else None
        )


class CompletionChunkChoice(BaseModel):
    delta: CompletionChunkChoiceDelta
    finish_reason: FinishReason | None
    index: int
    logprobs: OpenAIChunkChoiceLogprobs | LiteLLMChoiceLogprobs | Any | None = None


class CompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str | None
    name: str | None = None
    system_fingerprint: str | None = None
    choices: list[CompletionChunkChoice]
    usage: Usage | None = None
    # LiteLLM-specific fields
    provider_specific_fields: dict[str, Any] | None = None
    response_ms: float | None = None
    hidden_params: dict[str, Any] | None = None

    def split_into_specialized(
        self,
    ) -> "list[CompletionChunk]":
        if len(self.choices) != 1:
            raise ValidationError(
                "CompletionChunk must have exactly one choice for specialization."
            )
        delta = self.choices[0].delta

        specialized_chunks: list[CompletionChunk] = []

        thinking_delta = delta.thinking_delta
        tool_call_deltas = delta.tool_call_deltas
        response_delta = delta.response_delta
        annotations_delta = delta.annotations_delta
        refusal_delta = delta.refusal_delta

        if thinking_delta is not None:
            new_choice = self.choices[0].model_copy(update={"delta": thinking_delta})
            new_chunk = self.model_copy(update={"choices": [new_choice]})
            specialized_chunks.append(
                ThinkingChunk.model_validate(new_chunk.model_dump())
            )

        if tool_call_deltas:
            for delta_tool_call in tool_call_deltas:
                new_choice = self.choices[0].model_copy(
                    update={"delta": delta_tool_call}
                )
                new_chunk = self.model_copy(update={"choices": [new_choice]})
                specialized_chunks.append(
                    ToolCallChunk.model_validate(new_chunk.model_dump())
                )

        if response_delta is not None:
            new_choice = self.choices[0].model_copy(update={"delta": response_delta})
            new_chunk = self.model_copy(update={"choices": [new_choice]})
            specialized_chunks.append(
                ResponseChunk.model_validate(new_chunk.model_dump())
            )

        if annotations_delta is not None:
            new_choice = self.choices[0].model_copy(update={"delta": annotations_delta})
            new_chunk = self.model_copy(update={"choices": [new_choice]})
            specialized_chunks.append(
                AnnotationsChunk.model_validate(new_chunk.model_dump())
            )

        if refusal_delta is not None:
            new_choice = self.choices[0].model_copy(update={"delta": refusal_delta})
            new_chunk = self.model_copy(update={"choices": [new_choice]})
            specialized_chunks.append(
                RefusalChunk.model_validate(new_chunk.model_dump())
            )

        return specialized_chunks


class ResponseChunk(CompletionChunk):
    @field_validator("choices")
    @classmethod
    def validate_response_chunk(
        cls, choices: list[CompletionChunkChoice]
    ) -> list[CompletionChunkChoice]:
        if len(choices) != 1:
            raise ValidationError("ResponseChunk must have exactly one choice.")

        delta = choices[0].delta

        if not delta.content:
            raise ValidationError("ResponseChunk must have content in deltas.")

        if (
            delta.reasoning_content is not None
            or delta.thinking_blocks is not None
            or delta.tool_calls is not None
            or delta.refusal is not None
            or delta.annotations is not None
        ):
            raise ValidationError(
                "ResponseChunk should not have reasoning content, thinking blocks, "
                "tool calls, refusal, or annotations in deltas."
            )

        return choices

    @property
    def response(self) -> str:
        assert self.choices[0].delta.content
        return self.choices[0].delta.content


class ThinkingChunk(CompletionChunk):
    @field_validator("choices")
    @classmethod
    def validate_thinking_chunk(
        cls, choices: list[CompletionChunkChoice]
    ) -> list[CompletionChunkChoice]:
        if len(choices) != 1:
            raise ValidationError("ThinkingChunk must have exactly one choice.")

        delta = choices[0].delta

        if not (delta.thinking_blocks or delta.reasoning_content):
            raise ValidationError(
                "ThinkingChunk must have reasoning content or "
                "at least one thinking block."
            )
        if (
            delta.content is not None
            or delta.tool_calls is not None
            or delta.refusal is not None
            or delta.annotations is not None
        ):
            raise ValidationError(
                "ThinkingChunk should not have content, tool calls, "
                "refusal, or annotations in deltas."
            )

        return choices

    @property
    def thinking(self) -> str | list[ThinkingBlock | RedactedThinkingBlock]:
        delta = self.choices[0].delta
        if delta.reasoning_content:
            return delta.reasoning_content
        if delta.thinking_blocks:
            return delta.thinking_blocks
        raise ValueError("ThinkingChunk has no reasoning_content or thinking_blocks")


class ToolCallChunk(CompletionChunk):
    @field_validator("choices")
    @classmethod
    def validate_tool_call_chunk(
        cls, choices: list[CompletionChunkChoice]
    ) -> list[CompletionChunkChoice]:
        if len(choices) != 1:
            raise ValidationError("ToolCallChunk must have exactly one choice.")

        delta = choices[0].delta

        if not delta.tool_calls:
            raise ValidationError("ToolCallChunk must have tool calls in deltas.")
        if len(delta.tool_calls) != 1:
            raise ValidationError(
                "ToolCallChunk must have exactly one tool call in deltas."
            )

        if (
            delta.reasoning_content is not None
            or delta.thinking_blocks is not None
            or delta.content is not None
            or delta.refusal is not None
            or delta.annotations is not None
        ):
            raise ValidationError(
                "ToolCallChunk should not have reasoning content, thinking blocks, "
                "content, refusal, or annotations in deltas."
            )

        return choices

    @property
    def tool_call(self) -> CompletionChunkDeltaToolCall:
        assert self.choices[0].delta.tool_calls is not None
        return self.choices[0].delta.tool_calls[0]


class AnnotationsChunk(CompletionChunk):
    @field_validator("choices")
    @classmethod
    def validate_annotations_chunk(
        cls, choices: list[CompletionChunkChoice]
    ) -> list[CompletionChunkChoice]:
        if len(choices) != 1:
            raise ValidationError("AnnotationsChunk must have exactly one choice.")

        delta = choices[0].delta

        if not delta.annotations:
            raise ValidationError("AnnotationsChunk must have annotations in deltas.")

        if (
            delta.reasoning_content is not None
            or delta.thinking_blocks is not None
            or delta.content is not None
            or delta.tool_calls is not None
            or delta.refusal is not None
        ):
            raise ValidationError(
                "AnnotationsChunk should not have reasoning content, thinking blocks, "
                "content, tool calls, or refusal in deltas."
            )

        return choices

    @property
    def annotations(self) -> list[LiteLLMAnnotation]:
        assert self.choices[0].delta.annotations is not None
        return self.choices[0].delta.annotations


class RefusalChunk(CompletionChunk):
    @field_validator("choices")
    @classmethod
    def validate_refusal_chunk(
        cls, choices: list[CompletionChunkChoice]
    ) -> list[CompletionChunkChoice]:
        if len(choices) != 1:
            raise ValidationError("RefusalChunk must have exactly one choice.")

        delta = choices[0].delta

        if not delta.refusal:
            raise ValidationError("RefusalChunk must have refusal in deltas.")

        if (
            delta.reasoning_content is not None
            or delta.thinking_blocks is not None
            or delta.content is not None
            or delta.tool_calls is not None
            or delta.annotations is not None
        ):
            raise ValidationError(
                "RefusalChunk should not have reasoning content, thinking blocks, "
                "content, tool calls, or annotations in deltas."
            )

        return choices

    @property
    def refusal(self) -> str | None:
        return self.choices[0].delta.refusal


def combine_completion_chunks(chunks: list[CompletionChunk]) -> Completion:
    if not chunks:
        raise CombineCompletionChunksError(
            "Cannot combine an empty list of completion chunks."
        )

    model_list = {chunk.model for chunk in chunks}
    if len(model_list) > 1:
        raise CombineCompletionChunksError("All chunks must have the same model.")
    model = model_list.pop()

    name_list = {chunk.name for chunk in chunks}
    if len(name_list) > 1:
        raise CombineCompletionChunksError("All chunks must have the same name.")
    name = name_list.pop()

    system_fingerprints_list = {chunk.system_fingerprint for chunk in chunks}
    if len(system_fingerprints_list) > 1:
        raise CombineCompletionChunksError(
            "All chunks must have the same system fingerprint."
        )
    system_fingerprint = system_fingerprints_list.pop()

    created_list = [chunk.created for chunk in chunks]
    created = max(created_list)

    # Usage is found in the last completion chunk if requested
    usage = chunks[-1].usage

    logp_contents_per_choice: defaultdict[int, list[OpenAITokenLogprob]] = defaultdict(
        list
    )
    logp_refusals_per_choice: defaultdict[int, list[OpenAITokenLogprob]] = defaultdict(
        list
    )
    logprobs_per_choice: defaultdict[int, OpenAIChoiceLogprobs | None] = defaultdict(
        lambda: None
    )
    thinking_blocks_per_choice: defaultdict[
        int, list[ThinkingBlock | RedactedThinkingBlock]
    ] = defaultdict(list)
    annotations_per_choice: defaultdict[int, list[LiteLLMAnnotation]] = defaultdict(
        list
    )

    finish_reasons_per_choice: defaultdict[int, FinishReason | None] = defaultdict(
        lambda: None
    )

    contents_per_choice: defaultdict[int, str] = defaultdict(lambda: "")
    reasoning_contents_per_choice: defaultdict[int, str] = defaultdict(lambda: "")
    refusals_per_choice: defaultdict[int, str] = defaultdict(lambda: "")

    tool_calls_per_choice: defaultdict[
        int, Sequence[CompletionChunkDeltaToolCall] | None
    ] = defaultdict(lambda: None)

    messages_per_choice: dict[int, AssistantMessage] = {}

    for chunk in chunks:
        for choice in chunk.choices:
            index = choice.index

            # Concatenate content and refusal tokens for each choice
            contents_per_choice[index] += choice.delta.content or ""
            reasoning_contents_per_choice[index] += choice.delta.reasoning_content or ""
            refusals_per_choice[index] += choice.delta.refusal or ""

            # Concatenate logprobs for content and refusal tokens for each choice
            if choice.logprobs is not None:
                logp_contents_per_choice[index].extend(choice.logprobs.content or [])  # type: ignore
                logp_refusals_per_choice[index].extend(choice.logprobs.refusal or [])  # type: ignore
                thinking_blocks_per_choice[index].extend(
                    choice.delta.thinking_blocks or []
                )
                annotations_per_choice[index].extend(choice.delta.annotations or [])

            # Take the last finish reason for each choice
            finish_reasons_per_choice[index] = choice.finish_reason

            # Tool calls should be in the last chunk for each choice
            tool_calls_per_choice[index] = choice.delta.tool_calls

    for index in finish_reasons_per_choice:
        tool_calls: list[ToolCall] = []
        if tool_calls_per_choice[index] is not None:
            for _tool_call in tool_calls_per_choice[index]:  # type: ignore
                if (
                    _tool_call.id is None
                    or _tool_call.tool_name is None
                    or _tool_call.tool_arguments is None
                ):
                    raise CombineCompletionChunksError(
                        "Completion chunk tool calls must have id, tool_name, "
                        "and tool_arguments set."
                    )
                tool_calls.append(
                    ToolCall(
                        id=_tool_call.id,
                        tool_name=_tool_call.tool_name,
                        tool_arguments=_tool_call.tool_arguments,
                    )
                )

        messages_per_choice[index] = AssistantMessage(
            name=name,
            content=contents_per_choice[index] or "<empty>",
            reasoning_content=(reasoning_contents_per_choice[index] or None),
            thinking_blocks=(thinking_blocks_per_choice[index] or None),
            annotations=(annotations_per_choice[index] or None),
            refusal=(refusals_per_choice[index] or None),
            tool_calls=(tool_calls or None),
        )

        if logp_contents_per_choice[index] or logp_refusals_per_choice[index]:
            logprobs_per_choice[index] = OpenAIChoiceLogprobs(
                content=logp_contents_per_choice[index],
                refusal=logp_refusals_per_choice[index],
            )

    choices = [
        CompletionChoice(
            index=index,
            message=message,
            finish_reason=finish_reasons_per_choice[index],
            logprobs=logprobs_per_choice[index],
        )
        for index, message in messages_per_choice.items()
    ]

    return Completion(
        model=model,
        name=name,
        created=created,
        system_fingerprint=system_fingerprint,
        choices=choices,
        usage=usage,
    )
