import time
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
from .completion import Completion, FinishReason, Usage
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


class CompletionChunkDelta(BaseModel):
    content: str | None = None
    refusal: str | None = None
    role: Role | None = None
    tool_calls: list[CompletionChunkDeltaToolCall] | None = None
    reasoning_content: str | None = None
    thinking_blocks: list[ThinkingBlock | RedactedThinkingBlock] | None = None
    annotations: list[LiteLLMAnnotation] | None = None
    provider_specific_fields: dict[str, Any] | None = None

    @property
    def thinking_delta(self) -> "CompletionChunkDelta | None":
        return (
            CompletionChunkDelta(
                reasoning_content=self.reasoning_content,
                thinking_blocks=self.thinking_blocks,
                role=self.role,
                provider_specific_fields=self.provider_specific_fields,
            )
            if self.reasoning_content or self.thinking_blocks
            else None
        )

    @property
    def tool_call_deltas(self) -> "list[CompletionChunkDelta] | None":
        return (
            [
                CompletionChunkDelta(
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
    def response_delta(self) -> "CompletionChunkDelta | None":
        return (
            CompletionChunkDelta(
                content=self.content,
                role=self.role,
                provider_specific_fields=self.provider_specific_fields,
            )
            if self.content
            else None
        )

    @property
    def annotations_delta(self) -> "CompletionChunkDelta | None":
        return (
            CompletionChunkDelta(
                annotations=self.annotations,
                role=self.role,
                provider_specific_fields=self.provider_specific_fields,
            )
            if self.annotations
            else None
        )

    @property
    def refusal_delta(self) -> "CompletionChunkDelta | None":
        return (
            CompletionChunkDelta(
                refusal=self.refusal,
                role=self.role,
                provider_specific_fields=self.provider_specific_fields,
            )
            if self.refusal
            else None
        )


class CompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str | None
    name: str | None = None
    system_fingerprint: str | None = None
    usage: Usage | None = None

    # Removed choices to add delta directly to CompletionChunk
    delta: CompletionChunkDelta
    finish_reason: FinishReason | None
    logprobs: OpenAIChunkChoiceLogprobs | LiteLLMChoiceLogprobs | Any | None = None

    # LiteLLM-specific fields
    provider_specific_fields: dict[str, Any] | None = None
    response_ms: float | None = None
    hidden_params: dict[str, Any] | None = None

    def split_into_specialized(
        self,
    ) -> "list[CompletionChunk]":
        delta = self.delta

        specialized_chunks: list[CompletionChunk] = []

        thinking_delta = delta.thinking_delta
        tool_call_deltas = delta.tool_call_deltas
        response_delta = delta.response_delta
        annotations_delta = delta.annotations_delta
        refusal_delta = delta.refusal_delta

        if thinking_delta is not None:
            new_chunk = self.model_copy(update={"delta": thinking_delta})
            specialized_chunks.append(
                ThinkingChunk.model_validate(new_chunk.model_dump())
            )

        if tool_call_deltas:
            for delta_tool_call in tool_call_deltas:
                new_chunk = self.model_copy(update={"delta": delta_tool_call})
                specialized_chunks.append(
                    ToolCallChunk.model_validate(new_chunk.model_dump())
                )

        if response_delta is not None:
            new_chunk = self.model_copy(update={"delta": response_delta})
            specialized_chunks.append(
                ResponseChunk.model_validate(new_chunk.model_dump())
            )

        if annotations_delta is not None:
            new_chunk = self.model_copy(update={"delta": annotations_delta})
            specialized_chunks.append(
                AnnotationsChunk.model_validate(new_chunk.model_dump())
            )

        if refusal_delta is not None:
            new_chunk = self.model_copy(update={"delta": refusal_delta})
            specialized_chunks.append(
                RefusalChunk.model_validate(new_chunk.model_dump())
            )

        return specialized_chunks


class ResponseChunk(CompletionChunk):
    @field_validator("delta")
    @classmethod
    def validate_delta(cls, delta: CompletionChunkDelta) -> CompletionChunkDelta:
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

        return delta

    @property
    def response(self) -> str:
        assert self.delta.content
        return self.delta.content


class ThinkingChunk(CompletionChunk):
    @field_validator("delta")
    @classmethod
    def validate_delta(cls, delta: CompletionChunkDelta) -> CompletionChunkDelta:
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

        return delta

    @property
    def thinking(self) -> str | list[ThinkingBlock | RedactedThinkingBlock]:
        delta = self.delta
        if delta.reasoning_content:
            return delta.reasoning_content
        if delta.thinking_blocks:
            return delta.thinking_blocks
        raise ValueError("ThinkingChunk has no reasoning_content or thinking_blocks")


class ToolCallChunk(CompletionChunk):
    @field_validator("delta")
    @classmethod
    def validate_delta(cls, delta: CompletionChunkDelta) -> CompletionChunkDelta:
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

        return delta

    @property
    def tool_call(self) -> CompletionChunkDeltaToolCall:
        assert self.delta.tool_calls is not None
        return self.delta.tool_calls[0]


class AnnotationsChunk(CompletionChunk):
    @field_validator("delta")
    @classmethod
    def validate_annotations_chunk(
        cls, delta: CompletionChunkDelta
    ) -> CompletionChunkDelta:
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

        return delta

    @property
    def annotations(self) -> list[LiteLLMAnnotation]:
        assert self.delta.annotations is not None
        return self.delta.annotations


class RefusalChunk(CompletionChunk):
    @field_validator("delta")
    @classmethod
    def validate_refusal_chunk(
        cls, delta: CompletionChunkDelta
    ) -> CompletionChunkDelta:
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

        return delta

    @property
    def refusal(self) -> str | None:
        return self.delta.refusal


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

    content: str = ""
    reasoning_content: str = ""
    refusal: str = ""

    delta_tool_calls: list[CompletionChunkDeltaToolCall] | None = None
    logp_contents: list[OpenAITokenLogprob] = []
    logp_refusals: list[OpenAITokenLogprob] = []
    thinking_blocks: list[ThinkingBlock | RedactedThinkingBlock] = []
    annotations: list[LiteLLMAnnotation] = []

    logprobs: OpenAIChoiceLogprobs | None = None
    finish_reason: FinishReason | None = None
    usage: Usage | None = None

    if chunks:
        last_chunk = chunks[-1]

        # Usage is found in the last completion chunk if requested
        usage = last_chunk.usage

        # Take the last finish reason
        finish_reason = last_chunk.finish_reason

        # NOTE: this won't work because tool calls need to be merged across chunks
        delta_tool_calls = last_chunk.delta.tool_calls

    for chunk in chunks:
        # Concatenate content and refusal tokens
        content += chunk.delta.content or ""
        reasoning_content += chunk.delta.reasoning_content or ""
        refusal += chunk.delta.refusal or ""

        # Concatenate logprobs for content and refusal tokens
        if chunk.logprobs is not None:
            logp_contents.extend(chunk.logprobs.content or [])  # type: ignore
            logp_refusals.extend(chunk.logprobs.refusal or [])  # type: ignore
            thinking_blocks.extend(chunk.delta.thinking_blocks or [])
            annotations.extend(chunk.delta.annotations or [])

    tool_calls: list[ToolCall] | None = []
    if delta_tool_calls is not None:
        for _tool_call in delta_tool_calls:
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

    message = AssistantMessage(
        name=name,
        content=content or "<empty>",
        reasoning_content=(reasoning_content or None),
        thinking_blocks=(thinking_blocks or None),
        annotations=(annotations or None),
        refusal=(refusal or None),
        tool_calls=(tool_calls or None),
    )

    if logp_contents or logp_refusals:
        logprobs = OpenAIChoiceLogprobs(content=logp_contents, refusal=logp_refusals)

    return Completion(
        model=model,
        name=name,
        created=created,
        system_fingerprint=system_fingerprint,
        message=message,
        finish_reason=finish_reason,
        logprobs=logprobs,
        usage=usage,
    )
