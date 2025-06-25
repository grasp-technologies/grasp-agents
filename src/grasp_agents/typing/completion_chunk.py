import time
from collections import defaultdict
from collections.abc import Sequence
from uuid import uuid4

from openai.types.chat.chat_completion_chunk import (
    ChoiceLogprobs as CompletionChunkChoiceLogprobs,
)
from openai.types.chat.chat_completion_token_logprob import (
    ChatCompletionTokenLogprob as CompletionTokenLogprob,
)
from pydantic import BaseModel, Field

from ..errors import CombineCompletionChunksError
from .completion import (
    Completion,
    CompletionChoice,
    CompletionChoiceLogprobs,
    FinishReason,
    Usage,
)
from .message import AssistantMessage, ToolCall


class CompletionChunkDeltaToolCall(BaseModel):
    id: str | None
    index: int
    tool_name: str | None
    tool_arguments: str | None


class CompletionChunkChoiceDelta(BaseModel):
    content: str | None = None
    refusal: str | None
    role: str | None
    tool_calls: list[CompletionChunkDeltaToolCall] | None


class CompletionChunkChoice(BaseModel):
    delta: CompletionChunkChoiceDelta
    finish_reason: FinishReason | None
    index: int
    logprobs: CompletionChunkChoiceLogprobs | None = None


class CompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    name: str | None = None
    system_fingerprint: str | None = None
    choices: list[CompletionChunkChoice]
    usage: Usage | None = None


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

    logp_contents_per_choice: defaultdict[int, list[CompletionTokenLogprob]] = (
        defaultdict(list)
    )
    logp_refusals_per_choice: defaultdict[int, list[CompletionTokenLogprob]] = (
        defaultdict(list)
    )
    logprobs_per_choice: defaultdict[int, CompletionChoiceLogprobs | None] = (
        defaultdict(lambda: None)
    )

    finish_reasons_per_choice: defaultdict[int, FinishReason | None] = defaultdict(
        lambda: None
    )

    contents_per_choice: defaultdict[int, str] = defaultdict(lambda: "")
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
            refusals_per_choice[index] += choice.delta.refusal or ""

            # Concatenate logprobs for content and refusal tokens for each choice
            if choice.logprobs is not None:
                logp_contents_per_choice[index].extend(choice.logprobs.content or [])
                logp_refusals_per_choice[index].extend(choice.logprobs.refusal or [])

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
            refusal=(refusals_per_choice[index] or None),
            tool_calls=(tool_calls or None),
        )

        if logp_contents_per_choice[index] or logp_refusals_per_choice[index]:
            logprobs_per_choice[index] = CompletionChoiceLogprobs(
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
