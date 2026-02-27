"""Convert OpenAI Chat Completions message fields → internal item types."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openai.types.chat.chat_completion import ChoiceLogprobs
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)
from openai.types.responses.response_output_text import (
    AnnotationURLCitation,
    Logprob,
    LogprobTopLogprob,
)

from ...typing.content import (
    OutputContent,
    OutputRefusalContent,
    OutputTextContent,
)
from ...typing.items import (
    FunctionToolCallItem,
    ItemStatus,
    OutputItem,
    OutputMessageItem,
)

if TYPE_CHECKING:
    from . import OpenAICompletionMessage


def from_openai_completions_message(
    raw_message: OpenAICompletionMessage,
    output_message_status: ItemStatus,
    raw_logprobs: ChoiceLogprobs | None = None,
    refusal: str | None = None,
) -> list[OutputItem]:
    output_items: list[OutputItem] = []

    # Output message item

    output_message = _convert_output_message(
        raw_message=raw_message,
        raw_logprobs=raw_logprobs,
        refusal=refusal,
        status=output_message_status,
    )
    if output_message is not None:
        output_items.append(output_message)

    # Tool call items

    if raw_message.tool_calls:
        for tc in raw_message.tool_calls:
            if isinstance(tc, ChatCompletionMessageFunctionToolCall):
                output_items.append(
                    FunctionToolCallItem(
                        call_id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                        status="completed",
                    )
                )

    return output_items


def _convert_output_message(
    raw_message: OpenAICompletionMessage,
    status: ItemStatus,
    raw_logprobs: ChoiceLogprobs | None = None,
    refusal: str | None = None,
) -> OutputMessageItem | None:
    content_parts: list[OutputContent] = []
    logprobs: list[Logprob] | None = None

    if raw_logprobs is not None and raw_logprobs.content:
        logprobs = []
        for lp in raw_logprobs.content:
            top_logprobs = [
                LogprobTopLogprob(
                    token=tlp.token,
                    bytes=tlp.bytes or list(tlp.token.encode("utf-8")),
                    logprob=tlp.logprob,
                )
                for tlp in lp.top_logprobs
            ]
            logprobs.append(
                Logprob(
                    token=lp.token,
                    bytes=lp.bytes or list(lp.token.encode("utf-8")),
                    logprob=lp.logprob,
                    top_logprobs=top_logprobs,
                )
            )

    annotations = [
        AnnotationURLCitation(type="url_citation", **ann.url_citation.model_dump())
        for ann in (raw_message.annotations or [])
        if ann.type == "url_citation"
    ]
    if raw_message.content:
        content_parts.append(
            OutputTextContent(
                text=raw_message.content,
                annotations=annotations,  # type: ignore[arg-type]
                logprobs=logprobs,
            )
        )

    if refusal:
        content_parts.append(OutputRefusalContent(refusal=refusal))

    return (
        OutputMessageItem(status=status, content_ext=content_parts)
        if content_parts
        else None
    )
