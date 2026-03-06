"""Convert OpenAI Chat Completions message fields → internal item types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openai.types.chat.chat_completion import (
    ChoiceLogprobs as ChatCompletionChoiceLogprobs,
)
from openai.types.chat.chat_completion_message import (
    Annotation as ChatCompletionAnnotation,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)
from pydantic import TypeAdapter, ValidationError

from grasp_agents.types.content import (
    OutputMessageContentPart,
    OutputRefusal,
    OutputTextContentPart,
    UrlCitation,
)
from grasp_agents.types.items import (
    FunctionToolCallItem,
    ItemStatus,
    OutputItem,
    OutputMessageItem,
    ReasoningItem,
)
from grasp_agents.types.reasoning import OpenRouterReasoningDetails

from .logprob_converters import convert_logprobs

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessage


_REASONING_DETAILS_ADAPTER: TypeAdapter[OpenRouterReasoningDetails] = TypeAdapter(
    OpenRouterReasoningDetails
)


def generated_message_to_items(
    raw_message: ChatCompletionMessage,
    output_message_status: ItemStatus,
    raw_logprobs: ChatCompletionChoiceLogprobs | None = None,
) -> list[OutputItem]:
    output_items: list[OutputItem] = []

    output_message = _extract_output_message_item(
        raw_message=raw_message, raw_logprobs=raw_logprobs, status=output_message_status
    )
    if output_message is not None:
        output_items.append(output_message)

    output_items.extend(_extract_reasoning_items(raw_message=raw_message))
    output_items.extend(_extract_tool_call_items(raw_message=raw_message))

    return output_items


def convert_annotations(
    raw_annotations: list[ChatCompletionAnnotation] | list[dict[str, Any]],
) -> list[UrlCitation]:
    """Convert raw annotation dicts or Pydantic models to typed Citation objects."""
    citations: list[UrlCitation] = []
    for ann in raw_annotations:
        if isinstance(ann, ChatCompletionAnnotation):
            citation = ann.url_citation
            citations.append(
                UrlCitation(
                    end_index=citation.end_index,
                    start_index=citation.start_index,
                    title=citation.title,
                    url=citation.url,
                )
            )
        elif "url_citation" in ann:
            uc = ann["url_citation"]
            end_index = uc.get("end_index")
            start_index = uc.get("start_index")
            title = uc.get("title")
            url = uc.get("url")
            if (
                end_index is not None
                and start_index is not None
                and title is not None
                and url is not None
            ):
                citations.append(
                    UrlCitation(
                        end_index=end_index,
                        start_index=start_index,
                        title=title,
                        url=url,
                    )
                )

    return citations


def _extract_output_message_item(
    raw_message: ChatCompletionMessage,
    status: ItemStatus,
    raw_logprobs: ChatCompletionChoiceLogprobs | None = None,
) -> OutputMessageItem | None:
    content_parts: list[OutputMessageContentPart] = []

    logprobs = convert_logprobs(raw_logprobs) if raw_logprobs is not None else None

    citations = convert_annotations(raw_message.annotations or [])

    if raw_message.content:
        content_parts.append(
            OutputTextContentPart(
                text=raw_message.content, citations=citations, logprobs=logprobs
            )
        )

    if raw_message.refusal:
        content_parts.append(OutputRefusal(refusal=raw_message.refusal))

    return (
        OutputMessageItem(status=status, content_parts=content_parts)
        if content_parts
        else None
    )


def _extract_reasoning_items(raw_message: ChatCompletionMessage) -> list[ReasoningItem]:
    """
    Try to extract reasoning from `message.reasoning_content`
    or by assuming the OpenRouter-specific format
    """
    reasoning_items: list[ReasoningItem] = []
    reasoning_details = getattr(raw_message, "reasoning_details", [])

    for block_dict in reasoning_details:
        try:
            block = _REASONING_DETAILS_ADAPTER.validate_python(block_dict)
        except ValidationError:
            continue

        reasoning_items.append(ReasoningItem.from_open_router_reasoning_details(block))

    if not reasoning_items:
        reasoning_content = getattr(raw_message, "reasoning_content", None) or getattr(
            raw_message, "reasoning", None
        )
        if reasoning_content is not None:
            reasoning_items.append(
                ReasoningItem.from_reasoning_content(reasoning_content)
            )

    return reasoning_items


def _extract_tool_call_items(
    raw_message: ChatCompletionMessage,
) -> list[FunctionToolCallItem]:
    output_items: list[FunctionToolCallItem] = []

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
            # TODO: handle other tool call type when supported by grasp-agents

    return output_items
