"""Convert OpenAI Chat Completions message fields → internal item types."""

from typing import Any

from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion import (
    ChoiceLogprobs as ChatCompletionChoiceLogprobs,
)
from openai.types.chat.chat_completion_message import (
    Annotation as ChatCompletionAnnotation,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)
from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_status import ResponseStatus
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)
from pydantic import TypeAdapter, ValidationError

from grasp_agents.types.content import (
    OutputMessagePart,
    OutputMessageRefusal,
    OutputMessageText,
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
from grasp_agents.types.response import Response, ResponseUsage

from .logprob_converters import convert_logprobs
from .utils import validate_completion

_REASONING_DETAILS_ADAPTER: TypeAdapter[OpenRouterReasoningDetails] = TypeAdapter(
    OpenRouterReasoningDetails
)


def _chat_completion_to_items(
    raw_message: ChatCompletionMessage,
    output_message_status: ItemStatus,
    raw_logprobs: ChatCompletionChoiceLogprobs | None = None,
) -> list[OutputItem]:
    output_items: list[OutputItem] = []

    output_items.extend(_extract_reasoning_items(raw_message=raw_message))

    output_message = _extract_output_message_item(
        raw_message=raw_message, raw_logprobs=raw_logprobs, status=output_message_status
    )
    if output_message is not None:
        output_items.append(output_message)

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
    content_parts: list[OutputMessagePart] = []

    logprobs = convert_logprobs(raw_logprobs) if raw_logprobs is not None else None

    citations = convert_annotations(raw_message.annotations or [])

    if raw_message.content:
        content_parts.append(
            OutputMessageText(
                text=raw_message.content, citations=citations, logprobs=logprobs
            )
        )

    if raw_message.refusal:
        content_parts.append(OutputMessageRefusal(refusal=raw_message.refusal))

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


def convert_usage(raw_usage: CompletionUsage) -> ResponseUsage:
    cached_tokens = 0
    reasoning_tokens = 0

    if raw_usage.prompt_tokens_details is not None:
        cached_tokens = raw_usage.prompt_tokens_details.cached_tokens or 0

    if raw_usage.completion_tokens_details is not None:
        reasoning_tokens = raw_usage.completion_tokens_details.reasoning_tokens or 0

    return ResponseUsage(
        input_tokens=raw_usage.prompt_tokens,
        output_tokens=raw_usage.completion_tokens,
        total_tokens=raw_usage.total_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=cached_tokens),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=reasoning_tokens),
    )


def provider_output_to_response(provider_output: ChatCompletion) -> Response:
    """Convert an OpenAI Chat Completion → internal Response."""
    validate_completion(provider_output)

    # Completion-level fields

    _ = provider_output.system_fingerprint

    usage: ResponseUsage | None = None
    raw_usage = provider_output.usage
    if raw_usage:
        usage = convert_usage(raw_usage)

    # Choice-level fields

    raw_choice = provider_output.choices[0]
    finish_reason = raw_choice.finish_reason
    raw_logprobs = raw_choice.logprobs

    incomplete_details: IncompleteDetails | None = None
    status: ResponseStatus = "completed"

    if finish_reason == "length":
        status = "incomplete"
        incomplete_details = IncompleteDetails(reason="max_output_tokens")
    elif finish_reason == "content_filter":
        status = "incomplete"
        incomplete_details = IncompleteDetails(reason="content_filter")

    # Message-level fields

    raw_message = raw_choice.message

    output_items = _chat_completion_to_items(
        raw_message=raw_message,
        raw_logprobs=raw_logprobs,
        output_message_status=status,
    )

    return Response(
        id=provider_output.id,
        created_at=float(provider_output.created),
        model=provider_output.model,
        output_items=output_items,
        usage_with_cost=usage,
        status=status,
        incomplete_details=incomplete_details,
        service_tier=provider_output.service_tier,
    )
