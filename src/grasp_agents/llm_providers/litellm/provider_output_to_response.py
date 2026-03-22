from datetime import UTC, datetime
from typing import Any

from openai.types.chat.chat_completion import (
    ChoiceLogprobs as ChatCompletionChoiceLogprobs,
)
from openai.types.chat.chat_completion import ChoiceLogprobs as CompletionLogprobs
from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_status import ResponseStatus

from grasp_agents.llm_providers.openai_completions.logprob_converters import (
    convert_logprobs,
)
from grasp_agents.llm_providers.openai_completions.provider_output_to_response import (
    convert_usage,
)
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
    prefixed_id,
)
from grasp_agents.types.response import Response, ResponseUsage
from litellm.litellm_core_utils.prompt_templates.factory import (
    _get_thought_signature_from_tool,  # type: ignore
)
from litellm.types.llms.openai import (
    ChatCompletionAnnotation as LiteLLMChatCompletionAnnotation,
)
from litellm.types.llms.openai import (
    ChatCompletionRedactedThinkingBlock as LiteLLMChatCompletionRedactedThinkingBlock,
)
from litellm.types.llms.openai import (
    ChatCompletionThinkingBlock as LiteLLMChatCompletionThinkingBlock,
)
from litellm.types.llms.openai import OpenAIChatCompletionFinishReason
from litellm.types.utils import Choices as LiteLLMChoice
from litellm.types.utils import Message as LiteLLMChatCompletionMessage
from litellm.types.utils import ModelResponse as LiteLLMCompletion
from litellm.types.utils import Usage as LiteLLMUsage

from .utils import validate_completion

LiteLLMThinkingBlock = (
    LiteLLMChatCompletionThinkingBlock | LiteLLMChatCompletionRedactedThinkingBlock
)


def _litellm_chat_completion_to_items(
    raw_message: LiteLLMChatCompletionMessage,
    output_message_status: ItemStatus,
    model: str,
    raw_logprobs: ChatCompletionChoiceLogprobs | None = None,
) -> list[OutputItem]:
    output_items: list[OutputItem] = []

    output_message, message_thought_sigs = _extract_output_message_item(
        raw_message=raw_message,
        raw_logprobs=raw_logprobs,
        status=output_message_status,
    )

    tool_call_items, tool_thought_sigs = _extract_tool_call_items(
        raw_message=raw_message, model=model
    )

    reasoning_items = _extract_reasoning_items(
        raw_message=raw_message,
        thought_sigs=message_thought_sigs + tool_thought_sigs,
    )

    output_items.extend(reasoning_items)
    if output_message is not None:
        output_items.append(output_message)
    output_items.extend(tool_call_items)

    return output_items


def convert_annotations(
    raw_annotations: list[LiteLLMChatCompletionAnnotation],
) -> list[UrlCitation]:
    """Convert LiteLLM ChatCompletionAnnotation TypedDicts → UrlCitation."""
    result: list[UrlCitation] = []
    for ann in raw_annotations:
        if "url_citation" in ann:
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
                result.append(
                    UrlCitation(
                        end_index=end_index,
                        start_index=start_index,
                        title=title,
                        url=url,
                    )
                )

    return result


def _extract_output_message_item(
    raw_message: LiteLLMChatCompletionMessage,
    status: ItemStatus,
    raw_logprobs: ChatCompletionChoiceLogprobs | None = None,
) -> tuple[OutputMessageItem | None, list[str]]:
    content_parts: list[OutputMessagePart] = []

    logprobs = convert_logprobs(raw_logprobs) if raw_logprobs is not None else None
    citations = convert_annotations(getattr(raw_message, "annotations", []))

    if raw_message.content:
        content_parts.append(
            OutputMessageText(
                text=raw_message.content, citations=citations, logprobs=logprobs
            )
        )

    refusal: str | None = getattr(raw_message, "refusal", None)
    # if refusal is None and finish_reason == "guardrail_intervened":
    #     refusal = "guardrail_intervened"
    if refusal:
        content_parts.append(OutputMessageRefusal(refusal=refusal))

    provider_specific_fields = raw_message.provider_specific_fields
    thought_sigs = (provider_specific_fields or {}).get("thought_signatures", [])

    return (
        OutputMessageItem(
            status=status,
            content_parts=content_parts,
            provider_specific_fields=provider_specific_fields,
        )
        if content_parts
        else None
    ), thought_sigs


def _extract_tool_call_items(
    raw_message: LiteLLMChatCompletionMessage, model: str
) -> tuple[list[FunctionToolCallItem], list[str]]:
    tool_call_items: list[FunctionToolCallItem] = []
    thought_sigs: list[str] = []

    if raw_message.tool_calls:
        for tc in raw_message.tool_calls:
            if tc.function.name is not None:  # can be None for chunks
                tc_dict = tc.model_dump()
                provider_specific_fields = tc_dict.get("provider_specific_fields")

                tc_signature = _get_thought_signature_from_tool(
                    tool=tc_dict, model=model
                )

                tool_call_items.append(
                    FunctionToolCallItem(
                        call_id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                        status="completed",
                        provider_specific_fields=provider_specific_fields,
                    )
                )
                if tc_signature:
                    thought_sigs.append(tc_signature)

    return tool_call_items, thought_sigs


def _extract_reasoning_items(
    raw_message: LiteLLMChatCompletionMessage,
    thought_sigs: list[str] | None = None,
) -> list[ReasoningItem]:
    reasoning_items: list[ReasoningItem] = []

    thinking_blocks: list[LiteLLMThinkingBlock] = getattr(
        raw_message, "thinking_blocks", []
    )
    for block in thinking_blocks:
        reasoning_items.append(ReasoningItem.from_thinking_block(block))

    reasoning_content: str | None = getattr(raw_message, "reasoning_content", None)

    if not reasoning_items and reasoning_content is not None:
        reasoning_items.append(
            ReasoningItem.from_reasoning_content(
                reasoning_content,
                encrypted_content=thought_sigs[-1] if thought_sigs else None,
            )
        )

    return reasoning_items


def provider_output_to_response(provider_output: LiteLLMCompletion) -> Response:
    validate_completion(provider_output)

    # Completion-level fields

    response_id = provider_output.id or prefixed_id("resp")
    created_at = provider_output.created or datetime.now(UTC).timestamp()
    model = provider_output.model or "unspecified-model"
    _ = provider_output.system_fingerprint

    hidden_params: dict[str, Any] = provider_output._hidden_params  # type: ignore # noqa: SLF001
    response_headers: dict[str, Any] = provider_output._response_headers  # type: ignore # noqa: SLF001
    # request_id = provider_output._request_id

    response_ms: float | None = getattr(provider_output, "_response_ms", None)

    raw_usage: LiteLLMUsage | None = getattr(provider_output, "usage", None)
    usage: ResponseUsage | None = None
    if raw_usage:
        usage = convert_usage(raw_usage)
        usage.cost = hidden_params.get("response_cost")

    # Choice-level fields

    raw_choice = provider_output.choices[0]
    assert isinstance(
        raw_choice, LiteLLMChoice
    )  # validate_completion should have already checked this
    finish_reason: OpenAIChatCompletionFinishReason | None = raw_choice.finish_reason

    raw_logprobs = getattr(raw_choice, "logprobs", None)
    if not isinstance(raw_logprobs, CompletionLogprobs):
        raw_logprobs = None

    incomplete_details: IncompleteDetails | None = None
    status: ResponseStatus = "completed"

    if finish_reason == "length":
        status = "incomplete"
        incomplete_details = IncompleteDetails(reason="max_output_tokens")

    elif finish_reason == "content_filter":
        status = "incomplete"

        incomplete_details = IncompleteDetails(reason="content_filter")

    elif finish_reason == "guardrail_intervened":
        status = "incomplete"

    # Message-level fields

    output_items = _litellm_chat_completion_to_items(
        raw_message=raw_choice.message,
        raw_logprobs=raw_logprobs,
        model=model,
        output_message_status=status,
    )

    return Response(
        id=response_id,
        created_at=created_at,
        output_items=output_items,
        usage_with_cost=usage,
        error=None,
        metadata=None,
        incomplete_details=incomplete_details,
        status=status,
        model=model,
        response_ms=response_ms,
        provider_specific_fields=raw_choice.provider_specific_fields,
        hidden_params=hidden_params,
        response_headers=response_headers,
    )
