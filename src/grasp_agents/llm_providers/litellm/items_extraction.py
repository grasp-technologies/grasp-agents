from __future__ import annotations

from typing import TYPE_CHECKING

from openai.types.chat.chat_completion import (
    ChoiceLogprobs as ChatCompletionChoiceLogprobs,
)

from grasp_agents.llm_providers.openai_completions.logprob_converters import (
    convert_logprobs,
)
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

if TYPE_CHECKING:
    from litellm.types.utils import Message as LiteLLMChatCompletionMessage

LiteLLMThinkingBlock = (
    LiteLLMChatCompletionThinkingBlock | LiteLLMChatCompletionRedactedThinkingBlock
)


def generated_message_to_items(
    raw_message: LiteLLMChatCompletionMessage,
    output_message_status: ItemStatus,
    model: str,
    raw_logprobs: ChatCompletionChoiceLogprobs | None = None,
    refusal: str | None = None,
) -> list[OutputItem]:
    output_items: list[OutputItem] = []

    output_message = _extract_output_message_item(
        raw_message=raw_message,
        raw_logprobs=raw_logprobs,
        refusal=refusal,
        status=output_message_status,
    )
    if output_message is not None:
        output_items.append(output_message)

    tool_call_items, thought_signatures = _extract_tool_call_items(
        raw_message=raw_message, model=model
    )
    output_items.extend(tool_call_items)

    thought_signatures.extend(
        getattr(raw_message, "provider_specific_fields", {}).get(
            "thought_signatures", []
        )
    )
    message_thought_signature = thought_signatures[-1] if thought_signatures else None

    reasoning_items = _extract_reasoning_items(
        raw_message=raw_message, message_thought_signature=message_thought_signature
    )
    output_items.extend(reasoning_items)

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
    refusal: str | None = None,
) -> OutputMessageItem | None:
    content_parts: list[OutputMessageContentPart] = []

    logprobs = convert_logprobs(raw_logprobs) if raw_logprobs is not None else None
    citations = convert_annotations(getattr(raw_message, "annotations", []))

    if raw_message.content:
        content_parts.append(
            OutputTextContentPart(
                text=raw_message.content,
                citations=citations,
                logprobs=logprobs,
            )
        )

    if refusal:
        content_parts.append(OutputRefusal(refusal=refusal))

    return (
        OutputMessageItem(status=status, content_parts=content_parts)
        if content_parts
        else None
    )


def _extract_reasoning_items(
    raw_message: LiteLLMChatCompletionMessage,
    message_thought_signature: str | None = None,
) -> list[ReasoningItem]:
    reasoning_items: list[ReasoningItem] = []

    thinking_blocks: list[LiteLLMThinkingBlock] = getattr(
        raw_message, "thinking_blocks", []
    )
    reasoning_content: str | None = getattr(raw_message, "reasoning_content", None)

    for block in thinking_blocks:
        reasoning_items.append(ReasoningItem.from_thinking_block(block))

    if not reasoning_items and reasoning_content is not None:
        reasoning_items.append(
            ReasoningItem.from_reasoning_content(
                reasoning_content, encrypted_content=message_thought_signature
            )
        )

    return reasoning_items


def _extract_tool_call_items(
    raw_message: LiteLLMChatCompletionMessage, model: str
) -> tuple[list[FunctionToolCallItem], list[str]]:
    tool_call_items: list[FunctionToolCallItem] = []
    thought_signatures: list[str] = []

    if raw_message.tool_calls:
        for tc in raw_message.tool_calls:
            if tc.function.name is not None:
                tc_signature = _get_thought_signature_from_tool(
                    tool=tc.model_dump(), model=model
                )
                tool_call_items.append(
                    FunctionToolCallItem(
                        call_id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                        status="completed",
                        provider_specific_fields=(
                            {"thought_signature": tc_signature}
                            if tc_signature
                            else None
                        ),
                    )
                )
                if tc_signature:
                    thought_signatures.append(tc_signature)

    return tool_call_items, thought_signatures
