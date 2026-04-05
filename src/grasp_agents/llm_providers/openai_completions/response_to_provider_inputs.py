"""
Convert OpenResponses items → Chat Completions message params.

Two public functions:

- ``response_to_provider_input``: Convert a single ``Response`` to a
  typed assistant message param (content + tool_calls + reasoning).
- ``items_to_provider_inputs``: Convert a full memory item list to a
  list of typed Chat Completions message params, grouping consecutive output
  items (ReasoningItem, OutputMessageItem, FunctionToolCallItem) into a
  single assistant message.
"""

from collections.abc import Sequence
from typing import Any, Literal

from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_content_part_image_param import (
    ChatCompletionContentPartImageParam,
    ImageURL,
)
from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam,
)
from openai.types.chat.chat_completion_developer_message_param import (
    ChatCompletionDeveloperMessageParam,
)
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_message_tool_call_param import (
    ChatCompletionMessageToolCallParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import (
    Function as ToolCallFunction,
)
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)

from grasp_agents.types.content import InputImage, InputText
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputItem,
    InputMessageItem,
    OutputItem,
    OutputMessageItem,
    ReasoningItem,
)
from grasp_agents.types.reasoning import (
    OpenRouterReasoningDetails,
    OpenRouterReasoningEncrypted,
    OpenRouterReasoningSummary,
    OpenRouterReasoningText,
)
from grasp_agents.types.response import Response

ReasoningBlockFormat = Literal["anthropic", "openrouter"]


class ChatCompletionAssistantMessageParamExt(
    ChatCompletionAssistantMessageParam, total=False
):
    """
    Extension of ChatCompletionAssistantMessageParam
    to allow provider-specific fields.
    """

    reasoning_content: str | None
    reasoning: str | None
    thinking_blocks: list[dict[str, Any]] | None
    reasoning_details: list[dict[str, Any]] | None
    provider_specific_fields: dict[str, Any] | None


def items_to_provider_inputs(
    items: Sequence[InputItem],
    *,
    reasoning_block_format: ReasoningBlockFormat | None = "anthropic",
) -> list[ChatCompletionMessageParam]:
    """Convert a sequence of memory items to Chat Completions messages."""
    messages: list[ChatCompletionMessageParam] = []
    i = 0
    n = len(items)

    while i < n:
        item = items[i]

        if isinstance(item, InputMessageItem):
            messages.append(_input_message_to_message_param(item))
            i += 1

        elif isinstance(item, FunctionToolOutputItem):
            messages.append(_tool_output_to_message_param(item))
            i += 1

        else:
            # Collect consecutive assistant-side items into one message
            group: list[OutputItem] = [item]
            i += 1
            while i < n and isinstance(
                items[i], (ReasoningItem, OutputMessageItem, FunctionToolCallItem)
            ):
                group.append(items[i])  # type: ignore[arg-type]
                i += 1
            messages.append(
                _output_items_to_message_param(group, reasoning_block_format)
            )

    return messages


def _input_message_to_message_param(
    item: InputMessageItem,
) -> ChatCompletionMessageParam:
    """Convert an InputMessageItem to a user/system/developer message param."""
    if item.role == "system":
        return ChatCompletionSystemMessageParam(role="system", content=item.text)
    if item.role == "developer":
        return ChatCompletionDeveloperMessageParam(role="developer", content=item.text)

    has_images = any(isinstance(part, InputImage) for part in item.content_parts)
    if not has_images:
        return ChatCompletionUserMessageParam(role="user", content=item.text)

    content: list[
        ChatCompletionContentPartTextParam | ChatCompletionContentPartImageParam
    ] = []

    for part in item.content_parts:
        if isinstance(part, InputText):
            content.append(
                ChatCompletionContentPartTextParam(type="text", text=part.text)
            )
        elif isinstance(part, InputImage):
            content.append(
                ChatCompletionContentPartImageParam(
                    type="image_url",
                    image_url=ImageURL(url=part.to_str(), detail=part.detail),  # type: ignore[call-arg]
                )
            )

    return ChatCompletionUserMessageParam(role="user", content=content)


def _tool_output_to_message_param(
    item: FunctionToolOutputItem,
) -> ChatCompletionToolMessageParam:
    """Convert a FunctionToolOutputItem to a tool message param."""
    if isinstance(item.output_parts, list):
        has_images = any(isinstance(part, InputImage) for part in item.output_parts)
        if has_images:
            raise ValueError("Image tool outputs are not supported by Completions API")
        content = "\n".join(
            part.text for part in item.output_parts if isinstance(part, InputText)
        )
    else:
        content = item.output_parts

    return ChatCompletionToolMessageParam(
        role="tool", tool_call_id=item.call_id, content=content
    )


def _output_items_to_message_param(
    output_items: Sequence[OutputItem],
    reasoning_block_format: ReasoningBlockFormat | None = "anthropic",
) -> ChatCompletionAssistantMessageParamExt:
    """Convert a group of consecutive output items to an assistant message param."""
    msg = ChatCompletionAssistantMessageParamExt(role="assistant")

    reasoning_items: list[ReasoningItem] = []
    output_message_items: list[OutputMessageItem] = []
    tool_call_items: list[FunctionToolCallItem] = []

    for item in output_items:
        if isinstance(item, ReasoningItem):
            reasoning_items.append(item)

        elif isinstance(item, OutputMessageItem):
            output_message_items.append(item)

        elif isinstance(item, FunctionToolCallItem):
            tool_call_items.append(item)
        # NOTE: skipping WebSearchCallItem

    thought_sigs: list[str] = []

    if output_message_items:
        message_thought_sigs = _add_output_message_items(msg, output_message_items)
        thought_sigs.extend(message_thought_sigs)

    if tool_call_items:
        tool_thought_sigs = _add_tool_call_items(msg, tool_call_items)
        thought_sigs.extend(tool_thought_sigs)

    if reasoning_items:
        reasoning_thought_sigs = _add_reasoning_items(
            msg, reasoning_items, reasoning_block_format
        )
        thought_sigs.extend(reasoning_thought_sigs)

    # NOTE: Do we need this?

    # if (
    #     "provider_specific_fields" in msg
    #     and isinstance(msg["provider_specific_fields"], dict)
    #     and "thought_signatures" in msg["provider_specific_fields"]
    # ):
    #     msg["provider_specific_fields"]["thought_signatures"] = thought_sigs

    return msg


def response_to_provider_input(
    response: Response,
    *,
    reasoning_block_format: ReasoningBlockFormat | None = "anthropic",
) -> ChatCompletionAssistantMessageParamExt:
    # NOTE: Do we need response-level provider_specific_fields?
    return _output_items_to_message_param(response.output_items, reasoning_block_format)


def _add_output_message_items(
    msg: ChatCompletionAssistantMessageParamExt,
    output_message_items: list[OutputMessageItem],
) -> list[str]:
    thought_sigs: list[str] = []
    text_parts: list[str] = []

    for item in output_message_items:
        if item.text:
            text_parts.append(item.text)

        if item.provider_specific_fields:
            msg["provider_specific_fields"] = item.provider_specific_fields

            _thought_sigs = item.provider_specific_fields.get("thought_signatures", [])
            thought_sigs.extend(_thought_sigs)

    # NOTE: Empty content may not be allowed by some providers
    msg["content"] = "\n".join(text_parts)

    return thought_sigs


def _add_tool_call_items(
    msg: ChatCompletionAssistantMessageParamExt,
    tool_call_items: list[FunctionToolCallItem],
) -> list[str]:
    thought_sigs: list[str] = []
    tc_message_params: list[ChatCompletionMessageToolCallParam] = []

    for tc in tool_call_items:
        if (
            tc.provider_specific_fields
            and "thought_signature" in tc.provider_specific_fields
        ):
            thought_sig = tc.provider_specific_fields["thought_signature"]
            thought_sigs.append(thought_sig)

        tc_message_params.append(
            ChatCompletionMessageToolCallParam(
                id=tc.call_id,
                type="function",
                function=ToolCallFunction(name=tc.name, arguments=tc.arguments),
            )
        )

    if tc_message_params:
        msg["tool_calls"] = tc_message_params

    return thought_sigs


def _add_reasoning_items(
    msg: ChatCompletionAssistantMessageParamExt,
    reasoning_items: list[ReasoningItem],
    reasoning_block_format: ReasoningBlockFormat | None,
) -> list[str]:
    thought_sigs = [
        item.encrypted_content for item in reasoning_items if item.encrypted_content
    ]

    # NOTE: Pass either plain reasoning or summaries, not both

    contents = [r.content_text for r in reasoning_items if r.content_text]
    summaries = [r.summary_text for r in reasoning_items if r.summary_text]

    reasoning_text = "\n".join(contents or summaries)
    if reasoning_text:
        msg["reasoning_content"] = reasoning_text
        if reasoning_block_format == "openrouter":
            msg["reasoning"] = reasoning_text

    if reasoning_block_format == "anthropic":
        msg["thinking_blocks"] = [
            _reasoning_item_to_thinking_block(r) for r in reasoning_items
        ]
    elif reasoning_block_format == "openrouter":
        msg["reasoning_details"] = [
            _reasoning_item_to_openrouter_details(r).model_dump(exclude_none=True)
            for r in reasoning_items
        ]

    return thought_sigs


def _reasoning_item_to_thinking_block(r: ReasoningItem) -> dict[str, Any]:
    if r.redacted:
        return {"type": "redacted_thinking", "data": r.encrypted_content or ""}

    text = r.content_text or r.summary_text or ""
    block = {"type": "thinking", "thinking": text}
    if r.encrypted_content:
        block["signature"] = r.encrypted_content

    return block


def _reasoning_item_to_openrouter_details(
    r: ReasoningItem,
) -> OpenRouterReasoningDetails:
    if r.redacted and r.encrypted_content:
        return OpenRouterReasoningEncrypted(data=r.encrypted_content)

    text = r.content_text or r.summary_text or ""

    if r.encrypted_content:
        return OpenRouterReasoningText(text=text, signature=r.encrypted_content)

    return OpenRouterReasoningSummary(summary=text)
