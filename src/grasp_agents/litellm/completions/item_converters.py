from __future__ import annotations

from typing import TYPE_CHECKING, Any

from litellm.types.llms.openai import (
    ChatCompletionRedactedThinkingBlock,
    ChatCompletionThinkingBlock,
)
from litellm.types.utils import ChoiceLogprobs
from openai.types.responses.response_output_text import (
    AnnotationURLCitation,
    Logprob,
    LogprobTopLogprob,
)

from grasp_agents.typing.content import (
    OutputContent,
    OutputRefusalContent,
    OutputTextContent,
    ReasoningSummaryContent,
)
from grasp_agents.typing.items import (
    FunctionToolCallItem,
    ItemStatus,
    OutputItem,
    OutputMessageItem,
    ReasoningItem,
)

if TYPE_CHECKING:
    from . import LiteLLMCompletionMessage

LiteLLMThinkingBlock = ChatCompletionThinkingBlock | ChatCompletionRedactedThinkingBlock


def from_litellm_completions_message(
    raw_message: LiteLLMCompletionMessage,
    output_message_status: ItemStatus,
    raw_logprobs: ChoiceLogprobs | Any | None = None,
    refusal: str | None = None,
) -> list[OutputItem]:
    output_items: list[OutputItem] = []

    # Reasoning items

    reasoning_items = _convert_reasoning_items(
        reasoning_content=raw_message.reasoning_content,
        thinking_blocks=raw_message.thinking_blocks,
    )
    output_items.extend(reasoning_items)

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
            # function.name can be none in deltas only
            if tc.function.name is not None:
                output_items.append(
                    FunctionToolCallItem(
                        call_id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                        status="completed",
                    )
                )

    return output_items


def _convert_reasoning_items(
    reasoning_content: str | None = None,
    thinking_blocks: list[LiteLLMThinkingBlock] | None = None,
) -> list[ReasoningItem]:
    reasoning_items: list[ReasoningItem] = []

    if thinking_blocks is not None:
        for block in thinking_blocks or []:
            if block["type"] == "thinking":
                summary_text = block.get("thinking")  # type: ignore[reportUnknownMemberType]
                summary_parts = (
                    [ReasoningSummaryContent(text=summary_text)] if summary_text else []
                )
                reasoning_items.append(
                    ReasoningItem(
                        status="completed",
                        summary_ext=summary_parts,
                        encrypted_content=block.get("signature"),  # type: ignore[reportUnknownMemberType]
                        cache_control=block.get("cache_control"),  # type: ignore[reportUnknownMemberType]
                        redacted=False,
                    )
                )
            elif block["type"] == "redacted_thinking":
                reasoning_items.append(
                    ReasoningItem(
                        status="completed",
                        encrypted_content=block.get("data"),  # type: ignore[reportUnknownMemberType]
                        cache_control=block.get("cache_control"),  # type: ignore[reportUnknownMemberType]
                        redacted=True,
                    )
                )

    elif reasoning_content is not None:
        reasoning_items.append(
            ReasoningItem(
                status="completed",
                summary_ext=[ReasoningSummaryContent(text=reasoning_content)],
            )
        )

    return reasoning_items


def _convert_output_message(
    raw_message: LiteLLMCompletionMessage,
    status: ItemStatus,
    raw_logprobs: ChoiceLogprobs | Any | None = None,
    refusal: str | None = None,
) -> OutputMessageItem | None:
    content_parts: list[OutputContent] = []
    logprobs: list[Logprob] | None = None

    if isinstance(raw_logprobs, ChoiceLogprobs) and raw_logprobs.content:
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
        AnnotationURLCitation(type="url_citation", **ann["url_citation"])
        for ann in (raw_message.annotations or [])
        if "url_citation" in ann
    ]
    if raw_message.content:
        content_parts.append(
            OutputTextContent(
                text=raw_message.content,
                annotations=annotations,  # type: ignore
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
