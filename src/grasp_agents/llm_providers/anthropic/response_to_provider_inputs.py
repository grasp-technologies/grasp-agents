"""
Convert grasp-agents InputItem[] → Anthropic MessageParam[].

The main entry point is ``items_to_anthropic_messages``, which returns
``(system, messages)`` — system is extracted separately since Anthropic
takes it as a top-level parameter, not inside the messages array.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal

from anthropic.types import (
    ServerToolUseBlockParam,
    WebSearchResultBlockParam,
    WebSearchToolResultBlockParam,
)
from grasp_agents.types.content import (
    BASE64_DATA_PREFIX,
    InputImage,
    InputTextContentPart,
)
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
    OutputItem,
    OutputMessageItem,
    ReasoningItem,
    WebSearchCallItem,
)

from . import (
    Base64ImageSourceParam,
    ImageBlockParam,
    MessageParam,
    RedactedThinkingBlockParam,
    TextBlockParam,
    ThinkingBlockParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
    URLImageSourceParam,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anthropic.types import ContentBlockParam


def items_to_provider_inputs(
    items: Sequence[InputMessageItem | OutputItem | FunctionToolOutputItem],
) -> tuple[str | list[TextBlockParam] | None, list[MessageParam]]:
    """
    Convert memory items to Anthropic message format.

    Returns ``(system, messages)`` where *system* is extracted from
    system/developer role items.
    """
    system_parts: list[str] = []
    messages: list[MessageParam] = []
    i = 0
    n = len(items)

    while i < n:
        item = items[i]

        if isinstance(item, InputMessageItem):
            if item.role in {"system", "developer"}:
                system_parts.append(item.text)
                i += 1
            else:
                messages.append(_input_to_user_message(item))
                i += 1

        elif isinstance(item, FunctionToolOutputItem):
            messages.append(_tool_output_to_message(item))
            i += 1

        else:
            # Collect consecutive assistant-side items into one message
            group: list[OutputItem] = [item]
            i += 1
            while i < n and isinstance(
                items[i],
                (
                    ReasoningItem,
                    OutputMessageItem,
                    FunctionToolCallItem,
                    WebSearchCallItem,
                ),
            ):
                group.append(items[i])  # type: ignore[arg-type]
                i += 1
            messages.append(_output_group_to_message(group))

    system: str | list[TextBlockParam] | None = None
    if system_parts:
        system = "\n\n".join(system_parts)

    return system, messages


def _input_to_user_message(item: InputMessageItem) -> MessageParam:
    content: list[ContentBlockParam] = []

    for part in item.content_parts:
        if isinstance(part, InputTextContentPart):
            content.append(TextBlockParam(type="text", text=part.text))
        elif isinstance(part, InputImage):
            content.append(_image_to_block(part))

    if len(content) == 1 and content[0]["type"] == "text":
        return MessageParam(role="user", content=content[0]["text"])  # type: ignore[typeddict-item]

    return MessageParam(role="user", content=content)


def _image_to_block(img: InputImage) -> ImageBlockParam:
    if img.is_base64 and img.image_url:
        data = img.image_url.removeprefix(BASE64_DATA_PREFIX)
        return ImageBlockParam(
            type="image",
            source=Base64ImageSourceParam(
                type="base64",
                data=data,
                media_type="image/jpeg",
            ),
        )
    if img.is_url and img.image_url:
        return ImageBlockParam(
            type="image",
            source=URLImageSourceParam(type="url", url=img.image_url),
        )
    raise ValueError("InputImage must have either a URL or base64 data")


def _tool_output_to_message(item: FunctionToolOutputItem) -> MessageParam:
    if isinstance(item.output_parts, list):
        content_str = "\n".join(
            part.text
            for part in item.output_parts
            if isinstance(part, InputTextContentPart)
        )
    else:
        content_str = item.output_parts

    return MessageParam(
        role="user",
        content=[
            ToolResultBlockParam(
                type="tool_result",
                tool_use_id=item.call_id,
                content=content_str,
            )
        ],
    )


def _output_group_to_message(
    group: Sequence[OutputItem],
) -> MessageParam:
    content: list[Any] = []

    for item in group:
        if isinstance(item, ReasoningItem):
            content.append(_reasoning_to_block(item))
        elif isinstance(item, OutputMessageItem):
            if item.text:
                content.append(TextBlockParam(type="text", text=item.text))
        elif isinstance(item, WebSearchCallItem):
            content.extend(_web_search_to_blocks(item))
        else:
            content.append(
                ToolUseBlockParam(
                    type="tool_use",
                    id=item.call_id,
                    name=item.name,
                    input=json.loads(item.arguments),
                )
            )

    return MessageParam(role="assistant", content=content)


_WS_TOOL_NAME: Literal[
    "web_search",
    "web_fetch",
    "code_execution",
    "bash_code_execution",
    "text_editor_code_execution",
    "tool_search_tool_regex",
    "tool_search_tool_bm25",
] = "web_search"


def _web_search_to_blocks(
    item: WebSearchCallItem,
) -> list[ServerToolUseBlockParam | WebSearchToolResultBlockParam]:
    """Reconstruct server_tool_use + web_search_tool_result blocks."""
    query: str = ""
    action: Any = item.action
    if hasattr(action, "query"):
        query = action.query

    server_block = ServerToolUseBlockParam(
        type="server_tool_use",
        id=item.id,
        name=_WS_TOOL_NAME,
        input={"query": query},
    )

    results: list[WebSearchResultBlockParam] = []
    sources: list[Any] = getattr(action, "sources", None) or []
    psf = item.provider_specific_fields or {}
    encrypted: dict[str, str] = psf.get("web_search_encrypted_content", {})
    for source in sources:
        url: str = source.url
        result = WebSearchResultBlockParam(
            type="web_search_result",
            url=url,
            title="",
            encrypted_content=encrypted.get(url, ""),
        )
        results.append(result)

    result_block = WebSearchToolResultBlockParam(
        type="web_search_tool_result",
        tool_use_id=item.id,
        content=results,
    )

    return [server_block, result_block]


def _reasoning_to_block(
    item: ReasoningItem,
) -> ThinkingBlockParam | RedactedThinkingBlockParam:
    if item.redacted:
        return RedactedThinkingBlockParam(
            type="redacted_thinking",
            data=item.encrypted_content or "",
        )

    text = item.content_text or item.summary_text or ""
    return ThinkingBlockParam(
        type="thinking",
        thinking=text,
        signature=item.encrypted_content or "",
    )
