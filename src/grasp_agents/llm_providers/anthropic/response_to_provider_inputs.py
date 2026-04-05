"""
Convert grasp-agents InputItem[] → Anthropic MessageParam[].

The main entry point is ``items_to_anthropic_messages``, which returns
``(system, messages)`` — system is extracted separately since Anthropic
takes it as a top-level parameter, not inside the messages array.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal, cast

from anthropic.types import (
    Base64ImageSourceParam,
    CacheControlEphemeralParam,
    CitationWebSearchResultLocationParam,
    DocumentBlockParam,
    ImageBlockParam,
    MessageParam,
    PlainTextSourceParam,
    RedactedThinkingBlockParam,
    ServerToolUseBlockParam,
    TextBlockParam,
    ThinkingBlockParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
    URLImageSourceParam,
    WebFetchBlockParam,
    WebFetchToolResultBlockParam,
    WebFetchToolResultErrorBlockParam,
    WebSearchResultBlockParam,
    WebSearchToolResultBlockParam,
)
from anthropic.types.web_search_tool_request_error_param import (
    WebSearchToolRequestErrorParam,
)
from grasp_agents.types.content import (
    BASE64_DATA_PREFIX,
    InputImage,
    InputText,
    OutputMessageRefusal,
)
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
    OpenPageAction,
    OutputItem,
    OutputMessageItem,
    ReasoningItem,
    SearchAction,
    SearchSource,
    WebSearchCallItem,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

SUPPORTED_MIME_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

_CACHE_CONTROL_KEY = "anthropic:cache_control"


def _get_cache_control(
    psf: dict[str, Any] | None,
) -> CacheControlEphemeralParam | None:
    if psf is None:
        return None
    cc: Any = psf.get(_CACHE_CONTROL_KEY)
    if not isinstance(cc, dict):
        return None
    return cast("CacheControlEphemeralParam", cc)


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
                messages.append(
                    MessageParam(
                        role=item.role,  # type: ignore[assignment]
                        content=_convert_content_parts(item.content_parts),  # type: ignore[assignment]
                    )
                )
                i += 1

        elif isinstance(item, FunctionToolOutputItem):
            # Group consecutive tool results into a single user message
            # (required for parallel tool use to work correctly)
            tool_results: list[ToolResultBlockParam] = []
            while i < n and isinstance(items[i], FunctionToolOutputItem):
                tool_item: FunctionToolOutputItem = items[i]  # type: ignore[assignment]
                tool_results.append(
                    ToolResultBlockParam(
                        type="tool_result",
                        tool_use_id=tool_item.call_id,
                        content=_convert_content_parts(tool_item.output_parts),  # type: ignore[assignment]
                        cache_control=_get_cache_control(
                            tool_item.provider_specific_fields
                        ),
                    )
                )
                i += 1
            messages.append(MessageParam(role="user", content=tool_results))

        else:
            # Collect consecutive assistant-side items into one message
            output_items: list[OutputItem] = [item]
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
                output_items.append(items[i])  # type: ignore[arg-type]
                i += 1
            messages.append(_output_group_to_message_param(output_items))

    system: str | list[TextBlockParam] | None = None
    if system_parts:
        system = "\n\n".join(system_parts)

    return system, messages


def _image_to_block(img: InputImage) -> ImageBlockParam:
    cc = _get_cache_control(img.provider_specific_fields)

    if img.is_base64 and img.image_url:
        data = img.image_url.removeprefix(BASE64_DATA_PREFIX)
        if img.mime_type not in SUPPORTED_MIME_TYPES:
            raise ValueError(f"Unsupported MIME type for base64 image: {img.mime_type}")

        return ImageBlockParam(
            type="image",
            source=Base64ImageSourceParam(
                type="base64",
                data=data,
                media_type=img.mime_type,  # type: ignore[union-attr]
            ),
            cache_control=cc,
        )
    if img.is_url and img.image_url:
        return ImageBlockParam(
            type="image",
            source=URLImageSourceParam(type="url", url=img.image_url),
            cache_control=cc,
        )

    raise ValueError("InputImage must have either a URL or base64 data")


def _convert_content_parts(
    content_parts: list[InputImage | InputText],
) -> str | list[TextBlockParam | ImageBlockParam | WebSearchResultBlockParam]:
    content: list[TextBlockParam | ImageBlockParam | WebSearchResultBlockParam] = []
    has_cache_control = False

    for part in content_parts:
        if isinstance(part, InputText):
            cc = _get_cache_control(part.provider_specific_fields)
            if cc:
                has_cache_control = True
            content.append(
                TextBlockParam(
                    type="text",
                    text=part.text,
                    cache_control=cc,
                )
            )

        elif isinstance(part, InputImage):  # type: ignore[unreachable]
            if _get_cache_control(part.provider_specific_fields):
                has_cache_control = True
            content.append(_image_to_block(part))

    # String shortcut only when there's no cache_control to carry
    if not has_cache_control and len(content) == 1 and content[0]["type"] == "text":
        return content[0]["text"]

    return content


def _output_group_to_message_param(output_items: Sequence[OutputItem]) -> MessageParam:
    content: list[Any] = []

    for item in output_items:
        if isinstance(item, ReasoningItem):
            content.append(_reasoning_to_block(item))

        elif isinstance(item, OutputMessageItem):
            if item.text:
                content.extend(_output_message_to_blocks(item))

        elif isinstance(item, WebSearchCallItem):
            if isinstance(item.action, OpenPageAction):
                content.extend(_web_fetch_to_blocks(item))
            else:
                content.extend(_web_search_to_blocks(item))

        else:
            content.append(
                ToolUseBlockParam(
                    type="tool_use",
                    id=item.call_id,
                    name=item.name,
                    input=json.loads(item.arguments),
                    cache_control=_get_cache_control(item.provider_specific_fields),
                )
            )

    return MessageParam(role="assistant", content=content)


def _output_message_to_blocks(item: OutputMessageItem) -> list[TextBlockParam]:
    blocks: list[TextBlockParam] = []

    for part in item.content_parts:
        if isinstance(part, OutputMessageRefusal):
            continue

        citation_params: list[CitationWebSearchResultLocationParam] = []
        for citation in item.citations:
            citation_params.append(
                CitationWebSearchResultLocationParam(
                    type="web_search_result_location",
                    title=citation.title,
                    url=citation.url,
                    cited_text=(citation.provider_specific_fields or {}).get(
                        "anthropic:cited_text", ""
                    ),
                    encrypted_index=(citation.provider_specific_fields or {}).get(
                        "anthropic:encrypted_index", ""
                    ),
                )
            )

        cc = _get_cache_control(part.provider_specific_fields)
        blocks.append(
            TextBlockParam(
                type="text",
                text=part.text,
                citations=citation_params,
                cache_control=cc,
            )
        )

    return blocks


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
    action = item.action

    if not isinstance(action, SearchAction):
        raise TypeError(f"Expected SearchAction, got {type(action)}")

    server_block = ServerToolUseBlockParam(
        type="server_tool_use",
        id=item.id,
        name=_WS_TOOL_NAME,
        input={"query": action.queries[0] if action.queries else ""},
        cache_control=_get_cache_control(item.provider_specific_fields),
    )

    psf = item.provider_specific_fields or {}

    if item.status == "failed":
        result_content: (
            list[WebSearchResultBlockParam] | WebSearchToolRequestErrorParam
        ) = WebSearchToolRequestErrorParam(
            type="web_search_tool_result_error",
            error_code=psf.get("anthropic:error_code", "unavailable"),
        )
    else:
        results: list[WebSearchResultBlockParam] = []
        sources: list[SearchSource] = action.sources or []
        encrypted: dict[str, str] = psf.get("anthropic:encrypted_content", {})

        for source in sources:
            url: str = source.url
            results.append(
                WebSearchResultBlockParam(
                    type="web_search_result",
                    url=url,
                    title=source.title,
                    encrypted_content=encrypted.get(url, ""),
                )
            )
        result_content = results

    result_block = WebSearchToolResultBlockParam(
        type="web_search_tool_result",
        tool_use_id=item.id,
        content=result_content,
    )

    return [server_block, result_block]


_WF_TOOL_NAME: Literal[
    "web_search",
    "web_fetch",
    "code_execution",
    "bash_code_execution",
    "text_editor_code_execution",
    "tool_search_tool_regex",
    "tool_search_tool_bm25",
] = "web_fetch"


def _web_fetch_to_blocks(
    item: WebSearchCallItem,
) -> list[ServerToolUseBlockParam | WebFetchToolResultBlockParam]:
    """Reconstruct server_tool_use + web_fetch_tool_result blocks."""
    action = item.action

    if not isinstance(action, OpenPageAction):
        raise TypeError(f"Expected OpenPageAction, got {type(action)}")

    url = action.url
    psf = item.provider_specific_fields or {}

    server_block = ServerToolUseBlockParam(
        type="server_tool_use",
        id=item.id,
        name=_WF_TOOL_NAME,
        input={"url": url},
    )

    if item.status == "failed":
        result_content: WebFetchBlockParam | WebFetchToolResultErrorBlockParam = (
            WebFetchToolResultErrorBlockParam(
                type="web_fetch_tool_result_error",
                error_code=psf.get("anthropic:error_code", "unavailable"),
            )
        )
    else:
        result_content = WebFetchBlockParam(
            type="web_fetch_result",
            url=url,  # type: ignore[arg-type]  # always set on success
            retrieved_at=psf.get("anthropic:retrieved_at"),
            content=DocumentBlockParam(
                type="document",
                title=psf.get("anthropic:title"),
                source=PlainTextSourceParam(
                    type="text",
                    media_type="text/plain",
                    data=psf.get("anthropic:data", ""),
                ),
            ),
        )

    result_block = WebFetchToolResultBlockParam(
        type="web_fetch_tool_result",
        tool_use_id=item.id,
        content=result_content,
    )

    return [server_block, result_block]
