"""Convert Anthropic Message content blocks → internal item types."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from openai.types.responses.response_function_web_search import (
    ActionSearch,
    ActionSearchSource,
)
from openai.types.responses.response_output_text import Annotation

from grasp_agents.types.content import (
    OutputTextContentPart,
    ReasoningSummaryPart,
    UrlCitation,
)
from grasp_agents.types.items import (
    FunctionToolCallItem,
    OutputItem,
    OutputMessageItem,
    ReasoningItem,
    WebSearchCallItem,
)
from grasp_agents.types.response import WebSearchInfo, WebSearchSource

if TYPE_CHECKING:
    from anthropic.types import Message as AnthropicMessage
    from anthropic.types import RedactedThinkingBlock as AnthropicRedactedThinkingBlock
    from anthropic.types import TextBlock as AnthropicTextBlock
    from anthropic.types import ThinkingBlock as AnthropicThinkingBlock
    from anthropic.types import ToolUseBlock as AnthropicToolUseBlock

    from . import (
        AnthropicServerToolUseBlock,
        AnthropicWebSearchCitation,
        AnthropicWebSearchToolResultBlock,
    )


def generated_message_to_items(
    message: AnthropicMessage,
) -> tuple[list[OutputItem], WebSearchInfo | None]:
    """
    Convert an Anthropic Message's content blocks to output items.

    Returns (items, web_search_info). Consecutive TextBlocks are merged into
    a single OutputMessageItem with citation annotations extracted.
    Server tool blocks (server_tool_use + web_search_tool_result) are converted
    to WebSearchCallItem with encrypted_content for round-trip.
    """
    items: list[OutputItem] = []
    pending_text_blocks: list[AnthropicTextBlock] = []
    pending_thinking_blocks: list[AnthropicThinkingBlock] = []
    web_search_sources: list[WebSearchSource] = []
    # Track server_tool_use blocks to pair with web_search_tool_result
    pending_server_tools: dict[str, AnthropicServerToolUseBlock] = {}

    def _flush_text() -> None:
        if pending_text_blocks:
            items.append(_merge_text_blocks(pending_text_blocks))
            pending_text_blocks.clear()

    def _flush_thinking() -> None:
        if pending_thinking_blocks:
            items.append(_merge_thinking_blocks(pending_thinking_blocks))
            pending_thinking_blocks.clear()

    for block in message.content:
        if block.type == "text":
            _flush_thinking()
            pending_text_blocks.append(block)
            continue

        if block.type == "thinking":
            _flush_text()
            pending_thinking_blocks.append(block)
            continue

        # Any other block type flushes both pending lists
        _flush_text()
        _flush_thinking()

        if block.type == "redacted_thinking":
            items.append(_redacted_to_reasoning(block))
        elif block.type == "tool_use":
            items.append(_tool_use_to_item(block))
        elif block.type == "server_tool_use":
            pending_server_tools[block.id] = block
        elif block.type == "web_search_tool_result":
            server_block = pending_server_tools.pop(block.tool_use_id, None)
            search_item, sources = _web_search_to_item(block, server_block)
            items.append(search_item)
            web_search_sources.extend(sources)

    _flush_text()
    _flush_thinking()

    web_search = (
        WebSearchInfo(sources=web_search_sources) if web_search_sources else None
    )
    return items, web_search


def _merge_text_blocks(
    blocks: list[AnthropicTextBlock],
) -> OutputMessageItem:
    parts: list[OutputTextContentPart] = []
    for block in blocks:
        annotations: list[Annotation] = list(_extract_citations(block))
        parts.append(OutputTextContentPart(text=block.text, annotations=annotations))
    return OutputMessageItem(
        status="completed",
        content_parts=list(parts),  # type: ignore[arg-type]
    )


def _extract_citations(block: AnthropicTextBlock) -> list[UrlCitation]:
    """Extract URL citations from a TextBlock's citations list."""
    if not block.citations:
        return []

    citations: list[UrlCitation] = []
    for cit in block.citations:
        if cit.type == "web_search_result_location":
            cit_ws: AnthropicWebSearchCitation = cit  # type: ignore[assignment]
            citations.append(
                UrlCitation(
                    type="url_citation",
                    url=cit_ws.url,
                    title=cit_ws.title or "",
                    start_index=0,
                    end_index=len(block.text),
                    cited_text=cit_ws.cited_text,
                )
            )
    return citations


def _web_search_to_item(
    block: AnthropicWebSearchToolResultBlock,
    server_block: AnthropicServerToolUseBlock | None,
) -> tuple[WebSearchCallItem, list[WebSearchSource]]:
    """Convert server_tool_use + web_search_tool_result pair to a WebSearchCallItem."""
    query = ""
    if server_block is not None:
        query = str(server_block.input.get("query", ""))
        _ = server_block.caller

    sources: list[ActionSearchSource] = []
    web_sources: list[WebSearchSource] = []
    encrypted: dict[str, str] = {}

    if isinstance(block.content, list):
        for result in block.content:
            if not hasattr(result, "url"):
                continue
            sources.append(ActionSearchSource(type="url", url=result.url))
            web_sources.append(
                WebSearchSource(
                    url=result.url,
                    title=result.title,
                    page_age=result.page_age,
                )
            )
            if hasattr(result, "encrypted_content") and result.encrypted_content:
                encrypted[result.url] = result.encrypted_content

    item = WebSearchCallItem(
        id=server_block.id if server_block else block.tool_use_id,
        status="completed",
        action=ActionSearch(
            type="search",
            query=query,
            sources=sources or None,
        ),
        provider_specific_fields=(
            {"web_search_encrypted_content": encrypted} if encrypted else None
        ),
    )
    return item, web_sources


def _merge_thinking_blocks(
    blocks: list[AnthropicThinkingBlock],
) -> ReasoningItem:
    return ReasoningItem(
        status="completed",
        summary_parts=[ReasoningSummaryPart(text=b.thinking) for b in blocks],
        encrypted_content=blocks[-1].signature,
    )


def _redacted_to_reasoning(
    block: AnthropicRedactedThinkingBlock,
) -> ReasoningItem:
    return ReasoningItem(
        status="completed",
        summary_parts=[],
        redacted=True,
        encrypted_content=block.data,
    )


def _tool_use_to_item(
    block: AnthropicToolUseBlock,
) -> FunctionToolCallItem:
    return FunctionToolCallItem(
        call_id=block.id,
        name=block.name,
        arguments=json.dumps(block.input, indent=2),
        status="completed",
    )
