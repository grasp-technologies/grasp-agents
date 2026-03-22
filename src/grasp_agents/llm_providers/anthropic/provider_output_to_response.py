"""Convert Anthropic Message content blocks → internal item types."""

import json
from typing import TYPE_CHECKING, Any

from openai.types.responses import ResponseStatus
from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_function_web_search import (
    ActionOpenPage,
    ActionSearch,
    ActionSearchSource,
)
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)

from anthropic.types import Message as AnthropicMessage
from anthropic.types import RedactedThinkingBlock as AnthropicRedactedThinkingBlock
from anthropic.types import ServerToolUseBlock as AnthropicServerToolUseBlock
from anthropic.types import StopReason as AnthropicStopReason
from anthropic.types import TextBlock as AnthropicTextBlock
from anthropic.types import ThinkingBlock as AnthropicThinkingBlock
from anthropic.types import ToolUseBlock as AnthropicToolUseBlock
from anthropic.types import Usage as AnthropicUsage
from anthropic.types import (
    WebFetchBlock as AnthropicWebFetchBlock,
)
from anthropic.types import (
    WebFetchToolResultBlock as AnthropicWebFetchToolResultBlock,
)
from anthropic.types.web_search_tool_result_block import (
    WebSearchToolResultBlock as AnthropicWebSearchToolResultBlock,
)
from anthropic.types.web_search_tool_result_error import (
    WebSearchToolResultError as AnthropicWebSearchToolResultError,
)
from grasp_agents.types.content import (
    Citation,
    OutputMessageText,
    ReasoningSummary,
    UrlCitation,
)
from grasp_agents.types.items import (
    FunctionToolCallItem,
    OutputItem,
    OutputMessageItem,
    ReasoningItem,
    WebSearchCallItem,
)
from grasp_agents.types.response import (
    Response,
    ResponseUsage,
    WebSearchInfo,
    WebSearchSource,
)

if TYPE_CHECKING:
    from anthropic.types.citations_web_search_result_location import (
        CitationsWebSearchResultLocation as AnthropicWebSearchCitation,
    )


def _anthropic_message_to_items_and_web_search_info(
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

    # Track server_tool_use blocks to pair with web_search_tool_result
    pending_server_tools: dict[str, AnthropicServerToolUseBlock] = {}

    web_search_queries: list[str] = []
    web_search_sources: list[WebSearchSource] = []

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
            items.append(_redacted_thinking_block_to_reasoning_item(block))

        elif block.type == "tool_use":
            items.append(_tool_use_block_to_tool_call_item(block))

        elif block.type == "server_tool_use":
            pending_server_tools[block.id] = block

        elif block.type == "web_search_tool_result":
            call_block = pending_server_tools.pop(block.tool_use_id, None)
            search_item, queries, sources = _extract_web_search_data(
                call_block=call_block, result_block=block
            )
            items.append(search_item)
            web_search_queries.extend(queries)
            web_search_sources.extend(sources)

        elif block.type == "web_fetch_tool_result":
            call_block = pending_server_tools.pop(block.tool_use_id, None)
            items.append(_extract_web_fetch_data(call_block, block))

    _flush_text()
    _flush_thinking()

    web_search_info = (
        WebSearchInfo(queries=web_search_queries, sources=web_search_sources)
        if web_search_sources
        else None
    )

    return items, web_search_info


def _extract_citations(block: AnthropicTextBlock) -> list[UrlCitation]:
    """Extract URL citations from a TextBlock's citations list."""
    if not block.citations:
        return []

    citations: list[UrlCitation] = []
    for cit in block.citations:
        if cit.type == "web_search_result_location":
            cit_ws: AnthropicWebSearchCitation = cit  # type: ignore[assignment]
            provide_specific_fields = {
                "anthropic:cited_text": cit_ws.cited_text,
                "anthropic:encrypted_index": cit_ws.encrypted_index,
            }
            citations.append(
                UrlCitation(
                    type="url_citation",
                    url=cit_ws.url,
                    title=cit_ws.title or "",
                    start_index=0,
                    end_index=len(block.text),
                    provider_specific_fields=provide_specific_fields,
                )
            )

        # NOTE: Consider handling other citation types

    return citations


def _extract_web_search_data(
    call_block: AnthropicServerToolUseBlock | None,
    result_block: AnthropicWebSearchToolResultBlock,
) -> tuple[WebSearchCallItem, list[str], list[WebSearchSource]]:
    """Convert server_tool_use + web_search_tool_result pair to a WebSearchCallItem."""
    query = ""
    if call_block is not None:
        query = str(call_block.input.get("query", ""))
        _ = call_block.caller

    action_search_sources: list[ActionSearchSource] = []
    web_search_sources: list[WebSearchSource] = []
    encrypted: dict[str, str] = {}

    if isinstance(result_block.content, list):
        for result in result_block.content:
            action_search_sources.append(ActionSearchSource(type="url", url=result.url))
            web_search_sources.append(
                WebSearchSource(
                    url=result.url,
                    title=result.title,
                    page_age=result.page_age,
                )
            )
            if result.encrypted_content:
                encrypted[result.url] = result.encrypted_content

    elif isinstance(result_block.content, AnthropicWebSearchToolResultError):  # type: ignore[unreachable]
        item = WebSearchCallItem(
            id=call_block.id if call_block else result_block.tool_use_id,
            status="failed",
            action=ActionSearch(type="search", query=query),
            provider_specific_fields={
                "anthropic:error_code": result_block.content.error_code,
            },
        )
        return item, [query], []

    item = WebSearchCallItem(
        id=call_block.id if call_block else result_block.tool_use_id,
        status="completed",
        action=ActionSearch(
            type="search",
            query=query,
            queries=[query],
            sources=action_search_sources or None,
        ),
        provider_specific_fields=(
            {"anthropic:encrypted_content": encrypted} if encrypted else None
        ),
    )

    return item, [query], web_search_sources


def _extract_web_fetch_data(
    call_block: AnthropicServerToolUseBlock | None,
    result_block: AnthropicWebFetchToolResultBlock,
) -> WebSearchCallItem:
    """Convert server_tool_use + web_fetch_tool_result pair to a WebSearchCallItem."""
    content = result_block.content
    item_id = call_block.id if call_block else result_block.tool_use_id

    if isinstance(content, AnthropicWebFetchBlock):
        psf: dict[str, Any] = {}
        if content.retrieved_at:
            psf["anthropic:retrieved_at"] = content.retrieved_at
        doc = content.content
        psf["anthropic:title"] = doc.title or ""
        psf["anthropic:media_type"] = doc.source.media_type
        psf["anthropic:data"] = doc.source.data

        return WebSearchCallItem(
            id=item_id,
            status="completed",
            action=ActionOpenPage(type="open_page", url=content.url),
            provider_specific_fields=psf or None,
        )

    # WebFetchToolResultErrorBlock
    call_url = str(call_block.input.get("url", "")) if call_block else ""
    return WebSearchCallItem(
        id=item_id,
        status="failed",
        action=ActionOpenPage(type="open_page", url=call_url),
        provider_specific_fields={
            "anthropic:error_code": content.error_code,
        },
    )


def _merge_text_blocks(
    blocks: list[AnthropicTextBlock],
) -> OutputMessageItem:
    parts: list[OutputMessageText] = []
    for block in blocks:
        citations: list[Citation] = list(_extract_citations(block))
        parts.append(OutputMessageText(text=block.text, citations=citations))

    return OutputMessageItem(status="completed", content_parts=list(parts))


def _merge_thinking_blocks(blocks: list[AnthropicThinkingBlock]) -> ReasoningItem:
    return ReasoningItem(
        status="completed",
        summary_parts=[ReasoningSummary(text=b.thinking) for b in blocks],
        encrypted_content=blocks[-1].signature if blocks else None,
    )


def _redacted_thinking_block_to_reasoning_item(
    block: AnthropicRedactedThinkingBlock,
) -> ReasoningItem:
    return ReasoningItem(
        status="completed",
        summary_parts=[],
        redacted=True,
        encrypted_content=block.data,
    )


def _tool_use_block_to_tool_call_item(
    block: AnthropicToolUseBlock,
) -> FunctionToolCallItem:
    return FunctionToolCallItem(
        call_id=block.id,
        name=block.name,
        arguments=json.dumps(block.input, indent=2),
        status="completed",
    )


def _convert_usage(usage: AnthropicUsage) -> ResponseUsage:
    # TODO: more cached token details (extend ResponseUsage?)

    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens
    cached = getattr(usage, "cache_read_input_tokens", None) or 0

    return ResponseUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=cached),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )


def _map_stop_reason(
    stop_reason: AnthropicStopReason | None,
) -> tuple[ResponseStatus, IncompleteDetails | None]:
    if stop_reason in {"end_turn", "tool_use", "stop_sequence", "pause_turn"}:
        return "completed", None

    if stop_reason == "max_tokens":
        return "incomplete", IncompleteDetails(reason="max_output_tokens")

    if stop_reason == "refusal":
        return "incomplete", IncompleteDetails(reason="content_filter")

    return "completed", None


def provider_output_to_response(provider_output: AnthropicMessage) -> Response:
    """Convert an Anthropic ``Message`` to a grasp-agents ``Response``."""
    # NOTE: ignored AnthropicMessage fields: `container`, `model``, `stop_sequence`

    output_items, web_search_info = _anthropic_message_to_items_and_web_search_info(
        provider_output
    )
    usage = _convert_usage(provider_output.usage)
    status, incomplete_details = _map_stop_reason(provider_output.stop_reason)

    return Response(
        id=provider_output.id,
        model=provider_output.model,
        status=status,
        incomplete_details=incomplete_details,
        output_items=output_items,
        usage_with_cost=usage,
        web_search=web_search_info,
    )
