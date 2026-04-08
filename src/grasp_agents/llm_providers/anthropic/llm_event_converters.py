"""
Stateful converter: Anthropic RawMessageStreamEvent → LlmEvent stream.

Consumes Anthropic's content-block-based streaming protocol and emits
OpenResponses lifecycle events using the shared BaseLlmStreamConverter.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from grasp_agents.llm.llm_stream_converter import BaseLlmStreamConverter
from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)

from anthropic.types import (
    CitationsWebSearchResultLocation as _AnthropicWebSearchCitation,
)
from anthropic.types import WebFetchBlock as _AnthropicWebFetchBlock
from anthropic.types import (
    WebFetchToolResultBlock as _AnthropicWebFetchToolResultBlock,
)
from grasp_agents.types.content import Citation, UrlCitation
from grasp_agents.types.items import (
    OpenPageAction,
    SearchAction,
    SearchSource,
    WebSearchCallItem,
)
from grasp_agents.types.response import ResponseUsage

from . import AnthropicStreamEvent

if TYPE_CHECKING:
    from collections.abc import Iterator

    from openai.types.responses import ResponseStatus

    from grasp_agents.types.llm_events import LlmEvent, ResponseCompleted

    from . import (
        AnthropicContentBlockDeltaEvent,
        AnthropicContentBlockStartEvent,
        AnthropicContentBlockStopEvent,
        AnthropicMessageDeltaEvent,
        AnthropicMessageStartEvent,
        AnthropicWebSearchToolResultBlock,
    )


@dataclass
class ServerToolState:
    """Accumulated state for a server_tool_use block being streamed."""

    tool_id: str
    name: str
    input_json: str = ""


class AnthropicStreamConverter(BaseLlmStreamConverter[AnthropicStreamEvent]):
    """
    Converts an Anthropic RawMessageStreamEvent async stream into LlmEvents.

    Maps Anthropic's content-block-based protocol (message_start →
    content_block_start → content_block_delta → content_block_stop →
    message_delta → message_stop) to the OpenResponses event lifecycle.
    """

    def __init__(self) -> None:
        super().__init__()
        # Track block type by index for content_block_stop dispatch
        self._block_types: dict[int, str] = {}
        # Web search sources from web_search_tool_result blocks
        self._web_search_sources: list[SearchSource] = []
        # Pending server_tool_use blocks keyed by tool id
        self._pending_server_tools: dict[str, ServerToolState] = {}
        # Map block index → server tool id for input_json_delta buffering
        self._server_tool_block_idx: dict[int, str] = {}
        # Accumulated citations for current text block
        self._citations: list[UrlCitation] = []

    def _process_event(self, raw_event: AnthropicStreamEvent) -> Iterator[LlmEvent]:
        event_type = raw_event.type

        if event_type == "message_start":
            yield from self._on_message_start(raw_event)  # type: ignore[arg-type]
        elif event_type == "content_block_start":
            yield from self._on_block_start(raw_event)  # type: ignore[arg-type]
        elif event_type == "content_block_delta":
            yield from self._on_block_delta(raw_event)  # type: ignore[arg-type]
        elif event_type == "content_block_stop":
            yield from self._on_block_stop(raw_event)  # type: ignore[arg-type]
        elif event_type == "message_delta":
            self._on_message_delta(raw_event)  # type: ignore[arg-type]
        # message_stop: no action needed, _close_response runs after loop

    # ==== Event handlers ====

    def _on_message_start(
        self, event: AnthropicMessageStartEvent
    ) -> Iterator[LlmEvent]:
        msg = event.message
        yield from self._start_response(
            id=msg.id, model=msg.model, created_at=time.time()
        )

        # Capture initial usage from message_start
        usage = msg.usage
        self._usage = ResponseUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
            input_tokens_details=InputTokensDetails(
                cached_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0
            ),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        )

    def _on_block_start(
        self, event: AnthropicContentBlockStartEvent
    ) -> Iterator[LlmEvent]:
        block = event.content_block
        idx = event.index
        self._block_types[idx] = block.type

        match block.type:
            case "thinking":
                if not self._reasoning_open:
                    yield from self._open_reasoning()
                else:
                    # New thinking block within same item → close previous part
                    yield from self._close_reasoning_summary_part()
                yield from self._open_reasoning_summary_part()

            case "redacted_thinking":
                # Redacted blocks are always separate items
                if self._reasoning_open:
                    yield from self._close_reasoning()

                yield from self._open_reasoning()

                self._reasoning_redacted = True
                self._reasoning_encrypted_content = block.data

            case "text":
                if self._reasoning_open:
                    yield from self._close_reasoning()

                if not self._message_open:
                    yield from self._open_message()

                self._citations = []

                yield from self._open_text()

            case "tool_use":
                if self._reasoning_open:
                    yield from self._close_reasoning()

                if self._message_open:
                    yield from self._close_message()

                yield from self._open_tool_call(
                    call_id=block.id, name=block.name, idx=idx
                )

            case "server_tool_use":
                tool_id: str = block.id
                name: str = block.name
                self._pending_server_tools[tool_id] = ServerToolState(
                    tool_id=tool_id, name=name
                )
                self._server_tool_block_idx[idx] = tool_id

                if "web_search" in name or "web_fetch" in name:
                    yield from self._open_web_search(tool_id)

            case "web_search_tool_result":
                yield from self._on_web_search_result(block)

            case "web_fetch_tool_result":
                yield from self._on_web_fetch_result(block)
            case "code_execution_tool_result":
                pass
            case "bash_code_execution_tool_result":
                pass
            case "text_editor_code_execution_tool_result":
                pass
            case "tool_search_tool_result":
                pass
            case "container_upload":
                pass

    def _on_block_delta(
        self, event: AnthropicContentBlockDeltaEvent
    ) -> Iterator[LlmEvent]:
        delta = event.delta
        idx = event.index

        match delta.type:
            case "text_delta":
                yield from self._on_text(delta.text)

            case "thinking_delta":
                yield from self._on_reasoning_content(delta.thinking)

            case "signature_delta":
                self._reasoning_encrypted_content = delta.signature

            case "citations_delta":
                citation = delta.citation
                if isinstance(citation, _AnthropicWebSearchCitation):
                    self._citations.append(
                        UrlCitation(
                            url=citation.url,
                            title=citation.title or "",
                            start_index=0,
                            end_index=len(self._text or ""),
                            provider_specific_fields=(
                                {"anthropic:cited_text": citation.cited_text}
                                if citation.cited_text
                                else None
                            ),
                        )
                    )

            case "input_json_delta":
                block_type = self._block_types.get(idx)
                if block_type == "tool_use":
                    yield from self._on_tool_call_args(idx, delta.partial_json)

                elif block_type == "server_tool_use":
                    tool_id = self._server_tool_block_idx.get(idx)
                    if tool_id and tool_id in self._pending_server_tools:
                        state = self._pending_server_tools[tool_id]
                        is_first = not state.input_json
                        state.input_json += delta.partial_json

                        if is_first and "web_search" in state.name:
                            yield from self._on_web_search_searching(tool_id)

    def _on_block_stop(
        self, event: AnthropicContentBlockStopEvent
    ) -> Iterator[LlmEvent]:
        idx = event.index
        block_type = self._block_types.get(idx)

        if block_type in {"thinking", "redacted_thinking"}:
            yield from self._close_reasoning()
        elif block_type == "text" and self._text_open:
            yield from self._close_text()

    def _on_message_delta(self, event: AnthropicMessageDeltaEvent) -> None:
        delta = event.delta
        if delta.stop_reason:
            self._finish_reason = delta.stop_reason

        # Update usage with final output token count
        msg_usage: Any = event.usage
        if msg_usage and self._usage:
            output_tokens = getattr(msg_usage, "output_tokens", None)
            if output_tokens is not None:
                self._usage = ResponseUsage(
                    input_tokens=self._usage.input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=self._usage.input_tokens + output_tokens,
                    input_tokens_details=self._usage.input_tokens_details,
                    output_tokens_details=self._usage.output_tokens_details,
                )

    # ==== Web search ====

    def _on_web_search_result(
        self, block: AnthropicWebSearchToolResultBlock
    ) -> Iterator[LlmEvent]:
        """Handle web_search_tool_result block: create WebSearchCallItem."""
        tool_use_id: str = block.tool_use_id
        server_state = self._pending_server_tools.pop(tool_use_id, None)

        query = ""
        if server_state and server_state.input_json:
            try:
                parsed = json.loads(server_state.input_json)
                query = str(parsed.get("query", ""))
            except (json.JSONDecodeError, AttributeError):
                pass

        sources: list[SearchSource] = []
        encrypted: dict[str, str] = {}

        if isinstance(block.content, list):
            for result in block.content:
                source = SearchSource(
                    url=result.url,
                    title=result.title,
                    page_age=result.page_age,
                )
                sources.append(source)
                self._web_search_sources.append(source)
                if result.encrypted_content:
                    encrypted[result.url] = result.encrypted_content

        item = WebSearchCallItem(
            id=server_state.tool_id if server_state else tool_use_id,
            status="completed",
            action=SearchAction(
                queries=[query] if query else [],
                sources=sources,
            ),
            provider_specific_fields=(
                {"anthropic:encrypted_content": encrypted} if encrypted else None
            ),
        )

        yield from self._close_web_search(item)

    def _on_web_fetch_result(
        self, block: _AnthropicWebFetchToolResultBlock
    ) -> Iterator[LlmEvent]:
        """Handle web_fetch_tool_result block: create WebSearchCallItem."""
        tool_use_id: str = block.tool_use_id
        server_state = self._pending_server_tools.pop(tool_use_id, None)
        item_id = server_state.tool_id if server_state else tool_use_id
        content = block.content

        if isinstance(content, _AnthropicWebFetchBlock):
            psf: dict[str, Any] = {}
            if content.retrieved_at:
                psf["anthropic:retrieved_at"] = content.retrieved_at
            doc = content.content
            psf["anthropic:title"] = doc.title or ""
            psf["anthropic:media_type"] = doc.source.media_type
            psf["anthropic:data"] = doc.source.data

            item = WebSearchCallItem(
                id=item_id,
                status="completed",
                action=OpenPageAction(url=content.url),
                provider_specific_fields=psf or None,
            )
        else:
            # WebFetchToolResultErrorBlock
            call_url = ""
            if server_state and server_state.input_json:
                try:
                    parsed = json.loads(server_state.input_json)
                    call_url = str(parsed.get("url", ""))
                except (json.JSONDecodeError, AttributeError):
                    pass
            item = WebSearchCallItem(
                id=item_id,
                status="failed",
                action=OpenPageAction(url=call_url),
                provider_specific_fields={
                    "anthropic:error_code": content.error_code,
                },
            )

        yield from self._close_web_search(item)

    def _build_text_citations(self) -> list[Citation]:
        return list(self._citations)  # type: ignore[list-item]

    def _build_response_completed(self) -> ResponseCompleted:
        return super()._build_response_completed()

    # ==== Hooks ====

    def _map_finish_reason(
        self,
    ) -> tuple[ResponseStatus, IncompleteDetails | None]:
        if self._finish_reason in {
            "end_turn",
            "tool_use",
            "stop_sequence",
            "pause_turn",
        }:
            return "completed", None
        if self._finish_reason == "max_tokens":
            return "incomplete", IncompleteDetails(reason="max_output_tokens")
        if self._finish_reason == "refusal":
            return "incomplete", IncompleteDetails(reason="content_filter")
        return "completed", None
