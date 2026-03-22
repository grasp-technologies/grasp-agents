"""
Tests for Gemini provider converters.

Tests the full conversion pipeline: Google GenAI SDK types → grasp-agents types.
Uses real google-genai SDK model objects (not mocks) to ensure type compatibility.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

from google.genai.types import (
    Candidate,
    Citation,
    CitationMetadata,
    Content,
    FinishReason,
    FunctionCall,
    GenerateContentResponse,
    GenerateContentResponseUsageMetadata,
    GroundingChunk,
    GroundingChunkWeb,
    GroundingMetadata,
    GroundingSupport,
    LogprobsResult,
    LogprobsResultCandidate,
    LogprobsResultTopCandidates,
    Part,
    Segment,
    UrlContextMetadata,
    UrlMetadata,
    UrlRetrievalStatus,
)
from openai.types.responses.response_function_web_search import ActionOpenPage
from pydantic import BaseModel

from grasp_agents.llm_providers.gemini.llm_event_converters import (
    GeminiStreamConverter,
)
from grasp_agents.llm_providers.gemini.provider_output_to_response import (
    _gemini_response_to_items_and_web_search_info,
    provider_output_to_response,
)
from grasp_agents.llm_providers.gemini.response_to_provider_inputs import (
    items_to_provider_inputs,
)
from grasp_agents.llm_providers.gemini.tool_converters import (
    to_api_tool_config,
    to_api_tools,
)
from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
    OutputMessageItem,
    ReasoningItem,
    WebSearchCallItem,
)
from grasp_agents.types.llm_events import (
    FunctionCallArgumentsDelta,
    FunctionCallArgumentsDone,
    OutputItemAdded,
    OutputItemDone,
    ReasoningSummaryDelta,
    ResponseCompleted,
    OutputMessageTextDone,
)
from grasp_agents.types.llm_events import (
    OutputMessageTextDelta as LlmTextDelta,
)
from grasp_agents.types.tool import BaseTool, NamedToolChoice

# ==== Helpers ====


def _make_response(
    parts: list[Part],
    *,
    finish_reason: FinishReason = FinishReason.STOP,
    prompt_tokens: int = 100,
    output_tokens: int = 50,
    response_id: str = "resp_test123",
    grounding_metadata: GroundingMetadata | None = None,
    citation_metadata: CitationMetadata | None = None,
    logprobs_result: LogprobsResult | None = None,
    url_context_metadata: UrlContextMetadata | None = None,
) -> GenerateContentResponse:
    return GenerateContentResponse(
        response_id=response_id,
        candidates=[
            Candidate(
                content=Content(role="model", parts=parts),
                finish_reason=finish_reason,
                grounding_metadata=grounding_metadata,
                citation_metadata=citation_metadata,
                logprobs_result=logprobs_result,
                url_context_metadata=url_context_metadata,
            )
        ],
        usage_metadata=GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt_tokens,
            candidates_token_count=output_tokens,
            total_token_count=prompt_tokens + output_tokens,
        ),
    )


async def _collect_events(
    chunks: list[GenerateContentResponse],
) -> list[Any]:
    """Run GeminiStreamConverter on a list of response chunks."""
    converter = GeminiStreamConverter(model="gemini-2.0-flash")

    async def chunk_stream():  # noqa: RUF029
        for chunk in chunks:
            yield chunk

    return [e async for e in converter.convert(chunk_stream())]


# ==== items_extraction ====


class TestItemsExtraction:
    def test_simple_text(self):
        """Single text part → OutputMessageItem."""
        resp = _make_response([Part(text="Hello world")])
        items, _ = _gemini_response_to_items_and_web_search_info(resp)

        assert len(items) == 1
        assert isinstance(items[0], OutputMessageItem)
        assert items[0].text == "Hello world"

    def test_consecutive_text_parts_merged(self):
        """Multiple text parts merge into one OutputMessageItem."""
        resp = _make_response(
            [
                Part(text="First"),
                Part(text="Second"),
            ]
        )
        items, _ = _gemini_response_to_items_and_web_search_info(resp)

        assert len(items) == 1
        assert isinstance(items[0], OutputMessageItem)
        # Two content parts, merged into one message item
        assert len(items[0].content_parts) == 2

    def test_thinking_then_text(self):
        """Thinking part → ReasoningItem, text → OutputMessageItem."""
        resp = _make_response(
            [
                Part(text="Let me think...", thought=True),
                Part(text="The answer is 42"),
            ]
        )
        items, _ = _gemini_response_to_items_and_web_search_info(resp)

        assert len(items) == 2
        assert isinstance(items[0], ReasoningItem)
        assert items[0].summary_text == "Let me think..."
        assert isinstance(items[1], OutputMessageItem)
        assert items[1].text == "The answer is 42"

    def test_consecutive_thinking_parts_merged(self):
        """Multiple consecutive thinking parts merge into one ReasoningItem."""
        resp = _make_response(
            [
                Part(text="Step 1: analyze", thought=True),
                Part(text="Step 2: decide", thought=True),
                Part(text="The answer"),
            ]
        )
        items, _ = _gemini_response_to_items_and_web_search_info(resp)

        assert len(items) == 2
        assert isinstance(items[0], ReasoningItem)
        # Should have two summary parts
        assert len(items[0].summary_parts) == 2
        assert items[0].summary_parts[0].text == "Step 1: analyze"
        assert items[0].summary_parts[1].text == "Step 2: decide"

    def test_thought_signature_captured(self):
        """thought_signature is preserved as encrypted_content."""
        sig = b"encrypted_signature_bytes"
        resp = _make_response(
            [
                Part(text="Thinking...", thought=True, thought_signature=sig),
                Part(text="Answer"),
            ]
        )
        items, _ = _gemini_response_to_items_and_web_search_info(resp)

        assert isinstance(items[0], ReasoningItem)
        assert items[0].encrypted_content is not None
        assert items[0].encrypted_content == base64.b64encode(sig).decode("ascii")

    def test_thought_signature_last_wins(self):
        """When multiple thinking parts have signatures, the last one wins."""
        resp = _make_response(
            [
                Part(text="Part 1", thought=True, thought_signature=b"sig_1"),
                Part(text="Part 2", thought=True, thought_signature=b"sig_2"),
                Part(text="Answer"),
            ]
        )
        items, _ = _gemini_response_to_items_and_web_search_info(resp)

        assert isinstance(items[0], ReasoningItem)
        assert items[0].encrypted_content == base64.b64encode(b"sig_2").decode("ascii")

    def test_function_call(self):
        """Function call part → FunctionToolCallItem."""
        resp = _make_response(
            [
                Part(
                    function_call=FunctionCall(
                        id="fc_123",
                        name="get_weather",
                        args={"city": "Paris"},
                    )
                ),
            ]
        )
        items, _ = _gemini_response_to_items_and_web_search_info(resp)

        assert len(items) == 1
        assert isinstance(items[0], FunctionToolCallItem)
        assert items[0].call_id == "fc_123"
        assert items[0].name == "get_weather"
        assert json.loads(items[0].arguments) == {"city": "Paris"}

    def test_function_call_no_id(self):
        """Function call without id gets UUID fallback."""
        resp = _make_response(
            [
                Part(
                    function_call=FunctionCall(
                        name="search",
                        args={"q": "test"},
                    )
                ),
            ]
        )
        items, _ = _gemini_response_to_items_and_web_search_info(resp)

        assert len(items) == 1
        assert isinstance(items[0], FunctionToolCallItem)
        assert items[0].call_id  # Non-empty UUID

    def test_text_then_function_call(self):
        """Text + function call: separate items."""
        resp = _make_response(
            [
                Part(text="I'll check the weather"),
                Part(
                    function_call=FunctionCall(
                        id="fc_456",
                        name="get_weather",
                        args={"city": "London"},
                    )
                ),
            ]
        )
        items, _ = _gemini_response_to_items_and_web_search_info(resp)

        assert len(items) == 2
        assert isinstance(items[0], OutputMessageItem)
        assert items[0].text == "I'll check the weather"
        assert isinstance(items[1], FunctionToolCallItem)
        assert items[1].name == "get_weather"

    def test_thinking_text_function_call_pipeline(self):
        """Full response: thinking + text + function call."""
        resp = _make_response(
            [
                Part(text="Planning...", thought=True),
                Part(text="I'll look that up"),
                Part(
                    function_call=FunctionCall(
                        id="fc_789",
                        name="search",
                        args={"query": "test"},
                    )
                ),
            ]
        )
        items, _ = _gemini_response_to_items_and_web_search_info(resp)

        assert len(items) == 3
        assert isinstance(items[0], ReasoningItem)
        assert isinstance(items[1], OutputMessageItem)
        assert isinstance(items[2], FunctionToolCallItem)

    def test_empty_response(self):
        """Response with no candidates returns empty list."""
        resp = GenerateContentResponse(response_id="empty")
        items, _ = _gemini_response_to_items_and_web_search_info(resp)
        assert items == []

    def test_grounding_annotations(self):
        """Grounding metadata → URL citation annotations on text parts."""
        grounding = GroundingMetadata(
            grounding_chunks=[
                GroundingChunk(
                    web=GroundingChunkWeb(
                        uri="https://example.com/article",
                        title="Example Article",
                    )
                ),
            ],
            grounding_supports=[
                GroundingSupport(
                    grounding_chunk_indices=[0],
                    segment=Segment(
                        start_index=0,
                        end_index=5,
                    ),
                ),
            ],
        )
        resp = _make_response(
            [Part(text="Hello world")],
            grounding_metadata=grounding,
        )
        items, _ = _gemini_response_to_items_and_web_search_info(resp)

        assert len(items) == 2
        msg = items[0]
        assert isinstance(msg, OutputMessageItem)
        text_part = msg.content_parts[0]
        assert isinstance(text_part, OutputMessageText)
        assert len(text_part.annotations) == 1
        ann = text_part.annotations[0]
        assert ann.type == "url_citation"
        assert ann.url == "https://example.com/article"
        assert ann.title == "Example Article"

        # Grounding also produces a WebSearchCallItem
        ws = items[1]
        assert isinstance(ws, WebSearchCallItem)

    def test_citation_annotations(self):
        """Citation metadata → URL citation annotations on text parts."""
        citation_meta = CitationMetadata(
            citations=[
                Citation(
                    uri="https://docs.example.com",
                    title="API Docs",
                    start_index=0,
                    end_index=10,
                ),
            ]
        )
        resp = _make_response(
            [Part(text="Check the docs for details")],
            citation_metadata=citation_meta,
        )
        items, _ = _gemini_response_to_items_and_web_search_info(resp)

        assert len(items) == 1
        msg = items[0]
        assert isinstance(msg, OutputMessageItem)
        text_part = msg.content_parts[0]
        assert isinstance(text_part, OutputMessageText)
        assert len(text_part.annotations) == 1
        ann = text_part.annotations[0]
        assert ann.url == "https://docs.example.com"
        assert ann.title == "API Docs"

    def test_logprobs_attached(self):
        """Logprobs from candidate → attached to first text content part."""
        logprobs_result = LogprobsResult(
            chosen_candidates=[
                LogprobsResultCandidate(token="Hello", log_probability=-0.1),
                LogprobsResultCandidate(token=" world", log_probability=-0.2),
            ],
            top_candidates=[
                LogprobsResultTopCandidates(
                    candidates=[
                        LogprobsResultCandidate(token="Hello", log_probability=-0.1),
                        LogprobsResultCandidate(token="Hi", log_probability=-0.5),
                    ]
                ),
                LogprobsResultTopCandidates(
                    candidates=[
                        LogprobsResultCandidate(token=" world", log_probability=-0.2),
                    ]
                ),
            ],
        )
        resp = _make_response(
            [Part(text="Hello world")],
            logprobs_result=logprobs_result,
        )
        items, _ = _gemini_response_to_items_and_web_search_info(resp)

        assert len(items) == 1
        msg = items[0]
        assert isinstance(msg, OutputMessageItem)
        text_part = msg.content_parts[0]
        assert isinstance(text_part, OutputMessageText)
        assert text_part.logprobs is not None
        assert len(text_part.logprobs) == 2
        assert text_part.logprobs[0].token == "Hello"
        assert text_part.logprobs[0].logprob == -0.1
        # First token has 2 top candidates
        assert len(text_part.logprobs[0].top_logprobs) == 2

    def test_multiple_grounding_supports(self):
        """Multiple grounding supports distributed to correct text parts."""
        grounding = GroundingMetadata(
            grounding_chunks=[
                GroundingChunk(
                    web=GroundingChunkWeb(
                        uri="https://a.com",
                        title="Source A",
                    )
                ),
                GroundingChunk(
                    web=GroundingChunkWeb(
                        uri="https://b.com",
                        title="Source B",
                    )
                ),
            ],
            grounding_supports=[
                GroundingSupport(
                    grounding_chunk_indices=[0],
                    segment=Segment(start_index=0, end_index=5),
                ),
                GroundingSupport(
                    grounding_chunk_indices=[1],
                    segment=Segment(start_index=5, end_index=10),
                ),
            ],
        )
        resp = _make_response(
            [Part(text="Hello world")],
            grounding_metadata=grounding,
        )
        items, _ = _gemini_response_to_items_and_web_search_info(resp)

        msg = items[0]
        assert isinstance(msg, OutputMessageItem)
        text_part = msg.content_parts[0]
        assert isinstance(text_part, OutputMessageText)
        assert len(text_part.annotations) == 2
        assert text_part.annotations[0].url == "https://a.com"
        assert text_part.annotations[1].url == "https://b.com"


# ==== provider_output_to_response ====


class TestResponseConverters:
    def test_basic_text(self):
        """Basic text response → Response with correct fields."""
        resp = _make_response(
            [Part(text="Hello")],
            prompt_tokens=10,
            output_tokens=5,
        )
        result = provider_output_to_response(resp)

        assert result.id == "resp_test123"
        assert result.status == "completed"
        assert len(result.output_items) == 1
        assert isinstance(result.output_items[0], OutputMessageItem)
        assert result.usage_with_cost is not None
        assert result.usage_with_cost.input_tokens == 10
        assert result.usage_with_cost.output_tokens == 5
        assert result.usage_with_cost.total_tokens == 15

    def test_max_tokens(self):
        """MAX_TOKENS finish reason → incomplete status."""
        resp = _make_response(
            [Part(text="Truncated...")],
            finish_reason=FinishReason.MAX_TOKENS,
        )
        result = provider_output_to_response(resp)

        assert result.status == "incomplete"
        assert result.incomplete_details is not None
        assert result.incomplete_details.reason == "max_output_tokens"

    def test_safety_filter(self):
        """SAFETY finish reason → incomplete with content_filter."""
        resp = _make_response(
            [Part(text="")],
            finish_reason=FinishReason.SAFETY,
        )
        result = provider_output_to_response(resp)

        assert result.status == "incomplete"
        assert result.incomplete_details is not None
        assert result.incomplete_details.reason == "content_filter"

    def test_usage_with_thinking_tokens(self):
        """Usage metadata with thoughts_token_count."""
        resp = GenerateContentResponse(
            response_id="resp_usage",
            candidates=[
                Candidate(
                    content=Content(
                        role="model",
                        parts=[Part(text="Answer")],
                    ),
                    finish_reason=FinishReason.STOP,
                )
            ],
            usage_metadata=GenerateContentResponseUsageMetadata(
                prompt_token_count=100,
                candidates_token_count=50,
                total_token_count=200,
                thoughts_token_count=50,
            ),
        )
        result = provider_output_to_response(resp)

        assert result.usage_with_cost is not None
        assert result.usage_with_cost.output_tokens_details.reasoning_tokens == 50


# ==== response_to_provider_inputs ====


class TestResponseToProviderInputs:
    def test_system_extraction(self):
        """System messages extracted as system_instruction."""
        items = [
            InputMessageItem.from_text("You are a helper", role="system"),
            InputMessageItem.from_text("Hello"),
        ]
        system, contents = items_to_provider_inputs(items)

        assert system == "You are a helper"
        assert len(contents) == 1
        assert contents[0].role == "user"

    def test_system_concatenation(self):
        """Multiple system messages concatenated."""
        items = [
            InputMessageItem.from_text("Instruction 1", role="system"),
            InputMessageItem.from_text("Instruction 2", role="developer"),
            InputMessageItem.from_text("Hello"),
        ]
        system, contents = items_to_provider_inputs(items)

        assert system == "Instruction 1\n\nInstruction 2"
        assert len(contents) == 1

    def test_user_text(self):
        """User message → Content(role='user', parts=[Part(text=...)])."""
        items = [InputMessageItem.from_text("Hello")]
        _system, contents = items_to_provider_inputs(items)

        assert len(contents) == 1
        assert contents[0].role == "user"
        assert contents[0].parts is not None
        assert contents[0].parts[0].text == "Hello"

    def test_assistant_output_group(self):
        """Assistant items grouped into Content(role='model')."""
        items = [
            InputMessageItem.from_text("Hi"),
            OutputMessageItem(
                status="completed",
                content_parts=[OutputMessageText(text="Hello!")],
            ),
        ]
        _system, contents = items_to_provider_inputs(items)

        assert len(contents) == 2
        assert contents[0].role == "user"
        assert contents[1].role == "model"
        assert contents[1].parts is not None
        assert contents[1].parts[0].text == "Hello!"

    def test_tool_roundtrip(self):
        """Tool call + tool output → model + user contents."""
        items = [
            InputMessageItem.from_text("What's the weather?"),
            FunctionToolCallItem(
                call_id="fc_abc",
                name="get_weather",
                arguments='{"city": "Paris"}',
                status="completed",
            ),
            FunctionToolOutputItem(
                call_id="fc_abc",
                output_parts="Sunny, 22°C",
            ),
            OutputMessageItem(
                status="completed",
                content_parts=[OutputMessageText(text="It's sunny in Paris!")],
            ),
        ]
        _system, contents = items_to_provider_inputs(items)

        assert len(contents) == 4
        # User message
        assert contents[0].role == "user"
        # Model with function call
        assert contents[1].role == "model"
        assert contents[1].parts is not None
        fc_part = contents[1].parts[0]
        assert fc_part.function_call is not None
        assert fc_part.function_call.name == "get_weather"  # pyright: ignore[reportUnknownMemberType]
        # User with function response
        assert contents[2].role == "user"
        assert contents[2].parts is not None
        fr_part = contents[2].parts[0]
        assert fr_part.function_response is not None
        assert fr_part.function_response.response == {  # pyright: ignore[reportUnknownMemberType]
            "result": "Sunny, 22°C"
        }
        # Model with text
        assert contents[3].role == "model"

    def test_reasoning_roundtrip(self):
        """ReasoningItem → Part(thought=True)."""
        items = [
            InputMessageItem.from_text("Think about this"),
            ReasoningItem(
                status="completed",
                summary_parts=[],
                encrypted_content=base64.b64encode(b"sig123").decode("ascii"),
            ),
            OutputMessageItem(
                status="completed",
                content_parts=[OutputMessageText(text="Done")],
            ),
        ]
        _system, contents = items_to_provider_inputs(items)

        assert len(contents) == 2
        assert contents[1].role == "model"
        assert contents[1].parts is not None
        # Reasoning part
        assert contents[1].parts[0].thought is True  # pyright: ignore[reportUnknownMemberType]
        # Text part
        assert contents[1].parts[1].text == "Done"  # pyright: ignore[reportUnknownMemberType]

    def test_reasoning_encrypted_content_ignored(self):
        """Gemini doesn't use signatures on thinking parts — encrypted_content is ignored."""
        items = [
            InputMessageItem.from_text("Think"),
            ReasoningItem(
                status="completed",
                summary_parts=[],
                encrypted_content=base64.b64encode(b"my_signature").decode("ascii"),
            ),
            OutputMessageItem(
                status="completed",
                content_parts=[OutputMessageText(text="Result")],
            ),
        ]
        _system, contents = items_to_provider_inputs(items)

        reasoning_part = contents[1].parts[0]  # type: ignore[index]
        assert reasoning_part.thought is True  # pyright: ignore[reportUnknownMemberType]
        assert reasoning_part.thought_signature is None  # pyright: ignore[reportUnknownMemberType]


# ==== tool_converters ====


class _WeatherInput(BaseModel):
    city: str
    units: str = "celsius"


class _WeatherTool(BaseTool[_WeatherInput, str, None]):
    def __init__(self) -> None:
        super().__init__(
            name="get_weather",
            description="Get current weather for a city",
        )

    async def _run(
        self,
        inp: _WeatherInput,
        *,
        ctx: Any = None,  # noqa: ARG002
        call_id: str | None = None,  # noqa: ARG002
        progress_callback: Any = None,  # noqa: ARG002
    ) -> str:
        return f"Weather in {inp.city}"


class TestToolConverters:
    def test_to_api_tools(self):
        """BaseTool → Gemini Tool with FunctionDeclaration."""
        tool = _WeatherTool()
        api_tool = to_api_tools({"weather": tool})  # pyright: ignore[reportArgumentType]

        assert api_tool.function_declarations is not None
        assert len(api_tool.function_declarations) == 1
        decl = api_tool.function_declarations[0]
        assert decl.name == "get_weather"
        assert decl.description == "Get current weather for a city"
        assert decl.parameters_json_schema is not None
        assert "properties" in decl.parameters_json_schema
        assert "city" in decl.parameters_json_schema["properties"]

    def test_auto_choice(self):
        config = to_api_tool_config("auto")
        assert config.function_calling_config is not None
        assert config.function_calling_config.mode == "AUTO"

    def test_required_choice(self):
        config = to_api_tool_config("required")
        assert config.function_calling_config is not None
        assert config.function_calling_config.mode == "ANY"

    def test_none_choice(self):
        config = to_api_tool_config("none")
        assert config.function_calling_config is not None
        assert config.function_calling_config.mode == "NONE"

    def test_named_choice(self):
        config = to_api_tool_config(NamedToolChoice(name="search"))
        assert config.function_calling_config is not None
        assert config.function_calling_config.mode == "ANY"
        assert config.function_calling_config.allowed_function_names == ["search"]


# ==== Stream converter helpers ====


def _text_chunk(
    text: str,
    *,
    response_id: str = "resp_stream",
    finish_reason: FinishReason | None = None,
    usage: GenerateContentResponseUsageMetadata | None = None,
) -> GenerateContentResponse:
    return GenerateContentResponse(
        response_id=response_id,
        candidates=[
            Candidate(
                content=Content(role="model", parts=[Part(text=text)]),
                finish_reason=finish_reason,
            )
        ],
        usage_metadata=usage,
    )


def _thinking_chunk(
    text: str,
    *,
    response_id: str = "resp_stream",
    thought_signature: bytes | None = None,
) -> GenerateContentResponse:
    return GenerateContentResponse(
        response_id=response_id,
        candidates=[
            Candidate(
                content=Content(
                    role="model",
                    parts=[
                        Part(
                            text=text,
                            thought=True,
                            thought_signature=thought_signature,
                        )
                    ],
                ),
            )
        ],
    )


def _fc_chunk(
    name: str,
    args: dict[str, Any],
    *,
    call_id: str = "fc_1",
    response_id: str = "resp_stream",
    finish_reason: FinishReason | None = None,
) -> GenerateContentResponse:
    return GenerateContentResponse(
        response_id=response_id,
        candidates=[
            Candidate(
                content=Content(
                    role="model",
                    parts=[
                        Part(
                            function_call=FunctionCall(id=call_id, name=name, args=args)
                        )
                    ],
                ),
                finish_reason=finish_reason,
            )
        ],
    )


def _final_chunk(
    *,
    response_id: str = "resp_stream",
    finish_reason: FinishReason = FinishReason.STOP,
    prompt_tokens: int = 10,
    output_tokens: int = 20,
    url_context_metadata: UrlContextMetadata | None = None,
) -> GenerateContentResponse:
    return GenerateContentResponse(
        response_id=response_id,
        candidates=[
            Candidate(
                finish_reason=finish_reason,
                url_context_metadata=url_context_metadata,
            )
        ],
        usage_metadata=GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt_tokens,
            candidates_token_count=output_tokens,
            total_token_count=prompt_tokens + output_tokens,
        ),
    )


# ==== Stream converter tests ====


class TestGeminiStreamConverter:
    def _run(self, chunks: list[GenerateContentResponse]) -> list[Any]:
        return asyncio.get_event_loop().run_until_complete(_collect_events(chunks))

    def test_simple_text_streaming(self):
        """Multiple text chunks → text deltas + done."""
        chunks = [
            _text_chunk("Hello"),
            _text_chunk(" world"),
            _text_chunk("!"),
            _final_chunk(),
        ]
        events = self._run(chunks)

        text_deltas = [e for e in events if isinstance(e, LlmTextDelta)]
        assert len(text_deltas) == 3
        assert text_deltas[0].delta == "Hello"
        assert text_deltas[1].delta == " world"
        assert text_deltas[2].delta == "!"

        text_dones = [e for e in events if isinstance(e, OutputMessageTextDone)]
        assert len(text_dones) == 1
        assert text_dones[0].text == "Hello world!"

    def test_thinking_then_text(self):
        """Thinking chunks followed by text chunks."""
        chunks = [
            _thinking_chunk("Let me "),
            _thinking_chunk("think..."),
            _text_chunk("Answer: 42"),
            _final_chunk(),
        ]
        events = self._run(chunks)

        reasoning_deltas = [
            e for e in events if isinstance(e, ReasoningSummaryDelta)
        ]
        assert len(reasoning_deltas) == 2
        assert reasoning_deltas[0].delta == "Let me "
        assert reasoning_deltas[1].delta == "think..."

        reasoning_items = [
            e.item
            for e in events
            if isinstance(e, OutputItemDone) and isinstance(e.item, ReasoningItem)
        ]
        assert len(reasoning_items) == 1
        assert reasoning_items[0].summary_text == "Let me think..."

        text_deltas = [e for e in events if isinstance(e, LlmTextDelta)]
        assert len(text_deltas) == 1
        assert text_deltas[0].delta == "Answer: 42"

    def test_thinking_parts_have_no_signature(self):
        """Gemini doesn't put signatures on thinking parts — encrypted_content stays None."""
        chunks = [
            _thinking_chunk("Reasoning..."),
            _text_chunk("Result"),
            _final_chunk(),
        ]
        events = self._run(chunks)

        reasoning_items = [
            e.item
            for e in events
            if isinstance(e, OutputItemDone) and isinstance(e.item, ReasoningItem)
        ]
        assert len(reasoning_items) == 1
        assert reasoning_items[0].encrypted_content is None

    def test_function_call_streaming(self):
        """Function call in stream → tool call events."""
        chunks = [
            _fc_chunk(
                "get_weather",
                {"city": "Paris"},
                call_id="fc_stream_1",
            ),
            _final_chunk(finish_reason=FinishReason.STOP),
        ]
        events = self._run(chunks)

        arg_deltas = [e for e in events if isinstance(e, FunctionCallArgumentsDelta)]
        assert len(arg_deltas) == 1

        arg_dones = [e for e in events if isinstance(e, FunctionCallArgumentsDone)]
        assert len(arg_dones) == 1
        assert arg_dones[0].name == "get_weather"
        assert json.loads(arg_dones[0].arguments) == {"city": "Paris"}

    def test_usage_tracking(self):
        """Usage metadata propagated to ResponseCompleted."""
        chunks = [
            _text_chunk("Hi"),
            _final_chunk(prompt_tokens=100, output_tokens=50),
        ]
        events = self._run(chunks)

        completed = [e for e in events if isinstance(e, ResponseCompleted)]
        usage = completed[0].response.usage_with_cost
        assert usage is not None
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

    def test_max_tokens_incomplete(self):
        """MAX_TOKENS finish reason → incomplete status."""
        chunks = [
            _text_chunk("Truncated"),
            _final_chunk(finish_reason=FinishReason.MAX_TOKENS),
        ]
        events = self._run(chunks)

        completed = [e for e in events if isinstance(e, ResponseCompleted)]
        assert completed[0].response.status == "incomplete"
        details = completed[0].response.incomplete_details
        assert details is not None
        assert details.reason == "max_output_tokens"

    def test_sequence_numbers_monotonic(self):
        """All sequence numbers are strictly increasing."""
        chunks = [
            _text_chunk("Hello"),
            _text_chunk(" world"),
            _final_chunk(),
        ]
        events = self._run(chunks)

        seq_nums = [e.sequence_number for e in events]
        for i in range(1, len(seq_nums)):
            assert seq_nums[i] > seq_nums[i - 1]

    def test_output_indices_correct(self):
        """Each output item gets a unique, increasing index."""
        chunks = [
            _thinking_chunk("Reasoning"),
            _text_chunk("Text"),
            _fc_chunk("tool", {"a": 1}, call_id="fc_idx"),
            _final_chunk(),
        ]
        events = self._run(chunks)

        added = [e for e in events if isinstance(e, OutputItemAdded)]
        assert len(added) == 3
        assert added[0].output_index == 0  # reasoning
        assert added[1].output_index == 1  # message
        assert added[2].output_index == 2  # tool call

    def test_multiple_function_calls(self):
        """Multiple function calls each get unique indices."""
        chunks = [
            _fc_chunk(
                "search",
                {"q": "foo"},
                call_id="fc_a",
            ),
            _fc_chunk(
                "fetch",
                {"url": "http://example.com"},
                call_id="fc_b",
            ),
            _final_chunk(),
        ]
        events = self._run(chunks)

        arg_dones = [e for e in events if isinstance(e, FunctionCallArgumentsDone)]
        assert len(arg_dones) == 2
        assert arg_dones[0].name == "search"
        assert arg_dones[1].name == "fetch"


# ==== URL Context extraction ====


class TestUrlContextExtraction:
    """UrlContextMetadata on candidate → WebSearchCallItem(ActionOpenPage)."""

    def test_success(self):
        url_ctx = UrlContextMetadata(
            url_metadata=[
                UrlMetadata(
                    retrieved_url="https://example.com/page",
                    url_retrieval_status=UrlRetrievalStatus.URL_RETRIEVAL_STATUS_SUCCESS,
                ),
            ]
        )
        resp = _make_response(
            [Part(text="Page content here")],
            url_context_metadata=url_ctx,
        )
        items, _ = _gemini_response_to_items_and_web_search_info(resp)

        assert len(items) == 2
        assert isinstance(items[0], OutputMessageItem)
        wf = items[1]
        assert isinstance(wf, WebSearchCallItem)
        assert wf.status == "completed"
        assert isinstance(wf.action, ActionOpenPage)
        assert wf.action.url == "https://example.com/page"
        assert wf.provider_specific_fields is None

    def test_multiple_urls(self):
        url_ctx = UrlContextMetadata(
            url_metadata=[
                UrlMetadata(
                    retrieved_url="https://a.com",
                    url_retrieval_status=UrlRetrievalStatus.URL_RETRIEVAL_STATUS_SUCCESS,
                ),
                UrlMetadata(
                    retrieved_url="https://b.com",
                    url_retrieval_status=UrlRetrievalStatus.URL_RETRIEVAL_STATUS_SUCCESS,
                ),
            ]
        )
        resp = _make_response(
            [Part(text="Content")],
            url_context_metadata=url_ctx,
        )
        items, _ = _gemini_response_to_items_and_web_search_info(resp)

        wf_items = [i for i in items if isinstance(i, WebSearchCallItem)]
        assert len(wf_items) == 2
        assert wf_items[0].action.url == "https://a.com"  # type: ignore[union-attr]
        assert wf_items[1].action.url == "https://b.com"  # type: ignore[union-attr]

    def test_error_status(self):
        url_ctx = UrlContextMetadata(
            url_metadata=[
                UrlMetadata(
                    retrieved_url="https://unreachable.invalid",
                    url_retrieval_status=UrlRetrievalStatus.URL_RETRIEVAL_STATUS_ERROR,
                ),
            ]
        )
        resp = _make_response(
            [Part(text="Could not fetch")],
            url_context_metadata=url_ctx,
        )
        items, _ = _gemini_response_to_items_and_web_search_info(resp)

        wf = [i for i in items if isinstance(i, WebSearchCallItem)][0]
        assert wf.status == "failed"
        assert isinstance(wf.action, ActionOpenPage)
        assert wf.provider_specific_fields is not None
        assert "gemini:url_retrieval_status" in wf.provider_specific_fields

    def test_paywall_status(self):
        url_ctx = UrlContextMetadata(
            url_metadata=[
                UrlMetadata(
                    retrieved_url="https://paywalled.com",
                    url_retrieval_status=UrlRetrievalStatus.URL_RETRIEVAL_STATUS_PAYWALL,
                ),
            ]
        )
        resp = _make_response(
            [Part(text="Paywalled")],
            url_context_metadata=url_ctx,
        )
        items, _ = _gemini_response_to_items_and_web_search_info(resp)

        wf = [i for i in items if isinstance(i, WebSearchCallItem)][0]
        assert wf.status == "failed"

    def test_empty(self):
        resp = _make_response([Part(text="No url context")])
        items, _ = _gemini_response_to_items_and_web_search_info(resp)

        wf_items = [i for i in items if isinstance(i, WebSearchCallItem)]
        assert len(wf_items) == 0

    def test_with_grounding(self):
        """url_context + grounding_metadata → both produce items."""
        grounding = GroundingMetadata(
            grounding_chunks=[
                GroundingChunk(
                    web=GroundingChunkWeb(
                        uri="https://search-result.com",
                        title="Search Result",
                    )
                ),
            ],
            web_search_queries=["test query"],
        )
        url_ctx = UrlContextMetadata(
            url_metadata=[
                UrlMetadata(
                    retrieved_url="https://fetched.com",
                    url_retrieval_status=UrlRetrievalStatus.URL_RETRIEVAL_STATUS_SUCCESS,
                ),
            ]
        )
        resp = _make_response(
            [Part(text="Combined results")],
            grounding_metadata=grounding,
            url_context_metadata=url_ctx,
        )
        items, web_search_info = _gemini_response_to_items_and_web_search_info(resp)

        ws_items = [i for i in items if isinstance(i, WebSearchCallItem)]
        assert len(ws_items) == 2
        assert web_search_info is not None


# ==== URL Context streaming ====


class TestUrlContextStream:
    """Streaming url_context → WebSearchCallItem events."""

    def _run(self, chunks: list[GenerateContentResponse]) -> list[Any]:
        return asyncio.get_event_loop().run_until_complete(_collect_events(chunks))

    def test_success(self):
        url_ctx = UrlContextMetadata(
            url_metadata=[
                UrlMetadata(
                    retrieved_url="https://example.com/streamed",
                    url_retrieval_status=UrlRetrievalStatus.URL_RETRIEVAL_STATUS_SUCCESS,
                ),
            ]
        )
        chunks = [
            _text_chunk("Hello"),
            _final_chunk(url_context_metadata=url_ctx),
        ]
        events = self._run(chunks)

        ws_done = [
            e
            for e in events
            if isinstance(e, OutputItemDone)
            and isinstance(e.item, WebSearchCallItem)
        ]
        assert len(ws_done) == 1
        wf = ws_done[0].item
        assert isinstance(wf, WebSearchCallItem)
        assert isinstance(wf.action, ActionOpenPage)
        assert wf.action.url == "https://example.com/streamed"
        assert wf.status == "completed"

    def test_error(self):
        url_ctx = UrlContextMetadata(
            url_metadata=[
                UrlMetadata(
                    retrieved_url="https://bad.invalid",
                    url_retrieval_status=UrlRetrievalStatus.URL_RETRIEVAL_STATUS_ERROR,
                ),
            ]
        )
        chunks = [
            _text_chunk("Failed"),
            _final_chunk(url_context_metadata=url_ctx),
        ]
        events = self._run(chunks)

        ws_done = [
            e
            for e in events
            if isinstance(e, OutputItemDone)
            and isinstance(e.item, WebSearchCallItem)
        ]
        assert len(ws_done) == 1
        wf = ws_done[0].item
        assert isinstance(wf, WebSearchCallItem)
        assert wf.status == "failed"
        assert isinstance(wf.action, ActionOpenPage)
        assert wf.provider_specific_fields is not None
