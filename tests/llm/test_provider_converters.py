"""
Provider converter behavior:

* empty / ``None`` tool-call arguments normalize to ``"{}"`` and never crash
  the next request build (streamed and non-streamed)
* ``InputFile`` parts convert for Anthropic / Gemini / Chat Completions
  instead of being silently dropped
* Gemini honors ``apply_output_schema_via_provider=False`` (the default):
  the schema is not baked into the request config
"""

from __future__ import annotations

import base64

from google.genai.types import FunctionCall as GeminiFunctionCall
from pydantic import BaseModel

from grasp_agents.llm.llm_stream_converter import ToolCallState
from grasp_agents.llm_providers.anthropic.response_to_provider_inputs import (
    items_to_provider_inputs as anthropic_items_to_inputs,
)
from grasp_agents.llm_providers.gemini.provider_output_to_response import (
    _function_call_to_tool_call_item,  # pyright: ignore[reportPrivateUsage]
)
from grasp_agents.llm_providers.gemini.response_to_provider_inputs import (
    _file_to_part,  # pyright: ignore[reportPrivateUsage]
    _tool_call_to_part,  # pyright: ignore[reportPrivateUsage]
)
from grasp_agents.llm_providers.openai_completions.response_to_provider_inputs import (
    items_to_provider_inputs as completions_items_to_inputs,
)
from grasp_agents.types.content import InputFile, InputText
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
)

_PDF_B64 = base64.b64encode(b"%PDF-1.4 fake").decode()


def _file_part() -> InputFile:
    return InputFile(filename="doc.pdf", file_data=_PDF_B64)


# ---------------------------------------------------------------------------
# Empty tool-call arguments
# ---------------------------------------------------------------------------


class TestEmptyToolCallArgs:
    def test_stream_state_with_no_deltas_commits_empty_object(self) -> None:
        state = ToolCallState(item_index=0, item_id="i1", call_id="c1", name="t")
        assert (state.arguments.strip() or "{}") == "{}"

    def test_gemini_none_args_serialize_as_empty_object(self) -> None:
        item = _function_call_to_tool_call_item(  # type: ignore[call-arg]
            function_call=GeminiFunctionCall(id="c1", name="t", args=None),
            thought_sig=None,
        )
        assert item.arguments == "{}"

    def test_gemini_request_build_tolerates_empty_args(self) -> None:
        for raw in ("", "null"):
            part = _tool_call_to_part(
                FunctionToolCallItem(call_id="c1", name="t", arguments=raw)
            )
            assert part.function_call is not None
            assert part.function_call.args == {}

    def test_anthropic_request_build_tolerates_empty_args(self) -> None:
        items = [
            FunctionToolCallItem(call_id="c1", name="t", arguments=""),
            FunctionToolOutputItem(call_id="c1", output="done"),
        ]
        _, messages = anthropic_items_to_inputs(items)
        tool_use = next(
            block
            for msg in messages
            for block in msg["content"]
            if isinstance(block, dict) and block.get("type") == "tool_use"
        )
        assert tool_use["input"] == {}


# ---------------------------------------------------------------------------
# InputFile conversion
# ---------------------------------------------------------------------------


class TestInputFileConversion:
    def test_anthropic_converts_file_to_document_block(self) -> None:
        msg = InputMessageItem(
            role="user",
            content=[InputText(text="read this"), _file_part()],
        )
        _, messages = anthropic_items_to_inputs([msg])
        blocks = messages[0]["content"]
        assert isinstance(blocks, list)
        kinds = [b["type"] for b in blocks if isinstance(b, dict)]
        assert "document" in kinds
        doc = next(b for b in blocks if isinstance(b, dict) and b["type"] == "document")
        assert doc["source"]["type"] == "base64"
        assert doc["source"]["data"] == _PDF_B64

    def test_gemini_converts_file_to_inline_data(self) -> None:
        part = _file_to_part(_file_part())
        assert part.inline_data is not None
        assert part.inline_data.mime_type == "application/pdf"
        assert part.inline_data.data == base64.b64decode(_PDF_B64)

    def test_completions_converts_file_part(self) -> None:
        msg = InputMessageItem(
            role="user",
            content=[InputText(text="read this"), _file_part()],
        )
        params = completions_items_to_inputs([msg])
        content = params[0]["content"]
        assert isinstance(content, list)
        file_parts = [p for p in content if p.get("type") == "file"]
        assert len(file_parts) == 1
        assert file_parts[0]["file"]["file_data"] == _PDF_B64
        assert file_parts[0]["file"]["filename"] == "doc.pdf"


# ---------------------------------------------------------------------------
# Gemini output-schema gate
# ---------------------------------------------------------------------------


class _Out(BaseModel):
    answer: str


class TestGeminiSchemaGate:
    def _llm(self, **kwargs: object):
        from grasp_agents.llm_providers.gemini.gemini_llm import GeminiLLM

        return GeminiLLM(
            model_name="gemini-2.5-flash",
            api_provider={"name": "google", "base_url": None, "api_key": "dummy"},
            **kwargs,  # type: ignore[arg-type]
        )

    def test_schema_not_baked_into_config(self) -> None:
        llm = self._llm()
        params = llm._make_api_input([], output_schema=_Out)
        config = params["extra_settings"]["config"]  # type: ignore[index]
        # The schema travels via api_output_schema (strippable by the gate),
        # never pre-baked into the request config.
        assert config.response_schema is None
        assert config.response_mime_type is None
        assert params["api_output_schema"] is _Out

    def test_apply_output_schema_bakes_on_demand(self) -> None:
        llm = self._llm()
        config = llm._make_api_input([], output_schema=_Out)["extra_settings"][  # type: ignore[index]
            "config"
        ]
        applied = llm._apply_output_schema(config, _Out)
        assert applied is not None
        assert applied.response_schema is _Out
        assert applied.response_mime_type == "application/json"
