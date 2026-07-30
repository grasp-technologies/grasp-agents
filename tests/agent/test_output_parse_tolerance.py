"""
The agent's output parse tracks its LLM's trailing-output tolerance.

`LLM._validate_response` and `LLMAgent.parse_output_default` both parse the
final answer. If only the first tolerates a provider that closes the JSON value
and keeps emitting, the second turns a response the LLM just accepted into a
failed run — worse than the re-sample the tolerance was meant to avoid, since
an agent defaults to `max_retries=0`.
"""

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.llm.llm import LLM
from grasp_agents.tools.base import BaseTool
from grasp_agents.types.errors import JSONSchemaValidationError
from grasp_agents.types.llm_events import LlmEvent
from grasp_agents.types.response import Response


@dataclass(frozen=True)
class _StubLLM(LLM):
    """Never called: the agent is built only to reach its output parser."""

    model_name: str = "stub"

    async def _generate_response_once(
        self,
        input: Sequence[Any],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra_llm_settings: Any,
    ) -> Response:
        raise NotImplementedError

    async def _generate_response_stream_once(
        self,
        input: Sequence[Any],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra_llm_settings: Any,
    ) -> AsyncIterator[LlmEvent]:
        raise NotImplementedError
        yield  # type: ignore[unreachable]  # makes this an async generator


class Answer(BaseModel):
    capital: str
    population_millions: int


_GOOD = '{"capital":"Paris","population_millions":68}'
# What Bedrock's decoder emits for some models: a complete object, then one
# redundant closing brace.
_WITH_STRAY_BRACE = _GOOD + "}"


def _agent(*, tolerate: bool) -> LLMAgent[str, Answer, None]:
    return LLMAgent[str, Answer, None](
        name=f"parser_agent_{tolerate}",
        llm=_StubLLM(tolerate_output_around_json=tolerate),
    )


class TestOutputParseTolerance:
    def test_tolerating_llm_parses_a_stray_brace(self) -> None:
        parsed = _agent(tolerate=True).parse_output_default(_WITH_STRAY_BRACE)
        assert parsed.capital == "Paris"
        assert parsed.population_millions == 68

    def test_strict_llm_still_rejects_it(self) -> None:
        with pytest.raises(JSONSchemaValidationError):
            _agent(tolerate=False).parse_output_default(_WITH_STRAY_BRACE)

    def test_clean_output_parses_either_way(self) -> None:
        for tolerate in (True, False):
            assert _agent(tolerate=tolerate).parse_output_default(_GOOD).capital == (
                "Paris"
            )

    def test_tolerance_does_not_accept_a_missing_field(self) -> None:
        """Only content around the value is forgiven, never the value itself."""
        with pytest.raises(JSONSchemaValidationError):
            _agent(tolerate=True).parse_output_default('{"capital":"Paris"}')
