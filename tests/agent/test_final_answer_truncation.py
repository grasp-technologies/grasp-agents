"""
A final answer taken from a truncated response says so, to a custom output
parser as well, unless it is parsed against a structured output schema.
"""

from __future__ import annotations

import pytest
from openai.types.responses.response import IncompleteDetails
from pydantic import BaseModel

from grasp_agents import LLMAgent, SessionContext
from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.items import OutputMessageItem
from grasp_agents.types.response import Response
from tests._helpers import MockLLM

_NOTICE = "\n\n[Response truncated: max_output_tokens]"


class _Answer(BaseModel):
    v: int


def _response(text: str, *, truncated: bool) -> Response:
    return Response(
        model="mock",
        output=[
            OutputMessageItem(
                content=[OutputMessageText(text=text)], status="completed"
            )
        ],
        status="incomplete" if truncated else "completed",
        incomplete_details=(
            IncompleteDetails(reason="max_output_tokens") if truncated else None
        ),
    )


@pytest.mark.asyncio
async def test_text_output_says_it_was_truncated() -> None:
    agent = LLMAgent[str, str, None](
        name="writer",
        ctx=SessionContext[None](state=None),
        llm=MockLLM(responses_queue=[_response("The sky is", truncated=True)]),
    )

    out = await agent.run("why is the sky blue?")

    assert out.payloads == ["The sky is" + _NOTICE]


@pytest.mark.asyncio
async def test_complete_text_output_is_unchanged() -> None:
    agent = LLMAgent[str, str, None](
        name="writer",
        ctx=SessionContext[None](state=None),
        llm=MockLLM(responses_queue=[_response("Rayleigh.", truncated=False)]),
    )

    out = await agent.run("why is the sky blue?")

    assert out.payloads == ["Rayleigh."]


@pytest.mark.asyncio
async def test_structured_output_is_parsed_from_the_raw_answer() -> None:
    agent = LLMAgent[str, _Answer, None](
        name="solver",
        ctx=SessionContext[None](state=None),
        llm=MockLLM(responses_queue=[_response('{"v": 1}', truncated=True)]),
    )

    out = await agent.run("solve")

    assert out.payloads == [_Answer(v=1)]


@pytest.mark.asyncio
async def test_custom_parser_of_a_text_output_sees_the_notice() -> None:
    agent = LLMAgent[str, str, None](
        name="writer",
        ctx=SessionContext[None](state=None),
        llm=MockLLM(responses_queue=[_response("<a>x</a>", truncated=True)]),
    )
    received: list[str] = []

    @agent.add_output_parser
    def _parse(final_answer: str, **_: object) -> str:  # pyright: ignore[reportUnusedFunction]
        received.append(final_answer)
        return final_answer

    await agent.run("tag it")

    assert received == ["<a>x</a>" + _NOTICE]
