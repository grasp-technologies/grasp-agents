"""
Live compaction on real providers: the summary request is the agent's own
request plus a compaction request, sent with the agent's tools and reasoning
settings, and the run carries on from the folded view.

``integration``-gated; each case skips without its provider key. Run:

    uv run pytest -m integration tests/integration/test_compaction_live.py --tb=short
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from grasp_agents import LLMAgent, SessionContext
from grasp_agents.context import (
    COMPACTION_REQUEST_TAG,
    ContextBudget,
    SummarizingCompactor,
)
from grasp_agents.llm.cloud_llm import APIProvider
from grasp_agents.tools.function_tool import function_tool
from grasp_agents.types.events import (
    CompactionDoneEvent,
    CompactionStartEvent,
    Event,
    GenerationEndEvent,
)
from grasp_agents.types.items import InputMessageItem, OutputMessageItem

if TYPE_CHECKING:
    from grasp_agents.llm.llm import LLM
    from grasp_agents.types.response import Response

pytestmark = [pytest.mark.integration]

_NOTES = "My notes for later. The launch code is ORCHID-7. " + " ".join(
    f"Paragraph {i}: the committee reviewed the budget, the venue and the "
    "schedule, and deferred every decision to the next meeting."
    for i in range(60)
)


@function_tool
def lookup(topic: str) -> str:
    """Look up a short fact about a topic."""
    return f"No stored facts about {topic}."


def _anthropic(key: str) -> LLM:
    from grasp_agents.llm_providers.anthropic.anthropic_llm import (
        AnthropicLLM,
    )

    return AnthropicLLM(
        model_name="claude-haiku-5-5",
        api_provider=APIProvider(name="anthropic", base_url=None, api_key=key),
        llm_settings={
            "max_tokens": 8000,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "medium"},
        },
    )


def _openai_responses(key: str) -> LLM:
    from grasp_agents.llm_providers.openai_responses import (
        OpenAIResponsesLLM,
    )

    return OpenAIResponsesLLM(
        model_name="gpt-6-luna",
        api_provider=APIProvider(name="openai", base_url=None, api_key=key),
        llm_settings={"reasoning": {"effort": "medium"}},
    )


_PROVIDERS = {
    "anthropic": ("anthropic_api_key", _anthropic),
    "openai_responses": ("openai_api_key", _openai_responses),
}


@pytest.mark.parametrize("provider", list(_PROVIDERS))
@pytest.mark.asyncio
async def test_compaction_summarizes_on_the_agents_own_request(
    provider: str, request: pytest.FixtureRequest
) -> None:
    key_fixture, build = _PROVIDERS[provider]
    llm = build(request.getfixturevalue(key_fixture))
    agent = LLMAgent[str, str, None](
        name="assistant",
        ctx=SessionContext(),
        llm=llm,
        tools=[lookup],
        sys_prompt="You are a concise assistant. Answer in one short sentence.",
        stream_llm=True,
    )
    # The long first note puts the view over the threshold by the next question,
    # whatever the model does with tools. Too long to pin, it is summarized.
    agent.add_compaction(
        SummarizingCompactor(
            budget=ContextBudget(llm.model_name, max_input_tokens=200_000),
            target_tokens=1500,
            keep_recent_rounds=1,
            pinned_user_tokens=100,
        )
    )
    observed: list[Response] = []

    @agent.add_after_llm_hook
    async def _observe(response: Response, *, exec_id: str, turn: int) -> None:
        del exec_id, turn
        observed.append(response)

    events: list[Event[Any]] = []
    for question in (
        f"{_NOTES}\n\nReply with just: noted.",
        "What is the launch code?",
    ):
        events += [event async for event in agent.run_stream(question)]

    starts = [i for i, e in enumerate(events) if isinstance(e, CompactionStartEvent)]
    dones = [e for e in events if isinstance(e, CompactionDoneEvent)]
    assert len(dones) == 1
    assert starts
    assert starts[0] < events.index(dones[0])
    assert dones[0].data.summary.strip()
    assert dones[0].data.folded_rounds >= 1

    # The summary call is an extra LLM call: observed by the hooks, never written
    # to the transcript, and the run answers from the folded view afterwards.
    generations = sum(isinstance(e, GenerationEndEvent) for e in events)
    assert len(observed) > generations
    assert not any(
        isinstance(m, InputMessageItem) and COMPACTION_REQUEST_TAG in m.text
        for m in agent.transcript
    )
    last = agent.transcript[-1]
    assert isinstance(last, OutputMessageItem)
    assert last.text.strip()
