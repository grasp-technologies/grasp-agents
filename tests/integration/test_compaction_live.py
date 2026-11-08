"""
Live compaction on real providers: the summary request is the agent's own
request up to the fold's end plus a compaction request, sent with the agent's
tools and reasoning settings, and the run carries on from the folded view. With
Anthropic's conversation caching (on by default), the summary request reads the
conversation from the prompt cache.

``integration``-gated; each case skips without its provider key. Run:

    uv run pytest -m integration tests/integration/test_compaction_live.py --tb=short
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from grasp_agents import LLMAgent, SessionContext
from grasp_agents.context import (
    COMPACTION_REQUEST_TAG,
    Compactor,
    ContextBudget,
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
from grasp_agents.types.response import Response

if TYPE_CHECKING:
    from grasp_agents.llm.llm import LLM

pytestmark = [pytest.mark.integration]

# Several thousand tokens: past every provider's minimum cacheable prompt length.
_NOTES = "My notes for later. The launch code is ORCHID-7. " + " ".join(
    f"Paragraph {i}: the committee reviewed the budget, the venue and the "
    "schedule, and deferred every decision to the next meeting."
    for i in range(250)
)

# Over a quarter of the compaction target: enough new conversation to fold again.
_AGENDA = "More notes. " + " ".join(
    f"Agenda item {i}: the team discussed catering, parking and signage, and "
    "asked for quotes."
    for i in range(30)
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


# provider -> (key fixture, LLM builder, whether a cache hit is guaranteed:
# Anthropic caches the conversation by default, OpenAI's caching is best-effort)
_PROVIDERS = {
    "anthropic": ("anthropic_api_key", _anthropic, True),
    "openai_responses": ("openai_api_key", _openai_responses, False),
}


@pytest.mark.parametrize("provider", list(_PROVIDERS))
@pytest.mark.asyncio
async def test_compaction_summarizes_on_the_agents_own_request(
    provider: str, request: pytest.FixtureRequest
) -> None:
    key_fixture, build, caches = _PROVIDERS[provider]
    llm = build(request.getfixturevalue(key_fixture))
    ctx: SessionContext[None] = SessionContext()
    agent = LLMAgent[str, str, None](
        name="assistant",
        ctx=ctx,
        llm=llm,
        tools=[lookup],
        sys_prompt="You are a concise assistant. Answer in one short sentence.",
        stream_llm=True,
    )
    # The long first note puts the view over the threshold from the next
    # question on. While it is the newest user message it stays pinned; once a
    # later question is, it is too long to pin and only the summary carries it.
    # The second question adds enough to fold again although the pinned note
    # keeps the view over the threshold.
    agent.add_compaction(
        Compactor(
            budget=ContextBudget(llm.model_name, max_input_tokens=200_000),
            target_tokens=1500,
            keep_recent_rounds=1,
            pinned_user_tokens=100,
        )
    )

    paid: list[Response] = []

    @agent.add_usage_hook
    async def _paid(response: Response, *, exec_id: str) -> None:
        del exec_id
        paid.append(response)

    events: list[Event[Any]] = []
    for question in (
        f"{_NOTES}\n\nReply with just: noted.",
        f"{_AGENDA}\n\nReply with just: ok.",
        "What is the launch code?",
    ):
        events += [event async for event in agent.run_stream(question)]

    starts = [i for i, e in enumerate(events) if isinstance(e, CompactionStartEvent)]
    dones = [i for i, e in enumerate(events) if isinstance(e, CompactionDoneEvent)]
    assert len(dones) == 2
    assert all(s < d for s, d in zip(starts, dones, strict=True))
    for i in dones:
        done = events[i]
        assert isinstance(done, CompactionDoneEvent)
        assert done.data.summary.strip()

    # The summary calls are extra LLM calls: counted in the usage, seen by the
    # usage hooks, never written to the transcript, and the run answers from the
    # folded view afterwards.
    generations = [e.data for e in events if isinstance(e, GenerationEndEvent)]
    total = ctx.usage_tracker.total_usage
    in_turns = sum(g.usage.input_tokens for g in generations if g.usage)
    assert total.input_tokens > in_turns
    assert sum(r.usage.input_tokens for r in paid if r.usage) == total.input_tokens
    if caches:
        cached_in_turns = sum(
            g.usage.input_tokens_details.cached_tokens for g in generations if g.usage
        )
        assert total.input_tokens_details.cached_tokens > cached_in_turns
    assert not any(
        isinstance(m, InputMessageItem) and COMPACTION_REQUEST_TAG in m.text
        for m in agent.transcript
    )
    last = agent.transcript[-1]
    assert isinstance(last, OutputMessageItem)
    assert "ORCHID-7" in last.text
