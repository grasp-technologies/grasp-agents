"""MemberCard.from_processor: input_type derivation, skill extraction, and raises."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.agent_team.agent_card import MemberCard
from grasp_agents.processors.processor import Processor
from grasp_agents.run_context import RunContext
from tests._helpers import MockLLM


class _Ticket(BaseModel):
    title: str
    points: int


def _llm() -> MockLLM:
    return MockLLM(responses_queue=[])


def test_from_processor_takes_name_and_typed_input() -> None:
    agent = LLMAgent[_Ticket, Any, None](name="planner", llm=_llm())
    card = MemberCard.from_processor(agent)
    assert card.name == "planner"
    assert card.input_type is _Ticket


def test_from_processor_str_input_is_free_text() -> None:
    agent = LLMAgent[str, Any, None](name="chat", llm=_llm())
    assert MemberCard.from_processor(agent).input_type is str


def test_from_processor_any_input_is_free_text() -> None:
    # LLMAgent[Any, ...] resolves its input type to ``object`` — accepts anything, so
    # it is advertised as free text rather than a false typed contract.
    agent = LLMAgent[Any, Any, None](name="chat", llm=_llm())
    assert MemberCard.from_processor(agent).input_type is str


def test_from_processor_raises_on_incompatible_input() -> None:
    # A non-str / non-model InT can't be honestly advertised: a peer sending text
    # would fail the agent's own input validation. Raise rather than mislead.
    agent = LLMAgent[int, Any, None](name="counter", llm=_llm())
    with pytest.raises(ValueError, match="neither a str nor a BaseModel"):
        MemberCard.from_processor(agent)


def test_from_processor_derives_scoped_skills() -> None:
    agent = LLMAgent[Any, Any, None](
        name="worker", llm=_llm(), skill_include=["zeta", "alpha"]
    )
    # The agent's skill allowlist, sorted.
    assert MemberCard.from_processor(agent).skills == ["alpha", "zeta"]


def test_from_processor_unscoped_agent_has_no_static_skills() -> None:
    # An agent that sees the whole (runtime) catalog has no static allowlist.
    agent = LLMAgent[Any, Any, None](name="chat", llm=_llm())
    assert MemberCard.from_processor(agent).skills == []


def test_from_processor_skills_and_metadata_override() -> None:
    agent = LLMAgent[Any, Any, None](name="worker", llm=_llm(), skill_include=["zeta"])
    card = MemberCard.from_processor(
        agent, description="does work", skills=["custom"], resident=False
    )
    assert card.skills == ["custom"]
    assert card.description == "does work"
    assert card.resident is False


class _Forward(Processor[_Ticket, _Ticket, None]):
    async def _process(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[_Ticket] | None = None,
        exec_id: str,
        step: int | None = None,
    ) -> list[_Ticket]:
        del chat_inputs, exec_id, step
        return list(in_args or [])


def test_processor_rejects_slash_in_name() -> None:
    # A name becomes a store-key path segment (recipient / checkpoint path); a
    # slash would silently nest directories and break mailbox key scoping.
    with pytest.raises(ValueError, match="store-key path segment"):
        _Forward(name="a/b", ctx=RunContext[None](state=None))


def test_from_processor_works_on_plain_processor() -> None:
    # Not agent-specific: a plain processor yields a card too (no skills, typed input).
    proc = _Forward(name="filer", ctx=RunContext[None](state=None))
    card = MemberCard.from_processor(proc)
    assert card.name == "filer"
    assert card.input_type is _Ticket
    assert card.skills == []
