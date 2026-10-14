"""
Starting over: what resetting or replacing an agent's conversation keeps,
drops and persists.
"""

from collections.abc import Iterable
from typing import Any

import pytest

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.durability import InMemoryCheckpointStore
from grasp_agents.evals.task import (
    _reset_transcripts,  # pyright: ignore[reportPrivateUsage]
)
from grasp_agents.session_context import SessionContext
from grasp_agents.tools.processor_tool import ProcessorTool
from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.folds import FoldSpec
from grasp_agents.types.items import InputItem, InputMessageItem, OutputMessageItem
from tests._helpers import MockLLM, _text_response
from tests.durability.test_sessions import _make_agent, load_agent_checkpoint

_KEY = "s1/agent/test_agent"


def _user(text: str) -> InputMessageItem:
    return InputMessageItem.from_text(text, role="user")


def _assistant(text: str) -> OutputMessageItem:
    return OutputMessageItem(content=[OutputMessageText(text=text)], status="completed")


def _user_texts(messages: Iterable[InputItem]) -> list[str]:
    return [
        m.text for m in messages if isinstance(m, InputMessageItem) and m.role == "user"
    ]


def _fold() -> FoldSpec:
    return FoldSpec(start=0, end=2, summary="an earlier conversation")


def _bare_agent() -> LLMAgent[str, str, None]:
    return LLMAgent[str, str, None](
        name="worker",
        ctx=SessionContext[None](state=None),
        llm=MockLLM(responses_queue=[]),
        stream_llm=True,
    )


class TestResetDropsFolds:
    def test_reset_transcript(self) -> None:
        agent, _ = _make_agent([])
        agent.replace_transcript([_user("q"), _assistant("a")])
        agent.agent_ctx.cw.folds = [_fold()]

        agent.reset_transcript()

        assert agent.transcript.is_empty
        assert agent.agent_ctx.cw.folds == []

    @pytest.mark.asyncio
    async def test_reset_on_run(self) -> None:
        agent, _ = _make_agent(
            [_text_response("one"), _text_response("two")],
            reset_transcript_on_run=True,
        )
        await agent.run("first")
        agent.agent_ctx.cw.folds = [_fold()]

        await agent.run("second")

        assert agent.agent_ctx.cw.folds == []
        assert _user_texts(agent.transcript) == ["second"]

    def test_processor_tool_clone(self) -> None:
        template = _bare_agent()
        template.agent_ctx.cw.add_messages([_user("q"), _assistant("a")])
        template.agent_ctx.cw.folds = [_fold()]
        tool = ProcessorTool[Any, Any, None](
            processor=template, name="sub", description="A sub-agent."
        )

        clone = tool._resolve_processor(  # pyright: ignore[reportPrivateUsage]
            ctx=SessionContext[None](state=None), path=["parent", "sub"]
        )

        assert isinstance(clone, LLMAgent)
        assert clone.transcript.is_empty
        assert clone.agent_ctx.cw.folds == []
        assert template.agent_ctx.cw.folds == [_fold()]  # the clone is a copy

    def test_evals_trial_reset(self) -> None:
        agent = _bare_agent()
        agent.agent_ctx.cw.add_messages([_user("q"), _assistant("a")])
        agent.agent_ctx.cw.folds = [_fold()]

        _reset_transcripts(agent)

        assert agent.transcript.is_empty
        assert agent.agent_ctx.cw.folds == []


class TestResetIsNotUndoneByReload:
    @pytest.mark.asyncio
    async def test_next_run_keeps_the_fresh_conversation(self) -> None:
        store = InMemoryCheckpointStore()
        agent, _ = _make_agent(
            [_text_response("ans1"), _text_response("ans2")],
            session_key="s1",
            store=store,
        )
        await agent.run("first")

        agent.reset_transcript()
        await agent.run("second")

        assert _user_texts(agent.transcript) == ["second"]

        # The persisted log was rewritten, not appended to.
        head = await load_agent_checkpoint(store, _KEY)
        assert head is not None
        assert _user_texts(head.messages) == ["second"]

        cold, _ = _make_agent([], session_key="s1", store=store)
        assert await cold.load_checkpoint() is not None
        assert _user_texts(cold.transcript) == ["second"]


class TestReplaceOnAColdInstance:
    @pytest.mark.asyncio
    async def test_replaced_conversation_overwrites_the_stored_one(self) -> None:
        store = InMemoryCheckpointStore()
        old, _ = _make_agent(
            [_text_response("old answer")], session_key="s1", store=store
        )
        await old.run("old question")

        fresh, _ = _make_agent(
            [_text_response("next answer")], session_key="s1", store=store
        )
        fresh.replace_transcript([_user("seed question"), _assistant("seed answer")])
        await fresh.run("next")

        assert _user_texts(fresh.transcript) == ["seed question", "next"]

        head = await load_agent_checkpoint(store, _KEY)
        assert head is not None
        assert head.current.message_count == len(head.messages) == 4
        raw = await store.read_messages(_KEY, version=head.current.log_version)
        assert len(raw) == head.current.message_count  # nothing past the head
        assert "old question" not in _user_texts(raw)

        cold, _ = _make_agent([], session_key="s1", store=store)
        assert await cold.load_checkpoint() is not None
        assert _user_texts(cold.transcript) == ["seed question", "next"]


class TestRollbackHistory:
    @pytest.mark.asyncio
    async def test_replace_drops_boundaries_of_the_old_log(self) -> None:
        store = InMemoryCheckpointStore()
        agent, _ = _make_agent(
            [_text_response("a0"), _text_response("a1"), _text_response("a2")],
            session_key="s1",
            store=store,
        )
        await agent.run("q0", step=0)
        await agent.run("q1", step=1)

        fresh: list[InputItem] = [_user("fresh"), _assistant("fresh answer")]
        agent.replace_transcript(fresh)
        with pytest.raises(KeyError):
            await agent.rollback_to_step(0)

        # Boundaries archived after the replace index the new log.
        await agent.run("q2", step=2)
        await agent.rollback_to_step(2)
        assert list(agent.transcript) == fresh

    @pytest.mark.asyncio
    async def test_reset_drops_boundaries_of_the_old_log(self) -> None:
        agent, _ = _make_agent([_text_response("a0"), _text_response("a1")])
        await agent.run("q0", step=0)
        await agent.run("q1", step=1)

        agent.reset_transcript()
        with pytest.raises(KeyError):
            await agent.rollback_to_step(1)

    @pytest.mark.asyncio
    async def test_reset_on_run_agent_keeps_its_step_boundaries(self) -> None:
        store = InMemoryCheckpointStore()
        agent, _ = _make_agent(
            [_text_response(a) for a in ("a0", "a1", "a2", "a1_new")],
            session_key="s1",
            store=store,
            reset_transcript_on_run=True,
        )
        await agent.run("q0", step=0)
        await agent.run("q1", step=1)
        await agent.run("q2", step=2)

        await agent.rollback_to_step(1)

        assert agent.step == 1
        assert agent.transcript.is_empty  # each step started from an empty log
        rerun = await agent.run("q1", step=1)
        assert rerun.payloads[0] == "a1_new"
