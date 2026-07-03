"""
Tests for the ephemeral initial context and the ``InitialContextBuilder`` hook.

The initial context (system-prompt message + leading messages) is composed
fresh each step and prepended to the model-facing view — it is NOT stored in
the transcript log (which stays pure conversation). A registered
``InitialContextBuilder`` transforms the default initial context (augment,
replace, reorder), and the system prompt's per-section ``cache_control`` is
preserved on the composed system message.
"""

from __future__ import annotations

from typing import Any

import pytest

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.context.prompt_builder import SystemPromptSection
from grasp_agents.types.content import CacheControl, InputText
from grasp_agents.types.items import InputItem, InputMessageItem

# Reuse the hand-rolled MockLLM / text-response helpers.
from tests.durability.test_sessions import (  # type: ignore[attr-defined]  # pyright: ignore[reportPrivateUsage]
    MockLLM,
    _text_response,
)


def _cached_section() -> SystemPromptSection:
    def compute(*, ctx: Any = None, exec_id: Any = None, **_: Any) -> str:
        del ctx, exec_id
        return "Stable preamble."

    return SystemPromptSection(
        name="cached", compute=compute, cache_control=CacheControl(ttl="1h")
    )


def _agent(**kwargs: Any) -> LLMAgent[str, str, None]:
    return LLMAgent[str, str, None](
        name="a",
        llm=MockLLM(responses_queue=[_text_response("ok")]),
        env_info=False,
        **kwargs,
    )


class TestEphemeralInitialContext:
    @pytest.mark.asyncio
    async def test_system_prompt_is_ephemeral_not_in_log(self) -> None:
        agent = _agent(sys_prompt="Base.")
        await agent.run("hi")

        # The log is pure conversation — no system message stored.
        assert all(
            not (isinstance(m, InputMessageItem) and m.role == "system")
            for m in agent.transcript.messages
        )
        # The system prompt lives in the ephemeral header (prepended to the view).
        header = agent._cw.initial_context
        assert len(header) == 1
        assert isinstance(header[0], InputMessageItem)
        assert header[0].role == "system"
        assert header[0].text == "Base."


class TestInitialContextBuilder:
    @pytest.mark.asyncio
    async def test_builder_receives_default_with_cache_control(self) -> None:
        agent = _agent(sys_prompt="Base prompt.")
        agent.add_system_prompt_section(_cached_section())

        received: list[list[InputItem]] = []

        @agent.add_initial_context_builder
        async def build(messages: list[InputItem], *, exec_id: str) -> list[InputItem]:
            del exec_id
            received.append(list(messages))
            return messages

        await agent.run("hi")

        # The default = one system message from sys_prompt + the section, with
        # the section's cache_control preserved on its part.
        assert len(received) == 1
        header = received[0]
        assert len(header) == 1
        sys_msg = header[0]
        assert isinstance(sys_msg, InputMessageItem)
        assert sys_msg.role == "system"
        assert [p.text for p in sys_msg.content if isinstance(p, InputText)] == [
            "Base prompt.",
            "Stable preamble.",
        ]
        cached_part = sys_msg.content[1]
        assert isinstance(cached_part, InputText)
        assert cached_part.cache_control == CacheControl(ttl="1h")

    @pytest.mark.asyncio
    async def test_builder_augments_with_leading_message(self) -> None:
        agent = _agent(sys_prompt="Base.")
        reminder = InputMessageItem.from_text("Remember X.", role="user")

        @agent.add_initial_context_builder
        async def build(messages: list[InputItem], *, exec_id: str) -> list[InputItem]:
            del exec_id
            return [*messages, reminder]

        await agent.run("hi")

        header = agent._cw.initial_context
        assert len(header) == 2
        assert isinstance(header[0], InputMessageItem)
        assert header[0].role == "system"
        assert header[1] is reminder
        # The leading message is ephemeral — not appended to the conversation log.
        assert reminder not in agent.transcript.messages

    @pytest.mark.asyncio
    async def test_subclass_override_replaces_initial_context(self) -> None:
        seen: list[int] = []

        class _MyAgent(LLMAgent[str, str, None]):
            async def build_initial_context_impl(
                self, messages: list[InputItem], *, exec_id: str
            ) -> list[InputItem]:
                del exec_id
                seen.append(len(messages))  # received the default [system message]
                return [InputMessageItem.from_text("Custom system.", role="system")]

        agent = _MyAgent(
            name="a",
            llm=MockLLM(responses_queue=[_text_response("ok")]),
            sys_prompt="Base.",
            env_info=False,
        )
        await agent.run("hi")

        assert seen == [1]
        header = agent._cw.initial_context
        assert len(header) == 1
        assert isinstance(header[0], InputMessageItem)
        assert header[0].text == "Custom system."
