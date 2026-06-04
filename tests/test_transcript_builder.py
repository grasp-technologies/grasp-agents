"""
Tests for the ``@agent.add_transcript_builder`` hook (PR #23).

The hook receives the system prompt as parts (``Sequence[InputText]``), not a
flattened string, so a custom builder can preserve each part's
``CacheControl`` — the same contract :meth:`LLMAgentTranscript.reset` honors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.agent.prompt_builder import SystemPromptSection
from grasp_agents.types.content import CacheControl, InputText
from grasp_agents.types.items import InputMessageItem

# Reuse the hand-rolled MockLLM / text-response helpers.
from .test_sessions import (  # type: ignore[attr-defined]  # pyright: ignore[reportPrivateUsage]
    MockLLM,
    _text_response,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def _cached_section() -> SystemPromptSection:
    def compute(*, ctx: Any = None, exec_id: Any = None, **_: Any) -> str:
        del ctx, exec_id
        return "Stable preamble."

    return SystemPromptSection(
        name="cached", compute=compute, cache_control=CacheControl(ttl="1h")
    )


class TestTranscriptBuilderReceivesParts:
    @pytest.mark.anyio
    async def test_hook_receives_parts_with_cache_control(self) -> None:
        # env_info off so the parts list is exactly base + our section.
        agent = LLMAgent[str, str, None](
            name="a",
            llm=MockLLM(responses_queue=[_text_response("ok")]),
            sys_prompt="Base prompt.",
            env_info=False,
        )
        agent.add_system_prompt_section(_cached_section())

        received: list[Sequence[InputText] | str | None] = []

        def seed(
            *,
            instructions: str | Sequence[InputText] | None = None,
            in_args: str | None = None,
            exec_id: str,
        ) -> None:
            del in_args, exec_id
            received.append(instructions)
            # Forward to reset exactly as a real builder would.
            agent.transcript.reset(instructions)

        agent.add_transcript_builder(seed)

        await agent.run("hi")

        assert len(received) == 1
        parts = received[0]
        # Parts, not a joined string.
        assert isinstance(parts, list)
        assert all(isinstance(p, InputText) for p in parts)
        assert [p.text for p in parts] == ["Base prompt.", "Stable preamble."]
        # The section's cache_control survives into the hook's input.
        assert parts[0].cache_control is None
        assert parts[1].cache_control == CacheControl(ttl="1h")
        # And the forwarded reset preserved it on the seeded system message.
        sys_msg = agent.transcript.messages[0]
        assert isinstance(sys_msg, InputMessageItem)
        cached_part = sys_msg.content_parts[1]
        assert isinstance(cached_part, InputText)
        assert cached_part.cache_control == CacheControl(ttl="1h")

    @pytest.mark.anyio
    async def test_subclass_override_receives_parts(self) -> None:
        """Overriding ``build_transcript_impl`` on a subclass works too."""
        seen: list[Sequence[InputText] | str | None] = []

        class _MyAgent(LLMAgent[str, str, None]):
            def build_transcript_impl(
                self,
                *,
                instructions: str | Sequence[InputText] | None = None,
                in_args: str | None = None,
                exec_id: str,
            ) -> None:
                del in_args, exec_id
                seen.append(instructions)
                self.transcript.reset(instructions)

        agent = _MyAgent(
            name="a",
            llm=MockLLM(responses_queue=[_text_response("ok")]),
            sys_prompt="Base.",
            env_info=False,
        )

        await agent.run("hi")

        assert len(seen) == 1
        assert isinstance(seen[0], list)
        assert [p.text for p in seen[0]] == ["Base."]
