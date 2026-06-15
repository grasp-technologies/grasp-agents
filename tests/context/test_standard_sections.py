"""Tests for the env_info and mcp_instructions standard sections."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from grasp_agents import CacheControl
from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.context import (
    make_current_time_attachment,
    make_env_info_section,
)
from grasp_agents.types.items import InputMessageItem
from tests.durability.test_sessions import (  # type: ignore[attr-defined]  # pyright: ignore[reportPrivateUsage]
    MockLLM,
)

if TYPE_CHECKING:
    from grasp_agents.context.prompt_builder import SystemPromptSection

# ---------- env_info ----------


def _run_compute(section: SystemPromptSection) -> str | None:
    """Synchronously run a section's compute (handles sync or async)."""
    result = section.compute(ctx=None, exec_id=None)
    if inspect.isawaitable(result):
        msg = "expected sync compute"
        raise TypeError(msg)
    return result


class TestEnvInfoSection:
    def test_default_includes_date_platform_os_cwd(self) -> None:
        section = make_env_info_section()
        assert section.name == "env_info"
        assert section.cache_control is None
        text = _run_compute(section)
        assert text is not None
        assert "<environment>" in text
        assert "Date:" in text
        assert "Platform:" in text
        assert "OS:" in text
        assert "CWD:" in text
        # Date matches today.
        today = datetime.now(tz=UTC).date().isoformat()
        assert today in text

    def test_include_subset(self) -> None:
        section = make_env_info_section(include=("date",))
        text = _run_compute(section)
        assert text is not None
        assert "Date:" in text
        assert "Platform:" not in text
        assert "CWD:" not in text

    def test_model_field_renders_when_provided(self) -> None:
        section = make_env_info_section(
            include=("model",), model_name="claude-opus-4-7"
        )
        text = _run_compute(section)
        assert text is not None
        assert "Model: claude-opus-4-7" in text

    def test_model_field_skipped_when_unset(self) -> None:
        section = make_env_info_section(include=("model",))
        # No model name + no other fields → empty section.
        text = _run_compute(section)
        assert text is None

    def test_extra_fields_appended(self) -> None:
        section = make_env_info_section(
            include=("date",),
            extra_fields={"Project": "grasp-agents", "Branch": "main"},
        )
        text = _run_compute(section)
        assert text is not None
        assert "Project: grasp-agents" in text
        assert "Branch: main" in text

    def test_unknown_field_silently_ignored(self) -> None:
        section = make_env_info_section(include=("date", "nope"))
        text = _run_compute(section)
        assert text is not None
        assert "Date:" in text

    def test_empty_include_returns_none(self) -> None:
        section = make_env_info_section(include=())
        assert _run_compute(section) is None

    def test_cache_control_propagates(self) -> None:
        section = make_env_info_section(
            include=("datetime",), cache_control=CacheControl()
        )
        assert section.cache_control == CacheControl()

    def test_custom_section_name(self) -> None:
        section = make_env_info_section(section_name="my_env")
        assert section.name == "my_env"


class TestEnvInfoAgentWiring:
    """The LLMAgent ``env_info`` kwarg: bool toggle OR a configured section."""

    @staticmethod
    def _names(agent: LLMAgent[str, str, None]) -> set[str]:
        return {s.name for s in agent.system_prompt_sections}

    def test_true_attaches_default_section(self) -> None:
        agent = LLMAgent[str, str, None](
            name="a", llm=MockLLM(responses_queue=[]), env_info=True
        )
        assert "env_info" in self._names(agent)

    def test_false_attaches_nothing(self) -> None:
        agent = LLMAgent[str, str, None](
            name="a", llm=MockLLM(responses_queue=[]), env_info=False
        )
        assert "env_info" not in self._names(agent)

    def test_custom_section_used_verbatim(self) -> None:
        custom = make_env_info_section(
            section_name="my_env", include=("date", "model"), model_name="m"
        )
        agent = LLMAgent[str, str, None](
            name="a", llm=MockLLM(responses_queue=[]), env_info=custom
        )
        # The exact section object is registered; the default isn't added.
        assert custom in agent.system_prompt_sections
        assert "env_info" not in self._names(agent)


# ---------- current_time attachment ----------


class TestCurrentTimeAttachment:
    def test_default_name_and_live_stamp(self) -> None:
        att = make_current_time_attachment()
        assert att.name == "current_time"
        assert att.wrap_in_system_reminder is True
        msg = InputMessageItem.from_text("hi", role="user")
        out = att.compute(
            user_message=msg, ctx=None, exec_id=None, messages=None, agent_ctx=None
        )
        assert isinstance(out, str)
        assert out.startswith("Current time: ")
        # A live ISO wall-clock stamp (date + time), e.g. 2026-06-07T14:32:01+...
        assert "T" in out
        assert str(datetime.now(tz=UTC).year) in out

    def test_custom_name(self) -> None:
        assert make_current_time_attachment(name="clock").name == "clock"


class TestTimeAwareAgentWiring:
    """The LLMAgent ``time_aware`` kwarg: bool toggle OR a configured attachment."""

    @staticmethod
    def _att_names(agent: LLMAgent[str, str, None]) -> set[str]:
        return {
            a.name
            for a in agent._prompt_builder.input_attachments  # pyright: ignore[reportPrivateUsage]
        }

    def test_true_attaches_current_time(self) -> None:
        agent = LLMAgent[str, str, None](
            name="a", llm=MockLLM(responses_queue=[]), time_aware=True
        )
        assert "current_time" in self._att_names(agent)

    def test_false_attaches_nothing(self) -> None:
        agent = LLMAgent[str, str, None](
            name="a", llm=MockLLM(responses_queue=[]), time_aware=False
        )
        assert "current_time" not in self._att_names(agent)

    def test_custom_attachment_used_verbatim(self) -> None:
        custom = make_current_time_attachment(name="clock")
        agent = LLMAgent[str, str, None](
            name="a", llm=MockLLM(responses_queue=[]), time_aware=custom
        )
        attachments = agent._prompt_builder.input_attachments  # pyright: ignore[reportPrivateUsage]
        assert custom in attachments
        assert "current_time" not in {a.name for a in attachments}


# ---------- mcp_instructions ----------


@pytest.fixture
def mcp_section_factory() -> Any:
    pytest.importorskip("mcp")
    from grasp_agents.mcp import make_mcp_instructions_section

    return make_mcp_instructions_section


@dataclass
class _StubClient:
    name: str
    instructions: str | None


async def _await_compute(section: SystemPromptSection) -> str | None:
    result = section.compute(ctx=None, exec_id=None)
    if inspect.isawaitable(result):
        return await result
    return result


class TestMcpInstructionsSection:
    @pytest.mark.anyio
    async def test_no_clients_returns_none(self, mcp_section_factory: Any) -> None:
        section = mcp_section_factory([])
        assert section.cache_control == CacheControl()
        assert await _await_compute(section) is None

    @pytest.mark.anyio
    async def test_clients_with_no_instructions_omitted(
        self, mcp_section_factory: Any
    ) -> None:
        section = mcp_section_factory(
            [_StubClient("alpha", None), _StubClient("beta", "")]
        )
        assert await _await_compute(section) is None

    @pytest.mark.anyio
    async def test_single_client_renders(self, mcp_section_factory: Any) -> None:
        section = mcp_section_factory([_StubClient("alpha", "Connect with --auth")])
        text = await _await_compute(section)
        assert text is not None
        assert "## MCP server instructions" in text
        assert "### alpha" in text
        assert "Connect with --auth" in text

    @pytest.mark.anyio
    async def test_multiple_clients_concatenated(
        self, mcp_section_factory: Any
    ) -> None:
        section = mcp_section_factory(
            [
                _StubClient("alpha", "Use tool foo for queries."),
                _StubClient("beta", "Set BAR=1 first."),
            ]
        )
        text = await _await_compute(section)
        assert text is not None
        assert "### alpha" in text
        assert "### beta" in text
        assert text.index("### alpha") < text.index("### beta")
        assert "Use tool foo for queries." in text
        assert "Set BAR=1 first." in text

    @pytest.mark.anyio
    async def test_custom_section_name(self, mcp_section_factory: Any) -> None:
        section = mcp_section_factory(
            [_StubClient("alpha", "x")], section_name="mcp_extra"
        )
        assert section.name == "mcp_extra"
