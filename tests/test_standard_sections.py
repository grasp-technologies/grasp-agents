"""Tests for the env_info and mcp_instructions standard sections."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from grasp_agents import make_env_info_section

if TYPE_CHECKING:
    from grasp_agents.agent.prompt_builder import SystemPromptSection

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
        assert section.cache_break is False
        text = _run_compute(section)
        assert text is not None
        assert "## Environment" in text
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

    def test_cache_break_propagates(self) -> None:
        section = make_env_info_section(
            include=("datetime",), cache_break=True
        )
        assert section.cache_break is True

    def test_custom_section_name(self) -> None:
        section = make_env_info_section(section_name="my_env")
        assert section.name == "my_env"


# ---------- mcp_instructions ----------


@pytest.fixture
def mcp_section_factory() -> Any:
    pytest.importorskip("mcp")
    from grasp_agents import make_mcp_instructions_section  # noqa: PLC0415

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
        assert section.cache_break is True
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
        section = mcp_section_factory(
            [_StubClient("alpha", "Connect with --auth")]
        )
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
