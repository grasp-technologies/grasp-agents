"""Unit tests for :class:`FileSearchToolkit`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from grasp_agents.tools.file_search import (
    FileSearchToolkit,
    GlobTool,
    GrepTool,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_tools_are_glob_and_grep(tmp_path: Path) -> None:
    toolkit = FileSearchToolkit(allowed_roots=[tmp_path])
    tools = toolkit.tools()
    assert len(tools) == 2
    kinds = {type(t).__name__ for t in tools}
    assert kinds == {"GlobTool", "GrepTool"}


def test_accessors(tmp_path: Path) -> None:
    toolkit = FileSearchToolkit(allowed_roots=[tmp_path])
    assert isinstance(toolkit.glob, GlobTool)
    assert isinstance(toolkit.grep, GrepTool)


def test_default_allowed_roots_is_cwd() -> None:
    toolkit = FileSearchToolkit()
    # Internal field is a list[Path]; public surface doesn't expose it
    # directly, but we want to verify the default applies.
    assert len(toolkit._allowed_roots) == 1  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]


def test_timeout_propagates_to_tools(tmp_path: Path) -> None:
    toolkit = FileSearchToolkit(allowed_roots=[tmp_path], tool_timeout=5.0)
    assert toolkit.glob.timeout == 5.0
    assert toolkit.grep.timeout == 5.0


@pytest.mark.parametrize("include_hidden", [True, False])
def test_glob_hidden_flag_propagates(tmp_path: Path, include_hidden: bool) -> None:
    toolkit = FileSearchToolkit(
        allowed_roots=[tmp_path], glob_include_hidden=include_hidden
    )
    assert toolkit.glob._include_hidden is include_hidden  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
