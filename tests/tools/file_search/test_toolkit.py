"""Unit tests for :class:`FileSearchToolkit`."""

from __future__ import annotations

import pytest

from grasp_agents.tools.file_search import (
    FileSearchToolkit,
    GlobTool,
    GrepTool,
)


def test_tools_are_glob_and_grep() -> None:
    toolkit = FileSearchToolkit()
    tools = toolkit.tools()
    assert len(tools) == 2
    kinds = {type(t).__name__ for t in tools}
    assert kinds == {"GlobTool", "GrepTool"}


def test_accessors() -> None:
    toolkit = FileSearchToolkit()
    assert isinstance(toolkit.glob, GlobTool)
    assert isinstance(toolkit.grep, GrepTool)


def test_timeout_propagates_to_tools() -> None:
    toolkit = FileSearchToolkit(tool_timeout=5.0)
    assert toolkit.glob.timeout == 5.0
    assert toolkit.grep.timeout == 5.0


@pytest.mark.parametrize("include_hidden", [True, False])
def test_glob_hidden_flag_propagates(include_hidden: bool) -> None:
    toolkit = FileSearchToolkit(glob_include_hidden=include_hidden)
    # Private-member access — verifying configuration flow-through.
    assert toolkit.glob._include_hidden is include_hidden  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
