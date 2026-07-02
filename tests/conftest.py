"""Shared test configuration and fixtures."""

from __future__ import annotations

import os
from typing import Any

import pytest
from dotenv import load_dotenv

from grasp_agents.session_context import reset_default_session_context
from grasp_agents.tools.base import BaseTool

from ._helpers import AddTool, MultiplyTool

# Load .env file (gitignored) for local development
load_dotenv()


@pytest.fixture(autouse=True)
def _fresh_default_session_context() -> Any:
    """
    Give every test a fresh process-default ``SessionContext``.

    Bare-constructed agents now share one process-wide default (so uncomposed
    agents stay in one session); without this reset that default would
    accumulate state / responses / usage across unrelated tests. Tests that
    pass an explicit ``ctx`` are unaffected.
    """
    reset_default_session_context()
    yield
    reset_default_session_context()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Regenerate golden snapshot files",
    )


# ------------------------------------------------------------------ #
#  API key fixtures                                                    #
# ------------------------------------------------------------------ #


class _SecretStr(str):  # noqa: FURB189  (must BE a str: passed to SDKs as api_key)
    """
    A ``str`` whose ``repr`` hides the value.

    The value is still the real key everywhere it is used (e.g. passed to a
    provider client as ``api_key``), but pytest renders a failing test's fixture
    arguments and assertion operands via ``repr`` — so this keeps secrets out of
    tracebacks and CI logs. ``str(key)`` is unchanged, so authentication works.

    Every API-key fixture must return its key through :func:`_require_env_key`
    so a leaked key can never reach a traceback.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "'***'"


def _require_env_key(var: str) -> _SecretStr:
    """Fetch a required secret from the environment, redacted for display."""
    value = os.environ.get(var)
    if not value:
        pytest.skip(f"{var} not set")
    return _SecretStr(value)


@pytest.fixture
def anthropic_api_key() -> str:
    return _require_env_key("ANTHROPIC_API_KEY")


@pytest.fixture
def google_api_key() -> str:
    return _require_env_key("GEMINI_API_KEY")


@pytest.fixture
def openai_api_key() -> str:
    return _require_env_key("OPENAI_API_KEY")


# ------------------------------------------------------------------ #
#  Shared tool fixtures for integration tests                          #
# ------------------------------------------------------------------ #


@pytest.fixture
def tools() -> dict[str, BaseTool[Any, Any, Any]]:
    return {"add": AddTool()}


@pytest.fixture
def parallel_tools() -> dict[str, BaseTool[Any, Any, Any]]:
    return {"add": AddTool(), "multiply": MultiplyTool()}
