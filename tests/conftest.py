"""Shared test configuration and fixtures."""

from __future__ import annotations

import os
from typing import Any

import pytest
from dotenv import load_dotenv

from grasp_agents.run_context import reset_default_run_context
from grasp_agents.tools.base import BaseTool

from ._helpers import AddTool, MultiplyTool

# Load .env file (gitignored) for local development
load_dotenv()


@pytest.fixture(autouse=True)
def _fresh_default_run_context() -> Any:
    """
    Give every test a fresh process-default ``RunContext``.

    Bare-constructed agents now share one process-wide default (so uncomposed
    agents stay in one session); without this reset that default would
    accumulate state / responses / usage across unrelated tests. Tests that
    pass an explicit ``ctx`` are unaffected.
    """
    reset_default_run_context()
    yield
    reset_default_run_context()


@pytest.fixture
def anyio_backend() -> str:
    """Run ``anyio``-marked tests on the asyncio backend."""
    return "asyncio"


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


@pytest.fixture
def anthropic_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return key


@pytest.fixture
def google_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        pytest.skip("GEMINI_API_KEY not set")
    return key


@pytest.fixture
def openai_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return key


# ------------------------------------------------------------------ #
#  Shared tool fixtures for integration tests                          #
# ------------------------------------------------------------------ #


@pytest.fixture
def tools() -> dict[str, BaseTool[Any, Any, Any]]:
    return {"add": AddTool()}


@pytest.fixture
def parallel_tools() -> dict[str, BaseTool[Any, Any, Any]]:
    return {"add": AddTool(), "multiply": MultiplyTool()}
