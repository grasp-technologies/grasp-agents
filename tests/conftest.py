"""Shared test configuration and fixtures."""

from __future__ import annotations

import os
from typing import Any

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from grasp_agents.types.tool import BaseTool

# Load .env file (gitignored) for local development
load_dotenv()


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


class AddInput(BaseModel):
    a: int = Field(description="First integer")
    b: int = Field(description="Second integer")


class AddTool(BaseTool[AddInput, int, Any]):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name="add",
            description="Add two integers and return their sum.",
            **kwargs,
        )

    async def _run(
        self,
        inp: AddInput,
        *,
        ctx: Any = None,
        exec_id: str | None = None,
        progress_callback: Any = None,  # noqa: ARG002
    ) -> int:
        return inp.a + inp.b


class MultiplyInput(BaseModel):
    a: int = Field(description="First integer")
    b: int = Field(description="Second integer")


class MultiplyTool(BaseTool[MultiplyInput, int, Any]):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name="multiply",
            description="Multiply two integers and return their product.",
            **kwargs,
        )

    async def _run(
        self,
        inp: MultiplyInput,
        *,
        ctx: Any = None,
        exec_id: str | None = None,
        progress_callback: Any = None,  # noqa: ARG002
    ) -> int:
        return inp.a * inp.b


@pytest.fixture
def tools() -> dict[str, BaseTool[Any, Any, Any]]:
    return {"add": AddTool()}


@pytest.fixture
def parallel_tools() -> dict[str, BaseTool[Any, Any, Any]]:
    return {"add": AddTool(), "multiply": MultiplyTool()}
