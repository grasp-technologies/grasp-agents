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
    name: str = "add"
    description: str = "Add two integers and return their sum."

    async def run(
        self, inp: AddInput, *, ctx: Any = None, call_id: str | None = None  # noqa: ARG002
    ) -> int:
        return inp.a + inp.b


@pytest.fixture
def tools() -> dict[str, BaseTool[Any, Any, Any]]:
    return {"add": AddTool()}
