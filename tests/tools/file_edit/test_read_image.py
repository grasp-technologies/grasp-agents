"""
Unit tests for :class:`ReadImageTool`.

Like the other file tools, exercised through the public ``.run(...)`` API:
errors come back as a ``ToolErrorInfo`` value (not an exception), matching how
the agent loop sees a tool call. The tool reads bytes through the backend (so
the sandbox / allowed-roots policy applies) and returns an ``InputImage``.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

import pytest

from grasp_agents.agent.agent_context import AgentContext
from grasp_agents.file_backend import LocalFileBackend
from grasp_agents.run_context import RunContext
from grasp_agents.tools.file_edit import ReadImageInput, ReadImageTool
from grasp_agents.types.content import InputImage, InputText
from grasp_agents.types.events import ToolErrorInfo

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.asyncio

# A valid 1x1 PNG.
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _error_message(result: Any) -> str:
    assert isinstance(result, ToolErrorInfo), (
        f"Expected a ToolErrorInfo, got {type(result).__name__}: {result!r}"
    )
    return result.error


@pytest.fixture
def ctx(tmp_path: Path) -> RunContext[Any]:
    backend = LocalFileBackend(allowed_roots=[tmp_path])
    return RunContext[Any](file_backend=backend, session_key="test")


@pytest.fixture
def read_image() -> ReadImageTool:
    return ReadImageTool()


async def test_reads_png_as_image(
    tmp_path: Path, ctx: RunContext[Any], agent_ctx: AgentContext
) -> None:
    f = tmp_path / "pixel.png"
    f.write_bytes(_PNG_BYTES)

    result = await ReadImageTool().run(
        ReadImageInput(path=str(f)), ctx=ctx, agent_ctx=agent_ctx
    )

    assert isinstance(result, list)
    note, image = result
    assert isinstance(note, InputText)
    assert "pixel.png" in note.text
    assert isinstance(image, InputImage)
    assert image.mime_type == "image/png"
    assert image.is_base64
    # the embedded base64 must round-trip back to the original bytes
    b64 = image.image_url.split("base64,", 1)[1]  # type: ignore[union-attr]
    assert base64.b64decode(b64) == _PNG_BYTES


async def test_detail_is_passed_through(
    tmp_path: Path, ctx: RunContext[Any], agent_ctx: AgentContext
) -> None:
    f = tmp_path / "pixel.png"
    f.write_bytes(_PNG_BYTES)

    result = await ReadImageTool().run(
        ReadImageInput(path=str(f), detail="high"), ctx=ctx, agent_ctx=agent_ctx
    )
    assert isinstance(result, list)
    image = result[1]
    assert isinstance(image, InputImage)
    assert image.detail == "high"


async def test_jpeg_extension_maps_to_jpeg_mime(
    tmp_path: Path, ctx: RunContext[Any], agent_ctx: AgentContext
) -> None:
    # mime is keyed off the extension; content bytes aren't validated here.
    f = tmp_path / "photo.jpeg"
    f.write_bytes(_PNG_BYTES)

    result = await ReadImageTool().run(
        ReadImageInput(path=str(f)), ctx=ctx, agent_ctx=agent_ctx
    )
    assert isinstance(result, list)
    image = result[1]
    assert isinstance(image, InputImage)
    assert image.mime_type == "image/jpeg"


async def test_non_image_extension_refused(
    tmp_path: Path, ctx: RunContext[Any], agent_ctx: AgentContext
) -> None:
    f = tmp_path / "notes.txt"
    f.write_text("hello")

    result = await ReadImageTool().run(
        ReadImageInput(path=str(f)), ctx=ctx, agent_ctx=agent_ctx
    )
    msg = _error_message(result)
    assert "Supported formats" in msg


async def test_oversized_image_refused(
    tmp_path: Path, ctx: RunContext[Any], agent_ctx: AgentContext
) -> None:
    f = tmp_path / "big.png"
    f.write_bytes(_PNG_BYTES)

    tool = ReadImageTool(max_image_bytes=10)  # smaller than the PNG
    result = await tool.run(ReadImageInput(path=str(f)), ctx=ctx, agent_ctx=agent_ctx)
    msg = _error_message(result)
    assert "exceeding the maximum" in msg


async def test_missing_file_refused(
    tmp_path: Path, ctx: RunContext[Any], agent_ctx: AgentContext
) -> None:
    result = await ReadImageTool().run(
        ReadImageInput(path=str(tmp_path / "nope.png")), ctx=ctx, agent_ctx=agent_ctx
    )
    assert isinstance(result, ToolErrorInfo)


async def test_toolkit_opt_in_includes_read_image() -> None:
    from grasp_agents.tools import FileToolkit

    assert not any(t.name == "ReadImage" for t in FileToolkit().tools()), (
        "ReadImage should be opt-in (off by default)"
    )
    tk = FileToolkit(include_image=True)
    assert any(t.name == "ReadImage" for t in tk.tools())
    assert any(t.name == "ReadImage" for t in tk.read_only_tools())
