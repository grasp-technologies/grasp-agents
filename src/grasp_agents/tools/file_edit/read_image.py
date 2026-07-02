r"""
``ReadImage`` — read an image file as visual content via ``ctx.file_backend``.

The generic ``Read`` tool refuses binary files; this is its image counterpart.
It reads the raw bytes through the backend (so the sandbox / allowed-roots
policy still applies), base64-encodes them, and returns an ``InputImage`` the
model can actually see. Limited to the raster formats every major vision
provider accepts (PNG, JPEG, GIF, WebP).
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from grasp_agents.file_backend.paths import PathAccessError
from grasp_agents.tools.base import BaseTool, ToolProgressCallback
from grasp_agents.types.content import ImageDetail, InputImage, InputText

if TYPE_CHECKING:
    from grasp_agents.agent.agent_context import AgentContext
    from grasp_agents.session_context import SessionContext

# Raster formats every major vision provider (OpenAI / Anthropic / Gemini)
# accepts. Other image types (bmp, tiff, svg, …) are refused with a clear note.
_EXT_TO_MIME: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# Base64 inflates bytes ~33% and image tokens are dense, so cap well below the
# text ``Read`` ceiling. ~5 MB is around the per-image limit providers accept;
# larger images are refused here rather than rejected opaquely by the API.
DEFAULT_MAX_IMAGE_BYTES = 5_000_000


class ReadImageInput(BaseModel):
    """Input schema for the ``ReadImage`` tool."""

    path: str = Field(
        description=(
            "Absolute, relative, or ~-expanded path to an image file (PNG, "
            "JPEG, GIF, or WebP). Must resolve under one of the backend's "
            "allowed roots."
        )
    )
    detail: ImageDetail = Field(
        default="auto",
        description=(
            "Vision fidelity hint: `low` (cheaper, coarse), `high` (more "
            "tokens, fine detail), or `auto` (default)."
        ),
    )


class ReadImageTool(BaseTool[ReadImageInput, list[InputText | InputImage], Any]):
    """
    Read an image file as visual content via ``ctx.file_backend``.

    Stateless: the backend (and its allowed-roots / sandbox policy) lives on
    :attr:`SessionContext.file_backend`. Returns an :class:`InputImage` the model
    can see, preceded by a short text note naming the file.
    """

    name = "ReadImage"
    description = (
        "Read an image file (PNG, JPEG, GIF, WebP) so you can see it. Use this "
        "for images; the `Read` tool handles text files and refuses binaries.\n"
        "\n"
        "* The image is returned as visual content you can inspect directly.\n"
        "* Unsupported formats and oversized images are refused with a note."
    )

    def __init__(
        self,
        *,
        max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
        timeout: float | None = None,
    ) -> None:
        super().__init__(timeout=timeout)
        self._max_image_bytes = max_image_bytes

    async def _run(
        self,
        inp: ReadImageInput,
        *,
        ctx: SessionContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        path: list[str] | None = None,
        agent_ctx: AgentContext | None = None,
    ) -> list[InputText | InputImage]:
        del exec_id, progress_callback, path

        if ctx is None or ctx.file_backend is None:
            raise ValueError(
                "ReadImage requires ctx.file_backend. Wire a FileBackend on "
                "SessionContext before running the agent."
            )

        mime = _EXT_TO_MIME.get(Path(inp.path).suffix.lower())
        if mime is None:
            raise ValueError(
                f"Cannot read {inp.path!r} as an image. Supported formats: "
                f"{', '.join(sorted(_EXT_TO_MIME))}."
            )

        backend = ctx.file_backend
        state = agent_ctx.file_edit_state if agent_ctx is not None else None
        overrides = (
            set(state.dotfile_overrides)
            if state is not None and state.dotfile_overrides
            else None
        )
        try:
            resolved = await backend.validate_path(
                Path(inp.path), must_exist=True, dotfile_overrides=overrides
            )
        except PathAccessError as exc:
            raise ValueError(str(exc)) from exc

        size = (await backend.stat(resolved)).size
        if size > self._max_image_bytes:
            raise ValueError(
                f"Image is {size:,} bytes, exceeding the maximum readable size "
                f"({self._max_image_bytes:,}). Resize or downsample it first."
            )

        data, _ = await backend.read_bytes(resolved)
        b64 = base64.b64encode(data).decode("utf-8")
        image = InputImage.from_base64(b64, mime_type=mime, detail=inp.detail)
        note = InputText(text=f"Read image {resolved} ({size:,} bytes, {mime}):")
        return [note, image]
