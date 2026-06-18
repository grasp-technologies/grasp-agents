"""
Shared ``InputFile`` decoding for provider converters.

Private to ``grasp_agents.llm_providers`` — providers that need the raw
base64 payload + mime type (Anthropic, Gemini) parse it identically here;
OpenAI-shaped APIs accept the ``file_data`` string as-is.
"""

from __future__ import annotations

import mimetypes


def file_part_data(
    file_data: str,
    filename: str | None,
    default_mime: str = "application/pdf",
) -> tuple[str, str]:
    """
    Split an ``InputFile.file_data`` payload into ``(base64_data, mime_type)``.

    Accepts a raw base64 string or a ``data:<mime>;base64,<data>`` URI. The
    mime type falls back to the filename's extension, then ``default_mime``.
    """
    mime = (mimetypes.guess_type(filename)[0] if filename else None) or default_mime
    if file_data.startswith("data:") and "," in file_data:
        header, file_data = file_data.split(",", 1)
        mime = header.removeprefix("data:").removesuffix(";base64") or mime
    return file_data, mime
