"""
Render rich execution outputs (nbformat-shaped MIME bundles) to model content
parts, and sanitize them for safe persistence.

Shared by the notebook tools (:mod:`.file_edit.notebook`), ``RunCell``
(:mod:`.notebook_exec`), and ``RunPython`` (:mod:`.code_interpreter`): all three
turn the same output union — ``stream`` / ``execute_result`` / ``display_data``
/ ``error`` — into :class:`InputText` + :class:`InputImage` parts, and the tools
that *persist* outputs strip browser-executable payloads first. Kept free of any
``nbformat`` dependency so the code-interpreter path need not pull the notebook
layer in.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from ..types.content import InputImage, InputText

if TYPE_CHECKING:
    from collections.abc import Mapping

# Raster image MIME types the model can actually view (provider-supported).
# SVG / HTML are text and surface as a note, not an image.
VIEWABLE_IMAGE_MIMES: tuple[str, ...] = (
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
)
# Cap on images surfaced to the model in one read/run; the full set stays in
# the source (.ipynb / on disk). Bounds context when many figures are present.
DEFAULT_MAX_IMAGES = 10
# Cap on the text summary returned for a set of outputs.
DEFAULT_OUTPUT_TEXT_CHARS = 20_000

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def coerce_text(value: Any) -> str:
    """Join an nbformat list-of-lines (or stringify a scalar) into one string."""
    if isinstance(value, list):
        return "".join(str(v) for v in value)  # type: ignore[reportUnknownArgumentType]
    return str(value)


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def image_parts_from_data(data: Any, budget: int) -> list[InputImage]:
    """InputImage parts for each viewable raster MIME in a bundle (up to ``budget``)."""
    parts: list[InputImage] = []
    for mime in VIEWABLE_IMAGE_MIMES:
        if len(parts) >= budget:
            break
        if mime in data:
            # nbformat may store base64 image data as a list of lines.
            b64 = coerce_text(data[mime])
            parts.append(InputImage.from_base64(b64, mime_type=mime))
    return parts


def render_outputs_as_parts(
    outputs: list[Any],
    *,
    header: str | None = None,
    include_images: bool = True,
    max_text_chars: int = DEFAULT_OUTPUT_TEXT_CHARS,
    max_images: int = DEFAULT_MAX_IMAGES,
) -> list[InputText | InputImage]:
    """
    Render nbformat output dicts as model-facing content parts: a text summary
    (stream text, results, ANSI-stripped tracebacks, and notes for non-viewable
    rich MIME types) followed by the viewable images (png/jpeg/gif/webp).
    Shared by ``RunCell`` / ``RunPython`` (fresh outputs) and ``NotebookRead``
    (stored outputs).
    """
    segments: list[str] = []
    images: list[InputImage] = []
    for out in outputs:
        output_type = out.get("output_type")
        if output_type == "stream":
            segments.append(_strip_ansi(coerce_text(out.get("text", ""))))
        elif output_type == "error":
            tb = "\n".join(str(t) for t in out.get("traceback", []))
            segments.append(
                _strip_ansi(tb or f"{out.get('ename', '')}: {out.get('evalue', '')}")
            )
        elif output_type in {"execute_result", "display_data"}:
            data = out.get("data", {})
            if "text/plain" in data:
                segments.append(coerce_text(data["text/plain"]))
            if include_images and len(images) < max_images:
                images += image_parts_from_data(data, max_images - len(images))
            for mime in data:
                if mime == "text/plain" or mime in VIEWABLE_IMAGE_MIMES:
                    continue
                segments.append(f"[{mime}]")
    body = "\n\n".join(s for s in segments if s).strip()
    if len(body) > max_text_chars:
        body = body[:max_text_chars] + "\n[output truncated]"
    text = (f"{header}\n\n" if header else "") + (body or "(no text output)")
    return [InputText(text=text), *images]


# MIME types whose payload the notebook frontend executes as browser JS.
EXECUTABLE_OUTPUT_MIMES: tuple[str, ...] = (
    "application/javascript",
    "application/x-javascript",
    "text/javascript",
)
_SCRIPT_RE = re.compile(r"<script\b[^>]*>.*?</script\s*>", re.IGNORECASE | re.DOTALL)
_SCRIPT_OPEN_RE = re.compile(r"</?script\b[^>]*>", re.IGNORECASE)
_ON_HANDLER_RE = re.compile(
    r"""\son\w+\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)""", re.IGNORECASE
)
_JS_URI_RE = re.compile(
    r"""(href|src)\s*=\s*("javascript:[^"]*"|'javascript:[^']*'|javascript:[^\s>]+)""",
    re.IGNORECASE,
)


def _sanitize_html(html: str) -> tuple[str, bool]:
    cleaned = _SCRIPT_RE.sub("", html)
    cleaned = _SCRIPT_OPEN_RE.sub("", cleaned)
    cleaned = _ON_HANDLER_RE.sub("", cleaned)
    cleaned = _JS_URI_RE.sub(r'\1="#"', cleaned)
    return cleaned, cleaned != html


def sanitize_output_data(data: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    """
    Strip browser-executable payloads from a display MIME bundle, returning
    ``(clean_data, modified)``.

    A notebook output's JS runs in the *frontend that renders the .ipynb* — a
    human reviewer's Jupyter/nbviewer session, outside any exec sandbox. So an
    agent-authored output is a stored-XSS surface. This drops
    ``application/javascript`` (+ variants) and removes ``<script>``, ``on*=``
    handlers, and ``javascript:`` URIs from ``text/html`` before the output is
    persisted. Defense-in-depth layered on Jupyter's trusted-notebook model —
    *not* a full HTML sanitizer (it targets the executable vectors, not every
    conceivable one).
    """
    clean = dict(data)
    modified = False
    for mime in EXECUTABLE_OUTPUT_MIMES:
        if mime in clean:
            del clean[mime]
            modified = True
    if "text/html" in clean:
        sanitized, changed = _sanitize_html(coerce_text(clean["text/html"]))
        if changed:
            clean["text/html"] = sanitized
            modified = True
    if modified and not clean:
        # The bundle was executable-only; leave a marker so it isn't blank.
        clean["text/plain"] = "[executable output removed by sanitizer]"
    return clean, modified


__all__ = [
    "DEFAULT_MAX_IMAGES",
    "DEFAULT_OUTPUT_TEXT_CHARS",
    "EXECUTABLE_OUTPUT_MIMES",
    "VIEWABLE_IMAGE_MIMES",
    "coerce_text",
    "image_parts_from_data",
    "render_outputs_as_parts",
    "sanitize_output_data",
]
