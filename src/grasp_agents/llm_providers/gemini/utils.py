"""Validation helpers for Gemini responses."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import GeminiResponse


def validate_response(response: GeminiResponse) -> None:
    """Raise if the response has no candidates or content."""
    if not response.candidates:
        raise ValueError(f"Gemini response {response.response_id} has no candidates")
    candidate = response.candidates[0]
    if not candidate.content or not candidate.content.parts:
        raise ValueError(
            f"Gemini response {response.response_id} has no content "
            f"(finish_reason={candidate.finish_reason})"
        )


def encode_thought_signature(sig: bytes | str) -> str:
    """Base64-encode a Gemini thought signature (no-op if already a string)."""
    if isinstance(sig, bytes):  # type: ignore[unreachable]
        return base64.b64encode(sig).decode("ascii")
    return sig  # type: ignore[return-value]
