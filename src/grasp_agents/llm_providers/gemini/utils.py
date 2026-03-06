"""Validation helpers for Gemini responses."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import GeminiResponse


def validate_response(response: GeminiResponse) -> None:
    """Raise if the response has no candidates or content."""
    if not response.candidates:
        raise ValueError(
            f"Gemini response {response.response_id} has no candidates"
        )
    candidate = response.candidates[0]
    if not candidate.content or not candidate.content.parts:
        raise ValueError(
            f"Gemini response {response.response_id} has no content "
            f"(finish_reason={candidate.finish_reason})"
        )
