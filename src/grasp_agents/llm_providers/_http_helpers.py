"""
Shared parsing helpers for provider error-mapping modules.

Private to ``grasp_agents.llm_providers``. Kept here rather than in each
provider's ``utils.py`` because these parsers should produce identical
output across providers that all speak standard HTTP headers, or that
share the OpenAI SDK's exception shape.
"""

from __future__ import annotations

import math

import httpx
import openai

_QUOTA_CODES = frozenset({"insufficient_quota", "credit_balance_too_low"})
_QUOTA_PHRASES = (
    "insufficient_quota",
    "exceeded your current quota",
    "credit balance is too low",
)


def parse_retry_after(response: httpx.Response) -> float | None:
    """
    Extract ``Retry-After`` as a float number of seconds.

    Only the integer/float form is parsed; the HTTP-date form (RFC 7231
    §7.1.3) is not supported — returns ``None`` in that case.
    Returns ``None`` if the header is absent, blank, or otherwise unparseable.
    Non-finite and negative values are rejected so callers receive a sane,
    finite delay or ``None``.
    """
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return value


def is_quota_message(text: str) -> bool:
    """
    Detect billing/quota exhaustion from an error message.

    The last resort, for providers that discard the structured error body:
    LiteLLM rewrites every exception with a fixed ``code``/``type``, and
    OpenAI-compatible gateways often put the marker in the message alone.
    """
    lowered = text.lower()
    return any(phrase in lowered for phrase in _QUOTA_PHRASES)


def is_quota_error(err: openai.APIError) -> bool:
    """
    Detect billing/quota exhaustion on an OpenAI-shaped SDK error.

    Distinguishes a spent account from an ordinary 429: the former never
    clears, so it must fail over instead of being retried. ``code`` and
    ``type`` are both checked — a mid-stream error frame populates only
    one of them, depending on the provider.
    """
    if err.code in _QUOTA_CODES or err.type in _QUOTA_CODES:
        return True
    return is_quota_message(str(err))
