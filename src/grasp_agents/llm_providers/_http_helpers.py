"""
Shared HTTP parsing helpers for provider error-mapping modules.

Private to ``grasp_agents.llm_providers``. Kept here rather than in each
provider's ``utils.py`` because these parsers should produce identical
output across providers that all speak standard HTTP headers.
"""

from __future__ import annotations

import math

import httpx


def parse_retry_after(response: httpx.Response) -> float | None:
    """
    Extract ``Retry-After`` as a float number of seconds.

    Only the integer/float form is parsed. The HTTP-date form (RFC 7231
    §7.1.3) is not currently supported — returns ``None`` in that case,
    matching historical behavior of the provider-local helpers this
    function replaces. Returns ``None`` if the header is absent, blank,
    or otherwise unparseable. Non-finite and negative values are rejected:
    this is the single source of ``LlmRateLimitError.retry_after``, so the
    retry layer can trust the value to be a sane, finite delay.
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
