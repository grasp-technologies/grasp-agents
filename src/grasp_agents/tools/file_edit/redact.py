"""
Secret-redaction utilities for file-edit tool output.

Defense-in-depth paired with the sensitive-path deny list: even when
the model ends up reading a file that happens to contain a secret
(outside the deny list — e.g. a dev script with a hardcoded API key),
the redactor replaces obvious secret shapes with ``<REDACTED:KIND>``
markers before the content enters the model's context.

Ships two implementations:

- :class:`DefaultSecretRedactor` — regex-based, conservative. Matches
  well-formatted secrets (AWS keys, GitHub PATs, OpenAI/Anthropic keys,
  JWTs, private-key blocks). Tuned for high precision: an exotic secret
  shape may slip through, but ordinary code is never mangled.

- :class:`NullRedactor` — pass-through. Use to opt out of redaction.

Consumers with higher-risk workloads can plug in a richer implementation
(wrapping ``detect-secrets`` or similar) via the :class:`SecretRedactor`
:class:`Protocol`.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable


@runtime_checkable
class SecretRedactor(Protocol):
    """Replace secret-shaped substrings in ``text`` with redaction markers."""

    def __call__(self, text: str) -> str: ...


class NullRedactor:
    """Pass-through redactor. Use when redaction is opted off."""

    def __call__(self, text: str) -> str:
        return text


# (kind, pattern, replacement). Patterns are deliberately high-precision
# — prefix-anchored, length-constrained — so they don't trip on random
# long-looking strings in normal code.
_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    # AWS access key: 'AKIA' + 16 uppercase alphanums.
    (
        "AWS_ACCESS_KEY",
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "<REDACTED:AWS_ACCESS_KEY>",
    ),
    # GitHub personal / OAuth / fine-grained tokens.
    (
        "GITHUB_PAT",
        re.compile(r"\bghp_[A-Za-z0-9]{36}\b"),
        "<REDACTED:GITHUB_PAT>",
    ),
    (
        "GITHUB_OAUTH",
        re.compile(r"\bgho_[A-Za-z0-9]{36}\b"),
        "<REDACTED:GITHUB_OAUTH>",
    ),
    # Anthropic must come before OpenAI: the OpenAI shape otherwise
    # swallows ``sk-ant-...`` as a generic ``sk-<long>`` match.
    (
        "ANTHROPIC_KEY",
        re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
        "<REDACTED:ANTHROPIC_KEY>",
    ),
    # OpenAI: both ``sk-`` and ``sk-proj-`` shapes. 20+ chars after the prefix.
    (
        "OPENAI_KEY",
        re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
        "<REDACTED:OPENAI_KEY>",
    ),
    # JWT: three base64url segments separated by dots. Require at least
    # 16 chars per segment so random "a.b.c" strings don't match.
    (
        "JWT",
        re.compile(
            r"\beyJ[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\b"
        ),
        "<REDACTED:JWT>",
    ),
    # Multi-line PEM private keys (RSA / EC / OPENSSH / generic).
    (
        "PRIVATE_KEY_BLOCK",
        re.compile(
            r"-----BEGIN [A-Z ]+PRIVATE KEY-----.+?-----END [A-Z ]+PRIVATE KEY-----",
            re.DOTALL,
        ),
        "<REDACTED:PRIVATE_KEY_BLOCK>",
    ),
)


class DefaultSecretRedactor:
    """
    Regex-based secret redactor with conservative default patterns.

    Applies :data:`_PATTERNS` in order. Patterns are anchored with word
    boundaries and length floors to avoid false positives in ordinary
    code; the trade-off is that exotic secret shapes may not be caught.
    """

    def __call__(self, text: str) -> str:
        for _kind, pattern, replacement in _PATTERNS:
            text = pattern.sub(replacement, text)
        return text
