"""
Unit tests for the default secret redactor.

The contract is **high precision**: obvious secret shapes are masked,
normal code is never touched. If a test here fails, think carefully
about whether you're tightening or loosening that invariant.
"""

from __future__ import annotations

import pytest

from grasp_agents.tools.file_edit.redact import (
    DefaultSecretRedactor,
    NullRedactor,
    SecretRedactor,
)


@pytest.fixture
def redact() -> DefaultSecretRedactor:
    return DefaultSecretRedactor()


def test_protocol_satisfied() -> None:
    assert isinstance(DefaultSecretRedactor(), SecretRedactor)
    assert isinstance(NullRedactor(), SecretRedactor)


def test_null_redactor_passes_through() -> None:
    r = NullRedactor()
    sample = "anything goes AKIAIOSFODNN7EXAMPLE"
    assert r(sample) == sample


def test_aws_access_key_redacted(redact: DefaultSecretRedactor) -> None:
    out = redact("export AWS_KEY=AKIAIOSFODNN7EXAMPLE")
    assert "AKIA" not in out
    assert "<REDACTED:AWS_ACCESS_KEY>" in out


def test_github_pat_redacted(redact: DefaultSecretRedactor) -> None:
    out = redact("token: ghp_1234567890abcdefghijklmnopqrstuvwxyz")
    assert "ghp_" not in out
    assert "<REDACTED:GITHUB_PAT>" in out


def test_openai_key_redacted(redact: DefaultSecretRedactor) -> None:
    out = redact("OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz")
    assert "<REDACTED:OPENAI_KEY>" in out


def test_openai_proj_key_redacted(redact: DefaultSecretRedactor) -> None:
    out = redact("key: sk-proj-abcdefghijklmnopqrst-UVWXYZ_1234567890")
    assert "<REDACTED:OPENAI_KEY>" in out


def test_anthropic_key_redacted(redact: DefaultSecretRedactor) -> None:
    out = redact("sk-ant-api03-abcdefghijklmnopqrstuvwxyz")
    assert "<REDACTED:ANTHROPIC_KEY>" in out


def test_jwt_redacted(redact: DefaultSecretRedactor) -> None:
    # Generic JWT shape; 16+ chars per segment.
    jwt = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        ".eyJzdWIiOiIxMjM0NTY3ODkwIn0"
        ".abc123xyz456hello"
    )
    out = redact(f"Authorization: Bearer {jwt}")
    assert "<REDACTED:JWT>" in out


def test_private_key_block_redacted(redact: DefaultSecretRedactor) -> None:
    pem = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEpAIBAAKCAQEAabc...(many lines)...xyz\n"
        "-----END RSA PRIVATE KEY-----"
    )
    out = redact(f"ssh key:\n{pem}\ntail")
    assert "BEGIN RSA PRIVATE KEY" not in out
    assert "<REDACTED:PRIVATE_KEY_BLOCK>" in out
    # Text outside the block is preserved.
    assert "tail" in out


# ---------------------------------------------------------------------------
# Negative cases — ordinary code must not trip the redactor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        # Normal Python code.
        "def add(a, b):\n    return a + b",
        # Import statements.
        "from collections import defaultdict",
        # A three-dot version string (not a JWT shape — no 16-char segments).
        "version = '1.2.3'",
        # An 'sk-' that's too short.
        "short = 'sk-abc'",
        # An 'AKIA' that's not the full 20-char shape.
        "akia_prefix = 'AKIASHORT'",
        # A hash / checksum (no prefix match).
        "sha256: 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
    ],
)
def test_ordinary_code_untouched(redact: DefaultSecretRedactor, text: str) -> None:
    assert redact(text) == text


def test_multiple_secrets_in_one_pass(redact: DefaultSecretRedactor) -> None:
    text = (
        "aws = AKIAIOSFODNN7EXAMPLE\n"
        "gh  = ghp_1234567890abcdefghijklmnopqrstuvwxyz\n"
        "oai = sk-abcdefghijklmnopqrstuvwxyz\n"
    )
    out = redact(text)
    assert "AKIAIOSFODNN7EXAMPLE" not in out
    assert "ghp_" not in out
    # Three different redaction markers are present.
    assert "<REDACTED:AWS_ACCESS_KEY>" in out
    assert "<REDACTED:GITHUB_PAT>" in out
    assert "<REDACTED:OPENAI_KEY>" in out
