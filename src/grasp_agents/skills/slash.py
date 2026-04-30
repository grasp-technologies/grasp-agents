"""
Slash-command parsing helpers for skill invocation.

Frontends parse user input via :func:`parse_slash_command` and render the
matched skill body via :meth:`SkillRegistry.render_invocation`. The framework
itself does not own a CLI loop; these helpers are utilities for downstream
apps (CLI / chat UI / etc.).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Skill name regex from agentskills.io: lowercase a-z + 0-9 + non-leading,
# non-trailing, non-consecutive hyphens. Length 1-64.
_SLASH_RE = re.compile(
    r"^/(?P<name>[a-z0-9](?:-?[a-z0-9])*)"
    r"(?:\s+(?P<args>.*))?$",
    re.DOTALL,
)

_NAMED_ARG_RE = re.compile(
    r"--(?P<key>[A-Za-z_][A-Za-z0-9_-]*)(?:=(?P<value>\S*))?"
)


@dataclass(frozen=True)
class ParsedSlashCommand:
    """Result of :func:`parse_slash_command`."""

    name: str
    args: str  # raw args string (everything after the command name)


def parse_slash_command(text: str) -> ParsedSlashCommand | None:
    """
    Parse ``/name [args...]`` from a user input string.

    Returns ``None`` when ``text`` does not start with a valid slash command
    (e.g., raw user message, blank input, malformed name). The trailing args
    are kept as a single string — splitting / quoting is left to consumers.

    Trims surrounding whitespace before matching, but preserves arg spacing.
    """
    stripped = text.strip()
    if not stripped:
        return None
    match = _SLASH_RE.match(stripped)
    if match is None:
        return None
    name = match.group("name")
    if len(name) > 64:
        return None
    raw_args = match.group("args") or ""
    return ParsedSlashCommand(name=name, args=raw_args.strip())


def parse_named_args(raw: str) -> dict[str, str]:
    """
    Pull ``--key=value`` pairs out of an args string.

    Bare ``--flag`` (no ``=value``) maps to ``{"flag": ""}``. Tokens that are
    not in ``--key`` form are ignored — consumers wanting positional args
    should look at the original ``raw`` string. Whitespace is the token
    separator; values may not contain spaces.
    """
    return {
        match.group("key"): match.group("value") or ""
        for match in _NAMED_ARG_RE.finditer(raw)
    }
