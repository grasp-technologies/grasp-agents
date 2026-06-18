"""
The marker protocol shared by persistent :class:`ExecSession` implementations
(:class:`~grasp_agents.sandbox.local.session.LocalExecSession` over pipes,
:class:`~grasp_agents.sandbox.e2b.session.E2BExecSession` over E2B callbacks).

A persistent shell never exits between commands, so — unlike a one-shot
``sh -c`` — there is no EOF or per-command exit code. Each command is therefore
bracketed by a unique marker on *both* stdout and stderr to delimit the end of
its output; the stdout marker also carries ``$?``. Sessions feed
:func:`frame_command`'s payload to the shell's stdin and scan each stream for
the marker, recovering the exit code via :func:`parse_exit_code`.
"""

from __future__ import annotations

from uuid import uuid4


def frame_command(command: str) -> tuple[str, str]:
    """
    Return ``(marker, payload)`` for one command in a persistent shell.

    ``payload`` runs ``command`` then prints a unique ``marker`` on stdout (with
    the command's exit code) and on stderr, so a reader scanning each stream for
    ``marker`` knows when that stream's output for this command is complete.
    """
    marker = f"__GRASP_{uuid4().hex[:12]}__"
    payload = (
        f"{command}\n"
        f"printf '%s %d\\n' '{marker}' \"$?\"\n"
        f"printf '%s\\n' '{marker}' 1>&2\n"
    )
    return marker, payload


def parse_exit_code(after_marker: str) -> int:
    """Parse the exit code printed after the marker on the stdout sentinel line."""
    tokens = after_marker.strip().split(maxsplit=1)
    if not tokens:
        return -1
    try:
        return int(tokens[0])
    except ValueError:
        return -1


__all__ = ["frame_command", "parse_exit_code"]
