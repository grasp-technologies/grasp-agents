"""
Consistent ``<system-reminder>`` wrapping for framework-injected notes.

The model treats a ``<system-reminder>`` as out-of-band guidance from the runtime
(not the user). Used for injected memory, prompt sections, and the conversation
summaries produced by context compaction — so special messages are identified by
one consistent tag rather than ad-hoc phrasing.
"""

import re

SYSTEM_REMINDER_TAG = "system-reminder"

# Well-known subject of the resume framing note injected by
# ``BackgroundTaskManager.resume_durable`` — shared with the UIs that render
# it specially, so producer and detector cannot drift.
SESSION_RESUMED_SUBJECT = "session resumed"

_SUBJECT_RE = re.compile(rf'^<{SYSTEM_REMINDER_TAG} subject="(?P<subject>[^"]*)">')


def wrap_in_system_reminder(text: str, *, subject: str | None = None) -> str:
    """
    Wrap ``text`` in ``<system-reminder>`` tags, with an optional ``subject``
    attribute describing the kind of reminder (e.g. ``"conversation summary"``).
    """
    attr = f' subject="{subject}"' if subject else ""
    return f"<{SYSTEM_REMINDER_TAG}{attr}>\n{text}\n</{SYSTEM_REMINDER_TAG}>"


def match_system_reminder_subject(text: str) -> str | None:
    """
    The ``subject`` attribute if ``text`` is a subject-tagged
    ``<system-reminder>`` message, else ``None`` — the inverse of
    :func:`wrap_in_system_reminder`, for a UI telling framework notices apart
    from raw user input.
    """
    match = _SUBJECT_RE.match(text.lstrip())
    return match.group("subject") if match else None
