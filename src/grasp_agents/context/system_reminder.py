"""
Consistent ``<system-reminder>`` wrapping for framework-injected notes.

The model treats a ``<system-reminder>`` as out-of-band guidance from the runtime
(not the user). Used for injected memory, prompt sections, and the conversation
summaries produced by context compaction — so special messages are identified by
one consistent tag rather than ad-hoc phrasing.
"""

SYSTEM_REMINDER_TAG = "system-reminder"


def wrap_in_system_reminder(text: str, *, subject: str | None = None) -> str:
    """
    Wrap ``text`` in ``<system-reminder>`` tags, with an optional ``subject``
    attribute describing the kind of reminder (e.g. ``"conversation summary"``).
    """
    attr = f' subject="{subject}"' if subject else ""
    return f"<{SYSTEM_REMINDER_TAG}{attr}>\n{text}\n</{SYSTEM_REMINDER_TAG}>"
