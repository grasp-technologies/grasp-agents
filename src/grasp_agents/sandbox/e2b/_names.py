"""
How the E2B environment marks what it creates: a metadata key on every
sandbox, and a unique, timestamped name on every snapshot (E2B snapshots take
no metadata). Cleanup finds grasp-agents' sandboxes and snapshots by these.
"""

from __future__ import annotations

import re
import secrets
from datetime import UTC, datetime

METADATA_KEY = "grasp-agents"
"""Sandbox metadata key the environment sets to its label on every create."""

DEFAULT_LABEL = "default"

LABEL_PATTERN = r"^[a-z0-9-]{1,32}$"

_PREFIX = "grasp-agents"
_TIME_FORMAT = "%Y%m%dt%H%M%S"


def check_label(label: str | None) -> str:
    """The label to tag with: ``label``, or the default when ``None``."""
    if label is None:
        return DEFAULT_LABEL
    if re.fullmatch(LABEL_PATTERN, label) is None:
        raise ValueError(
            f"E2B label {label!r} must be 1-32 lowercase letters, digits or "
            "dashes: it goes into snapshot names."
        )
    return label


def snapshot_name(label: str, now: datetime) -> str:
    """
    A new snapshot's name. Unique per snapshot: E2B re-points a name's tag to
    its newest snapshot, so a shared name would change what earlier refs
    restore.
    """
    return (
        f"{_PREFIX}-{label}-{now.astimezone(UTC):{_TIME_FORMAT}}-{secrets.token_hex(4)}"
    )


def snapshot_created_at(name: str, label: str) -> datetime | None:
    """
    When the snapshot named ``name`` was taken, if the environment named it
    with ``label``; else ``None``. E2B prefixes names with the team's slug.
    """
    bare = name.rsplit("/", 1)[-1]
    found = re.fullmatch(
        rf"{_PREFIX}-{re.escape(label)}-(\d{{8}}t\d{{6}})-[0-9a-f]{{8}}", bare
    )
    if found is None:
        return None
    return datetime.strptime(found.group(1), _TIME_FORMAT).replace(tzinfo=UTC)
