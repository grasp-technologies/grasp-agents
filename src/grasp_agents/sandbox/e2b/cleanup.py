"""
:func:`sweep_e2b`: remove what E2B environments left on the account, such as
paused sandboxes nothing resumed and the snapshots of sessions nothing holds
any more.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal

from e2b import AsyncSandbox, SandboxException, SandboxQuery, SandboxState

from ._names import METADATA_KEY, check_label, snapshot_created_at

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from e2b.connection_config import ApiParams

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SweptSandbox:
    sandbox_id: str
    state: Literal["paused", "running"]
    # How long it has been paused, or running.
    age: timedelta


@dataclass(frozen=True)
class SweptSnapshot:
    snapshot_id: str
    # Since it was taken, from its name.
    age: timedelta


@dataclass
class E2BSweepReport:
    """
    What :func:`sweep_e2b` did, or in a dry run would do: ``killed`` and
    ``deleted`` are acted on (or due); ``in_use`` are snapshots E2B refused
    because a sandbox still uses them; ``kept`` are old enough but held by
    ``keep_snapshots``.
    """

    dry_run: bool
    killed: list[SweptSandbox] = field(default_factory=list[SweptSandbox])
    deleted: list[SweptSnapshot] = field(default_factory=list[SweptSnapshot])
    in_use: list[SweptSnapshot] = field(default_factory=list[SweptSnapshot])
    kept: list[SweptSnapshot] = field(default_factory=list[SweptSnapshot])


def _template(ref: str) -> str:
    """A snapshot ref without its tag: E2B deletes a snapshot's template whole."""
    return ref.rsplit(":", 1)[0]


async def sweep_e2b(
    *,
    older_than: timedelta,
    keep_snapshots: Collection[str],
    label: str | None = None,
    states: Collection[Literal["paused", "running"]] = ("paused",),
    metadata: Mapping[str, str] | None = None,
    dry_run: bool = True,
    api_key: str | None = None,
    domain: str | None = None,
) -> E2BSweepReport:
    """
    Kill the sandboxes and delete the snapshots E2B environments with
    ``label`` left on the account for longer than ``older_than``.

    1. Sandboxes whose ``"grasp-agents"`` metadata is ``label`` (and that
       match ``metadata``, if given), in one of ``states``, paused or running
       for longer than ``older_than``, are killed. Only paused ones by
       default: a running sandbox may be serving a session right now.
    2. Then snapshots named by such an environment, taken longer ago than
       ``older_than``, are deleted, except those in ``keep_snapshots``.
       Killing the sandboxes first frees the snapshots they were created
       from, which E2B refuses to delete while a sandbox uses them; the
       report lists any it still refuses.

    ``keep_snapshots`` is required: pass every ref a resume or a rollback may
    still restore, such as ``read_fs_snapshot_refs(store)`` over each store
    your sessions use. A snapshot a live record holds but this set misses is
    deleted, and restoring it then fails. ``older_than`` must also exceed the
    longest checkpoint, since a snapshot is in no record until its
    checkpoint lands; hours are a safe order.

    Unnamed snapshots and sandboxes without the metadata key are never
    touched. With ``dry_run=True`` (the default) nothing changes and the
    report says what would.
    """
    if older_than < timedelta(0):
        raise ValueError(f"older_than must not be negative, got {older_than}.")
    label = check_label(label)
    opts: ApiParams = {}
    if api_key is not None:
        opts["api_key"] = api_key
    if domain is not None:
        opts["domain"] = domain
    now = datetime.now(UTC)
    report = E2BSweepReport(dry_run=dry_run)

    query = SandboxQuery(
        metadata={**(metadata or {}), METADATA_KEY: label},
        state=[SandboxState(state) for state in states],
    )
    pages = AsyncSandbox.list(query=query, **opts)
    while pages.has_next:
        for info in await pages.next_items():
            paused = info.state == SandboxState.PAUSED
            age = now - (info.end_at if paused else info.started_at)
            if age <= older_than:
                continue
            report.killed.append(
                SweptSandbox(
                    str(info.sandbox_id), "paused" if paused else "running", age
                )
            )
            if not dry_run:
                logger.info("killing E2B sandbox %s (%s)", info.sandbox_id, age)
                await AsyncSandbox.kill(info.sandbox_id, **opts)

    keep = {_template(ref) for ref in keep_snapshots}
    snapshots = AsyncSandbox.list_snapshots(**opts)
    while snapshots.has_next:
        for snapshot in await snapshots.next_items():
            taken = next(
                (
                    created
                    for name in snapshot.names
                    if (created := snapshot_created_at(name, label)) is not None
                ),
                None,
            )
            if taken is None or now - taken <= older_than:
                continue
            swept = SweptSnapshot(snapshot.snapshot_id, now - taken)
            if _template(snapshot.snapshot_id) in keep:
                report.kept.append(swept)
                continue
            if dry_run:
                report.deleted.append(swept)
                continue
            try:
                await AsyncSandbox.delete_snapshot(snapshot.snapshot_id, **opts)
            except SandboxException as exc:
                if "sandboxes using it" not in str(exc):
                    raise
                report.in_use.append(swept)
                continue
            logger.info("deleted E2B snapshot %s (%s)", snapshot.snapshot_id, swept.age)
            report.deleted.append(swept)
    return report


__all__ = ["E2BSweepReport", "SweptSandbox", "SweptSnapshot", "sweep_e2b"]
