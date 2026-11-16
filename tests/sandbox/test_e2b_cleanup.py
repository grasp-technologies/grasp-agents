"""
``sweep_e2b``: finding and removing what E2B environments left on the
account, against a stand-in for ``AsyncSandbox`` that lists and filters the
way E2B does.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("e2b")

from e2b import SandboxException, SandboxState

from grasp_agents.sandbox import sweep_e2b
from grasp_agents.sandbox.e2b import cleanup

if TYPE_CHECKING:
    from e2b import SandboxQuery

pytestmark = pytest.mark.asyncio

NOW = dt.datetime.now(dt.UTC)
HOUR = dt.timedelta(hours=1)


def _snapshot_name(label: str, taken: dt.datetime, token: str) -> str:
    return f"team/grasp-agents-{label}-{taken:%Y%m%dt%H%M%S}-{token}"


@dataclass
class _Pages:
    items: list[Any]
    has_next: bool = True

    async def next_items(self) -> list[Any]:
        self.has_next = False
        return self.items


@dataclass
class _Account:
    """Stands in for ``AsyncSandbox``'s account-level class methods."""

    sandboxes: list[SimpleNamespace] = field(default_factory=list)
    snapshots: list[SimpleNamespace] = field(default_factory=list)
    in_use: set[str] = field(default_factory=set)
    calls: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)

    def sandbox(
        self,
        sandbox_id: str,
        *,
        state: SandboxState = SandboxState.PAUSED,
        age: dt.timedelta,
        metadata: dict[str, str],
    ) -> None:
        self.sandboxes.append(
            SimpleNamespace(
                sandbox_id=sandbox_id,
                state=state,
                metadata=metadata,
                started_at=NOW - age - HOUR,
                end_at=NOW - age if state is SandboxState.PAUSED else NOW + HOUR,
            )
        )

    def snapshot(self, name: str | None, template_id: str = "anon0") -> str:
        snapshot_id = f"{name or template_id}:default"
        bare = [name] if name else []
        self.snapshots.append(SimpleNamespace(snapshot_id=snapshot_id, names=bare))
        return snapshot_id

    def list(self, query: SandboxQuery | None = None, **opts: Any) -> _Pages:
        self.calls.append(("list", "", opts))
        wanted = query.metadata or {} if query is not None else {}
        states = query.state if query is not None and query.state else None
        return _Pages(
            [
                info
                for info in self.sandboxes
                if wanted.items() <= info.metadata.items()
                and (states is None or info.state in states)
            ]
        )

    def list_snapshots(self, **opts: Any) -> _Pages:
        self.calls.append(("list_snapshots", "", opts))
        return _Pages(list(self.snapshots))

    async def kill(self, sandbox_id: str, **opts: Any) -> bool:
        self.calls.append(("kill", sandbox_id, opts))
        return True

    async def delete_snapshot(self, snapshot_id: str, **opts: Any) -> bool:
        self.calls.append(("delete_snapshot", snapshot_id, opts))
        if snapshot_id in self.in_use:
            raise SandboxException(
                f"400: cannot delete template '{snapshot_id}' because there are "
                "paused sandboxes using it"
            )
        return True

    def acted(self) -> list[tuple[str, str]]:
        return [
            (name, target)
            for name, target, _ in self.calls
            if name in {"kill", "delete_snapshot"}
        ]


@pytest.fixture
def account(monkeypatch: pytest.MonkeyPatch) -> _Account:
    fake = _Account()
    monkeypatch.setattr(cleanup, "AsyncSandbox", fake)
    return fake


def _seed(account: _Account) -> dict[str, str]:
    tag = {"grasp-agents": "prod"}
    account.sandbox("old-paused", age=3 * HOUR, metadata=tag)
    account.sandbox("young-paused", age=HOUR / 2, metadata=tag)
    account.sandbox(
        "old-running", state=SandboxState.RUNNING, age=3 * HOUR, metadata=tag
    )
    account.sandbox("other-label", age=3 * HOUR, metadata={"grasp-agents": "dev"})
    account.sandbox("untagged", age=3 * HOUR, metadata={})
    return {
        "old": account.snapshot(_snapshot_name("prod", NOW - 3 * HOUR, "0000000a")),
        "kept": account.snapshot(_snapshot_name("prod", NOW - 3 * HOUR, "0000000b")),
        "in_use": account.snapshot(_snapshot_name("prod", NOW - 3 * HOUR, "0000000c")),
        "young": account.snapshot(_snapshot_name("prod", NOW - HOUR / 2, "0000000d")),
        "other": account.snapshot(_snapshot_name("dev", NOW - 3 * HOUR, "0000000e")),
        "unnamed": account.snapshot(None, template_id="anon1"),
    }


async def test_a_dry_run_reports_and_changes_nothing(account: _Account) -> None:
    refs = _seed(account)

    report = await sweep_e2b(
        older_than=2 * HOUR, keep_snapshots={refs["kept"]}, label="prod"
    )

    assert report.dry_run
    assert [s.sandbox_id for s in report.killed] == ["old-paused"]
    assert {s.snapshot_id for s in report.deleted} == {refs["old"], refs["in_use"]}
    assert [s.snapshot_id for s in report.kept] == [refs["kept"]]
    assert account.acted() == []


async def test_a_sweep_kills_sandboxes_then_deletes_snapshots(
    account: _Account,
) -> None:
    refs = _seed(account)
    account.in_use.add(refs["in_use"])

    report = await sweep_e2b(
        older_than=2 * HOUR,
        keep_snapshots={refs["kept"]},
        label="prod",
        dry_run=False,
    )

    assert account.acted() == [
        ("kill", "old-paused"),
        ("delete_snapshot", refs["old"]),
        ("delete_snapshot", refs["in_use"]),
    ]
    assert [s.snapshot_id for s in report.deleted] == [refs["old"]]
    assert [s.snapshot_id for s in report.in_use] == [refs["in_use"]]
    (killed,) = report.killed
    assert dt.timedelta(hours=2.9) < killed.age < dt.timedelta(hours=3.1)


async def test_a_kept_ref_is_matched_whatever_its_tag(account: _Account) -> None:
    refs = _seed(account)

    report = await sweep_e2b(
        older_than=2 * HOUR,
        keep_snapshots={refs["kept"].replace(":default", ":v2")},
        label="prod",
    )

    assert [s.snapshot_id for s in report.kept] == [refs["kept"]]


async def test_running_sandboxes_are_swept_only_when_asked(account: _Account) -> None:
    _seed(account)

    report = await sweep_e2b(
        older_than=2 * HOUR,
        keep_snapshots=(),
        label="prod",
        states=("paused", "running"),
    )

    assert [s.sandbox_id for s in report.killed] == ["old-paused", "old-running"]


async def test_the_default_label_and_extra_metadata_narrow_the_sandboxes(
    account: _Account,
) -> None:
    account.sandbox("a", age=3 * HOUR, metadata={"grasp-agents": "default", "app": "x"})
    account.sandbox("b", age=3 * HOUR, metadata={"grasp-agents": "default"})

    report = await sweep_e2b(
        older_than=2 * HOUR, keep_snapshots=(), metadata={"app": "x"}
    )

    assert [s.sandbox_id for s in report.killed] == ["a"]


async def test_the_api_key_and_domain_reach_every_call(account: _Account) -> None:
    refs = _seed(account)

    await sweep_e2b(
        older_than=2 * HOUR,
        keep_snapshots={refs["kept"]},
        label="prod",
        dry_run=False,
        api_key="key-1",
        domain="e2b.example",
    )

    opts = {"api_key": "key-1", "domain": "e2b.example"}
    assert all(call_opts == opts for _, _, call_opts in account.calls)


async def test_a_negative_age_is_refused(account: _Account) -> None:
    with pytest.raises(ValueError, match="older_than"):
        await sweep_e2b(older_than=-HOUR, keep_snapshots=())
