"""
Tool-call approval with a pending queue, scoped allowlists, and
async resolve.

Use this when approvals need to:

* be **scoped** (``once`` / ``session`` / ``always``) so repeat calls
  of the same tool skip re-prompting;
* be **asynchronous** — a UI, HTTP handler, or chat bot resolves
  pending requests later from different code paths;
* optionally **persist** the ``always`` allowlist across process
  restarts via a JSON file at ``persist_path``.

The agent task awaits an :class:`asyncio.Future`; external code
completes it by calling :meth:`ApprovalStore.resolve`. The requesting
task must stay alive between submit and resolve — if the process
dies, pending approvals are lost.

Usage::

    store = InMemoryApprovalStore(persist_path=Path(".approvals.json"))

    # Poll or surface ``store.list_pending(session_key)`` from your UI
    # and resolve each one via ``store.resolve(...)``.

    agent.add_before_tool_hook(
        build_store_approval(tool_names={"delete_file", "send_email"})
    )

    ctx = RunContext(
        approval_store=store,
        approval_session_key="user-42",
    )
    await agent.run(..., ctx=ctx)
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .tool_decision import RejectToolContent, ToolCallDecision

if TYPE_CHECKING:
    from collections.abc import Callable, Container, Mapping, Sequence
    from pathlib import Path

    from ..run_context import RunContext
    from ..types.hooks import BeforeToolHook
    from ..types.items import FunctionToolCallItem


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class ApprovalScope(StrEnum):
    """How long an allow decision is remembered."""

    ONCE = "once"
    """Allow just this invocation. Nothing is remembered."""

    SESSION = "session"
    """Allow future calls with the same ``approval_key`` in this
    session (keyed by ``session_key`` in the store)."""

    ALWAYS = "always"
    """Allow permanently. Written to the persistent allowlist and,
    if the store has ``persist_path`` set, to disk."""


@dataclass(frozen=True, slots=True)
class ApprovalAllow:
    """
    Let the tool run. ``scope`` controls whether future calls with
    the same ``approval_key`` skip the prompt.
    """

    scope: ApprovalScope = ApprovalScope.ONCE


@dataclass(frozen=True, slots=True)
class ApprovalDeny:
    """
    Block the tool. ``reason`` becomes the
    :class:`~grasp_agents.agent.tool_decision.RejectToolContent`
    content shown to the LLM.
    """

    reason: str = "User denied tool call"


ApprovalDecision = ApprovalAllow | ApprovalDeny


@dataclass(frozen=True, slots=True)
class PendingApproval:
    """
    A tool call waiting for a human (or policy) decision.

    ``approval_key`` is the stable identifier the store uses for
    allowlist matching. Default: the tool name. For finer control
    (per-args, per-path), pass ``approval_key_fn`` to
    :func:`build_store_approval`.
    """

    session_key: str
    call_id: str
    tool_name: str
    arguments: str
    approval_key: str


if TYPE_CHECKING:
    ApprovalKeyFn = Callable[[FunctionToolCallItem], str]


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


@runtime_checkable
class ApprovalStore(Protocol):
    """
    Contract for pending-approval storage and allowlist management.
    Implementations may back pending state and allowlists with memory,
    a file, a database, or a shared queue. :func:`build_store_approval`
    depends only on this protocol.
    """

    async def submit_pending(
        self, pending: PendingApproval
    ) -> asyncio.Future[ApprovalDecision]:
        """Register a pending request; return a future to await the decision."""
        ...

    async def resolve(
        self,
        session_key: str,
        call_id: str,
        decision: ApprovalDecision,
    ) -> bool:
        """
        Complete the pending future with ``decision``.

        Returns ``True`` if a pending request was found and resolved,
        ``False`` if the ``(session_key, call_id)`` pair had no pending
        request (e.g. already resolved, never submitted, or timed out).

        When ``decision`` is :class:`ApprovalAllow` with
        ``scope=SESSION`` or ``ALWAYS`` the store also records the
        pending's ``approval_key`` in the matching allowlist.
        """
        ...

    async def list_pending(self, session_key: str) -> list[PendingApproval]:
        """Snapshot of unresolved requests for ``session_key``."""
        ...

    async def is_pre_approved(
        self, approval_key: str, *, session_key: str
    ) -> bool:
        """True if ``approval_key`` is in the session or persistent allowlist."""
        ...

    async def add_persistent(self, approval_key: str) -> None:
        """Unconditionally add ``approval_key`` to the persistent allowlist."""
        ...

    async def add_session(self, session_key: str, approval_key: str) -> None:
        """Unconditionally add ``approval_key`` to ``session_key``'s allowlist."""
        ...

    async def clear_session(self, session_key: str) -> None:
        """Drop session allowlist + any pending requests for ``session_key``."""
        ...


class InMemoryApprovalStore:
    """
    Default :class:`ApprovalStore` backed by ``asyncio.Future`` objects.
    State survives across agent turns within the same process. The
    ``always`` allowlist also survives process restart when
    ``persist_path`` is set (JSON file).
    """

    def __init__(self, *, persist_path: Path | None = None) -> None:
        self._persist_path = persist_path
        self._lock = asyncio.Lock()
        self._pending: dict[str, dict[str, PendingApproval]] = {}
        self._futures: dict[
            tuple[str, str], asyncio.Future[ApprovalDecision]
        ] = {}
        self._session_allowlist: dict[str, set[str]] = {}
        self._persistent_allowlist: set[str] = set()
        self._load_persistent()

    # --- Persistence (best-effort) ---

    def _load_persistent(self) -> None:
        """
        Populate ``_persistent_allowlist`` from ``persist_path``.

        Silently ignores malformed files — the allowlist is a user
        preference cache, not load-bearing state.
        """
        if self._persist_path is None or not self._persist_path.is_file():
            return
        try:
            raw = json.loads(self._persist_path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(raw, dict):
            return
        keys: Any = raw.get("always_approved")  # type: ignore[misc]
        if isinstance(keys, list):
            self._persistent_allowlist = {str(k) for k in keys}  # type: ignore[misc]

    def _save_persistent(self) -> None:
        if self._persist_path is None:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist_path.write_text(
            json.dumps(
                {"always_approved": sorted(self._persistent_allowlist)},
                indent=2,
            )
        )

    # --- ApprovalStore methods ---

    async def submit_pending(
        self, pending: PendingApproval
    ) -> asyncio.Future[ApprovalDecision]:
        loop = asyncio.get_running_loop()
        async with self._lock:
            self._pending.setdefault(pending.session_key, {})[
                pending.call_id
            ] = pending
            fut: asyncio.Future[ApprovalDecision] = loop.create_future()
            self._futures[pending.session_key, pending.call_id] = fut
        return fut

    async def resolve(
        self,
        session_key: str,
        call_id: str,
        decision: ApprovalDecision,
    ) -> bool:
        async with self._lock:
            fut = self._futures.pop((session_key, call_id), None)
            pending = self._pending.get(session_key, {}).pop(call_id, None)
            if not self._pending.get(session_key):
                self._pending.pop(session_key, None)
            if isinstance(decision, ApprovalAllow) and pending is not None:
                if decision.scope is ApprovalScope.SESSION:
                    self._session_allowlist.setdefault(
                        session_key, set()
                    ).add(pending.approval_key)
                elif decision.scope is ApprovalScope.ALWAYS:
                    self._persistent_allowlist.add(pending.approval_key)
                    self._save_persistent()
        if fut is None:
            return False
        if not fut.done():
            fut.set_result(decision)
        return True

    async def list_pending(self, session_key: str) -> list[PendingApproval]:
        async with self._lock:
            return list(self._pending.get(session_key, {}).values())

    async def is_pre_approved(
        self, approval_key: str, *, session_key: str
    ) -> bool:
        async with self._lock:
            if approval_key in self._persistent_allowlist:
                return True
            return approval_key in self._session_allowlist.get(
                session_key, set()
            )

    async def add_persistent(self, approval_key: str) -> None:
        async with self._lock:
            self._persistent_allowlist.add(approval_key)
            self._save_persistent()

    async def add_session(
        self, session_key: str, approval_key: str
    ) -> None:
        async with self._lock:
            self._session_allowlist.setdefault(session_key, set()).add(
                approval_key
            )

    async def clear_session(self, session_key: str) -> None:
        async with self._lock:
            self._session_allowlist.pop(session_key, None)
            pending_map = self._pending.pop(session_key, {})
            for call_id in pending_map:
                fut = self._futures.pop((session_key, call_id), None)
                if fut is not None and not fut.done():
                    fut.cancel()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_store_approval(
    *,
    tool_names: Container[str] | None = None,
    approval_key_fn: ApprovalKeyFn | None = None,
    timeout: float | None = None,
) -> BeforeToolHook[Any]:
    """
    Build a :class:`BeforeToolHook` that consults the
    :class:`ApprovalStore` on the active
    :class:`~grasp_agents.run_context.RunContext`.

    Set ``ctx.approval_store`` to enable the gate; if it is ``None``
    the hook is a no-op and all calls run. ``ctx.approval_session_key``
    scopes the ``session`` allowlist so different users don't share
    each other's decisions.

    Each matched call goes through the approval flow:
    pre-approval check (skip if the key is already allowed), submit a
    :class:`PendingApproval`, await the resulting future. All requests
    in a batch are submitted up front so a UI can present them
    together; awaits are sequential.

    ``tool_names`` restricts gating to named tools (all when ``None``).
    ``approval_key_fn`` controls the allowlist key (defaults to the
    tool name; use ``lambda c: f"{c.name}:{c.arguments}"`` for
    per-arguments matching). ``timeout`` (seconds, per-call) denies
    the call automatically if no decision arrives; ``None`` waits
    forever.
    """
    key_fn: ApprovalKeyFn = approval_key_fn or (lambda call: call.name)

    async def hook(
        *,
        tool_calls: Sequence[FunctionToolCallItem],
        ctx: RunContext[Any],
        exec_id: str,  # noqa: ARG001 — BeforeToolHook protocol signature
    ) -> Mapping[str, ToolCallDecision] | None:
        store = ctx.approval_store
        if store is None:
            return None
        session_key = ctx.approval_session_key

        decisions: dict[str, ToolCallDecision] = {}
        pending_futures: list[
            tuple[FunctionToolCallItem, asyncio.Future[ApprovalDecision]]
        ] = []

        for call in tool_calls:
            if tool_names is not None and call.name not in tool_names:
                continue
            approval_key = key_fn(call)
            if await store.is_pre_approved(
                approval_key, session_key=session_key
            ):
                continue
            pending = PendingApproval(
                session_key=session_key,
                call_id=call.call_id,
                tool_name=call.name,
                arguments=call.arguments,
                approval_key=approval_key,
            )
            fut = await store.submit_pending(pending)
            pending_futures.append((call, fut))

        for call, fut in pending_futures:
            try:
                if timeout is not None:
                    decision = await asyncio.wait_for(fut, timeout=timeout)
                else:
                    decision = await fut
            except TimeoutError:
                decisions[call.call_id] = RejectToolContent(
                    content=(
                        f"Approval for '{call.name}' timed out "
                        f"after {timeout}s."
                    )
                )
                continue
            except asyncio.CancelledError:
                decisions[call.call_id] = RejectToolContent(
                    content=f"Approval for '{call.name}' was cancelled.",
                )
                continue
            if isinstance(decision, ApprovalDeny):
                decisions[call.call_id] = RejectToolContent(
                    content=decision.reason
                )
            # ApprovalAllow → AllowTool default → no entry needed

        return decisions or None

    return hook
