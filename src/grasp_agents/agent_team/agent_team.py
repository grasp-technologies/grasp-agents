"""
``AgentTeam`` — a standalone host for a team of peer agents that communicate
asynchronously by sending each other messages.

A sibling to :class:`~grasp_agents.runner.Runner` over the **same** actor runtime
(:class:`~grasp_agents.runtime.ActorDriver` + :class:`~grasp_agents.runtime.
Transport`): where ``Runner`` drives an orchestrated graph to a single result
packet, ``AgentTeam`` lets members each run their own loop and message one another
over a shared mailbox ``Transport`` (in-memory or durable). A *resident* (an
agent with no static recipients) runs **resident** — one long-lived loop consuming
its inbox between turns; a *transform* (a processor, or an agent with static
recipients) is **triggered** by the shared :class:`ActorDriver`, one activation per
message. Both consume the one transport. The session ends at **quiescence** (no
member running, every inbox empty, no background work) or at the ``max_hops`` budget.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import timedelta
from typing import TYPE_CHECKING, Any, ClassVar, cast
from uuid import uuid4

from pydantic import BaseModel, Field

from grasp_agents.durability.checkpoint_mixin import CheckpointPersistMixin
from grasp_agents.durability.checkpoints import CheckpointKind, TeamCheckpoint
from grasp_agents.mailbox import CheckpointMailboxTransport, resolve_session_transport
from grasp_agents.runtime import ActorDriver
from grasp_agents.session_context import SessionContext, current_session_context

from ._roles import activate_member, is_llm_agent, is_resident, resident_idle
from .agent_card import MemberCard
from .events import (
    MessageDeliveredEvent,
    TeamEndedEvent,
    TeamRunInfo,
    TeamStartedEvent,
    TeamStopReason,
)
from .message import CONTROL_PRIORITY, USER_SENDER, TeamMessage
from .prompt import (
    make_rewind_notice,
    make_sender_attribution_attachment,
    make_team_section,
)
from .tools import (
    SCHEDULE_WAKEUP_TOOL_NAME,
    SEND_MESSAGE_TOOL_NAME,
    ScheduleWakeupTool,
    SendMessageTool,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

    from grasp_agents.agent.llm_agent import LLMAgent
    from grasp_agents.processors.processor import Processor
    from grasp_agents.runtime import Transport
    from grasp_agents.tools.base import BaseTool
    from grasp_agents.types.events import Event

logger = logging.getLogger(__name__)

# Prefix for the entry-seed's deterministic message id. The 21-char all-zero
# timestamp shape sorts before any real (``_new_message_id``) id, so the seed is
# consumed first; the fixed value makes a resume re-post idempotent.
_SEED_ID_PREFIX = "00000000T000000000000-seed-"

# How often a running team reclaims its durable mailbox ``processed/`` records, and
# how long a record is kept before it is eligible (a forensic grace window past the
# point it is no longer needed for redelivery dedup). A daemon team never quiesces,
# so it must sweep on its own rather than relying on the host to call
# ``CheckpointMailboxTransport.prune_processed``.
_MAILBOX_GC_INTERVAL_S = 300.0
_MAILBOX_PROCESSED_RETENTION = timedelta(hours=1)
# Dead-lettered (corrupt) records are kept longer — they exist for post-hoc
# inspection, and corruption is rare (records are written atomically).
_MAILBOX_CORRUPT_RETENTION = timedelta(days=7)


def _seed_message_id(entry: str) -> str:
    return f"{_SEED_ID_PREFIX}{entry}"


def _is_seed_id(message_id: str) -> bool:
    """
    A permanent entry-seed marker — pinned against mailbox GC so a later resume
    still finds it and does not re-seed the entry (see :meth:`_drive`).
    """
    return message_id.startswith(_SEED_ID_PREFIX)


class TeamRunResult(BaseModel):
    """Summary of a completed team session (returned by :meth:`AgentTeam.run`)."""

    messages: list[TeamMessage] = Field(default_factory=list[TeamMessage])
    activations: int = 0
    stop_reason: TeamStopReason = TeamStopReason.QUIESCED


class AgentTeam[CtxT](CheckpointPersistMixin):
    _checkpoint_kind: ClassVar[CheckpointKind | None] = CheckpointKind.TEAM

    def __init__(
        self,
        members: Sequence[Processor[Any, Any, CtxT]],
        *,
        entry: Processor[Any, Any, CtxT] | str | None = None,
        cards: Sequence[MemberCard] | None = None,
        ctx: SessionContext[CtxT] | None = None,
        name: str | None = None,
        path: list[str] | None = None,
        max_hops: int = 50,
        max_tokens: int | None = None,
    ) -> None:
        if not members:
            raise ValueError("AgentTeam requires at least one member.")
        names = [m.name for m in members]
        dups = sorted({n for n in names if names.count(n) > 1})
        if dups:
            raise ValueError(f"Duplicate member names {dups}; names must be unique.")

        self._members = list(members)
        self._members_by_name = {m.name: m for m in self._members}
        self._name = name or f"team-{uuid4().hex[:6]}"
        self._path = path or []
        self._max_hops = max_hops
        self._max_tokens = max_tokens

        # A member with static recipients hands its output off by name; those names
        # must be team members. (A resident routes dynamically via SendMessage
        # and declares no static recipients, so there is nothing to validate.)
        member_names = set(self._members_by_name)
        for member in self._members:
            for r in member.recipients or []:
                if r not in member_names:
                    raise ValueError(
                        f"Member {member.name!r} routes to unknown recipient "
                        f"{r!r}; recipients must be team members: "
                        f"{', '.join(sorted(member_names))}."
                    )

        if entry is None:
            self._entry_name = names[0]
        elif isinstance(entry, str):
            if entry not in self._members_by_name:
                raise ValueError(f"Entry {entry!r} is not a team member.")
            self._entry_name = entry
        else:
            self._entry_name = entry.name

        self._cards = self._resolve_cards(cards)
        self._cards_by_name = {c.name: c for c in self._cards}

        leads = [c.name for c in self._cards if c.lead]
        if len(leads) > 1:
            raise ValueError(
                f"Team {self._name!r} declares more than one lead: {leads}; "
                "at most one member may be the lead."
            )
        # The lead's role — priority mail, rewind right, rewind announcements —
        # presumes a persistent loop; a triggered member is activated fresh per
        # message and cannot hold it.
        if leads and not is_resident(
            self._members_by_name[leads[0]], self._cards_by_name[leads[0]]
        ):
            raise ValueError(
                f"Team {self._name!r} lead {leads[0]!r} is a triggered member; "
                "the lead must run resident (an LLM agent consuming its inbox)."
            )

        # Bind the session: explicit ctx, else the ambient / process-default one.
        self._ctx: SessionContext[CtxT] = (
            ctx if ctx is not None else current_session_context()  # type: ignore[assignment]
        )

        # The lead holds the session's environment-rewind right; claiming here
        # makes a conflict (e.g. a ctx that already declares a different
        # rewinder) a construction error, not a mid-run one.
        if leads:
            self._ctx.claim_environment_rewind(leads[0])

        # When the rewinder restores a snapshot mid-run, the filesystem (and any
        # kernels) change under every other member — tell them so they re-verify
        # state instead of panicking over the shift.
        self._ctx.add_environment_restored_callback(self._notify_environment_rewind)

        # The one shared mailbox Transport every member views — residents
        # consume it via their AgentInbox, transforms via the ActorDriver.
        # Always the session's (``ctx.transport``), created and installed on
        # first use (durable over the session store, else in-memory), so a
        # host rebuilt mid-session reuses the same mailbox — and its live
        # consumption counters — instead of opening a fresh one.
        self._transport: Transport[TeamMessage] = resolve_session_transport(self._ctx)

        # Partition members by execution mode (explicit on the card, else inferred):
        # a resident runs a persistent loop off its inbox; everything else
        # (processors, agents with recipients) is a triggered transform.
        self._residents: dict[str, LLMAgent[Any, Any, Any]] = {}
        self._transforms: dict[str, Processor[Any, Any, CtxT]] = {}
        for member in self._members:
            if is_resident(member, self._cards_by_name.get(member.name)):
                self._residents[member.name] = cast("LLMAgent[Any, Any, Any]", member)
            else:
                self._transforms[member.name] = member

        # Give every resident the shared messaging + self-wakeup tools, then
        # adopt all members so ctx / path / tracing cascade down (as Runner adopts
        # its procs). SendMessage routes through the team (``self`` is the sink);
        # ScheduleWakeup is a self-contained background tool (no scheduler needed).
        send_tool = cast(
            "BaseTool[BaseModel, Any, CtxT]",
            SendMessageTool(self._cards, transport_resolver=lambda _ctx: self),
        )
        wakeup_tool = cast("BaseTool[BaseModel, Any, CtxT]", ScheduleWakeupTool())
        team_section = make_team_section(self._cards)
        for resident in self._residents.values():
            resident.tools[SEND_MESSAGE_TOOL_NAME] = send_tool
            resident.tools[SCHEDULE_WAKEUP_TOOL_NAME] = wakeup_tool
            resident.add_system_prompt_section(team_section)

        # A triggered member renders a peer hand-off through its own input pipeline,
        # which has no sender fence; give every LLM member the attribution attachment
        # so its turns name the teammate they came from (inert for a resident, which
        # gets attribution from the fence on its drained messages).
        attribution = make_sender_attribution_attachment()
        for member in self._members:
            if is_llm_agent(member):
                member.add_input_attachment(attribution)
            member.on_adopted(self)

        # Session-scoped checkpoint bookkeeping (the team persists only its
        # coordinator scalars — see TeamCheckpoint).
        self._checkpoint_number = 0
        self._checkpoint_lock = asyncio.Lock()

        # Per-run state (reset in run_stream).

        self._driver: ActorDriver[TeamMessage] | None = None
        self._resident_tasks: list[asyncio.Task[None]] = []

        self._activations = 0
        self._failed: list[str] = []

        self._hop_exhausted = False
        self._token_exhausted = False
        self._tokens_at_start = 0

        self._stop_requested = False
        self._daemon = False
        self._poll_interval = 0.05

    # -- properties read by Processor.on_adopted's duck-typing (mirror Runner) --

    @property
    def name(self) -> str:
        return self._name

    @property
    def ctx(self) -> SessionContext[CtxT]:
        return self._ctx

    @property
    def path(self) -> list[str]:
        return self._path

    # -- session persistence (coordinator scalars only) --

    async def _load_checkpoint(self) -> TeamCheckpoint | None:
        checkpoint = await self._deserialize_checkpoint(self._ctx, TeamCheckpoint)
        if checkpoint is not None:
            logger.info(
                "Loaded team checkpoint %s (activations=%d)",
                self._checkpoint_store_key(self._ctx),
                checkpoint.activations,
            )
        return checkpoint

    async def _save_checkpoint(self) -> None:
        checkpoint = TeamCheckpoint(
            session_key=self._ctx.session_key,
            processor_name=self._name,
            activations=self._activations,
        )
        await self._serialize_checkpoint(self._ctx, checkpoint)

    def _resolve_cards(self, cards: Sequence[MemberCard] | None) -> list[MemberCard]:
        if cards is None:
            return [MemberCard(name=n) for n in self._members_by_name]

        unknown = sorted({c.name for c in cards} - set(self._members_by_name))
        if unknown:
            raise ValueError(f"Cards reference non-members {unknown}.")

        provided = {c.name: c for c in cards}

        return [provided.get(n, MemberCard(name=n)) for n in self._members_by_name]

    async def _push(self, event: Event[Any]) -> None:
        driver = self._driver
        if driver is not None:
            await driver.push_to_stream(event)

    # -- routing (the team is the MessageSink every send goes through) --

    async def post(self, envelope: TeamMessage) -> None:
        """
        Route a message to its recipient(s): announce the delivery, count it against
        the hop budget, then deposit it on the shared transport (where the
        recipient's resident loop or the driver picks it up). This is the single
        interception point — ``SendMessage``, processor hand-off, the entry seed, and
        external sources / wakeups all go through it.

        Outside a live run (no driver) the post is dropped rather than half-applied —
        e.g. a wakeup timer that fires after the run quiesced must not mutate the next
        run's accounting or deposit an orphan message no one consumes.
        """
        if self._driver is None:
            logger.debug(
                "AgentTeam %s: post outside a live run; dropping message from %r",
                self._name,
                envelope.sender,
            )
            return

        for single in envelope.split_by_recipient():
            recipient = single.recipient
            if recipient not in self._members_by_name:
                logger.warning(
                    "AgentTeam %s dropping message to unknown recipient %r",
                    self._name,
                    recipient,
                )
                continue

            # Count + persist the hop under the lock so concurrent sends can't
            # both slip past the budget or double-bump the checkpoint number.
            # The count is saved BEFORE depositing on the transport (mirroring
            # the Runner's save-before-post): a crash then leaves the budget
            # counted rather than under-counted — conservative for a safety cap.
            async with self._checkpoint_lock:
                if not self._daemon and self._activations >= self._max_hops:
                    # Budget spent: refuse further deliveries and ask to stop.
                    self._hop_exhausted = True
                    self._stop_requested = True
                    return

                if not self._daemon and self._over_token_budget():
                    # Token ceiling reached: stop spawning new activations (those
                    # already in flight finish). A daemon opts out, like max_hops.
                    self._token_exhausted = True
                    self._stop_requested = True
                    return

                self._activations += 1
                await self._save_checkpoint()

            await self._push(
                MessageDeliveredEvent(
                    source=single.sender, destination=recipient, data=single
                )
            )
            await self._transport.post(single)

    async def submit_message(self, to: str, text: str) -> None:
        """
        Inject human input to a member mid-run — a control-plane message (drains
        ahead of peer mail) routed like any other send. Use this to talk to a team
        agent from a UI pane. A no-op outside a live run (``post`` drops it); a
        bounded run counts it against ``max_hops`` (a daemon run does not).
        """
        await self.post(
            TeamMessage.from_text(
                sender=USER_SENDER, to=to, text=text, priority=CONTROL_PRIORITY
            )
        )

    async def _notify_environment_rewind(self, fs_snapshot_ref: str) -> None:
        """
        Tell every other resident the environment was rewound (control-plane, so
        the notice drains ahead of queued peer mail and stale context is caught
        before it is acted on). Residents only: a triggered member is activated
        fresh per message and holds no cross-turn view of the filesystem. A no-op
        outside a live run (``post`` drops it) — with no member mid-turn there is
        no one to startle.
        """
        del fs_snapshot_ref
        rewinder = self._ctx.environment_rewinder
        recipients = [name for name in self._residents if name != rewinder]
        if rewinder is None or not recipients:
            return
        await self.post(
            TeamMessage.from_text(
                sender=rewinder,
                to=recipients,
                text=make_rewind_notice(rewinder),
                priority=CONTROL_PRIORITY,
            )
        )

    # -- lifecycle (session-scoped; the embedder closes the team) --

    async def aclose(self) -> None:
        for member in self._members:
            try:
                await member.aclose()
            except Exception:
                logger.warning(
                    "Failed to close team member %r during teardown",
                    member.name,
                    exc_info=True,
                )

    async def __aenter__(self) -> AgentTeam[CtxT]:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # -- running --

    async def run(
        self, chat_inputs: Any = None, *, to: str | None = None, **run_kwargs: Any
    ) -> TeamRunResult:
        """Drive the team to quiescence and summarize the session."""
        messages: list[TeamMessage] = []
        activations = 0
        stop_reason = TeamStopReason.QUIESCED

        async for event in self.run_stream(chat_inputs, to=to, **run_kwargs):
            if isinstance(event, MessageDeliveredEvent):
                messages.append(event.data)
            elif isinstance(event, TeamEndedEvent):
                activations = event.data.activations
                stop_reason = event.data.stop_reason or TeamStopReason.QUIESCED

        return TeamRunResult(
            messages=messages, activations=activations, stop_reason=stop_reason
        )

    async def run_stream(
        self,
        chat_inputs: Any = None,
        *,
        to: str | None = None,
        daemon: bool = False,
        poll_interval: float = 0.05,
        **run_kwargs: Any,
    ) -> AsyncIterator[Event[Any]]:
        """
        Seed the entry member with ``chat_inputs`` and stream every member's events.

        By default the run ends at **quiescence** (no member running, every inbox
        empty, no background work) — or at a budget: the ``max_hops`` delivery count
        or the ``max_tokens`` ceiling (this run's token spend). With ``daemon=True``
        it never self-terminates: it keeps serving mail (e.g. from an external
        source), opts out of both budgets, and dead-letters a member failure rather
        than stopping the team. Stop a daemon by cancelling the stream. Events carry
        ``source = <member name>``.
        """
        entry = to or self._entry_name
        if entry not in self._members_by_name:
            raise ValueError(f"Recipient {entry!r} is not a team member.")

        self._daemon = daemon
        self._poll_interval = poll_interval
        self._activations = 0
        self._failed = []

        # Baseline the per-run token budget here; on resume this rebases (the budget
        # bounds this run's own spend, generous across restarts — like max_hops).
        self._tokens_at_start = self._ctx.usage_tracker.total_usage.total_tokens
        self._token_exhausted = False
        self._hop_exhausted = False

        self._stop_requested = False

        # Cleared until the new driver is wired (a few lines down), so a source /
        # wakeup racing the run start is dropped by ``post`` rather than landing on a
        # stale driver from a prior run.
        self._driver = None

        self._resident_tasks = []

        # Session-scoped restore (ctx.state + shared filesystem) before any
        # member runs; idempotent, so member runs below no-op on it.
        await self._ctx.load_checkpoint()

        # Resume: restore the session-wide hop count from a prior checkpoint. The
        # entry is re-seeded idempotently in ``_drive`` (deterministic id +
        # processed-dedup), so it is delivered exactly once across runs without a
        # separate "seeded" flag — a flag keyed on checkpoint existence could
        # outlive an undelivered seed (saved before the deposit) and strand it.
        checkpoint = await self._load_checkpoint()
        if checkpoint is not None:
            self._activations = checkpoint.activations

        yield TeamStartedEvent(source=self._name, data=TeamRunInfo(team=self._name))

        stop_reason = TeamStopReason.QUIESCED

        # Daemon mode: the team owns quiescence + the hop budget, so the driver never
        # self-terminates — it just runs the triggered transforms and merges events.
        driver: ActorDriver[TeamMessage] = ActorDriver(
            self._transport, termination="daemon"
        )
        self._driver = driver

        try:
            async for event in self._drive(driver, entry, chat_inputs, run_kwargs):
                yield event

        except asyncio.CancelledError:
            self._driver = None
            raise

        except Exception:
            logger.exception("AgentTeam %s failed", self._name)
            stop_reason = TeamStopReason.ERROR

        else:
            stop_reason = self._final_stop_reason()

        self._driver = None

        yield TeamEndedEvent(
            source=self._name,
            data=TeamRunInfo(
                team=self._name, activations=self._activations, stop_reason=stop_reason
            ),
        )

    async def _drive(
        self,
        driver: ActorDriver[TeamMessage],
        entry: str,
        chat_inputs: Any,
        run_kwargs: dict[str, Any],
    ) -> AsyncIterator[Event[Any]]:
        async with driver:
            # Triggered transforms run on the shared driver; resident residents
            # run their own loop off the same transport.
            for member in self._transforms.values():
                driver.register_handler(
                    member.name, self._make_transform_handler(member, run_kwargs)
                )

            for member_name, resident in self._residents.items():
                resident.attach_inbox()
                self._resident_tasks.append(
                    asyncio.create_task(
                        self._run_resident(resident, run_kwargs),
                        name=f"resident:{member_name}",
                    )
                )

            monitor = asyncio.create_task(
                self._monitor(driver), name=f"{self._name}-monitor"
            )

            try:
                # Seed the entry idempotently. Unlike a peer send (re-issued when
                # its sender's turn re-runs on resume), the seed has no retry
                # source, so it carries a deterministic id: a re-post on resume
                # overwrites a still-pending copy and is dedup-skipped once the
                # entry has processed it. So the entry is seeded exactly once
                # across the session — never dropped (a crash can leave the hop
                # checkpoint saved with the seed undelivered), never duplicated.
                if chat_inputs is not None:
                    seed = TeamMessage.from_text(
                        sender=USER_SENDER,
                        to=entry,
                        text=str(chat_inputs),
                        message_id=_seed_message_id(entry),
                    )
                    if not await self._transport.was_processed(entry, seed.message_id):
                        await self.post(seed)

                async for event in driver.stream_events():
                    yield event

            finally:
                monitor.cancel()
                for task in self._resident_tasks:
                    task.cancel()
                await asyncio.gather(
                    monitor, *self._resident_tasks, return_exceptions=True
                )

                for resident in self._residents.values():
                    resident.detach_inbox()

    async def _run_resident(
        self, resident: LLMAgent[Any, Any, Any], run_kwargs: dict[str, Any]
    ) -> None:
        """
        Run one resident member: a single seedless ``run_stream`` whose loop
        starts parked, consumes its inbox between turns, and ends only when this
        task is cancelled (at quiescence / daemon stop). Its events bubble to the
        team stream.
        """
        try:
            async for event in resident.run_stream(**run_kwargs):
                await self._push(event)

        except asyncio.CancelledError:
            raise

        except Exception:
            logger.warning("Resident member %r failed", resident.name, exc_info=True)
            self._failed.append(resident.name)
            if not self._daemon:
                self._stop_requested = True

    def _make_transform_handler(
        self, member: Processor[Any, Any, CtxT], run_kwargs: dict[str, Any]
    ) -> Callable[[TeamMessage], Awaitable[None]]:
        """
        A triggered member's activation: run it over the one inbound message, bubble
        its events, and route any output back through the team.
        """

        async def handler(message: TeamMessage) -> None:
            try:
                await activate_member(
                    member,
                    message,
                    transport=self._transport,
                    run_kwargs=run_kwargs,
                    push=self._push,
                    post=self.post,
                )

            except asyncio.CancelledError:
                raise

            except Exception:
                # Dead-letter a member failure rather than tearing down the team;
                # a non-daemon run stops once this activation unwinds.
                logger.warning(
                    "Team member %r failed handling a message",
                    member.name,
                    exc_info=True,
                )
                self._failed.append(member.name)
                if not self._daemon:
                    self._stop_requested = True

        return handler

    async def _monitor(self, driver: ActorDriver[TeamMessage]) -> None:
        """
        Detect quiescence (or a stop request) and tear the run down: cancel the
        resident loops and shut the driver, which ends the event stream.

        Quiescence must hold across two consecutive polls with no new activation in
        between — a cheap guard against the small window between an idle observation
        and a delivery that races it.
        """
        last_idle_activations: int | None = None
        last_gc = time.monotonic()

        while True:
            await asyncio.sleep(self._poll_interval)
            now = time.monotonic()
            if now - last_gc >= _MAILBOX_GC_INTERVAL_S:
                last_gc = now
                await self._gc_mailbox()

            if self._stop_requested:
                break

            if self._daemon:
                continue

            if not await self._is_quiescent(driver):
                last_idle_activations = None
                continue

            if last_idle_activations == self._activations:
                break

            last_idle_activations = self._activations

        for task in self._resident_tasks:
            task.cancel()

        await driver.shutdown()

    async def _gc_mailbox(self) -> None:
        """
        Reclaim old durable ``processed/`` records, pinning the entry seeds. A
        no-op on the in-memory transport (nothing to reclaim). Housekeeping, so a
        transient sweep failure is logged and the run continues — the delivery path
        crashes on its own if the store is truly broken.
        """
        transport = self._transport
        if not isinstance(transport, CheckpointMailboxTransport):
            return
        try:
            await transport.prune_processed(
                older_than=_MAILBOX_PROCESSED_RETENTION,
                keep=_is_seed_id,
                corrupt_older_than=_MAILBOX_CORRUPT_RETENTION,
            )
        except Exception:
            logger.warning("AgentTeam %s: mailbox GC failed", self._name, exc_info=True)

    async def _is_quiescent(self, driver: ActorDriver[TeamMessage]) -> bool:
        for resident in self._residents.values():
            if not await resident_idle(resident):
                return False
        return await driver.is_quiescent()

    def _over_token_budget(self) -> bool:
        """
        Whether this run's token spend (delta from its start baseline) has
        reached ``max_tokens``. Always ``False`` when no budget is set.
        """
        if self._max_tokens is None:
            return False
        spent = self._ctx.usage_tracker.total_usage.total_tokens - self._tokens_at_start
        return spent >= self._max_tokens

    def _final_stop_reason(self) -> TeamStopReason:
        if self._failed:
            logger.warning(
                "AgentTeam %s: members failed: %s",
                self._name,
                ", ".join(self._failed),
            )
            return TeamStopReason.MEMBER_ERROR

        if self._hop_exhausted:
            logger.warning(
                "AgentTeam %s reached max_hops=%d with mail still pending",
                self._name,
                self._max_hops,
            )
            return TeamStopReason.HOP_BUDGET_EXHAUSTED

        if self._token_exhausted:
            logger.warning(
                "AgentTeam %s reached max_tokens=%s with mail still pending",
                self._name,
                self._max_tokens,
            )
            return TeamStopReason.TOKEN_BUDGET_EXHAUSTED

        return TeamStopReason.QUIESCED
