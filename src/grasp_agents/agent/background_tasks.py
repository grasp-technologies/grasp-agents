import asyncio
import contextlib
import json
import operator
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from logging import getLogger
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from grasp_agents.durability.checkpoint_store import CheckpointStore
from grasp_agents.durability.checkpoints import CheckpointKind
from grasp_agents.durability.store_keys import (
    is_direct_child,
    make_store_key,
    make_tool_call_path,
    task_prefix,
)
from grasp_agents.durability.task_record import TaskRecord, TaskStatus
from grasp_agents.session_context import SessionContext
from grasp_agents.tools.base import BaseTool
from grasp_agents.types.events import (
    BackgroundTaskCompletedEvent,
    BackgroundTaskInfo,
    BackgroundTaskLaunchedEvent,
    Event,
    ToolErrorEvent,
    ToolErrorInfo,
    ToolOutputEvent,
    ToolStreamEvent,
    UserMessageEvent,
)
from grasp_agents.types.items import FunctionToolCallItem, InputMessageItem

from .llm_agent_transcript import LLMAgentTranscript
from .task_progress import (
    append_task_log,
    excerpt_for_inline,
    open_task_log,
    task_log_name,
    write_result_file,
)

if TYPE_CHECKING:
    from grasp_agents.file_backend.base import FileBackend

    from .agent_context import AgentContext

logger = getLogger(__name__)

# Safety ceiling on a single background task's progress log. Bounds disk for a
# runaway command (e.g. ``yes``); once hit, a one-time marker is appended and
# further output is dropped from the file (the live event stream is unaffected).
MAX_TASK_LOG_BYTES = 64 * 1024 * 1024


def _fmt_duration(seconds: float) -> str:
    """Compact human duration: ``45s`` / ``4m12s`` / ``1h05m``."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


class KillTaskResult(BaseModel):
    """
    What ``KillTask`` returns for a stopped background task.

    ``status`` is ``cancelled`` when the task was still running and ``KillTask``
    stopped it (the common case), or ``completed`` / ``failed`` when it had
    already reached that terminal state before the kill arrived. ``output`` is
    an excerpt of the text the task produced (the full output is in its
    ``.grasp`` log); ``result`` is the tool's own terminal output (e.g. a
    ``BashResult``) when it had finished, else ``None``.
    """

    task_id: str
    tool_name: str
    status: Literal["cancelled", "completed", "failed"]
    output: str = ""
    result: Any = None


def _task_notification(
    *,
    task_id: str,
    tool_name: str,
    status: str,
    result: str | None = None,
    error: str | None = None,
    log_path: str | None = None,
    elapsed_s: float | None = None,
) -> str:
    """Build an XML-tagged task notification for LLM consumption."""
    parts = [
        "<task_notification>",
        f"<task_id>{task_id}</task_id>",
        f"<tool_name>{tool_name}</tool_name>",
        f"<status>{status}</status>",
    ]
    if elapsed_s is not None:
        parts.append(f"<ran_for>{_fmt_duration(elapsed_s)}</ran_for>")
    if result is not None:
        parts.append(f"<result>\n{result}\n</result>")
    if error is not None:
        parts.append(f"<error>\n{error}\n</error>")
    if log_path is not None:
        parts.append(f"<output_file>\n{log_path}\n</output_file>")
    parts.append("</task_notification>")

    return "\n".join(parts)


@dataclass
class PendingTask:
    """
    A backgrounded unit of work — kind-agnostic.

    The manager drives ``tool.run_stream`` in ``task`` and appends every event
    to ``events`` (the buffer). ``cursor`` tracks how far :meth:`drain` has
    consumed that buffer at the turn boundary — in one pass it re-emits each new
    event to the parent stream (live progress) and mirrors stream text to the
    on-disk progress log. ``task_key`` is set only for a *resumable* tool (its
    ``TaskRecord`` is persisted so a restart can re-spawn it); a backgrounded
    shell command is not resumable, so it has none.
    """

    task_id: str
    tool_name: str
    exec_id: str
    task: asyncio.Task[None]
    events: list[Event[Any]] = field(default_factory=list[Event[Any]])
    # Monotonic per-agent launch sequence number (mirrored on the durable
    # ``TaskRecord``). Watermarks record its high-water value; a transcript
    # rewind cancels tasks above the watermark — their launching calls are
    # no longer in the history the model sees.
    launch_seq: int = 0
    # Whether this task's result gates the final answer (it is part of the
    # answer). Read from ``tool.blocks_final_answer`` at launch — independent of
    # how the task was backgrounded (immediate vs deadline).
    blocks_final_answer: bool = True
    # Cap on inlined result chars in the completion note (``tool``'s setting); a
    # larger result is excerpted, with the note pointing at the on-disk log.
    max_inline_result_chars: int | None = None
    tool_call_id: str | None = None
    task_key: str | None = None
    cursor: int = 0  # events consumed by drain (bubbled to parent + flushed to log)
    announced: bool = False  # completion note already emitted
    started_at: float = field(default_factory=time.monotonic)  # for live elapsed
    log_path: str | None = None  # resolved .grasp/tasks log file, once written
    log_bytes: int = 0  # bytes appended to the log so far (for the size cap)


async def _consume(
    stream: AsyncIterator[Event[Any]],
    events: list[Event[Any]],
    *,
    store: CheckpointStore | None = None,
    task_key: str | None = None,
) -> None:
    """
    Drive a tool's ``run_stream``, buffering every event into ``events``.

    On finish, persist the outcome to the ``TaskRecord`` *before* the
    done-callback fires — ``COMPLETED`` (carrying the result) or ``FAILED``
    (carrying the error). This closes the window between a task finishing and
    :meth:`BackgroundTaskManager.drain` delivering it: a crash in that window
    leaves a terminal record that resume re-injects (see
    :meth:`resume_durable`), instead of a ``PENDING`` record that would force a
    re-run or lose the outcome.
    """
    result: Any = None
    failed = False
    try:
        async for event in stream:
            events.append(event)
            if isinstance(event, ToolOutputEvent):
                result = event.data
            elif isinstance(event, ToolErrorEvent):
                result = event.data
                failed = True
    except asyncio.CancelledError:
        # A genuine interruption (KillTask / shutdown / process death): leave the
        # record PENDING so a later resume can re-spawn or report the task.
        raise
    except Exception as exc:
        # The task errored (e.g. a sub-agent that could not resume). Record it as
        # a clean failure: don't leak a dangling task exception, and don't leave
        # a PENDING record that a later resume would re-spawn forever.
        logger.warning("Background task errored: %r", exc)
        err = ToolErrorEvent(data=ToolErrorInfo(tool_name="", error=f"{exc}"))
        events.append(err)
        result = err.data
        failed = True

    if store is not None and task_key is not None:
        await _persist_outcome(store, task_key, result, failed=failed)


async def _save_record_update(
    store: CheckpointStore, task_key: str, **updates: Any
) -> None:
    """
    Load → apply ``updates`` (stamping ``updated_at``) → save a ``TaskRecord``.
    A no-op when no record exists at ``task_key``.
    """
    existing = await store.load(task_key)
    if not existing:
        return
    record = TaskRecord.model_validate_json(existing)
    record = record.model_copy(update={**updates, "updated_at": datetime.now(UTC)})
    await store.save(task_key, record.model_dump_json().encode())


async def _persist_outcome(
    store: CheckpointStore, task_key: str, result: Any, *, failed: bool
) -> None:
    """Flip an existing ``TaskRecord`` to its terminal outcome (no-op if absent)."""
    if failed:
        await _save_record_update(
            store, task_key, status=TaskStatus.FAILED, error=_serialize_result(result)
        )
    else:
        await _save_record_update(
            store,
            task_key,
            status=TaskStatus.COMPLETED,
            result=_serialize_result(result),
        )


def _result_of(events: list[Event[Any]]) -> tuple[Any, bool]:
    """The task's terminal result and whether it failed (last writer wins)."""
    result: Any = None
    failed = False
    for event in events:
        if isinstance(event, ToolErrorEvent):
            result = event.data
            failed = True
        elif isinstance(event, ToolOutputEvent):
            result = event.data
    return result, failed


def _stream_text(events: list[Event[Any]]) -> str:
    """Concatenated incremental output text across ``events`` (rendered via str)."""
    return "".join(str(e.data) for e in events if isinstance(e, ToolStreamEvent))


def _serialize_result(result: Any) -> str:
    """
    Serialize a task result the way the foreground transcript does
    (``FunctionToolOutputItem.from_tool_result``): a string passes through
    verbatim (no spurious quotes), anything else is JSON via ``to_jsonable_python``
    so a pydantic result renders as data, not a Python ``repr`` (which would leak
    ``<Enum.X: 'x'>``-style values into the model's completion note).
    """
    if isinstance(result, str):
        return result
    return json.dumps(to_jsonable_python(result), indent=2)


# Cap on inlined chars when a completion note is REBUILT from a durable record
# (resume re-injection / rollback re-delivery). The tool's own
# ``max_inline_result_chars`` is not persisted, so a conservative default
# keeps a large stored result from being inlined whole into the transcript.
_REINJECT_INLINE_CAP = 8000


def _record_note(record: TaskRecord, *, failed: bool) -> InputMessageItem:
    """A task's completion note rebuilt from its durable record."""
    raw = record.error if failed else record.result
    body: str | None = None
    if raw is not None:
        body, _ = excerpt_for_inline(
            raw, _REINJECT_INLINE_CAP, output_file=record.output_path
        )
    return InputMessageItem.from_text(
        _task_notification(
            task_id=record.task_id,
            tool_name=record.tool_name,
            status="failed" if failed else "completed",
            result=None if failed else body,
            error=body if failed else None,
            log_path=record.output_path,
        ),
        role="user",
    )


class BackgroundTaskManager[CtxT]:
    """
    Tracks background work of any kind, generic over the tool that produces it.

    One mechanism: run a tool's ``run_stream`` as a task and buffer the events
    it yields, keyed by ``task_id``. :meth:`run_backgroundable` owns both
    backgrounding modes — ``auto_background_at == 0`` launches a task now (e.g. a
    sub-agent), a positive value runs the call in the foreground and *sidelines*
    it as a background task only if it outlives the deadline (e.g. a long shell
    command). When a task backgrounds is the only difference; whether it gates
    the final answer and how much of its result is inlined on completion are the
    tool's own ``blocks_final_answer`` / ``max_inline_result_chars``, read the
    same way for both. Every task's events are bubbled to the parent stream as
    live progress;
    its streamed output is also mirrored to an agent-readable ``.grasp`` log
    (``Read`` / ``Grep`` it to inspect a running task), and :meth:`kill_task`
    stops one by id. One drain / idle-wait / cancel path serves every task;
    every backgrounded task gets a persisted ``TaskRecord`` so a restart can
    surface it — resumable ones re-spawned on resume, others reported to the
    agent as interrupted.
    """

    def __init__(
        self,
        *,
        agent_name: str,
        transcript: LLMAgentTranscript,
        tools: dict[str, BaseTool[BaseModel, Any, CtxT]] | None,
        path: list[str] | None = None,
        max_background: int = 16,
        max_task_log_bytes: int = MAX_TASK_LOG_BYTES,
    ) -> None:
        self._agent_name = agent_name
        self._transcript = transcript
        self._tools = tools
        self._tasks: dict[str, PendingTask] = {}
        self._bg_counter = 0
        self._max_background = max_background
        self._max_task_log_bytes = max_task_log_bytes

        # Completed task ids, pushed by each task's done-callback (single
        # completion seam): :meth:`drain` pops them to deliver notes, and
        # :meth:`wait_idle` blocks on the next one — one queue, every task kind.
        self._completions: asyncio.Queue[str] = asyncio.Queue()

        # Deferred record updates (task_key → field update), applied by
        # :meth:`flush_delivered` once a checkpoint has persisted the
        # transcript that carries the corresponding notification.
        self._pending_delivered: dict[str, dict[str, Any]] = {}

        # Deferred CANCELLED flips for tasks killed by
        # :meth:`cancel_launched_after`, also applied by :meth:`flush_delivered`.
        # Kept apart from ``_pending_delivered`` because a watermark restore
        # wholesale-replaces that map — a kill must survive it (the killed
        # launch is gone from the transcript no matter which boundary is live).
        self._pending_killed: dict[str, dict[str, Any]] = {}

        self.path = path

    @property
    def has_live_tasks(self) -> bool:
        """
        True while any background task is still running.

        Tasks are session-scoped — they survive run boundaries and are
        released only by :meth:`cancel_all` (via ``LLMAgent.aclose``) or
        ``KillTask``.
        """
        return any(not pt.task.done() for pt in self._tasks.values())

    @property
    def has_pending(self) -> bool:
        """
        True while any *answer-blocking* background task is undelivered.

        Consulted by the JUDGE phase: such a task's result is part of the
        answer, so it gates even a final answer. Non-blocking tasks (e.g. a
        backgrounded shell command) are excluded — they must never hold the run
        hostage. A task stops blocking once its completion note is delivered
        (``announced``).
        """
        return any(
            pt.blocks_final_answer and not pt.announced for pt in self._tasks.values()
        )

    @property
    def has_undelivered_completions(self) -> bool:
        """
        True when a task has finished but :meth:`drain` has not yet delivered its
        note — a completion is waiting to be surfaced at the next turn boundary.

        A level-triggered readiness signal: a caller that lets the agent idle between
        runs can read it to know a finished task's result is pending delivery, and run
        another turn so :meth:`drain` surfaces it.
        """
        return not self._completions.empty()

    async def wait_idle(self, timeout: float | None = None) -> None:  # noqa: ASYNC109
        """
        Block until the next tracked task completes — the loop's idle wait.

        Returns immediately if a completion is already queued, or if nothing is
        pending (so the loop never blocks with no work outstanding). The awaited
        id is requeued so :meth:`drain` still delivers it. ``timeout`` bounds the
        wait (the caller passes the remaining run-deadline budget) so an idle
        wait on a task that never completes cannot sail past ``run_timeout`` — on
        expiry it returns and the loop's next deadline check stops the run.
        """
        if not self._completions.empty():
            return
        if all(pt.announced for pt in self._tasks.values()):
            return
        try:
            task_id = await asyncio.wait_for(self._completions.get(), timeout)
        except TimeoutError:
            return
        self._completions.put_nowait(task_id)

    # --- Launch ---

    async def run_backgroundable(
        self,
        call: FunctionToolCallItem,
        tool: BaseTool[BaseModel, Any, CtxT],
        inp: BaseModel,
        *,
        ctx: SessionContext[CtxT],
        exec_id: str,
        agent_ctx: "AgentContext | None" = None,
    ) -> tuple[Any, BackgroundTaskLaunchedEvent | None]:
        """
        Run a tool call, backgrounding it per ``tool.auto_background_at`` (assumed
        set — the loop only routes a backgroundable tool here).

        The two modes differ only in *when* the hand-off happens:
        ``auto_background_at == 0`` backgrounds immediately (e.g. a worker
        sub-agent); a positive value runs the call in the foreground and
        sidelines it only if it outlives the deadline (e.g. a long shell
        command). Everything after the decision — the pending ``TaskRecord``, the
        ``.grasp`` log, the launch note, the launched event — is shared; whether
        a backgrounded call gates the final answer and how much of its result is
        inlined are the tool's own ``blocks_final_answer`` /
        ``max_inline_result_chars``, read the same way for both.

        Returns ``(output, launched)``: a call that finishes in the foreground
        yields its result and ``None``; a backgrounded one yields a launch note
        (its result is delivered on completion, or it is stopped early with
        ``KillTask``) and the :class:`BackgroundTaskLaunchedEvent` for the caller
        to bubble. The manager owns the whole decision — the tool knows nothing
        about backgrounding.

        ``agent_ctx`` is the parent loop's agent-scope state, forwarded to the
        backgrounded ``run_stream`` so a sub-agent-as-tool still resolves its
        parent's transcript / sibling tools off the call.
        """
        abg = tool.auto_background_at
        assert abg is not None

        task_key = self._task_store_key(ctx, call.call_id)
        child_path = make_tool_call_path(self.path, call.call_id)
        events: list[Event[Any]] = []

        if abg == 0:
            try:
                self._check_capacity()
            except RuntimeError as exc:
                return f"Tool '{tool.name}' could not be backgrounded: {exc}", None
            task_id, launch_seq = self._next_id()
            log_path = await self._resolve_log(call.call_id, tool, ctx=ctx)
            # Persist the PENDING record *before* starting the task so a
            # near-instant finish still updates an existing record, and wire
            # outcome persistence into ``_consume`` (resumable tools rely on it
            # for a result that lands between drains).
            if task_key is not None:
                await self._write_pending_record(
                    task_key,
                    call,
                    task_id,
                    launch_seq=launch_seq,
                    ctx=ctx,
                    output_path=log_path,
                )
            stream = tool.run_stream(
                inp, ctx=ctx, exec_id=exec_id, path=child_path, agent_ctx=agent_ctx
            )
            started_at = time.monotonic()
            task = asyncio.create_task(
                _consume(stream, events, store=ctx.checkpoint_store, task_key=task_key)
            )
            self._register(
                task_id=task_id,
                launch_seq=launch_seq,
                task=task,
                events=events,
                tool=tool,
                exec_id=exec_id,
                tool_call_id=call.call_id,
                task_key=task_key,
                started_at=started_at,
                log_path=log_path,
            )
            note = self._launch_note(
                tool, task_id, backgrounded_after=None, log_path=log_path
            )
            return note, self._make_launched_event(
                task_id, tool.name, call.call_id, log_path=log_path, exec_id=exec_id
            )

        # abg > 0: run in the foreground, racing the deadline. ``started_at`` is
        # stamped at launch (not at sideline), so a backgrounded task's reported
        # runtime includes its foreground portion. ``_consume`` gets the store
        # key up front: if the call backgrounds, its eventual finish must flip
        # the pending record to COMPLETED/FAILED (a foreground finish writes
        # nothing — no record exists yet).
        stream = tool.run_stream(
            inp, ctx=ctx, exec_id=exec_id, path=child_path, agent_ctx=agent_ctx
        )
        started_at = time.monotonic()
        task = asyncio.create_task(
            _consume(stream, events, store=ctx.checkpoint_store, task_key=task_key)
        )
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=abg)
        except TimeoutError:
            try:
                self._check_capacity()
            except RuntimeError as exc:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                return f"Tool '{tool.name}' could not be backgrounded: {exc}", None
            task_id, launch_seq = self._next_id()
            # Resolve the log once, up front, so the single pending-record write
            # already carries ``output_path`` and the note can cite the file the
            # loop will have flushed output to by the model's next turn.
            log_path = await self._resolve_log(call.call_id, tool, ctx=ctx)
            self._register(
                task_id=task_id,
                launch_seq=launch_seq,
                task=task,
                events=events,
                tool=tool,
                exec_id=exec_id,
                tool_call_id=call.call_id,
                task_key=task_key,
                started_at=started_at,
                log_path=log_path,
            )
            if task_key is not None:
                await self._write_pending_record(
                    task_key,
                    call,
                    task_id,
                    launch_seq=launch_seq,
                    ctx=ctx,
                    output_path=log_path,
                )
                if task.done() and ctx.checkpoint_store is not None:
                    # The call finished while the pending record was being
                    # written — ``_consume``'s own outcome write found no
                    # record to update, so persist the outcome here.
                    result, failed = _result_of(events)
                    await _persist_outcome(
                        ctx.checkpoint_store, task_key, result, failed=failed
                    )
            note = self._launch_note(
                tool, task_id, backgrounded_after=abg, log_path=log_path
            )
            return note, self._make_launched_event(
                task_id, tool.name, call.call_id, log_path=log_path, exec_id=exec_id
            )
        except asyncio.CancelledError:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            raise

        result, _ = _result_of(events)

        return result, None

    def _launch_note(
        self,
        tool: BaseTool[BaseModel, Any, CtxT],
        task_id: str,
        *,
        backgrounded_after: float | None,
        log_path: str | None,
    ) -> str:
        """
        Build the launch note returned as the immediate tool result.

        ``backgrounded_after`` is the deadline a positive-``auto_background_at``
        call outlived (``None`` for an immediate background); it shapes only the
        opening clause. The rest — result-on-completion, the optional log pointer
        (present iff the tool has a ``.grasp`` log), and the ``KillTask`` hint —
        is identical, so the note encodes only *how* the call backgrounded, never
        the task's kind.
        """
        parts: list[str] = []

        if backgrounded_after is None:
            opening = f"Task '{tool.name}' launched in the background (id: {task_id})."
        else:
            opening = (
                f"Tool '{tool.name}' is still running after {backgrounded_after:g}s "
                f"and was moved to the background (id: {task_id})."
            )
        parts.append(opening)

        waiting_guidance = (
            "You will receive a <task_notification> when it finishes — "
            "until then, assume it is still running and do NOT call it again. "
            "You can continue with other work. "
            "If the only thing left to do is wait for background tasks, "
            "report immediately that you are waiting for the task(s) to finish "
            "WITHOUT calling any tools."
        )
        parts.append(waiting_guidance)

        if log_path is not None:
            output_guidance = (
                f"Its output is streaming to {log_path} — Read or Grep that file "
                "to check progress so far (it is partial until the task "
                "finishes, and a read's own exit code reflects the read, not "
                "the task)."
            )
            parts.append(output_guidance)

        kill_guidance = (
            "Stop it with KillTask if you no longer need it "
            "or the user wants to cancel it."
        )
        parts.append(kill_guidance)

        return "\n".join(parts)

    def _next_id(self) -> tuple[str, int]:
        """Mint a task id and its ``launch_seq`` (the id's numeric suffix)."""
        self._bg_counter += 1
        return f"bg_{self._bg_counter}", self._bg_counter

    @property
    def last_launch_seq(self) -> int:
        """High-water launch seq: every launch so far has ``launch_seq <=`` this."""
        return self._bg_counter

    def seed_launch_seq(self, seq: int) -> None:
        """
        Seed the launch counter from a restored watermark.

        Never lowers it: after a rewind the cancelled launches' seqs stay
        burned, so a re-run's fresh launches can't be mistaken for them.
        """
        self._bg_counter = max(self._bg_counter, seq)

    def _reserve_task_id(self, task_id: str) -> None:
        """
        Advance the id counter past an existing task's id.

        A re-spawned task keeps its original id (e.g. ``bg_2``); the resumed
        manager's counter, however, restarts at zero. Without this, a *new* tool
        call in the same resumed run would be assigned a colliding ``bg_N``,
        overwriting the re-spawned task in ``_tasks`` and crossing their result
        deliveries (the re-spawned task's real result is persisted to its record
        but never delivered; the new call's result is delivered under its id).
        """
        _, _, suffix = task_id.rpartition("_")
        if suffix.isdigit():
            self._bg_counter = max(self._bg_counter, int(suffix))

    def _check_capacity(self) -> None:
        """
        Raise if the background-task ceiling is reached.

        Applies to both backgrounding modes; the caller turns the failure into a
        launch note rather than letting it abort the turn.
        """
        if len(self._tasks) >= self._max_background:
            raise RuntimeError(
                f"Too many background tasks ({self._max_background}); kill or "
                "drain existing ones first."
            )

    def _register(
        self,
        *,
        task_id: str,
        launch_seq: int,
        task: asyncio.Task[None],
        events: list[Event[Any]],
        tool: BaseTool[BaseModel, Any, CtxT],
        exec_id: str,
        tool_call_id: str | None,
        task_key: str | None,
        started_at: float,
        log_path: str | None = None,
    ) -> None:
        """
        Track a backgrounded task and enqueue its id when it ends.

        Builds the ``PendingTask`` from the tool — ``blocks_final_answer`` /
        ``max_inline_result_chars`` come from it, so both backgrounding modes and
        a resume re-spawn are tracked identically — and wires the done-callback
        that feeds the single completion queue. ``started_at`` is the task's
        launch stamp (for live elapsed); ``log_path`` its resolved ``.grasp`` log
        (``None`` for a tool without one).
        """
        pt = PendingTask(
            task_id=task_id,
            launch_seq=launch_seq,
            tool_name=tool.name,
            exec_id=exec_id,
            tool_call_id=tool_call_id,
            task=task,
            events=events,
            blocks_final_answer=tool.blocks_final_answer,
            max_inline_result_chars=tool.max_inline_result_chars,
            task_key=task_key,
            started_at=started_at,
            log_path=log_path,
        )
        self._tasks[task_id] = pt
        task.add_done_callback(lambda _, tid=task_id: self._completions.put_nowait(tid))

    async def _resolve_log(
        self,
        name: str,
        tool: BaseTool[BaseModel, Any, CtxT],
        *,
        ctx: SessionContext[CtxT],
    ) -> str | None:
        """
        Resolve a backgrounded task's ``.grasp`` log path, or ``None``.

        Gated on ``tool.has_progress_log``: only a tool that mirrors incremental
        output to a log gets one resolved (eagerly, at background time, so the
        pending record and launch note can cite it). A tool whose events are
        structural — a sub-agent — has no log, independent of *how* it
        backgrounded.
        """
        if not tool.has_progress_log or ctx.file_backend is None:
            return None
        return await open_task_log(ctx.file_backend, name=name)

    def _make_launched_event(
        self,
        task_id: str,
        tool_name: str,
        call_id: str,
        *,
        log_path: str | None,
        exec_id: str,
    ) -> BackgroundTaskLaunchedEvent:
        """
        Build the launched event both backgrounding modes bubble.

        ``output_name`` is surfaced only for a task that has a ``.grasp`` log
        (``log_path`` set); a sub-agent without one advertises ``None``.
        """
        return BackgroundTaskLaunchedEvent(
            source=self._agent_name,
            exec_id=exec_id,
            data=BackgroundTaskInfo(
                task_id=task_id,
                tool_name=tool_name,
                tool_call_id=call_id,
                output_name=task_log_name(call_id) if log_path else None,
            ),
        )

    # --- TaskRecord persistence ---

    def _task_store_key(self, ctx: SessionContext[CtxT], call_id: str) -> str | None:
        """Store key for a backgrounded call's ``TaskRecord`` (``None`` if none)."""
        child_path = make_tool_call_path(self.path, call_id)
        if ctx.checkpoint_store is None or child_path is None:
            return None
        return make_store_key(ctx.session_key, CheckpointKind.TASK, child_path)

    async def _write_pending_record(
        self,
        task_key: str,
        call: FunctionToolCallItem,
        task_id: str,
        *,
        launch_seq: int,
        ctx: SessionContext[CtxT],
        output_path: str | None = None,
    ) -> None:
        """
        Persist a PENDING ``TaskRecord`` so a restart can surface this task.

        ``output_path`` is the task's ``.grasp`` log — resolved up front for a
        tool with ``has_progress_log`` so the record carries it in a single
        write; ``None`` for a tool with no log.
        """
        store = ctx.checkpoint_store
        if store is None:
            return
        record = TaskRecord(
            session_key=ctx.session_key,
            task_id=task_id,
            launch_seq=launch_seq,
            tool_call_id=call.call_id,
            tool_name=call.name,
            tool_call_arguments=call.arguments,
            status=TaskStatus.PENDING,
            output_path=output_path,
        )
        await store.save(task_key, record.model_dump_json().encode())

    async def _mark_record(
        self,
        task_key: str | None,
        *,
        ctx: SessionContext[CtxT] | None,
        **updates: Any,
    ) -> None:
        """Load → update → save a ``TaskRecord`` (no-op if no store / key / record)."""
        store = ctx.checkpoint_store if ctx else None
        if store is None or task_key is None:
            return
        await _save_record_update(store, task_key, **updates)

    async def _append_log(
        self, pt: PendingTask, text: str, backend: "FileBackend"
    ) -> None:
        """
        Mirror new stream ``text`` to the task's ``.grasp/tasks/<call_id>.log``
        so a crash leaves a recoverable, Grep-able trace and a running task can
        be inspected with ``Read`` / ``Grep``. Called from :meth:`drain` with the
        delta bubbled this turn, so only new output is appended (O(new output),
        not a rewrite). No-op for a task with no log (``has_progress_log`` false →
        no resolved ``log_path``) or once ``max_task_log_bytes`` is hit (a final
        marker is appended, then output is dropped). Best-effort — never raises.
        """
        if not text or pt.log_path is None or pt.log_bytes >= self._max_task_log_bytes:
            return
        chunk = text.encode()
        remaining = self._max_task_log_bytes - pt.log_bytes
        if len(chunk) > remaining:
            chunk = chunk[:remaining] + b"\n... [log truncated] ...\n"
            pt.log_bytes = self._max_task_log_bytes  # saturate → stop appending
        else:
            pt.log_bytes += len(chunk)
        await append_task_log(backend, pt.log_path, chunk)

    def get(self, task_id: str) -> PendingTask:
        pt = self._tasks.get(task_id)
        if pt is None:
            known = ", ".join(sorted(self._tasks)) or "none"
            raise ValueError(
                f"Unknown background task id {task_id!r} (known: {known})."
            )
        return pt

    def remove(self, task_id: str) -> None:
        self._tasks.pop(task_id, None)

    # --- Stop (KillTask) ---

    async def kill_task(
        self, task_id: str, *, ctx: SessionContext[CtxT] | None = None
    ) -> KillTaskResult:
        """Cancel a task (its stream closes → process group killed) and read it."""
        pt = self.get(task_id)
        # Capture whether it had already finished *before* we cancel: a task
        # still tracked here that is already done finished on its own, so the
        # outcome is genuine; otherwise we are the ones stopping it.
        already_done = pt.task.done()
        if not already_done:
            pt.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pt.task

        result, failed = _result_of(pt.events)
        # A head+tail excerpt of what it produced, so the model sees why it was
        # killed without bloating the transcript; the full output is in the log.
        output, _ = excerpt_for_inline(
            _stream_text(pt.events),
            pt.max_inline_result_chars or 8000,
            output_file=pt.log_path,
        )
        status: Literal["cancelled", "completed", "failed"]
        if failed:
            status = "failed"
        elif already_done:
            status = "completed"
        else:
            status = "cancelled"
        # Deliberately stopped → record is terminal, so a later resume does not
        # report the killed task as interrupted.
        await self._mark_record(
            pt.task_key,
            ctx=ctx,
            status=TaskStatus.CANCELLED,
            error="Stopped by KillTask",
        )
        self._tasks.pop(task_id, None)
        return KillTaskResult(
            task_id=task_id,
            tool_name=pt.tool_name,
            status=status,
            output=output,
            result=result,
        )

    def cancel_launched_after(self, seq: int) -> None:
        """
        Cancel every task with ``launch_seq > seq`` — synchronous and I/O-free.

        Called when the transcript is rewound past the tool calls that
        launched these tasks (a failed-run settle or a step rollback): their
        launches are no longer in the history the model sees, so letting them
        finish would inject completion notes for calls that never happened.
        Cancelled tasks are dropped from tracking immediately, so a completion
        already queued for :meth:`drain` is suppressed rather than announced.
        Tasks at or below ``seq`` keep running — their launches are still in
        the kept transcript, and their notes deliver normally later.

        Durable record flips (→ CANCELLED) are deferred to
        :meth:`flush_delivered`, after the pruned transcript is durable; a
        crash before that is covered by the resume-side orphan guard in
        :meth:`resume_durable`.
        """
        for task_id, pt in list(self._tasks.items()):
            if pt.launch_seq <= seq:
                continue
            if not pt.task.done():
                pt.task.cancel()
            del self._tasks[task_id]
            if pt.task_key is not None:
                self._pending_delivered.pop(pt.task_key, None)
                self._pending_killed[pt.task_key] = {
                    "status": TaskStatus.CANCELLED,
                    "error": "Cancelled: the launching tool call was rolled back",
                }
            logger.info(
                "Cancelled background task %s (%s): launched after the rewind point",
                task_id,
                pt.tool_name,
            )

    # --- Turn-boundary drain ---

    async def drain(
        self, *, exec_id: str, ctx: SessionContext[CtxT]
    ) -> AsyncIterator[Event[Any]]:
        """
        The turn-boundary pass over every tracked task: in one sweep re-emit
        each task's new buffered events as live progress *and* mirror their
        stream text to the ``.grasp`` log, then deliver one completion
        notification per finished task.

        Bubbling and log-mirroring share a single ``cursor`` — the log is
        written from the very events being bubbled — so there is no separate
        flush step. The mirroring stays here (consumer-side, batched one append
        per task per turn) rather than in the producing task, so it is cheap
        even on a remote backend where a per-event write would be a round-trip.

        Delivery is uniform regardless of how the task was backgrounded
        (immediate vs deadline): the note inlines the task's result (or error)
        and the task is dropped. A result larger than the tool's
        ``max_inline_result_chars`` is excerpted in the note with a pointer to
        its ``.grasp`` log, which holds the full streamed output for ``Read`` /
        ``Grep``. Either way the task is ``announced`` once, so it stops gating
        the final answer.
        """
        # Bubble each task's new events to the parent stream (pure
        # observability — live progress for a backgrounded shell command just as
        # for a sub-agent; ``blocks_final_answer`` governs only the JUDGE gate)
        # and mirror this pass's stream text to its ``.grasp`` log in one append.
        backend = ctx.file_backend
        for pt in list(self._tasks.values()):
            start = pt.cursor
            while pt.cursor < len(pt.events):
                event = pt.events[pt.cursor]
                pt.cursor += 1
                yield event
            if backend is not None and pt.cursor > start:
                await self._append_log(
                    pt, _stream_text(pt.events[start : pt.cursor]), backend
                )

        while not self._completions.empty():
            task_id = self._completions.get_nowait()
            pt = self._tasks.get(task_id)
            if pt is None or pt.announced:
                continue

            result, failed = _result_of(pt.events)
            status = "failed" if failed else "completed"
            full = _serialize_result(result)
            cap = pt.max_inline_result_chars

            result_file: str | None = None
            if cap is not None and len(full) > cap and ctx.file_backend is not None:
                result_file = await write_result_file(
                    ctx.file_backend, name=pt.tool_call_id or pt.task_id, text=full
                )

            body, _ = excerpt_for_inline(
                full, cap, output_file=result_file or pt.log_path
            )
            note_result = None if failed else body
            note_error = body if failed else None

            notification = InputMessageItem.from_text(
                _task_notification(
                    task_id=pt.task_id,
                    tool_name=pt.tool_name,
                    status=status,
                    result=note_result,
                    error=note_error,
                    log_path=pt.log_path,
                    elapsed_s=time.monotonic() - pt.started_at,
                ),
                role="user",
            )
            self._transcript.update([notification])

            # Durable record → DELIVERED (resumable tasks only). Deferred to
            # :meth:`flush_delivered` after the next checkpoint persists the
            # transcript holding this note: flipping now would lose the
            # outcome on a crash before that checkpoint (resume skips
            # DELIVERED records). The note's transcript position rides along —
            # a step rollback that truncates below it re-injects the note
            # (:meth:`undeliver_after`). Records are kept for post-hoc
            # observability; reclaim with ``prune_delivered``.
            if ctx.checkpoint_store is not None and pt.task_key is not None:
                self._pending_delivered[pt.task_key] = {
                    "status": TaskStatus.DELIVERED,
                    "result": None if failed else _serialize_result(result),
                    "error": _serialize_result(result) if failed else None,
                    "delivered_msg_count": len(self._transcript.messages),
                }

            yield BackgroundTaskCompletedEvent(
                source=self._agent_name,
                exec_id=exec_id,
                data=BackgroundTaskInfo(
                    task_id=pt.task_id,
                    tool_name=pt.tool_name,
                    tool_call_id=pt.tool_call_id or "",
                ),
            )
            yield UserMessageEvent(
                source=pt.tool_name,
                destination=self._agent_name,
                exec_id=exec_id,
                data=notification,
            )

            pt.announced = True  # delivered → no longer gates the final answer
            # The full output lives in the .grasp log; nothing is retained for a
            # poll, so always drop the finished task.
            self._tasks.pop(task_id, None)

        # Front-trim each surviving task's buffer: leading events consumed by
        # BOTH the bubble (parent stream) and flush (.grasp log) cursors are
        # durable elsewhere, so drop them to bound memory for a long, chatty
        # backgrounded command. Stops before any terminal result event so
        # ``_result_of`` / ``kill_task`` still find it.
        for pt in self._tasks.values():
            self._trim_consumed(pt)

    @staticmethod
    def _trim_consumed(pt: PendingTask) -> None:
        """Drop drained (bubbled + flushed) leading events; keep results."""
        keep_from = pt.cursor
        for i in range(keep_from):
            if isinstance(pt.events[i], ToolErrorEvent | ToolOutputEvent):
                keep_from = i  # never drop a terminal result event
                break
        if keep_from <= 0:
            return
        del pt.events[:keep_from]
        pt.cursor -= keep_from

    # --- Cancel ---

    async def cancel_all(self, ctx: SessionContext[CtxT] | None = None) -> None:
        """
        Cancel all background tasks and wait for cleanup.

        Session teardown — reached via ``LLMAgent.aclose()``, never at run
        end (tasks are session-scoped and survive run boundaries).
        """
        store = ctx.checkpoint_store if ctx else None

        if store is not None and ctx is not None:
            for pt in self._tasks.values():
                if pt.task_key is None:
                    continue
                await _save_record_update(
                    store,
                    pt.task_key,
                    status=TaskStatus.CANCELLED,
                    error="Cancelled: agent session closed",
                )

        tasks = [pt.task for pt in self._tasks.values()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        # Cancelled tasks' done-callbacks enqueue their ids; drop them so a
        # reused manager starts clean.
        while not self._completions.empty():
            self._completions.get_nowait()

    # --- Session resume ---

    async def resume_durable(
        self,
        *,
        ctx: SessionContext[CtxT] | None = None,
        exec_id: str | None = None,
        agent_ctx: "AgentContext | None" = None,
        bg_launch_seq: int | None = None,
    ) -> list[InputMessageItem]:
        """
        On resume, re-spawn or notify about interrupted background tasks.

        Resumable tasks (e.g. sub-agents) are re-spawned silently; the rest are
        reported to the agent — interrupted ones it may want to redo, completed
        ones whose result never reached it are re-injected. Any such notice is
        prefixed with a one-line framing so the agent understands it was resumed.

        ``bg_launch_seq`` is the restored head's launch high-water: a record
        above it was launched by a tool call that never made it into the
        restored transcript (a crash before its checkpoint, or a rewind whose
        deferred CANCELLED flip was lost with the process), so it is
        dead-lettered — flipped CANCELLED, never re-spawned or reported.
        ``None`` disables the guard.

        Returns the notification messages injected into the transcript so the
        caller can stream them as events (no transcript message stays hidden).
        """
        store = ctx.checkpoint_store if ctx else None
        if store is None or ctx is None:
            return []

        # Scope to *this* manager's own background tasks. Records live at
        # ``<session>/task/<path>/tc_<call_id>``; sibling agents and nested
        # sub-agents own separate subtrees and resume via their own managers,
        # so a session-wide scan would make us handle (and mis-route) their
        # tasks. Keep only direct ``tc_*`` children — a deeper segment belongs
        # to a nested sub-agent.
        prefix = make_store_key(ctx.session_key, CheckpointKind.TASK, self.path) + "/"
        keys = [k for k in await store.list_keys(prefix) if is_direct_child(k, prefix)]

        notifications: list[InputMessageItem] = []
        deferred_keys: list[str] = []
        for key in keys:
            record = await store.load_json(key, TaskRecord, subject="task record")
            if record is None:
                continue

            # Reserve every prior id so a new tool call in this resumed run
            # can't be handed a ``bg_N`` that collides with a re-spawned task.
            self._reserve_task_id(record.task_id)

            # Terminal + already surfaced — nothing to do. (FAILED is NOT here:
            # a FAILED record is an errored task that the crash kept ``drain``
            # from delivering, so it still needs re-injecting below.)
            if record.status in {TaskStatus.CANCELLED, TaskStatus.DELIVERED}:
                continue

            # A deferred flip restored with the head means the head's
            # transcript already holds this record's note (the crash landed
            # between that save and its flush) — don't inject a duplicate;
            # the restored flip lands at the next save's flush.
            if key in self._pending_delivered:
                continue

            # A deferred kill restored with the head: the task was cancelled
            # by a rewind whose CANCELLED flip a crash kept from flushing.
            # Never re-spawn or report it; the flip lands at the next flush.
            if key in self._pending_killed:
                continue

            # Orphan guard: launched after the restored head's boundary, so its
            # launching tool call is not in the restored transcript — the agent
            # has no memory of it. Dead-letter instead of re-spawning/reporting.
            if bg_launch_seq is not None and record.launch_seq > bg_launch_seq:
                await _save_record_update(
                    store,
                    key,
                    status=TaskStatus.CANCELLED,
                    error="Cancelled: the launching tool call was never persisted",
                )
                logger.info(
                    "Dead-lettered orphan task record %s (%s): launch_seq %d > "
                    "restored head's %d",
                    record.task_id,
                    record.tool_name,
                    record.launch_seq,
                    bg_launch_seq,
                )
                continue

            if record.status == TaskStatus.PENDING:
                # A resumable task (sub-agent / workflow) is re-spawned silently
                # — it continues from its own checkpoint and reports back via its
                # own bubbled events on completion, so no resume notice is needed.
                if self._try_respawn_child(
                    record, task_key=key, ctx=ctx, exec_id=exec_id, agent_ctx=agent_ctx
                ):
                    continue

                elapsed = (record.updated_at - record.created_at).total_seconds()
                notification = InputMessageItem.from_text(
                    _task_notification(
                        task_id=record.task_id,
                        tool_name=record.tool_name,
                        status="interrupted",
                        error=(
                            "Task was stopped. "
                            "Any partial output it produced before stopping is in the "
                            "log below. Check the output and attempt to continue from "
                            "where it left off whenever possible, or re-run from "
                            "scratch if not."
                        ),
                        log_path=record.output_path,
                        elapsed_s=elapsed,
                    ),
                    role="user",
                )
                # The interrupted notice was delivered → terminal (DELIVERED),
                # not FAILED (which is an errored task, re-injected below).
                update = {
                    "status": TaskStatus.DELIVERED,
                    "error": "Interrupted: session restarted",
                }
            elif record.status in {TaskStatus.COMPLETED, TaskStatus.FAILED}:
                # Finished (ok or errored) but the crash kept ``drain`` from
                # delivering it — re-inject the outcome, then mark it delivered.
                notification = _record_note(
                    record, failed=record.status == TaskStatus.FAILED
                )
                update = {"status": TaskStatus.DELIVERED}
            else:
                continue

            # Deferred to ``flush_delivered`` (after the checkpoint that
            # persists the notice) — same loss-window reasoning as ``drain``.
            self._pending_delivered[key] = update
            deferred_keys.append(key)
            notifications.append(notification)

        injected: list[InputMessageItem] = []
        if notifications:
            # One framing line so the agent understands the per-task notices
            # that follow: it was resumed, in-memory state was reconstructed
            # (not continued), and interrupted tasks may need redoing.
            # (Silently re-spawned resumable tasks add no notice and so don't
            # trigger a framing on their own.)
            framing = InputMessageItem.from_text(
                "<session_resumed>\n"
                "Resumed from a checkpoint: your conversation and task records "
                "were restored from disk, but any in-flight work that was only "
                "in memory is gone. The background tasks below were running "
                "when the previous run stopped."
                "\n</session_resumed>",
                role="user",
            )
            injected = [framing, *notifications]
            self._transcript.update(injected)
            # Stamp each re-injected note's transcript position on its
            # deferred flip (mirrors ``drain``): a later rollback truncating
            # below the note must know to re-inject it. The notices sit at
            # the tail of ``injected``, after the framing line.
            total = len(self._transcript.messages)
            for offset, key in enumerate(deferred_keys):
                self._pending_delivered[key]["delivered_msg_count"] = (
                    total - len(deferred_keys) + offset + 1
                )

        logger.info(
            "Handled %d task records for session %s", len(keys), ctx.session_key
        )
        return injected

    def export_pending_delivered(self) -> dict[str, dict[str, Any]]:
        """The deferred record updates not yet flushed (rollback snapshot)."""
        return dict(self._pending_delivered)

    def restore_pending_delivered(self, pending: dict[str, dict[str, Any]]) -> None:
        """
        Restore a :meth:`export_pending_delivered` snapshot: on a rollback the
        deferred flips are discarded, so records whose notes were rolled back
        stay COMPLETED for a later resume to re-inject. A failed-run settle
        instead re-merges the flips for the notes it kept (see
        ``LLMAgent._settle_run``).
        """
        self._pending_delivered = dict(pending)

    def export_pending_killed(self) -> dict[str, dict[str, Any]]:
        """The deferred CANCELLED flips not yet flushed."""
        return dict(self._pending_killed)

    def restore_pending_killed(self, killed: dict[str, dict[str, Any]]) -> None:
        """
        Merge a snapshot's deferred CANCELLED flips UNDER the live ones: a
        kill is never discarded by a watermark restore (the killed launch is
        gone from the transcript no matter which boundary is live), and a
        cold resume re-arms kills whose flush a crash pre-empted — otherwise
        the restored head's ``bg_launch_seq`` (raised past the killed
        launches before the flush) would defeat the resume orphan guard and
        resurrect them.
        """
        self._pending_killed = {**killed, **self._pending_killed}

    async def flush_delivered(self, *, ctx: SessionContext[CtxT]) -> None:
        """
        Apply deferred terminal record updates (→ ``DELIVERED`` / ``CANCELLED``).

        Called after a checkpoint has persisted the transcript containing the
        corresponding completion/interruption notes. Flipping the records any
        earlier opens a loss window: a crash between the flip and the next
        checkpoint leaves a DELIVERED record whose note never became durable,
        so resume would never re-inject the outcome. (A crash between the
        checkpoint and this flush yields at worst a re-injected note — a
        duplicate beats a lost outcome.) The same holds for a
        :meth:`cancel_launched_after` kill: its CANCELLED flip waits until the
        transcript no longer referencing the launch is durable, so a crash in
        between resumes with the launch still on record.
        """
        store = ctx.checkpoint_store
        if store is None or not (self._pending_delivered or self._pending_killed):
            return
        # A kill wins over a delivery flip for the same record: the kill came
        # later, when the delivered note was rolled back. Each entry is
        # dropped only once applied, so a store error mid-flush keeps the
        # rest deferred for the next flush instead of losing them (a re-apply
        # is idempotent — updates are absolute field sets).
        pending = {**self._pending_delivered, **self._pending_killed}
        for key, update in pending.items():
            await _save_record_update(store, key, **update)
            self._pending_delivered.pop(key, None)
            self._pending_killed.pop(key, None)

    async def undeliver_after(
        self,
        *,
        message_count: int,
        bg_launch_seq: int,
        ctx: SessionContext[CtxT],
        deferred_delivered: Mapping[str, dict[str, Any]] | None = None,
    ) -> list[InputMessageItem]:
        """
        The bg-task half of a step rollback, called after the transcript is
        truncated to ``message_count``: a delivered record whose completion
        note sat past the cut (``delivered_msg_count > message_count``) while
        its launching call survived (``launch_seq <= bg_launch_seq``) just
        lost the only live copy of its outcome — the finished task is no
        longer tracked in memory, and resume skips DELIVERED records, so
        nothing else would ever re-surface it and the agent would wait on it
        forever. Re-inject each such note at the rewind point and re-defer its
        DELIVERED flip with the new position.

        ``deferred_delivered`` overlays not-yet-flushed DELIVERED flips onto
        the stored records: a drained-but-unflushed note (its record still
        COMPLETED) sits in the truncated span exactly like a flushed one's.
        The caller passes the map exported *before* the rollback's context
        restore replaced it. Records launched *after* the boundary are
        ``cancel_launched_after``'s business — their launching calls are
        gone, so their outcomes stay buried. Returns the injected messages
        for the caller to surface as events.
        """
        store = ctx.checkpoint_store
        if store is None:
            return []
        deferred = (
            dict(self._pending_delivered)
            if deferred_delivered is None
            else deferred_delivered
        )
        prefix = make_store_key(ctx.session_key, CheckpointKind.TASK, self.path) + "/"
        keys = [k for k in await store.list_keys(prefix) if is_direct_child(k, prefix)]

        matches: list[tuple[str, TaskRecord, int]] = []
        for key in keys:
            record = await store.load_json(key, TaskRecord, subject="task record")
            if record is None or record.launch_seq > bg_launch_seq:
                continue
            pending = deferred.get(key, {})
            is_delivered = (
                record.status is TaskStatus.DELIVERED
                or pending.get("status") is TaskStatus.DELIVERED
            )
            count = pending.get("delivered_msg_count", record.delivered_msg_count)
            if not is_delivered or count is None or count <= message_count:
                continue
            matches.append((key, record, count))
        # Original observation order: the positions the notes held before the
        # truncation (completion order, which can differ from launch order).
        matches.sort(key=operator.itemgetter(2))

        injected: list[InputMessageItem] = []
        for key, record, _count in matches:
            is_fail = record.result is None and record.error is not None
            notification = _record_note(record, failed=is_fail)
            self._transcript.update([notification])
            self._pending_delivered[key] = {
                "status": TaskStatus.DELIVERED,
                "delivered_msg_count": len(self._transcript.messages),
            }
            injected.append(notification)
            logger.info(
                "Re-injected background task %s (%s): its completion note was "
                "truncated by a rollback",
                record.task_id,
                record.tool_name,
            )
        return injected

    # --- Offline cleanup ---

    @staticmethod
    async def prune_delivered(
        ctx: SessionContext[Any],
        *,
        older_than: timedelta,
    ) -> int:
        """
        Delete ``DELIVERED`` task records older than ``older_than``.

        :meth:`drain` marks a task ``DELIVERED`` and keeps the record for
        post-hoc observability; this offline sweep reclaims the old ones.
        Returns the number pruned. Short-circuits with ``0`` when no store
        is attached.

        Unlike :meth:`resume_durable` (agent-scoped — it re-spawns tasks and
        injects into a specific agent's transcript), this is a static,
        session-wide GC: deleting a terminal ``DELIVERED`` record is safe
        regardless of which agent owns it, so one sweep cleans the session.
        """
        store = ctx.checkpoint_store
        if store is None:
            return 0

        cutoff = datetime.now(UTC) - older_than
        pruned = 0
        keys = await store.list_keys(task_prefix(ctx.session_key))

        for key in keys:
            record = await store.load_json(key, TaskRecord, subject="task record")
            if record is None:
                continue
            if record.status == TaskStatus.DELIVERED and record.updated_at < cutoff:
                await store.delete(key)
                pruned += 1
        return pruned

    def _try_respawn_child(
        self,
        record: TaskRecord,
        *,
        task_key: str,
        ctx: SessionContext[CtxT] | None,
        exec_id: str | None,
        agent_ctx: "AgentContext | None" = None,
    ) -> bool:
        """Re-spawn a child task from its session checkpoint."""
        if not ctx or not exec_id or not ctx.checkpoint_store:
            return False

        tool = self._tools.get(record.tool_name) if self._tools else None
        if not tool or not tool.resumable:
            return False

        child_path = make_tool_call_path(self.path, record.tool_call_id)

        events: list[Event[Any]] = []
        stream = tool.resume_stream(
            ctx=ctx,
            exec_id=exec_id,
            path=child_path,
            agent_ctx=agent_ctx,
            tool_call_arguments=record.tool_call_arguments,
        )
        async_task = asyncio.create_task(
            _consume(stream, events, store=ctx.checkpoint_store, task_key=task_key)
        )

        self._register(
            task_id=record.task_id,
            launch_seq=record.launch_seq,
            task=async_task,
            events=events,
            tool=tool,
            exec_id=exec_id,
            tool_call_id=record.tool_call_id,
            task_key=task_key,
            started_at=time.monotonic(),
            log_path=record.output_path,
        )

        logger.info(
            "Re-spawned child task %s (%s) at task key %s",
            record.task_id,
            record.tool_name,
            task_key,
        )
        return True
