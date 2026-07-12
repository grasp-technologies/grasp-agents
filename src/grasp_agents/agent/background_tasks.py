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

from pydantic import BaseModel, TypeAdapter
from pydantic_core import to_jsonable_python

from grasp_agents.context.prompt_builder import SystemPromptSection
from grasp_agents.context.system_reminder import (
    SESSION_RESUMED_SUBJECT,
    wrap_in_system_reminder,
)
from grasp_agents.durability.checkpoint_store import CheckpointStore
from grasp_agents.durability.checkpoints import CheckpointKind
from grasp_agents.durability.store_keys import (
    is_direct_child,
    make_store_key,
    make_tool_call_path,
    task_prefix,
)
from grasp_agents.durability.task_record import (
    TaskCancelledFlip,
    TaskDeliveredFlip,
    TaskRecord,
    TaskStatus,
)
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
from grasp_agents.utils.errors import format_error_chain

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


# --- Task notifications ---


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


BACKGROUND_TASKS_SECTION_NAME = "background_tasks"


def make_background_tasks_section(
    *, section_name: str = BACKGROUND_TASKS_SECTION_NAME
) -> SystemPromptSection:
    """
    Build the background-task guidance section (waiting behavior, progress
    logs, KillTask), auto-attached by ``LLMAgent``.

    Renders only when the toolset has a backgroundable tool
    (``auto_background_at`` set). The tools are read live off the call's
    ``agent_ctx`` at prompt-build time, so tools wired after construction
    (e.g. by a team onto its residents) are picked up at the next run entry.
    The KillTask line renders only when that tool is attached.
    """

    def compute(
        *,
        ctx: SessionContext[Any] | None = None,
        exec_id: str | None = None,
        agent_ctx: "AgentContext | None" = None,
        **_: Any,
    ) -> str | None:
        del ctx, exec_id
        if agent_ctx is None:
            return None
        tools = agent_ctx.tools
        if all(t.auto_background_at is None for t in tools.values()):
            return None

        lines = [
            "<background_tasks>",
            (
                "Some of your tools run as background tasks: a backgrounded "
                "call immediately returns a <task_notification> with status "
                '"running" and a task_id, while the work continues.'
            ),
            (
                "- You will receive another <task_notification> when a task "
                "finishes — until then, assume it is still running and do NOT "
                "repeat the call. You can continue with other work."
            ),
            (
                "- If the only thing left to do is wait for background tasks, "
                "report that you are waiting for them to finish WITHOUT "
                "calling any tools."
            ),
            (
                "- A task's <log_file> tag, when present (not every task has "
                "a log), names the file its streamed output is mirrored to — "
                "Read or Grep that file to check progress so far (it is "
                "partial until the task finishes, and a read's own exit code "
                "reflects the read, not the task)."
            ),
            (
                '- A notification with status "interrupted" means the task '
                "was stopped before finishing (e.g. by a restart): check any "
                "partial output it produced, then continue from where it "
                "left off when possible, or re-run it from scratch."
            ),
        ]
        if "KillTask" in tools:
            lines.append(
                "- Stop a task with KillTask if you no longer need it or the "
                "user wants to cancel it."
            )
        lines.append("</background_tasks>")

        return "\n".join(lines)

    return SystemPromptSection(name=section_name, compute=compute)


def _make_launch_note(
    *,
    task_id: str,
    tool_name: str,
    tool_call_id: str,
    backgrounded_after: float | None,
    log_path: str | None,
) -> str:
    """
    The immediate tool result for a backgrounded call: the same XML block as a
    completion note, with status RUNNING. ``backgrounded_after`` is the
    deadline a positive-``auto_background_at`` call outlived (``None`` for an
    immediate background); it shapes only the subject line, never the task's
    kind. The waiting/log/kill guidance is not repeated here — it lives in the
    auto-attached ``background_tasks`` system-prompt section
    (:func:`make_background_tasks_section`).
    """
    subject = (
        "Task launched and running in the background"
        if backgrounded_after is None
        else (
            f"Task was moved to the background after running "
            f"for {backgrounded_after:g}s"
        )
    )
    parts = [
        "<task_notification>",
        f"<subject> {subject} </subject>",
        f"<task_id> {task_id} </task_id>",
        f"<tool_name> {tool_name} </tool_name>",
        f"<tool_call_id> {tool_call_id} </tool_call_id>",
        f"<status> {TaskStatus.RUNNING.value} </status>",
    ]

    if log_path is not None:
        parts.append(f"<log_file> {log_path} </log_file>")

    parts.append("</task_notification>")

    return "\n".join(parts)


def _make_launch_event(
    agent_name: str,
    task_id: str,
    tool_name: str,
    tool_call_id: str,
    *,
    log_path: str | None,
    exec_id: str,
) -> BackgroundTaskLaunchedEvent:
    """
    Build the launched event both backgrounding modes bubble.

    ``output_name`` is surfaced only for a task that has a ``.grasp`` log
    (``log_path`` set); a sub-agent without one advertises ``None``. It equals
    the log's basename by construction (``open_task_log`` builds the filename
    with ``task_log_name``).
    """
    return BackgroundTaskLaunchedEvent(
        source=agent_name,
        exec_id=exec_id,
        data=BackgroundTaskInfo(
            task_id=task_id,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            output_name=task_log_name(tool_call_id) if log_path else None,
        ),
    )


# Fallback cap on inlined chars when the tool's own ``max_inline_result_chars``
# is unavailable — a completion note rebuilt from a durable record (resume
# re-injection / rollback re-delivery; the setting is not persisted) or a
# ``KillTask`` excerpt for a tool without a cap — so a large output is never
# inlined whole into the transcript.
_DEFAULT_INLINE_CAP = 8000


def _make_outcome_note(
    *,
    task_id: str,
    tool_name: str,
    tool_call_id: str,
    failed: bool,
    interrupted: bool = False,
    result: str | None = None,
    error: str | None = None,
    log_path: str | None = None,
    elapsed_s: float | None = None,
) -> str:
    """Build an XML-tagged task notification for LLM consumption."""
    status = "interrupted" if interrupted else "failed" if failed else "completed"
    subject = f"Task {status}"

    parts = [
        "<task_notification>",
        f"<subject> {subject} </subject>",
        f"<task_id> {task_id} </task_id>",
        f"<tool_name> {tool_name} </tool_name>",
        f"<tool_call_id> {tool_call_id} </tool_call_id>",
        f"<status> {status} </status>",
    ]

    if elapsed_s is not None:
        parts.append(f"<ran_for> {_fmt_duration(elapsed_s)} </ran_for>")

    if result is not None:
        parts.append(f"<result>\n{result}\n</result>")

    if error is not None:
        parts.append(f"<error>\n{error}\n</error>")

    if log_path is not None:
        parts.append(f"<log_file>\n{log_path}\n</log_file>")

    parts.append("</task_notification>")

    return "\n".join(parts)


def _task_record_to_input_message(
    record: TaskRecord, *, failed: bool
) -> InputMessageItem:
    """A task's completion note rebuilt from its durable record."""
    raw = record.error if failed else record.result
    body: str | None = None
    if raw is not None:
        body, _ = excerpt_for_inline(
            raw, _DEFAULT_INLINE_CAP, log_file=record.output_path
        )

    return InputMessageItem.from_text(
        _make_outcome_note(
            task_id=record.task_id,
            tool_name=record.tool_name,
            tool_call_id=record.tool_call_id,
            failed=failed,
            result=None if failed else body,
            error=body if failed else None,
            log_path=record.output_path,
        ),
        role="user",
    )


# --- Task record persistence ---

# Validate a restored flip map (a raw dict off a persisted head) back into its
# typed shape.
_delivered_map_adapter: TypeAdapter[dict[str, TaskDeliveredFlip]] = TypeAdapter(
    dict[str, TaskDeliveredFlip]
)
_cancelled_map_adapter: TypeAdapter[dict[str, TaskCancelledFlip]] = TypeAdapter(
    dict[str, TaskCancelledFlip]
)


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


async def _save_record_outcome(
    store: CheckpointStore,
    task_key: str,
    result: Any | None = None,
    error: ToolErrorInfo | None = None,
) -> None:
    if error is not None:
        await _save_record_update(
            store, task_key, status=TaskStatus.FAILED, error=_serialize_result(error)
        )
    else:
        await _save_record_update(
            store,
            task_key,
            status=TaskStatus.COMPLETED,
            result=_serialize_result(result),
        )


# --- Task event stream utilities ---


def _stream_text(events: list[Event[Any]]) -> str:
    """Concatenated incremental output text across ``events`` (rendered via str)."""
    return "".join(str(e.data) for e in events if isinstance(e, ToolStreamEvent))


def _outcome_from_events(
    events: list[Event[Any]],
) -> tuple[Any | None, ToolErrorInfo | None]:
    """The task's terminal result and whether it failed (last writer wins)."""
    result: Any = None
    error: ToolErrorInfo | None = None

    for event in events:
        if isinstance(event, ToolErrorEvent):
            error = event.data
        elif isinstance(event, ToolOutputEvent):
            result = event.data

    return result, error


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
    :meth:`resume_durable`), instead of a ``RUNNING`` record that would force a
    re-run or lose the outcome.
    """
    result: Any | None = None
    error: ToolErrorInfo | None = None

    try:
        async for event in stream:
            events.append(event)
            if isinstance(event, ToolOutputEvent):
                result = event.data
            elif isinstance(event, ToolErrorEvent):
                error = event.data

    except asyncio.CancelledError:
        # A genuine interruption (KillTask / shutdown / process death): leave the
        # record RUNNING so a later resume can re-spawn or report the task.
        raise

    except Exception as exc:
        # The task errored (e.g. a sub-agent that could not resume). Record it as
        # a clean failure: don't leak a dangling task exception, and don't leave
        # a RUNNING record that a later resume would re-spawn forever.
        logger.warning("Background task errored", exc_info=exc)
        err = ToolErrorEvent(
            data=ToolErrorInfo(tool_name="", error=format_error_chain(exc))
        )
        events.append(err)
        error = err.data

    if store is not None and task_key is not None:
        await _save_record_outcome(store, task_key, result=result, error=error)


@dataclass
class BackgroundTask:
    """
    A backgrounded unit of work — kind-agnostic.

    The manager drives ``tool.run_stream`` in ``consumer`` and appends every
    event to ``events`` (the buffer). ``cursor`` tracks how far :meth:`drain` has
    consumed that buffer at the turn boundary — in one pass it re-emits each new
    event to the parent stream (live progress) and mirrors stream text to the
    on-disk progress log. ``task_key`` is set only for a *resumable* tool (its
    ``TaskRecord`` is persisted so a restart can re-spawn it); a backgrounded
    shell command is not resumable, so it has none.
    """

    task_id: str
    tool_name: str
    tool_call_id: str
    exec_id: str

    consumer: asyncio.Task[None]
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

    task_key: str | None = None
    cursor: int = 0  # events consumed by drain (bubbled to parent + flushed to log)
    delivered: bool = False  # completion note reached the transcript

    log_path: str | None = None  # resolved .grasp/tasks log file, once written
    log_bytes: int = 0  # bytes appended to the log so far (for the size cap)

    started_at: float = field(default_factory=time.monotonic)  # for live elapsed

    def trim_consumed(self) -> None:
        """Drop drained (bubbled + flushed) leading events; keep results."""
        keep_from = self.cursor
        for i in range(keep_from):
            if isinstance(self.events[i], ToolErrorEvent | ToolOutputEvent):
                keep_from = i  # never drop a terminal result event
                break
        if keep_from <= 0:
            return
        del self.events[:keep_from]
        self.cursor -= keep_from


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

        # Stamped by the owning agent (with ``path``): False opts this
        # agent's ``TaskRecord``s out of the checkpoint store.
        self.durability_enabled: bool = True

        self._transcript = transcript
        self._tools = tools
        self._tasks: dict[str, BackgroundTask] = {}

        self._bg_counter = 0

        self._max_background = max_background
        self._max_task_log_bytes = max_task_log_bytes

        # Completed task ids, pushed by each task's done-callback (single
        # completion seam): :meth:`drain` pops them to deliver notes, and
        # :meth:`wait_idle` blocks on the next one — one queue, every task kind.
        self._completions: asyncio.Queue[str] = asyncio.Queue()

        # Deferred DELIVERED flips (task_key → field update), applied by
        # :meth:`flush_flips` once a checkpoint has persisted the
        # transcript that carries the corresponding notification.
        self._deferred_delivered: dict[str, TaskDeliveredFlip] = {}

        # Deferred CANCELLED flips for tasks cancelled by
        # :meth:`cancel_launched_after`, also applied by :meth:`flush_flips`.
        # Kept apart from ``_deferred_delivered`` because a watermark restore
        # wholesale-replaces that map — a cancellation must survive it (the
        # cancelled launch is gone from the transcript no matter which boundary
        # is live).
        self._deferred_cancelled: dict[str, TaskCancelledFlip] = {}

        self.path = path

    @property
    def has_live_tasks(self) -> bool:
        """
        True while any background task is still running.

        Tasks are session-scoped — they survive run boundaries and are
        released only by :meth:`cancel_all` (via ``LLMAgent.aclose``) or
        ``KillTask``.
        """
        return any(not bt.consumer.done() for bt in self._tasks.values())

    @property
    def has_blocking_tasks(self) -> bool:
        """
        True while any *answer-blocking* background task is undelivered.

        Consulted by the JUDGE phase: such a task's result is part of the
        answer, so it gates a bounded run's final answer — ending the run
        early would leave no loop for the result to be delivered to. It does
        not gate a resident's answer: the resident loop outlives it and
        receives the completion on wake. Non-blocking tasks (e.g. a
        backgrounded shell command) are excluded — they must never hold the run
        hostage. A task stops blocking once its completion note is delivered
        (``delivered``).
        """
        return any(
            bt.blocks_final_answer and not bt.delivered for bt in self._tasks.values()
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

    def _check_capacity(self) -> None:
        if len(self._tasks) >= self._max_background:
            raise RuntimeError(
                f"Too many background tasks ({self._max_background}); kill or "
                "drain existing ones first."
            )

    def _task_store_key(self, ctx: SessionContext[CtxT], call_id: str) -> str | None:
        """Store key for a backgrounded call's ``TaskRecord`` (``None`` if none)."""
        child_path = make_tool_call_path(self.path, call_id)
        if (
            ctx.checkpoint_store is None
            or child_path is None
            or not self.durability_enabled
        ):
            return None
        return make_store_key(ctx.session_key, CheckpointKind.TASK, child_path)

    async def _resolve_log(
        self,
        name: str,
        tool: BaseTool[BaseModel, Any, CtxT],
        *,
        ctx: SessionContext[CtxT],
    ) -> str | None:
        if not tool.has_progress_log or ctx.file_backend is None:
            return None
        return await open_task_log(ctx.file_backend, name=name)

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

    def _register(
        self,
        *,
        task_id: str,
        tool_call_id: str,
        launch_seq: int,
        consumer: asyncio.Task[None],
        events: list[Event[Any]],
        tool: BaseTool[BaseModel, Any, CtxT],
        exec_id: str,
        task_key: str | None,
        started_at: float,
        log_path: str | None = None,
    ) -> None:
        """
        Track a backgrounded task and enqueue its id when it ends.

        Builds the ``BackgroundTask`` from the tool — ``blocks_final_answer`` /
        ``max_inline_result_chars`` come from it, so both backgrounding modes and
        a resume re-spawn are tracked identically — and wires the done-callback
        that feeds the single completion queue. ``started_at`` is the task's
        launch stamp (for live elapsed); ``log_path`` its resolved ``.grasp`` log
        (``None`` for a tool without one).
        """
        bt = BackgroundTask(
            task_id=task_id,
            launch_seq=launch_seq,
            tool_name=tool.name,
            exec_id=exec_id,
            tool_call_id=tool_call_id,
            consumer=consumer,
            events=events,
            blocks_final_answer=tool.blocks_final_answer,
            max_inline_result_chars=tool.max_inline_result_chars,
            task_key=task_key,
            started_at=started_at,
            log_path=log_path,
        )
        self._tasks[task_id] = bt
        consumer.add_done_callback(
            lambda _, tid=task_id: self._completions.put_nowait(tid)
        )

    # --- Launch ---

    async def _write_running_record(
        self,
        task_key: str,
        call: FunctionToolCallItem,
        task_id: str,
        *,
        launch_seq: int,
        ctx: SessionContext[CtxT],
        log_path: str | None = None,
    ) -> None:
        """
        Persist a RUNNING ``TaskRecord`` so a restart can surface this task.

        ``log_path`` is the task's ``.grasp`` log — resolved up front for a
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
            status=TaskStatus.RUNNING,
            output_path=log_path,
        )
        await store.save(task_key, record.model_dump_json().encode())

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
        command). Everything after the decision — the RUNNING ``TaskRecord``, the
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

            # Persist the RUNNING record *before* starting the task so a
            # near-instant finish still updates an existing record, and wire
            # outcome persistence into ``_consume`` (resumable tools rely on it
            # for a result that lands between drains).

            if task_key is not None:
                await self._write_running_record(
                    task_key,
                    call,
                    task_id,
                    launch_seq=launch_seq,
                    ctx=ctx,
                    log_path=log_path,
                )

            # Create and register the task

            stream = tool.run_stream(
                inp, ctx=ctx, exec_id=exec_id, path=child_path, agent_ctx=agent_ctx
            )
            started_at = time.monotonic()
            consumer = asyncio.create_task(
                _consume(stream, events, store=ctx.checkpoint_store, task_key=task_key)
            )
            self._register(
                consumer=consumer,
                events=events,
                task_id=task_id,
                tool_call_id=call.call_id,
                launch_seq=launch_seq,
                tool=tool,
                exec_id=exec_id,
                task_key=task_key,
                started_at=started_at,
                log_path=log_path,
            )

            # Make a launch note for the agent

            launch_note = _make_launch_note(
                task_id=task_id,
                tool_name=tool.name,
                tool_call_id=call.call_id,
                backgrounded_after=None,
                log_path=log_path,
            )
            launch_event = _make_launch_event(
                agent_name=self._agent_name,
                task_id=task_id,
                tool_name=tool.name,
                tool_call_id=call.call_id,
                log_path=log_path,
                exec_id=exec_id,
            )

            return launch_note, launch_event

        # abg > 0: run in the foreground, racing the deadline. ``started_at`` is
        # stamped at launch (not at sideline), so a backgrounded task's reported
        # runtime includes its foreground portion. ``_consume`` gets the store
        # key up front: if the call backgrounds, its eventual finish must flip
        # the RUNNING record to COMPLETED/FAILED (a foreground finish writes
        # nothing — no record exists yet).

        stream = tool.run_stream(
            inp, ctx=ctx, exec_id=exec_id, path=child_path, agent_ctx=agent_ctx
        )
        started_at = time.monotonic()
        consumer = asyncio.create_task(
            _consume(stream, events, store=ctx.checkpoint_store, task_key=task_key)
        )

        try:
            await asyncio.wait_for(asyncio.shield(consumer), timeout=abg)
        except TimeoutError:
            try:
                self._check_capacity()
            except RuntimeError as exc:
                consumer.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await consumer
                return f"Tool '{tool.name}' could not be backgrounded: {exc}", None

            task_id, launch_seq = self._next_id()
            log_path = await self._resolve_log(call.call_id, tool, ctx=ctx)

            self._register(
                consumer=consumer,
                events=events,
                task_id=task_id,
                tool_call_id=call.call_id,
                launch_seq=launch_seq,
                tool=tool,
                exec_id=exec_id,
                task_key=task_key,
                started_at=started_at,
                log_path=log_path,
            )

            # Unlike the immediate mode, the record cannot be written before
            # the task starts: whether this call backgrounds at all is known
            # only once it outlives the deadline, and a foreground finish must
            # leave no record (resume would re-inject an outcome the model
            # already saw inline).
            if task_key is not None:
                await self._write_running_record(
                    task_key,
                    call,
                    task_id,
                    launch_seq=launch_seq,
                    ctx=ctx,
                    log_path=log_path,
                )
                if consumer.done() and ctx.checkpoint_store is not None:
                    # The call finished while the RUNNING record was being
                    # written — ``_consume``'s own outcome write found no
                    # record to update, so persist the outcome here.
                    result, error = _outcome_from_events(events)
                    await _save_record_outcome(
                        ctx.checkpoint_store, task_key, result=result, error=error
                    )

            launch_note = _make_launch_note(
                task_id=task_id,
                tool_name=tool.name,
                tool_call_id=call.call_id,
                backgrounded_after=abg,
                log_path=log_path,
            )
            launch_event = _make_launch_event(
                agent_name=self._agent_name,
                task_id=task_id,
                tool_name=tool.name,
                tool_call_id=call.call_id,
                log_path=log_path,
                exec_id=exec_id,
            )

            return launch_note, launch_event

        except asyncio.CancelledError:
            consumer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await consumer
            raise

        # The call finished in the foreground

        result, error = _outcome_from_events(events)

        return error or result, None

    async def wait_idle(self, timeout: float | None = None) -> None:  # noqa: ASYNC109
        """
        Block until the next tracked task completes — the loop's idle wait.

        Returns immediately if a completion is already queued, or if every
        tracked task has delivered (so the loop never blocks with no work
        outstanding). The awaited
        id is requeued so :meth:`drain` still delivers it. ``timeout`` bounds the
        wait (the caller passes the remaining run-deadline budget) so an idle
        wait on a task that never completes cannot sail past ``run_timeout`` — on
        expiry it returns and the loop's next deadline check stops the run.
        """
        if not self._completions.empty():
            return
        if all(bt.delivered for bt in self._tasks.values()):
            return
        try:
            task_id = await asyncio.wait_for(self._completions.get(), timeout)
        except TimeoutError:
            return
        self._completions.put_nowait(task_id)

    # --- Turn-boundary drain ---

    async def _append_log(
        self, bt: BackgroundTask, text: str, backend: "FileBackend"
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
        if not text or bt.log_path is None or bt.log_bytes >= self._max_task_log_bytes:
            return

        chunk = text.encode()
        remaining = self._max_task_log_bytes - bt.log_bytes

        if len(chunk) > remaining:
            chunk = chunk[:remaining] + b"\n... [log truncated] ...\n"
            bt.log_bytes = self._max_task_log_bytes  # saturate → stop appending
        else:
            bt.log_bytes += len(chunk)

        await append_task_log(backend, bt.log_path, chunk)

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
        ``Grep``. Either way the task is ``delivered`` once, so it stops gating
        the final answer.
        """
        # Bubble each task's new events to the parent stream (pure
        # observability — live progress for a backgrounded shell command just as
        # for a sub-agent; ``blocks_final_answer`` governs only the JUDGE gate)
        # and mirror this pass's stream text to its ``.grasp`` log in one append.
        backend = ctx.file_backend

        for bt in list(self._tasks.values()):
            start = bt.cursor

            while bt.cursor < len(bt.events):
                event = bt.events[bt.cursor]
                bt.cursor += 1
                if isinstance(event, ToolStreamEvent) and event.source == bt.tool_name:
                    event = event.model_copy(update={"task_id": bt.task_id})
                yield event

            if backend is not None and bt.cursor > start:
                await self._append_log(
                    bt, _stream_text(bt.events[start : bt.cursor]), backend
                )

        while not self._completions.empty():
            task_id = self._completions.get_nowait()
            bt = self._tasks.get(task_id)
            if bt is None or bt.delivered:
                continue

            result, error = _outcome_from_events(bt.events)
            failed = error is not None

            full = _serialize_result(error) if failed else _serialize_result(result)
            cap = bt.max_inline_result_chars

            log_file: str | None = None
            if cap is not None and len(full) > cap and ctx.file_backend is not None:
                log_file = await write_result_file(
                    ctx.file_backend, name=bt.tool_call_id or bt.task_id, text=full
                )

            body, _ = excerpt_for_inline(full, cap, log_file=log_file or bt.log_path)
            note_result = None if failed else body
            note_error = body if failed else None

            notification = InputMessageItem.from_text(
                _make_outcome_note(
                    task_id=bt.task_id,
                    tool_name=bt.tool_name,
                    tool_call_id=bt.tool_call_id,
                    failed=failed,
                    result=note_result,
                    error=note_error,
                    log_path=bt.log_path,
                    elapsed_s=time.monotonic() - bt.started_at,
                ),
                role="user",
            )
            self._transcript.update([notification])

            # Durable record → DELIVERED (resumable tasks only). Deferred to
            # :meth:`flush_flips` after the next checkpoint persists the
            # transcript holding this note: flipping now would lose the
            # outcome on a crash before that checkpoint (resume skips
            # DELIVERED records). The note's transcript position rides along —
            # a step rollback that truncates below it re-injects the note
            # (:meth:`redeliver_after`). Records are kept for post-hoc
            # observability; reclaim with ``prune_delivered``.
            if ctx.checkpoint_store is not None and bt.task_key is not None:
                self._deferred_delivered[bt.task_key] = {
                    "result": None if failed else _serialize_result(result),
                    "error": _serialize_result(error) if failed else None,
                    "note_transcript_pos": len(self._transcript.messages),
                }

            yield BackgroundTaskCompletedEvent(
                source=self._agent_name,
                exec_id=exec_id,
                data=BackgroundTaskInfo(
                    task_id=bt.task_id,
                    tool_name=bt.tool_name,
                    tool_call_id=bt.tool_call_id,
                ),
            )
            yield UserMessageEvent(
                source=bt.tool_name,
                destination=self._agent_name,
                exec_id=exec_id,
                data=notification,
            )

            bt.delivered = True  # no longer gates the final answer
            # The full output lives in the .grasp log; nothing is retained for a
            # poll, so always drop the finished task.
            self._tasks.pop(task_id, None)

        # Front-trim each surviving task's buffer: leading events consumed by
        # BOTH the bubble (parent stream) and flush (.grasp log) cursors are
        # durable elsewhere, so drop them to bound memory for a long, chatty
        # backgrounded command. Stops before any terminal result event so
        # ``_outcome_from_events`` / ``kill_task`` still find it.
        for bt in self._tasks.values():
            bt.trim_consumed()

    # --- Cancellation / kill ---
    #
    # "Kill" is the model-facing surface only (the ``KillTask`` tool →
    # :meth:`kill_task`): an explicit, targeted stop of one task. Every
    # framework-initiated stop — session teardown, transcript rewind — and the
    # terminal record status use "cancel(led)".

    async def cancel_all(self, ctx: SessionContext[CtxT] | None = None) -> None:
        """
        Cancel all background tasks and wait for cleanup.

        Session teardown — reached via ``LLMAgent.aclose()``, never at run
        end (tasks are session-scoped and survive run boundaries).
        """
        store = ctx.checkpoint_store if ctx else None

        if store is not None and ctx is not None:
            for bt in self._tasks.values():
                if bt.task_key is None:
                    continue
                await _save_record_update(
                    store,
                    bt.task_key,
                    status=TaskStatus.CANCELLED,
                    error="Cancelled: agent session closed",
                )

        consumers = [bt.consumer for bt in self._tasks.values()]
        for c in consumers:
            c.cancel()
        if consumers:
            await asyncio.gather(*consumers, return_exceptions=True)

        self._tasks.clear()

        # Cancelled tasks' done-callbacks enqueue their ids; drop them so a
        # reused manager starts clean.
        while not self._completions.empty():
            self._completions.get_nowait()

    async def kill_task(
        self, task_id: str, *, ctx: SessionContext[CtxT] | None = None
    ) -> KillTaskResult:
        """Cancel a task (its stream closes → process group killed) and read it."""
        bt = self._tasks.get(task_id)
        if bt is None:
            known = ", ".join(sorted(self._tasks)) or "none"
            raise ValueError(
                f"Unknown background task id {task_id!r} (known: {known})."
            )

        already_done = bt.consumer.done()
        if not already_done:
            bt.consumer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await bt.consumer

        result, error = _outcome_from_events(bt.events)

        status_at_kill: Literal["cancelled", "completed", "failed"]
        if error is not None:
            status_at_kill = "failed"
        elif already_done:
            status_at_kill = "completed"
        else:
            status_at_kill = "cancelled"

        # A head+tail excerpt of what it produced, so the model sees why it was
        # killed without bloating the transcript; the full output is in the log.
        output, _ = excerpt_for_inline(
            _stream_text(bt.events),
            bt.max_inline_result_chars or _DEFAULT_INLINE_CAP,
            log_file=bt.log_path,
        )

        # Mark as terminal, so a later resume does not
        # report the killed task as interrupted.

        store = ctx.checkpoint_store if ctx else None
        if store is not None and bt.task_key is not None:
            await _save_record_update(
                store,
                bt.task_key,
                status=TaskStatus.CANCELLED,
                error="Stopped by KillTask",
            )
        self._tasks.pop(task_id, None)

        return KillTaskResult(
            task_id=task_id,
            tool_name=bt.tool_name,
            status=status_at_kill,
            output=output,
            result=result,
        )

    # --- Resume ---

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
        consumer = asyncio.create_task(
            _consume(stream, events, store=ctx.checkpoint_store, task_key=task_key)
        )
        self._register(
            consumer=consumer,
            events=events,
            task_id=record.task_id,
            tool_call_id=record.tool_call_id,
            launch_seq=record.launch_seq,
            tool=tool,
            exec_id=exec_id,
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

    async def resume_durable(
        self,
        *,
        ctx: SessionContext[CtxT] | None = None,
        exec_id: str | None = None,
        agent_ctx: "AgentContext | None" = None,
        task_launch_seq: int | None = None,
    ) -> list[InputMessageItem]:
        """
        On resume, re-spawn or notify about interrupted background tasks.

        Resumable tasks (e.g. sub-agents) are re-spawned silently; the rest are
        reported to the agent — interrupted ones it may want to redo, completed
        ones whose result never reached it are re-injected. Any such notice is
        prefixed with a one-line framing so the agent understands it was resumed.

        ``task_launch_seq`` is the restored head's launch high-water: a record
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

        # Scope to this BG task manager
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
            if key in self._deferred_delivered:
                continue

            # A deferred cancellation restored with the head: the task was
            # cancelled by a rewind whose CANCELLED flip a crash kept from
            # flushing. Never re-spawn or report it; the flip lands at the
            # next flush.
            if key in self._deferred_cancelled:
                continue

            # Orphan guard: launched after the restored head's boundary, so its
            # launching tool call is not in the restored transcript — the agent
            # has no memory of it. Dead-letter instead of re-spawning/reporting.
            if task_launch_seq is not None and record.launch_seq > task_launch_seq:
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
                    task_launch_seq,
                )
                continue

            if record.status == TaskStatus.RUNNING:
                # A resumable task (sub-agent / workflow) is re-spawned silently
                # — it continues from its own checkpoint and reports back via its
                # own bubbled events on completion, so no resume notice is needed.
                if self._try_respawn_child(
                    record, task_key=key, ctx=ctx, exec_id=exec_id, agent_ctx=agent_ctx
                ):
                    continue

                elapsed = (record.updated_at - record.created_at).total_seconds()
                partial_output = (
                    "Any partial output it produced before stopping is in the log file."
                    if record.output_path
                    else "It has no output log."
                )
                error = f"The task was stopped before completing. {partial_output}"
                notification = InputMessageItem.from_text(
                    _make_outcome_note(
                        task_id=record.task_id,
                        tool_name=record.tool_name,
                        tool_call_id=record.tool_call_id,
                        failed=False,
                        interrupted=True,
                        error=error,
                        log_path=record.output_path,
                        elapsed_s=elapsed,
                    ),
                    role="user",
                )
                # The interrupted notice was delivered → terminal (DELIVERED),
                # not FAILED (which is an errored task, re-injected below).
                update: TaskDeliveredFlip = {
                    "error": "Interrupted: session restarted",
                }

            elif record.status in {TaskStatus.COMPLETED, TaskStatus.FAILED}:
                # Finished (ok or errored) but the crash kept ``drain`` from
                # delivering it — re-inject the outcome, then mark it delivered
                # (the record already carries it, so the flip adds nothing).
                notification = _task_record_to_input_message(
                    record,
                    failed=(record.status == TaskStatus.FAILED),
                )
                update = {}

            else:
                continue

            # Deferred to ``flush_flips`` (after the checkpoint that
            # persists the notice) — same loss-window reasoning as ``drain``.
            self._deferred_delivered[key] = update
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
                wrap_in_system_reminder(
                    "Resumed from a checkpoint: your conversation and task "
                    "records were restored from disk, but any in-flight work "
                    "that was only in memory is gone. The background tasks "
                    "below were running when the previous run stopped.",
                    subject=SESSION_RESUMED_SUBJECT,
                ),
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
                self._deferred_delivered[key]["note_transcript_pos"] = (
                    total - len(deferred_keys) + offset + 1
                )

        logger.info(
            "Handled %d task records for session %s", len(keys), ctx.session_key
        )

        return injected

    async def redeliver_after(
        self,
        *,
        message_count: int,
        task_launch_seq: int,
        ctx: SessionContext[CtxT],
        pre_restore_deferred_delivered: Mapping[str, Mapping[str, Any]],
    ) -> list[InputMessageItem]:
        """
        The bg-task half of a step rollback, called after the transcript is
        truncated to ``message_count``: a delivered record whose completion
        note sat past the cut (``note_transcript_pos > message_count``) while
        its launching call survived (``launch_seq <= task_launch_seq``) just
        lost the only live copy of its outcome — the finished task is no
        longer tracked in memory, and resume skips DELIVERED records, so
        nothing else would ever re-surface it and the agent would wait on it
        forever. Re-inject each such note at the rewind point and re-defer its
        DELIVERED flip with the new position.

        ``pre_restore_deferred_delivered`` overlays not-yet-flushed DELIVERED
        flips onto the stored records: a drained-but-unflushed note (its
        record still COMPLETED) sits in the truncated span exactly like a
        flushed one's. It must be the map exported *before* the rollback's
        context restore — the post-restore live map is never a substitute: it
        holds the boundary's flips, whose positions all sit at or before the
        cut. Records launched *after* the boundary are
        ``cancel_launched_after``'s business — their launching calls are
        gone, so their outcomes stay buried. Returns the injected messages
        for the caller to surface as events.
        """
        store = ctx.checkpoint_store
        if store is None:
            return []

        deferred = _delivered_map_adapter.validate_python(
            pre_restore_deferred_delivered
        )

        prefix = make_store_key(ctx.session_key, CheckpointKind.TASK, self.path) + "/"
        keys = [k for k in await store.list_keys(prefix) if is_direct_child(k, prefix)]

        matches: list[tuple[str, TaskRecord, int]] = []
        for key in keys:
            record = await store.load_json(key, TaskRecord, subject="task record")
            if record is None or record.launch_seq > task_launch_seq:
                continue

            flip = deferred.get(key)
            is_delivered = record.status is TaskStatus.DELIVERED or flip is not None
            note_pos = record.note_transcript_pos
            if flip is not None:
                note_pos = flip.get("note_transcript_pos", note_pos)
            if not is_delivered or note_pos is None or note_pos <= message_count:
                continue

            matches.append((key, record, note_pos))

        # Original observation order: the positions the notes held before the
        # truncation (completion order, which can differ from launch order).
        matches.sort(key=operator.itemgetter(2))

        injected: list[InputMessageItem] = []
        for key, record, _note_pos in matches:
            is_fail = record.result is None and record.error is not None
            notification = _task_record_to_input_message(record, failed=is_fail)

            self._transcript.update([notification])
            self._deferred_delivered[key] = {
                "note_transcript_pos": len(self._transcript.messages),
            }

            injected.append(notification)

            logger.info(
                "Re-injected background task %s (%s): its completion note was "
                "truncated by a rollback",
                record.task_id,
                record.tool_name,
            )
        return injected

    def cancel_launched_after(self, task_launch_seq: int) -> None:
        """
        Cancel every task with ``seq > task_launch_seq`` — synchronous and I/O-free.

        Applied by :meth:`AgentContext.restore` whenever the transcript is
        rewound past the tool calls that launched these tasks (a failed-run
        settle or a step rollback; vacuous on a cold reload): their
        launches are no longer in the history the model sees, so letting them
        finish would inject completion notes for calls that never happened.
        Cancelled tasks are dropped from tracking immediately, so a completion
        already queued for :meth:`drain` is suppressed rather than delivered.
        Tasks at or below ``task_launch_seq`` keep running — their launches are still in
        the kept transcript, and their notes deliver normally later.

        Durable record flips (→ CANCELLED) are deferred to
        :meth:`flush_flips`, after the pruned transcript is durable; a
        crash before that is covered by the resume-side orphan guard in
        :meth:`resume_durable`.
        """
        for task_id, bt in list(self._tasks.items()):
            if bt.launch_seq <= task_launch_seq:
                continue

            if not bt.consumer.done():
                bt.consumer.cancel()

            del self._tasks[task_id]

            if bt.task_key is not None:
                self._deferred_delivered.pop(bt.task_key, None)
                self._deferred_cancelled[bt.task_key] = {
                    "error": "Cancelled: the launching tool call was rolled back",
                }
            logger.info(
                "Cancelled background task %s (%s): launched after the rewind point",
                task_id,
                bt.tool_name,
            )

    # --- Deferred record flips (DELIVERED / CANCELLED) ---

    async def flush_flips(self, *, ctx: SessionContext[CtxT]) -> None:
        """
        Apply deferred terminal record updates (→ ``DELIVERED`` / ``CANCELLED``).

        Called after a checkpoint has persisted the transcript containing the
        corresponding completion/interruption notes. Flipping the records any
        earlier opens a loss window: a crash between the flip and the next
        checkpoint leaves a DELIVERED record whose note never became durable,
        so resume would never re-inject the outcome. (A crash between the
        checkpoint and this flush yields at worst a re-injected note — a
        duplicate beats a lost outcome.) The same holds for a
        :meth:`cancel_launched_after` cancellation: its CANCELLED flip waits
        until the transcript no longer referencing the launch is durable, so a
        crash in between resumes with the launch still on record.
        """
        store = ctx.checkpoint_store
        if store is None or not (self._deferred_delivered or self._deferred_cancelled):
            return

        # The status is implied by the map an entry sits in. A cancellation
        # wins over a delivery flip for the same record: it came later, when
        # the delivered note was rolled back. Each entry is dropped only once
        # applied, so a store error mid-flush keeps the rest deferred for the
        # next flush instead of losing them (a re-apply is idempotent —
        # updates are absolute field sets).
        for key, update in list(self._deferred_delivered.items()):
            if key not in self._deferred_cancelled:
                await _save_record_update(
                    store, key, status=TaskStatus.DELIVERED, **update
                )
            del self._deferred_delivered[key]

        for key, flip in list(self._deferred_cancelled.items()):
            await _save_record_update(store, key, status=TaskStatus.CANCELLED, **flip)
            del self._deferred_cancelled[key]

    def export_deferred_delivered(self) -> dict[str, dict[str, Any]]:
        """The deferred record updates not yet flushed (rollback snapshot)."""
        return {k: dict(v) for k, v in self._deferred_delivered.items()}

    def restore_deferred_delivered(
        self, deferred: Mapping[str, Mapping[str, Any]]
    ) -> None:
        """
        Restore a :meth:`export_deferred_delivered` snapshot, replacing the
        live map. The caller keeps a live flip by merging it into the snapshot
        first — ``AgentContext.restore`` does so exactly for the flips whose
        notes survived its transcript surgery, so a rolled-back note's record
        stays COMPLETED for a later re-injection.
        """
        self._deferred_delivered = _delivered_map_adapter.validate_python(deferred)

    def export_deferred_cancelled(self) -> dict[str, dict[str, Any]]:
        """The deferred CANCELLED flips not yet flushed."""
        return {k: dict(v) for k, v in self._deferred_cancelled.items()}

    def restore_deferred_cancelled(
        self, cancelled: Mapping[str, Mapping[str, Any]]
    ) -> None:
        """
        Restore an :meth:`export_deferred_cancelled` snapshot, replacing the
        live map. As with the delivered flips, the caller keeps live entries
        by merging them into the snapshot first — ``AgentContext.restore``
        keeps ALL of them (a cancellation is a physical fact no transcript
        boundary falsifies; the keep-rules for both maps live there).
        """
        self._deferred_cancelled = _cancelled_map_adapter.validate_python(cancelled)

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
