import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from logging import getLogger
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from grasp_agents.durability.checkpoint_store import CheckpointStore
from grasp_agents.durability.checkpoints import CheckpointKind
from grasp_agents.durability.store_keys import (
    make_store_key,
    make_tool_call_path,
    task_prefix,
)
from grasp_agents.durability.task_record import TaskRecord, TaskStatus
from grasp_agents.run_context import RunContext
from grasp_agents.types.events import (
    BackgroundTaskCompletedEvent,
    BackgroundTaskInfo,
    BackgroundTaskLaunchedEvent,
    Event,
    ToolErrorEvent,
    ToolOutputEvent,
    ToolStreamEvent,
    UserMessageEvent,
)
from grasp_agents.types.items import FunctionToolCallItem, InputMessageItem
from grasp_agents.types.tool import BaseTool

from .llm_agent_transcript import LLMAgentTranscript
from .task_progress import (
    append_task_log,
    open_task_log,
    task_log_name,
    write_result_file,
)

if TYPE_CHECKING:
    from ..tools.file_backend.base import FileBackend
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
    async for event in stream:
        events.append(event)
        if isinstance(event, ToolOutputEvent):
            result = event.data
        elif isinstance(event, ToolErrorEvent):
            result = event.data
            failed = True

    if store is not None and task_key is not None:
        existing = await store.load(task_key)
        if existing:
            record = TaskRecord.model_validate_json(existing)
            if failed:
                outcome = {
                    "status": TaskStatus.FAILED,
                    "error": _serialize_result(result),
                }
            else:
                outcome = {
                    "status": TaskStatus.COMPLETED,
                    "result": _serialize_result(result),
                }
            record = record.model_copy(
                update={**outcome, "updated_at": datetime.now(UTC)}
            )
            await store.save(task_key, record.model_dump_json().encode())


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


def _excerpt_for_inline(
    text: str, cap: int | None, *, output_file: str | None = None
) -> tuple[str, bool]:
    """
    Fit a result into a completion note: ``(text, truncated)``.

    Returns ``text`` unchanged when no cap applies or it already fits. Otherwise
    keeps a head + tail of ``cap`` chars total with a middle marker; when an
    ``output_file`` is known (the task's ``.grasp`` progress log), the marker
    points there so the model can ``Read`` / ``Grep`` the full output on demand
    rather than bloating the transcript with it.
    """
    if cap is None or len(text) <= cap:
        return text, False
    head = cap // 2
    tail = cap - head
    omitted = len(text) - cap
    pointer = f" — full output in {output_file}" if output_file else ""
    marker = f"\n... [{omitted} chars omitted{pointer}] ...\n"
    return text[:head] + marker + text[-tail:], True


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
        self._path = path
        self._tasks: dict[str, PendingTask] = {}
        self._bg_counter = 0
        self._max_background = max_background
        self._max_task_log_bytes = max_task_log_bytes
        # Completed task ids, pushed by each task's done-callback (single
        # completion seam): :meth:`drain` pops them to deliver notes, and
        # :meth:`wait_idle` blocks on the next one — one queue, every task kind.
        self._completions: asyncio.Queue[str] = asyncio.Queue()

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
        if not any(not pt.announced for pt in self._tasks.values()):
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
        ctx: RunContext[CtxT],
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
        child_path = make_tool_call_path(self._path, call.call_id)
        events: list[Event[Any]] = []

        if abg == 0:
            try:
                self._check_capacity()
            except RuntimeError as exc:
                return f"Tool '{tool.name}' could not be backgrounded: {exc}", None
            task_id = self._next_id()
            log_path = await self._resolve_log(call.call_id, tool, ctx=ctx)
            # Persist the PENDING record *before* starting the task so a
            # near-instant finish still updates an existing record, and wire
            # outcome persistence into ``_consume`` (resumable tools rely on it
            # for a result that lands between drains).
            if task_key is not None:
                await self._write_pending_record(
                    task_key, call, task_id, ctx=ctx, output_path=log_path
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
        # runtime includes its foreground portion.
        stream = tool.run_stream(
            inp, ctx=ctx, exec_id=exec_id, path=child_path, agent_ctx=agent_ctx
        )
        started_at = time.monotonic()
        task = asyncio.create_task(_consume(stream, events))
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
            task_id = self._next_id()
            # Resolve the log once, up front, so the single pending-record write
            # already carries ``output_path`` and the note can cite the file the
            # loop will have flushed output to by the model's next turn.
            log_path = await self._resolve_log(call.call_id, tool, ctx=ctx)
            self._register(
                task_id=task_id,
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
                    task_key, call, task_id, ctx=ctx, output_path=log_path
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
        if backgrounded_after is None:
            opening = f"Task '{tool.name}' launched in the background (id: {task_id})."
        else:
            opening = (
                f"Tool '{tool.name}' is still running after {backgrounded_after:g}s "
                f"and was moved to the background (id: {task_id})."
            )
        parts = [
            opening,
            "Its result and final status will be delivered to you when it "
            "finishes — until then, assume it is still running.",
        ]
        if log_path is not None:
            parts.append(
                f"Its output is streaming to {log_path} — Read or Grep that file "
                "to check progress so far (it is partial until the task "
                "finishes, and a read's own exit code reflects the read, not "
                "the task)."
            )
        parts.append("Stop it with KillTask if you no longer need it.")
        return "\n".join(parts)

    def _next_id(self) -> str:
        self._bg_counter += 1
        return f"bg_{self._bg_counter}"

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
        ctx: RunContext[CtxT],
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

    def _task_store_key(self, ctx: RunContext[CtxT], call_id: str) -> str | None:
        """Store key for a backgrounded call's ``TaskRecord`` (``None`` if none)."""
        child_path = make_tool_call_path(self._path, call_id)
        if ctx.checkpoint_store is None or child_path is None:
            return None
        return make_store_key(ctx.session_key, CheckpointKind.TASK, child_path)

    async def _write_pending_record(
        self,
        task_key: str,
        call: FunctionToolCallItem,
        task_id: str,
        *,
        ctx: RunContext[CtxT],
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
            tool_call_id=call.call_id,
            tool_name=call.name,
            tool_call_arguments=call.arguments,
            status=TaskStatus.PENDING,
            started_at=datetime.now(UTC),
            output_path=output_path,
        )
        await store.save(task_key, record.model_dump_json().encode())

    async def _mark_record(
        self,
        task_key: str | None,
        *,
        ctx: RunContext[CtxT] | None,
        **updates: Any,
    ) -> None:
        """Load → update → save a ``TaskRecord`` (no-op if no store / key / record)."""
        store = ctx.checkpoint_store if ctx else None
        if store is None or task_key is None:
            return
        existing = await store.load(task_key)
        if not existing:
            return
        record = TaskRecord.model_validate_json(existing)
        record = record.model_copy(update={**updates, "updated_at": datetime.now(UTC)})
        await store.save(task_key, record.model_dump_json().encode())

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
        self, task_id: str, *, ctx: RunContext[CtxT] | None = None
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
        output, _ = _excerpt_for_inline(
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

    # --- Turn-boundary drain ---

    async def drain(
        self, *, exec_id: str, ctx: RunContext[CtxT]
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

            body, _ = _excerpt_for_inline(
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

            # Durable record → DELIVERED (resumable tasks only). Records are kept
            # for post-hoc observability; reclaim with ``prune_delivered``.
            if ctx.checkpoint_store is not None and pt.task_key is not None:
                existing = await ctx.checkpoint_store.load(pt.task_key)
                if existing:
                    record = TaskRecord.model_validate_json(existing)
                    record = record.model_copy(
                        update={
                            "status": TaskStatus.DELIVERED,
                            "result": None if failed else _serialize_result(result),
                            "error": _serialize_result(result) if failed else None,
                            "updated_at": datetime.now(UTC),
                        }
                    )
                    await ctx.checkpoint_store.save(
                        pt.task_key, record.model_dump_json().encode()
                    )

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

    async def cancel_all(self, ctx: RunContext[CtxT] | None = None) -> None:
        """Cancel all pending background tasks and wait for cleanup."""
        store = ctx.checkpoint_store if ctx else None

        if store is not None and ctx is not None:
            for pt in self._tasks.values():
                if pt.task_key is None:
                    continue
                existing = await store.load(pt.task_key)
                if existing:
                    record = TaskRecord.model_validate_json(existing)
                    record = record.model_copy(
                        update={
                            "status": TaskStatus.CANCELLED,
                            "error": "Cancelled: agent loop terminated",
                            "updated_at": datetime.now(UTC),
                        }
                    )
                    await store.save(pt.task_key, record.model_dump_json().encode())

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
        ctx: RunContext[CtxT] | None = None,
        exec_id: str | None = None,
        agent_ctx: "AgentContext | None" = None,
    ) -> None:
        """
        On resume, re-spawn or notify about interrupted background tasks.

        Resumable tasks (e.g. sub-agents) are re-spawned silently; the rest are
        reported to the agent — interrupted ones it may want to redo, completed
        ones whose result never reached it are re-injected. Any such notice is
        prefixed with a one-line framing so the agent understands it was resumed.
        """
        store = ctx.checkpoint_store if ctx else None
        if store is None or ctx is None:
            return

        # Scope to *this* manager's own background tasks. Records live at
        # ``<session>/task/<path>/tc_<call_id>``; sibling agents and nested
        # sub-agents own separate subtrees and resume via their own managers,
        # so a session-wide scan would make us handle (and mis-route) their
        # tasks. Keep only direct ``tc_*`` children — a deeper segment belongs
        # to a nested sub-agent.
        prefix = make_store_key(ctx.session_key, CheckpointKind.TASK, self._path) + "/"
        keys = [k for k in await store.list_keys(prefix) if "/" not in k[len(prefix) :]]

        notifications: list[InputMessageItem] = []
        for key in keys:
            record = await store.load_json(key, TaskRecord, subject="task record")
            if record is None:
                continue

            # Terminal + already surfaced — nothing to do. (FAILED is NOT here:
            # a FAILED record is an errored task that the crash kept ``drain``
            # from delivering, so it still needs re-injecting below.)
            if record.status in {TaskStatus.CANCELLED, TaskStatus.DELIVERED}:
                continue

            if record.status == TaskStatus.PENDING:
                if self._try_respawn_child(
                    record, task_key=key, ctx=ctx, exec_id=exec_id, agent_ctx=agent_ctx
                ):
                    continue

                elapsed = (
                    (record.updated_at - record.started_at).total_seconds()
                    if record.started_at is not None
                    else None
                )
                notification = InputMessageItem.from_text(
                    _task_notification(
                        task_id=record.task_id,
                        tool_name=record.tool_name,
                        status="interrupted",
                        error="Session restarted before completion",
                        log_path=record.output_path,
                        elapsed_s=elapsed,
                    ),
                    role="user",
                )
                # The interrupted notice was delivered → terminal (DELIVERED),
                # not FAILED (which now means an errored task, re-injected below).
                record = record.model_copy(
                    update={
                        "status": TaskStatus.DELIVERED,
                        "error": "Interrupted: session restarted",
                        "updated_at": datetime.now(UTC),
                    }
                )
            elif record.status in {TaskStatus.COMPLETED, TaskStatus.FAILED}:
                # Finished (ok or errored) but the crash kept ``drain`` from
                # delivering it — re-inject the outcome, then mark it delivered.
                is_fail = record.status == TaskStatus.FAILED
                notification = InputMessageItem.from_text(
                    _task_notification(
                        task_id=record.task_id,
                        tool_name=record.tool_name,
                        status="failed" if is_fail else "completed",
                        result=None if is_fail else record.result,
                        error=record.error if is_fail else None,
                        log_path=record.output_path,
                    ),
                    role="user",
                )
                record = record.model_copy(
                    update={
                        "status": TaskStatus.DELIVERED,
                        "updated_at": datetime.now(UTC),
                    }
                )
            else:
                continue

            await store.save(key, record.model_dump_json().encode())
            notifications.append(notification)

        if notifications:
            # One framing line so the agent understands the per-task notices
            # that follow: it was resumed, in-memory state was reconstructed
            # (not continued), and interrupted tasks may need redoing.
            framing = InputMessageItem.from_text(
                "Session resumed from a checkpoint — in-memory state was "
                "reconstructed. The background tasks below were in flight when "
                "the session stopped; for any reported as interrupted, decide "
                "whether to redo the work (any output it produced is noted with "
                "it).",
                role="user",
            )
            self._transcript.update([framing, *notifications])

        logger.info(
            "Handled %d task records for session %s", len(keys), ctx.session_key
        )

    # --- Offline cleanup ---

    @staticmethod
    async def prune_delivered(
        ctx: RunContext[Any],
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
        ctx: RunContext[CtxT] | None,
        exec_id: str | None,
        agent_ctx: "AgentContext | None" = None,
    ) -> bool:
        """Re-spawn a child task from its session checkpoint."""
        if not ctx or not exec_id or not ctx.checkpoint_store:
            return False

        tool = self._tools.get(record.tool_name) if self._tools else None
        if not tool or not tool.resumable:
            return False

        child_path = make_tool_call_path(self._path, record.tool_call_id)

        events: list[Event[Any]] = []
        stream = tool.resume_stream(
            ctx=ctx, exec_id=exec_id, path=child_path, agent_ctx=agent_ctx
        )
        async_task = asyncio.create_task(
            _consume(stream, events, store=ctx.checkpoint_store, task_key=task_key)
        )

        self._register(
            task_id=record.task_id,
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
