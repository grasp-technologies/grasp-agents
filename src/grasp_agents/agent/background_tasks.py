import asyncio
import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from logging import getLogger
from typing import TYPE_CHECKING, Any, Generic, Literal

from pydantic import BaseModel

from grasp_agents.durability.checkpoint_store import CheckpointStore
from grasp_agents.durability.checkpoints import CheckpointKind
from grasp_agents.durability.store_keys import (
    make_store_key,
    make_tool_call_path,
    task_prefix,
)
from grasp_agents.durability.task_record import TaskRecord, TaskStatus
from grasp_agents.run_context import CtxT, RunContext
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

if TYPE_CHECKING:
    from .agent_context import AgentContext

logger = getLogger(__name__)


class TaskOutputResult(BaseModel):
    """
    What ``TaskOutput`` / ``KillTask`` return for a backgrounded task.

    Generic over any tool: ``output`` is the text produced since the last poll
    (assembled from the task's stream events), and ``result`` is the tool's own
    terminal output (e.g. a ``BashResult``) once it has finished — ``None``
    while it is still running or if it was killed before producing one.
    """

    task_id: str
    tool_name: str
    status: Literal["running", "completed", "failed"]
    output: str = ""
    result: Any = None


def _task_notification(
    *,
    task_id: str,
    tool_name: str,
    status: str,
    result: str | None = None,
    error: str | None = None,
) -> str:
    """Build an XML-tagged task notification for LLM consumption."""
    parts = [
        "<task_notification>",
        f"<task_id>{task_id}</task_id>",
        f"<tool_name>{tool_name}</tool_name>",
        f"<status>{status}</status>",
    ]
    if result is not None:
        parts.append(f"<result>\n{result}\n</result>")
    if error is not None:
        parts.append(f"<error>\n{error}\n</error>")
    parts.append("</task_notification>")

    return "\n".join(parts)


@dataclass
class PendingTask:
    """
    A backgrounded unit of work — kind-agnostic.

    The manager drives ``tool.run_stream`` in ``task`` and appends every event
    to ``events`` (the buffer). Two cursors read that one buffer: ``bubble``
    re-emits new events to the parent stream at the turn boundary (used for
    answer-blocking sub-agent tasks, so their nested progress is visible), and
    ``poll`` is advanced by ``TaskOutput`` for the model's incremental reads.
    ``task_key`` is set only for a *resumable* tool (its ``TaskRecord`` is
    persisted so a restart can re-spawn it); a backgrounded shell command is not
    resumable, so it has none.
    """

    task_id: str
    tool_name: str
    exec_id: str
    task: asyncio.Task[None]
    events: list[Event[Any]] = field(default_factory=list[Event[Any]])
    # Whether this task's result gates the final answer (it is part of the
    # answer). Read from ``tool.blocks_final_answer`` at launch — independent of
    # how the task was backgrounded (spawn vs deadline).
    blocks_final_answer: bool = True
    # Cap on inlined result chars in the completion note (``tool``'s setting); a
    # larger result is excerpted + the task kept for a ``TaskOutput`` read.
    max_inline_result_chars: int | None = None
    tool_call_id: str | None = None
    task_key: str | None = None
    bubble_cursor: int = 0  # events already re-emitted to the parent stream
    poll_cursor: int = 0  # events already read out via TaskOutput
    announced: bool = False  # completion note already emitted


async def _consume(
    stream: AsyncIterator[Event[Any]],
    events: list[Event[Any]],
    *,
    store: CheckpointStore | None = None,
    task_key: str | None = None,
) -> None:
    """
    Drive a tool's ``run_stream``, buffering every event into ``events``.

    On a successful finish, persist a ``COMPLETED`` ``TaskRecord`` carrying the
    result *before* the done-callback fires. This closes the window between a
    task finishing and :meth:`BackgroundTaskManager.drain` delivering it: a
    crash in that window leaves a ``COMPLETED`` record that resume re-injects
    (see :meth:`resume_durable`), instead of a ``PENDING`` record that would
    force a re-run or lose the result.
    """
    result: Any = None
    failed = False
    async for event in stream:
        events.append(event)
        if isinstance(event, ToolOutputEvent):
            result = event.data
        elif isinstance(event, ToolErrorEvent):
            result = event.data  # ToolErrorInfo
            failed = True

    if store is not None and task_key is not None and not failed:
        existing = await store.load(task_key)
        if existing:
            record = TaskRecord.model_validate_json(existing)
            record = record.model_copy(
                update={
                    "status": TaskStatus.COMPLETED,
                    "result": str(result),
                    "updated_at": datetime.now(UTC),
                }
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


def _excerpt_for_inline(
    text: str, cap: int | None, *, task_id: str
) -> tuple[str, bool]:
    """
    Fit a result into a completion note: ``(text, truncated)``.

    Returns ``text`` unchanged when no cap applies or it already fits. Otherwise
    keeps a head + tail of ``cap`` chars total with a middle marker pointing the
    model at ``TaskOutput`` for the full result — the basis of cap-and-defer:
    the note stays small, the finished task is retained so the full result can
    still be pulled.
    """
    if cap is None or len(text) <= cap:
        return text, False
    head = cap // 2
    tail = cap - head
    omitted = len(text) - cap
    marker = (
        f"\n... [{omitted} chars omitted — call TaskOutput(task_id={task_id!r}) "
        "for the full result] ...\n"
    )
    return text[:head] + marker + text[-tail:], True


class BackgroundTaskManager(Generic[CtxT]):
    """
    Tracks background work of any kind, generic over the tool that produces it.

    One mechanism: run a tool's ``run_stream`` as a task and buffer the events
    it yields, keyed by ``task_id``. :meth:`spawn` launches a task now (e.g. a
    sub-agent); :meth:`run_with_deadline` runs a call in the foreground and
    *sidelines* it as a pollable task only if it outlives its
    ``auto_background_at`` (e.g. a long shell command). Either path is just *how*
    a task is backgrounded; whether it gates the final answer and how much of
    its result is inlined on completion are the tool's own
    ``blocks_final_answer`` / ``max_inline_result_chars``, read the same way in
    both. ``TaskOutput`` / ``KillTask`` read / stop a task by id
    (:meth:`read_output` / :meth:`kill_task`). One drain / idle-wait / cancel
    path serves every task; every backgrounded task gets a persisted
    ``TaskRecord`` so a restart can surface it — resumable ones re-spawned on
    resume, others reported to the agent as interrupted.
    """

    def __init__(
        self,
        *,
        agent_name: str,
        transcript: LLMAgentTranscript,
        tools: dict[str, BaseTool[BaseModel, Any, CtxT]] | None,
        path: list[str] | None = None,
        max_background: int = 16,
    ) -> None:
        self._agent_name = agent_name
        self._transcript = transcript
        self._tools = tools
        self._path = path
        self._tasks: dict[str, PendingTask] = {}
        self._bg_counter = 0
        self._max_background = max_background
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
        (``announced``), even if it is retained for a later ``TaskOutput`` read
        (a large, excerpted result).
        """
        return any(
            pt.blocks_final_answer and not pt.announced for pt in self._tasks.values()
        )

    async def wait_idle(self) -> None:
        """
        Block until the next tracked task completes — the loop's idle wait.

        Returns immediately if a completion is already queued, or if nothing is
        pending (so the loop never blocks with no work outstanding). The awaited
        id is requeued so :meth:`drain` still delivers it.
        """
        if not self._completions.empty():
            return
        if not any(not pt.announced for pt in self._tasks.values()):
            return
        task_id = await self._completions.get()
        self._completions.put_nowait(task_id)

    # --- Launch ---

    async def spawn(
        self,
        call: FunctionToolCallItem,
        tool: BaseTool[BaseModel, Any, CtxT],
        inp: BaseModel,
        *,
        ctx: RunContext[CtxT],
        exec_id: str,
        agent_ctx: "AgentContext | None" = None,
    ) -> tuple[str, BackgroundTaskLaunchedEvent]:
        """
        Launch a background task from a tool call now; return ``(note, event)`` —
        a launch note (the immediate tool result, built here so both backgrounding
        paths craft their note in the manager) + the launched event to bubble.

        Whether the task gates the final answer is the tool's own
        ``blocks_final_answer`` (independent of persistence): a worker sub-agent
        leaves it ``True`` (its result is the answer, delivered inline at the
        turn boundary); a fire-and-forget tool sets it ``False``. Every
        backgrounded task gets a persisted PENDING ``TaskRecord`` so
        :meth:`resume_durable` can surface it after a restart — re-spawning it if
        the tool is resumable, else reporting it as interrupted.

        ``agent_ctx`` is the parent loop's agent-scope state, forwarded to the
        backgrounded ``run_stream`` so a sub-agent-as-tool still resolves its
        parent's transcript / sibling tools off the call.
        """
        task_id = self._next_id()
        store = ctx.checkpoint_store
        child_path = make_tool_call_path(self._path, call.call_id)

        task_key = self._task_store_key(ctx, call.call_id)
        if task_key is not None:
            await self._write_pending_record(task_key, call, task_id, ctx=ctx)

        events: list[Event[Any]] = []
        stream = tool.run_stream(
            inp, ctx=ctx, exec_id=exec_id, path=child_path, agent_ctx=agent_ctx
        )
        async_task = asyncio.create_task(
            _consume(stream, events, store=store, task_key=task_key)
        )
        self._register(
            PendingTask(
                task_id=task_id,
                tool_name=call.name,
                exec_id=exec_id,
                tool_call_id=call.call_id,
                task=async_task,
                events=events,
                blocks_final_answer=tool.blocks_final_answer,
                max_inline_result_chars=tool.max_inline_result_chars,
                task_key=task_key,
            )
        )

        note = (
            f"Task '{call.name}' launched in the background (id: {task_id}); "
            "its result will be delivered to you when it finishes."
        )
        event = BackgroundTaskLaunchedEvent(
            source=self._agent_name,
            exec_id=exec_id,
            data=BackgroundTaskInfo(
                task_id=task_id,
                tool_name=call.name,
                tool_call_id=call.call_id,
            ),
        )
        return note, event

    async def run_with_deadline(
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
        Run a call in the foreground, moving it to the background if it outlives
        ``tool.auto_background_at``.

        Returns ``(output, launched)``: if the call finishes within the deadline,
        ``output`` is the tool's result and ``launched`` is ``None``; otherwise
        the still-running task is sidelined as a non-blocking, *pollable* task,
        ``output`` is a launch note (the model reads its output with
        ``TaskOutput`` / stops it with ``KillTask``), and ``launched`` is the
        :class:`BackgroundTaskLaunchedEvent` for the caller to bubble — the same
        event :meth:`spawn` emits, so a deadline-backgrounded task is just as
        visible to observers. The manager owns the whole race — the tool knows
        nothing about backgrounding.
        """
        abg = tool.auto_background_at
        assert abg is not None

        task_key = self._task_store_key(ctx, call.call_id)
        events: list[Event[Any]] = []
        stream = tool.run_stream(
            inp,
            ctx=ctx,
            exec_id=exec_id,
            path=make_tool_call_path(self._path, call.call_id),
            agent_ctx=agent_ctx,
        )
        task = asyncio.create_task(_consume(stream, events))
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=abg)
        except TimeoutError:
            try:
                task_id = self._sideline(
                    task,
                    events,
                    tool=tool,
                    exec_id=exec_id,
                    tool_call_id=call.call_id,
                    task_key=task_key,
                )
            except RuntimeError as exc:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                return f"Tool '{tool.name}' could not be backgrounded: {exc}", None
            # Only now that it's backgrounded does it need a durable record: a
            # restart will report it as interrupted (it is non-resumable, so it
            # is not re-spawned — the agent reads its partial output / re-runs).
            if task_key is not None:
                await self._write_pending_record(task_key, call, task_id, ctx=ctx)
            note = (
                f"Tool '{tool.name}' is still running after {abg:g}s and was "
                f"moved to the background (id: {task_id}); poll it with "
                f"TaskOutput(task_id={task_id!r}), stop it with KillTask, or "
                "just wait — you are notified when it finishes."
            )
            launched = BackgroundTaskLaunchedEvent(
                source=self._agent_name,
                exec_id=exec_id,
                data=BackgroundTaskInfo(
                    task_id=task_id,
                    tool_name=tool.name,
                    tool_call_id=call.call_id,
                ),
            )
            return note, launched
        except asyncio.CancelledError:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            raise

        result, _ = _result_of(events)
        return result, None

    def _next_id(self) -> str:
        self._bg_counter += 1
        return f"bg_{self._bg_counter}"

    def _sideline(
        self,
        task: asyncio.Task[None],
        events: list[Event[Any]],
        *,
        tool: BaseTool[BaseModel, Any, CtxT],
        exec_id: str,
        tool_call_id: str | None,
        task_key: str | None = None,
    ) -> str:
        """
        Register a still-running deadline task as a pollable background task.

        Whether it gates the final answer and how much of its result is inlined
        come from the tool (``blocks_final_answer`` / ``max_inline_result_chars``)
        — the same source the immediate :meth:`spawn` path reads, so a deadline
        task is treated no differently from a spawned one once backgrounded.
        ``task_key`` links the (separately written) ``TaskRecord`` so drain /
        kill mark it terminal.
        """
        if len(self._tasks) >= self._max_background:
            raise RuntimeError(
                f"Too many background tasks ({self._max_background}); kill or "
                "drain existing ones first."
            )
        task_id = self._next_id()
        self._register(
            PendingTask(
                task_id=task_id,
                tool_name=tool.name,
                exec_id=exec_id,
                tool_call_id=tool_call_id,
                task=task,
                events=events,
                blocks_final_answer=tool.blocks_final_answer,
                max_inline_result_chars=tool.max_inline_result_chars,
                task_key=task_key,
            )
        )
        return task_id

    def _register(self, pt: PendingTask) -> None:
        """Track a task and push its id onto the completion queue when it ends."""
        self._tasks[pt.task_id] = pt
        pt.task.add_done_callback(
            lambda _t, tid=pt.task_id: self._completions.put_nowait(tid)
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
    ) -> None:
        """Persist a PENDING ``TaskRecord`` so a restart can surface this task."""
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

    # --- Poll / stop (TaskOutput / KillTask) ---

    async def read_output(
        self, task_id: str, *, ctx: RunContext[CtxT] | None = None
    ) -> TaskOutputResult:
        """Output since the last read; the terminal result once finished."""
        pt = self.get(task_id)
        new = pt.events[pt.poll_cursor :]
        pt.poll_cursor = len(pt.events)
        output = _stream_text(new)

        if pt.task.done():
            result, failed = _result_of(pt.events)
            status: Literal["running", "completed", "failed"] = (
                "failed" if failed else "completed"
            )
            # Finished + read here → the result reached the agent via this call,
            # so mark the record delivered (no resume re-injection) and drop it.
            await self._mark_record(pt.task_key, ctx=ctx, status=TaskStatus.DELIVERED)
            self._tasks.pop(task_id, None)
        else:
            result, status = None, "running"

        return TaskOutputResult(
            task_id=task_id,
            tool_name=pt.tool_name,
            status=status,
            output=output,
            result=result,
        )

    async def kill_task(
        self, task_id: str, *, ctx: RunContext[CtxT] | None = None
    ) -> TaskOutputResult:
        """Cancel a task (its stream closes → process group killed) and read it."""
        pt = self.get(task_id)
        if not pt.task.done():
            pt.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pt.task

        new = pt.events[pt.poll_cursor :]
        pt.poll_cursor = len(pt.events)
        result, failed = _result_of(pt.events)
        # Deliberately stopped → record is terminal, so a later resume does not
        # report the killed task as interrupted.
        await self._mark_record(
            pt.task_key,
            ctx=ctx,
            status=TaskStatus.CANCELLED,
            error="Stopped by KillTask",
        )
        self._tasks.pop(task_id, None)
        return TaskOutputResult(
            task_id=task_id,
            tool_name=pt.tool_name,
            status="failed" if failed else "completed",
            output=_stream_text(new),
            result=result,
        )

    # --- Turn-boundary drain ---

    async def drain(
        self, *, exec_id: str, ctx: RunContext[CtxT]
    ) -> AsyncIterator[Event[Any]]:
        """
        Re-emit buffered events for answer-blocking tasks (sub-agent progress),
        then deliver one completion notification per finished task.

        Delivery is uniform regardless of how the task was backgrounded (spawn
        vs deadline): the note inlines the task's result (or error) and the task
        is dropped. A result larger than the tool's ``max_inline_result_chars``
        is excerpted in the note with a pointer to ``TaskOutput`` and the
        finished task is *kept* so the full result can still be pulled
        (cap-and-defer). Either way the task is ``announced`` once, so it stops
        gating the final answer.
        """
        # Bubble new events for answer-blocking tasks — their progress is shown
        # live in the parent stream. A non-blocking task's output is pulled on
        # demand via ``TaskOutput``, so it is not bubbled.
        for pt in list(self._tasks.values()):
            if not pt.blocks_final_answer:
                continue
            while pt.bubble_cursor < len(pt.events):
                event = pt.events[pt.bubble_cursor]
                pt.bubble_cursor += 1
                yield event

        while not self._completions.empty():
            task_id = self._completions.get_nowait()
            pt = self._tasks.get(task_id)
            if pt is None or pt.announced:
                continue

            result, failed = _result_of(pt.events)
            status = "failed" if failed else "completed"
            # Inline the result (or error); excerpt + defer to TaskOutput when it
            # exceeds the tool's cap.
            body, truncated = _excerpt_for_inline(
                str(result), pt.max_inline_result_chars, task_id=pt.task_id
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
                            "result": None if failed else str(result),
                            "error": str(result) if failed else None,
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
            if not truncated:
                self._tasks.pop(task_id, None)  # full result delivered → drop
            # else: retain so the full result is still readable via TaskOutput.

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

            # Terminal states — already handled
            if record.status in {
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
                TaskStatus.DELIVERED,
            }:
                continue

            if record.status == TaskStatus.PENDING:
                if self._try_respawn_child(
                    record, task_key=key, ctx=ctx, exec_id=exec_id, agent_ctx=agent_ctx
                ):
                    continue

                notification = InputMessageItem.from_text(
                    _task_notification(
                        task_id=record.task_id,
                        tool_name=record.tool_name,
                        status="interrupted",
                        error="Session restarted before completion",
                    ),
                    role="user",
                )
                record = record.model_copy(
                    update={
                        "status": TaskStatus.FAILED,
                        "error": "Interrupted: session restarted",
                        "updated_at": datetime.now(UTC),
                    }
                )
            elif record.status == TaskStatus.COMPLETED:
                # Completed between drain and checkpoint — re-inject.
                notification = InputMessageItem.from_text(
                    _task_notification(
                        task_id=record.task_id,
                        tool_name=record.tool_name,
                        status="completed",
                        result=record.result,
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
                "the session stopped; for any reported as interrupted, inspect "
                "its output (TaskOutput) and decide whether to redo the work.",
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
            PendingTask(
                task_id=record.task_id,
                tool_name=record.tool_name,
                exec_id=exec_id,
                tool_call_id=record.tool_call_id,
                task=async_task,
                events=events,
                blocks_final_answer=tool.blocks_final_answer,
                max_inline_result_chars=tool.max_inline_result_chars,
                task_key=task_key,
            )
        )

        logger.info(
            "Re-spawned child task %s (%s) at task key %s",
            record.task_id,
            record.tool_name,
            task_key,
        )
        return True
