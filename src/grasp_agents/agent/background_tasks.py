import asyncio
import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from logging import getLogger
from typing import Any, Generic, Protocol, runtime_checkable
from uuid import uuid4

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
    ToolErrorInfo,
    ToolOutputEvent,
    UserMessageEvent,
)
from grasp_agents.types.items import FunctionToolCallItem, InputMessageItem
from grasp_agents.types.tool import BaseTool

from .llm_agent_transcript import LLMAgentTranscript

logger = getLogger(__name__)


@runtime_checkable
class BackgroundMonitor(Protocol):
    """
    A *pollable* background unit handed to :meth:`BackgroundTaskManager.track`.

    The manager uses only :meth:`summary` (for the turn-boundary completion
    note); the tool that created the monitor reads / stops it through its own
    concrete type. This is the single seam between the general task manager and
    a tool-specific background unit (e.g. a shell command's output buffer) — the
    manager stays kind-agnostic.
    """

    def summary(self) -> str:
        """A short, one-line completion summary for the notification."""
        ...


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
    A background unit of work tracked by the manager — kind-agnostic.

    A subagent-as-tool runs *durably* (``task_key`` set, ``blocks_final_answer``
    True, result delivered inline in the completion note). A backgrounded shell
    command is the same shape but *ephemeral* (no ``task_key`` — an OS process
    can't be resumed), *non-blocking*, and *pollable*: it carries an opaque
    ``monitor`` the owning tool uses to read / stop it, and its output is read
    back through that tool rather than inlined. The manager treats every task
    through these general fields and never inspects ``monitor``.
    """

    task_id: str
    tool_name: str
    exec_id: str
    task: asyncio.Task[Any]
    # Its completion gates the final answer (its result is part of the answer).
    blocks_final_answer: bool = True
    tool_call_id: str | None = None
    # Lifecycle store key for this task's TaskRecord; ``None`` when ephemeral
    # (no store / no checkpoint kind), in which case status updates are no-ops.
    task_key: str | None = None
    # Buffered events to bubble at the turn boundary (subagent tasks); ``None``
    # for tasks that deliver their output another way (e.g. a polled command).
    events: asyncio.Queue[Event[Any] | None] | None = None
    # Handle the owning tool uses to read / stop a *pollable* task; the manager
    # only calls ``monitor.summary()``. Its presence marks the task pollable:
    # the result is not inlined in the completion note, and the task is kept
    # after announcement so the owning tool can still read it.
    monitor: BackgroundMonitor | None = None
    announced: bool = False


async def _stream_to_result(
    stream: AsyncIterator[Event[Any]],
    queue: asyncio.Queue[Event[Any] | None],
    *,
    store: CheckpointStore | None = None,
    task_key: str | None = None,
) -> Any:
    """
    Run a tool's run_stream, forwarding events to ``queue``. Return result.

    On a successful result, persist a ``COMPLETED`` TaskRecord carrying the
    result *before* signalling completion. This closes the window between a
    task finishing and :meth:`BackgroundTaskManager.drain` delivering it: a
    crash in that window leaves a ``COMPLETED`` record that resume re-injects
    (see :meth:`handle_pending`), instead of a ``PENDING`` record that forces
    a re-run or loses the result.
    """
    result: Any = None
    failed = False
    async for event in stream:
        if isinstance(event, ToolOutputEvent):
            result = event.data
        elif isinstance(event, ToolErrorEvent):
            result = event.data  # ToolErrorInfo
            failed = True
        await queue.put(event)

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

    await queue.put(None)  # sentinel: stream finished
    return result


class BackgroundTaskManager(Generic[CtxT]):
    """
    Tracks background work of any kind: durable, answer-blocking, resumable
    tasks created with :meth:`spawn_durable` (e.g. a subagent-as-tool), and
    ephemeral, non-blocking ones registered with :meth:`track` (e.g. a
    backgrounded shell command). One drain / idle-wait / cancel path serves
    both; only the durable methods (``spawn_durable`` / ``resume_durable`` /
    ``prune_delivered``) touch the checkpoint store.
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
        # Completed task ids, pushed by each task's done-callback (Hermes-style
        # single completion seam): :meth:`drain` pops them to deliver notes, and
        # :meth:`wait_idle` blocks on the next one — one queue, every task kind.
        self._completions: asyncio.Queue[str] = asyncio.Queue()

    @property
    def has_pending(self) -> bool:
        """
        True while any *answer-blocking* background task is pending.

        Consulted by the JUDGE phase: such a task's result is part of the
        answer, so it gates even a final answer. Non-blocking tasks (e.g. a
        backgrounded shell command) are excluded — they must never hold the run
        hostage.
        """
        return any(pt.blocks_final_answer for pt in self._tasks.values())

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

    # --- generic tracking of externally-started background work ---

    def track(
        self,
        task: asyncio.Task[Any],
        *,
        label: str,
        exec_id: str,
        blocks_final_answer: bool = False,
        monitor: BackgroundMonitor | None = None,
        tool_call_id: str | None = None,
        id_prefix: str = "bg",
    ) -> str:
        """
        Register an already-running background task and return its id.

        Unlike :meth:`spawn`, ``track`` neither starts the work nor persists a
        ``TaskRecord``: the caller owns the lifecycle (the task dies with the
        host — nothing to resume), and the manager only tracks it for the idle
        wait, the turn-boundary completion note, and cancel-on-exit. ``monitor``
        is the handle the owning tool reads / stops; see :class:`PendingTask`.
        """
        if len(self._tasks) >= self._max_background:
            raise RuntimeError(
                f"Too many background tasks ({self._max_background}); kill or "
                "drain existing ones first."
            )
        self._bg_counter += 1
        task_id = f"{id_prefix}_{self._bg_counter}"
        self._register(
            PendingTask(
                task_id=task_id,
                tool_name=label,
                exec_id=exec_id,
                task=task,
                blocks_final_answer=blocks_final_answer,
                tool_call_id=tool_call_id,
                monitor=monitor,
            )
        )
        return task_id

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

    async def cancel(self, task_id: str) -> PendingTask:
        """Cancel one task's future; the owning tool reads any final output."""
        pt = self.get(task_id)
        if not pt.task.done():
            pt.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pt.task
        return pt

    def _register(self, pt: PendingTask) -> None:
        """Track a task and push its id onto the completion queue when it ends."""
        self._tasks[pt.task_id] = pt
        pt.task.add_done_callback(
            lambda _t, tid=pt.task_id: self._completions.put_nowait(tid)
        )

    async def spawn_durable(
        self,
        call: FunctionToolCallItem,
        tool: BaseTool[BaseModel, Any, CtxT],
        inp: BaseModel,
        *,
        ctx: RunContext[CtxT],
        exec_id: str,
    ) -> tuple[str, BackgroundTaskLaunchedEvent]:
        """
        Start a durable, answer-blocking background task from a tool call:
        run ``tool.run_stream`` in a task, persist a ``TaskRecord`` (so it can
        be resumed by :meth:`resume_durable`), and deliver its result inline in
        the completion note. For an ephemeral, already-running task use
        :meth:`track`.
        """
        task_id = str(uuid4())[:8]
        store = ctx.checkpoint_store

        child_path = make_tool_call_path(self._path, call.call_id)
        task_key: str | None = None
        if store is not None and child_path is not None:
            task_key = make_store_key(ctx.session_key, CheckpointKind.TASK, child_path)
            record = TaskRecord(
                session_key=ctx.session_key,
                task_id=task_id,
                tool_call_id=call.call_id,
                tool_name=call.name,
                tool_call_arguments=call.arguments,
                status=TaskStatus.PENDING,
            )
            await store.save(task_key, record.model_dump_json().encode())

        queue: asyncio.Queue[Event[Any] | None] = asyncio.Queue()
        stream = tool.run_stream(
            inp,
            ctx=ctx,
            exec_id=exec_id,
            path=child_path,
        )
        async_task = asyncio.create_task(
            _stream_to_result(stream, queue, store=store, task_key=task_key)
        )

        self._register(
            PendingTask(
                task_id=task_id,
                tool_name=call.name,
                exec_id=exec_id,
                tool_call_id=call.call_id,
                task=async_task,
                task_key=task_key,
                events=queue,
            )
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
        return task_id, event

    async def drain(
        self, *, exec_id: str, ctx: RunContext[CtxT]
    ) -> AsyncIterator[Event[Any]]:
        """
        Bubble buffered events, then deliver a completion notification for each
        finished task (its id pushed onto the completion queue by a done
        callback).

        A *pollable* task (one with a ``monitor``) is announced once and kept —
        its output is read back through its own tool; any other task has its
        result inlined in the note and is dropped.
        """
        # Bubble buffered events (event-bubbling tasks only — e.g. subagents).
        for pt in self._tasks.values():
            if pt.events is None:
                continue
            while not pt.events.empty():
                event = pt.events.get_nowait()
                if event is not None:
                    yield event

        while not self._completions.empty():
            task_id = self._completions.get_nowait()
            pt = self._tasks.get(task_id)
            if pt is None or pt.announced:
                continue
            pollable = pt.monitor is not None

            result: Any = None
            failed = False
            if not pollable:
                try:
                    result = pt.task.result()
                    failed = isinstance(result, ToolErrorInfo)
                except Exception as exc:
                    result = exc
                    failed = True

            notification = InputMessageItem.from_text(
                _task_notification(
                    task_id=pt.task_id,
                    tool_name=pt.tool_name,
                    status="failed" if failed else "completed",
                    result=(
                        pt.monitor.summary()
                        if pt.monitor is not None
                        else (None if failed else str(result))
                    ),
                    error=str(result) if failed else None,
                ),
                role="user",
            )
            self._transcript.update([notification])

            # Durable record → DELIVERED (subagent tasks only; an ephemeral task
            # has no ``task_key``). Records are kept for post-hoc observability;
            # reclaim with ``prune_delivered(older_than)``.
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

            # Pollable tasks stay (read later via their own tool), announced
            # once; any other task is delivered and dropped.
            if pollable:
                pt.announced = True
            else:
                self._tasks.pop(task_id, None)

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
    ) -> None:
        """On resume, re-spawn or notify about interrupted durable tasks."""
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
                    record, task_key=key, ctx=ctx, exec_id=exec_id
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
            self._transcript.update([notification])

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

        Unlike :meth:`handle_pending` (agent-scoped — it re-spawns tasks
        and injects into a specific agent's transcript), this is a static,
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
    ) -> bool:
        """Re-spawn a child task from its session checkpoint."""
        if not ctx or not exec_id or not ctx.checkpoint_store:
            return False

        tool = self._tools.get(record.tool_name) if self._tools else None
        if not tool or not tool.resumable:
            return False

        child_path = make_tool_call_path(self._path, record.tool_call_id)

        queue: asyncio.Queue[Event[Any] | None] = asyncio.Queue()
        stream = tool.resume_stream(
            ctx=ctx,
            exec_id=exec_id,
            path=child_path,
        )
        async_task = asyncio.create_task(
            _stream_to_result(
                stream, queue, store=ctx.checkpoint_store, task_key=task_key
            )
        )

        self._register(
            PendingTask(
                task_id=record.task_id,
                tool_name=record.tool_name,
                exec_id=exec_id,
                tool_call_id=record.tool_call_id,
                task=async_task,
                task_key=task_key,
                events=queue,
            )
        )

        logger.info(
            "Re-spawned child task %s (%s) at task key %s",
            record.task_id,
            record.tool_name,
            task_key,
        )
        return True
