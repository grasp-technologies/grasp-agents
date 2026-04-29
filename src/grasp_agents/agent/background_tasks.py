import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from logging import getLogger
from typing import Any, Generic
from uuid import uuid4

from pydantic import BaseModel

from grasp_agents.durability.checkpoints import CheckpointKind
from grasp_agents.durability.store_keys import (
    is_lifecycle_key,
    make_lifecycle_key,
    make_tool_call_path,
    session_prefix,
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

from .llm_agent_memory import LLMAgentMemory

logger = getLogger(__name__)


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
    """A background tool execution tracked by the manager."""

    task_id: str
    tool_name: str
    exec_id: str
    tool_call_id: str
    task: asyncio.Task[Any]
    # Lifecycle store key for this task's TaskRecord; ``None`` when no
    # store / no checkpoint kind, in which case status updates are no-ops.
    lifecycle_key: str | None = None
    events: asyncio.Queue[Event[Any] | None] = field(
        default_factory=lambda: asyncio.Queue[Event[Any] | None]()  # noqa: PLW0108
    )


async def _stream_to_result(
    stream: AsyncIterator[Event[Any]],
    queue: asyncio.Queue[Event[Any] | None],
) -> Any:
    """Run a tool's run_stream, forwarding events to queue. Return result."""
    result: Any = None
    async for event in stream:
        if isinstance(event, ToolOutputEvent):
            result = event.data
        elif isinstance(event, ToolErrorEvent):
            result = event.data  # ToolErrorInfo
        await queue.put(event)
    await queue.put(None)  # sentinel: stream finished
    return result


class BackgroundTaskManager(Generic[CtxT]):
    """Manages background tool executions: spawn, drain, cancel, resume."""

    def __init__(
        self,
        *,
        agent_name: str,
        memory: LLMAgentMemory,
        tools: dict[str, BaseTool[BaseModel, Any, CtxT]] | None,
        session_path: list[str] | None = None,
        parent_kind: CheckpointKind | None = CheckpointKind.AGENT,
    ) -> None:
        self._agent_name = agent_name
        self._memory = memory
        self._tools = tools
        self._session_path = session_path
        # Kind of the processor that owns this manager — used to compose
        # lifecycle keys under the parent's namespace.
        self._parent_kind = parent_kind
        self._tasks: dict[str, PendingTask] = {}

    @property
    def has_pending(self) -> bool:
        return bool(self._tasks)

    async def spawn(
        self,
        call: FunctionToolCallItem,
        tool: BaseTool[BaseModel, Any, CtxT],
        inp: BaseModel,
        *,
        ctx: RunContext[CtxT],
        exec_id: str,
    ) -> tuple[str, BackgroundTaskLaunchedEvent]:
        task_id = str(uuid4())[:8]
        store = ctx.checkpoint_store

        # Path the spawned tool runs at; lifecycle key sits in the parent's
        # kind namespace so every bg invocation has a record regardless of
        # whether the wrapped tool is itself resumable.
        child_session_path = make_tool_call_path(self._session_path, call.call_id)
        lifecycle_key: str | None = None
        if (
            store is not None
            and child_session_path is not None
            and self._parent_kind is not None
        ):
            lifecycle_key = make_lifecycle_key(
                ctx.session_key, self._parent_kind, child_session_path
            )
            record = TaskRecord(
                session_key=ctx.session_key,
                task_id=task_id,
                tool_call_id=call.call_id,
                tool_name=call.name,
                tool_call_arguments=call.arguments,
                status=TaskStatus.PENDING,
            )
            await store.save(lifecycle_key, record.model_dump_json().encode())

        queue: asyncio.Queue[Event[Any] | None] = asyncio.Queue()
        stream = tool.run_stream(
            inp,
            ctx=ctx,
            exec_id=exec_id,
            session_path=child_session_path,
        )
        async_task = asyncio.create_task(_stream_to_result(stream, queue))

        self._tasks[task_id] = PendingTask(
            task_id=task_id,
            tool_name=call.name,
            exec_id=exec_id,
            tool_call_id=call.call_id,
            task=async_task,
            lifecycle_key=lifecycle_key,
            events=queue,
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
        self, *, wait: bool, exec_id: str, ctx: RunContext[CtxT]
    ) -> AsyncIterator[Event[Any]]:
        """Check for completed background tasks, optionally waiting."""
        if not self._tasks:
            return

        if wait:
            pending = {pt.task for pt in self._tasks.values()}
            await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

        # Yield buffered subagent events from all tasks
        for pt in self._tasks.values():
            while not pt.events.empty():
                event = pt.events.get_nowait()
                if event is not None:
                    yield event

        # Handle completed tasks
        completed_ids = [tid for tid, pt in self._tasks.items() if pt.task.done()]

        for tid in completed_ids:
            pt = self._tasks.pop(tid)
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
                    result=None if failed else str(result),
                    error=str(result) if failed else None,
                ),
                role="user",
            )
            self._memory.update([notification])

            # Mark record as DELIVERED. Records are kept for post-hoc
            # observability / debugging; run ``prune_delivered(older_than)``
            # to reclaim space once they're no longer interesting.
            if ctx.checkpoint_store is not None and pt.lifecycle_key is not None:
                existing = await ctx.checkpoint_store.load(pt.lifecycle_key)

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
                        pt.lifecycle_key, record.model_dump_json().encode()
                    )

            yield BackgroundTaskCompletedEvent(
                source=self._agent_name,
                exec_id=exec_id,
                data=BackgroundTaskInfo(
                    task_id=pt.task_id,
                    tool_name=pt.tool_name,
                    tool_call_id=pt.tool_call_id,
                ),
            )
            yield UserMessageEvent(
                source=pt.tool_name,
                destination=self._agent_name,
                exec_id=exec_id,
                data=notification,
            )

    # --- Cancel ---

    async def cancel_all(self, ctx: RunContext[CtxT] | None = None) -> None:
        """Cancel all pending background tasks and wait for cleanup."""
        store = ctx.checkpoint_store if ctx else None

        if store is not None and ctx is not None:
            for pt in self._tasks.values():
                if pt.lifecycle_key is None:
                    continue

                existing = await store.load(pt.lifecycle_key)
                if existing:
                    record = TaskRecord.model_validate_json(existing)
                    record = record.model_copy(
                        update={
                            "status": TaskStatus.CANCELLED,
                            "error": "Cancelled: agent loop terminated",
                            "updated_at": datetime.now(UTC),
                        }
                    )
                    await store.save(
                        pt.lifecycle_key, record.model_dump_json().encode()
                    )

        tasks = [pt.task for pt in self._tasks.values()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    # --- Session resume ---

    async def handle_pending(
        self,
        *,
        ctx: RunContext[CtxT] | None = None,
        exec_id: str | None = None,
    ) -> None:
        """On resume, re-spawn or notify about interrupted background tasks."""
        store = ctx.checkpoint_store if ctx else None
        if store is None or ctx is None:
            return

        # Only this session's keys, only the lifecycle leaves.
        keys = [
            k
            for k in await store.list_keys(session_prefix(ctx.session_key))
            if is_lifecycle_key(k)
        ]

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
                    record, lifecycle_key=key, ctx=ctx, exec_id=exec_id
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
            self._memory.update([notification])

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

        Records are normally deleted at delivery (see :meth:`drain`), so
        this is primarily for stragglers left by older code or crashed
        deliveries. Returns the number of records pruned. Short-circuits
        with ``0`` when no store is attached.
        """
        store = ctx.checkpoint_store
        if store is None:
            return 0

        cutoff = datetime.now(UTC) - older_than
        pruned = 0
        keys = [
            k
            for k in await store.list_keys(session_prefix(ctx.session_key))
            if is_lifecycle_key(k)
        ]

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
        lifecycle_key: str,
        ctx: RunContext[CtxT] | None,
        exec_id: str | None,
    ) -> bool:
        """Re-spawn a child task from its session checkpoint."""
        if not ctx or not exec_id or not ctx.checkpoint_store:
            return False

        tool = self._tools.get(record.tool_name) if self._tools else None
        if not tool or not tool.resumable:
            return False

        child_session_path = make_tool_call_path(
            self._session_path, record.tool_call_id
        )

        queue: asyncio.Queue[Event[Any] | None] = asyncio.Queue()
        stream = tool.resume_stream(
            ctx=ctx,
            exec_id=exec_id,
            session_path=child_session_path,
        )
        async_task = asyncio.create_task(_stream_to_result(stream, queue))

        self._tasks[record.task_id] = PendingTask(
            task_id=record.task_id,
            tool_name=record.tool_name,
            exec_id=exec_id,
            tool_call_id=record.tool_call_id,
            task=async_task,
            lifecycle_key=lifecycle_key,
            events=queue,
        )

        logger.info(
            "Re-spawned child task %s (%s) at lifecycle key %s",
            record.task_id,
            record.tool_name,
            lifecycle_key,
        )
        return True
