import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from logging import getLogger
from typing import Any, Generic
from uuid import uuid4

from pydantic import BaseModel

from ..durability.checkpoints import CheckpointSchemaError
from ..durability.task_record import TaskRecord, TaskStatus
from ..run_context import CtxT, RunContext
from ..types.events import (
    BackgroundTaskCompletedEvent,
    BackgroundTaskInfo,
    BackgroundTaskLaunchedEvent,
    Event,
    ToolErrorEvent,
    ToolErrorInfo,
    ToolOutputEvent,
    UserMessageEvent,
)
from ..types.items import FunctionToolCallItem, InputMessageItem
from ..types.tool import BaseTool
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
    events: asyncio.Queue[Event[Any] | None] = field(
        default_factory=lambda: asyncio.Queue[Event[Any] | None]()
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
    ) -> None:
        self._agent_name = agent_name
        self._memory = memory
        self._tools = tools
        self._tasks: dict[str, PendingTask] = {}

        # Session persistence — wired by LLMAgent.setup_session()
        self.session_id: str | None = None

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
        child_session_id: str | None = None

        # Resumable tools get their own child session
        if tool.resumable and ctx.checkpoint_store and self.session_id:
            child_session_id = f"child/{self.session_id}/{task_id}"

        # Persist task record BEFORE spawning — survives crashes
        if ctx.checkpoint_store and self.session_id:
            record = TaskRecord(
                task_id=task_id,
                parent_session_id=self.session_id,
                tool_call_id=call.call_id,
                tool_name=call.name,
                tool_call_arguments=call.arguments,
                child_session_id=child_session_id,
            )
            await ctx.checkpoint_store.save(record.store_key, record.model_dump_json().encode())

        queue: asyncio.Queue[Event[Any] | None] = asyncio.Queue()
        stream = tool.run_stream(
            inp,
            ctx=ctx,
            exec_id=exec_id,
            session_id=child_session_id,
        )
        async_task = asyncio.create_task(_stream_to_result(stream, queue))

        self._tasks[task_id] = PendingTask(
            task_id=task_id,
            tool_name=call.name,
            exec_id=exec_id,
            tool_call_id=call.call_id,
            task=async_task,
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

    # --- Drain ---

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

            # Mark record as DELIVERED
            if ctx.checkpoint_store and self.session_id:
                key = f"task/{self.session_id}/{pt.task_id}"
                existing = await ctx.checkpoint_store.load(key)
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
                    await ctx.checkpoint_store.save(key, record.model_dump_json().encode())

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

        if store and self.session_id:
            for pt in self._tasks.values():
                key = f"task/{self.session_id}/{pt.task_id}"
                existing = await store.load(key)
                if existing:
                    record = TaskRecord.model_validate_json(existing)
                    record = record.model_copy(
                        update={
                            "status": TaskStatus.CANCELLED,
                            "error": "Cancelled: agent loop terminated",
                            "updated_at": datetime.now(UTC),
                        }
                    )
                    await store.save(key, record.model_dump_json().encode())

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
        if not store or not self.session_id:
            return

        keys = await store.list_keys(f"task/{self.session_id}/")
        for key in keys:
            data = await store.load(key)
            if data is None:
                continue

            try:
                record = TaskRecord.model_validate_json(data)
            except CheckpointSchemaError:
                raise
            except Exception:
                logger.warning(
                    "Corrupt task record at %s, skipping",
                    key,
                    exc_info=True,
                )
                continue

            # Terminal states — already handled
            if record.status in {
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
                TaskStatus.DELIVERED,
            }:
                continue

            if record.status == TaskStatus.PENDING:
                if self._try_respawn_child(record, ctx=ctx, exec_id=exec_id):
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
                # Completed between drain and checkpoint — re-inject
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

            self._memory.update([notification])
            await store.save(record.store_key, record.model_dump_json().encode())

        logger.info(
            "Handled %d task records for session %s",
            len(keys),
            self.session_id,
        )

    def _try_respawn_child(
        self,
        record: TaskRecord,
        *,
        ctx: RunContext[CtxT] | None,
        exec_id: str | None,
    ) -> bool:
        """Re-spawn a child task from its session checkpoint."""
        if not record.child_session_id or not ctx or not exec_id:
            return False
        if not ctx.checkpoint_store:
            return False

        tool = self._tools.get(record.tool_name) if self._tools else None
        if not tool or not tool.resumable:
            return False

        queue: asyncio.Queue[Event[Any] | None] = asyncio.Queue()
        stream = tool.resume_stream(
            ctx=ctx,
            exec_id=exec_id,
            session_id=record.child_session_id,
        )
        async_task = asyncio.create_task(_stream_to_result(stream, queue))

        self._tasks[record.task_id] = PendingTask(
            task_id=record.task_id,
            tool_name=record.tool_name,
            exec_id=exec_id,
            tool_call_id=record.tool_call_id,
            task=async_task,
            events=queue,
        )
        logger.info(
            "Re-spawned child task %s (%s) from session %s",
            record.task_id,
            record.tool_name,
            record.child_session_id,
        )
        return True
