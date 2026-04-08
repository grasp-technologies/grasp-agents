import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from logging import getLogger
from typing import TYPE_CHECKING, Any, Generic
from uuid import uuid4

from pydantic import BaseModel

from .llm_agent_memory import LLMAgentMemory
from .processors.processor_tool import ProcessorTool
from .run_context import CtxT, RunContext
from .sessions.task_record import TaskRecord, TaskStatus
from .types.events import (
    BackgroundTaskCompletedEvent,
    BackgroundTaskInfo,
    BackgroundTaskLaunchedEvent,
    Event,
    ToolErrorInfo,
    UserMessageEvent,
)
from .types.items import FunctionToolCallItem, InputMessageItem
from .types.tool import BaseTool

if TYPE_CHECKING:
    from .sessions.store import CheckpointStore

logger = getLogger(__name__)


@dataclass
class PendingTask:
    """A background tool execution tracked by the manager."""

    task_id: str
    tool_name: str
    exec_id: str
    tool_call_id: str
    task: asyncio.Task[Any]


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

        # Session persistence — set by LLMAgent when enabled
        self.store: CheckpointStore | None = None
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

        # Session-capable processor tools get their own child session
        is_session_tool = (
            isinstance(tool, ProcessorTool)
            and tool.resumable
            and self.store
            and self.session_id
        )
        if is_session_tool:
            child_session_id = f"child/{self.session_id}/{task_id}"

        # Persist task record BEFORE spawning — survives crashes
        if self.store and self.session_id:
            record = TaskRecord(
                task_id=task_id,
                parent_session_id=self.session_id,
                tool_call_id=call.call_id,
                tool_name=call.name,
                child_session_id=child_session_id,
            )
            await self.store.save(record.store_key, record.model_dump_json().encode())

        async_task: asyncio.Task[Any]
        if child_session_id and isinstance(tool, ProcessorTool) and self.store:
            child_proc = tool.processor.copy()
            child_proc.configure_session(child_session_id, self.store)  # LLMAgent
            async_task = asyncio.create_task(
                child_proc.run(in_args=inp, exec_id=exec_id, ctx=ctx)
            )
        else:
            async_task = asyncio.create_task(
                tool.run(inp, ctx=ctx, exec_id=exec_id, _validated=True)
            )

        self._tasks[task_id] = PendingTask(
            task_id=task_id,
            tool_name=call.name,
            exec_id=exec_id,
            tool_call_id=call.call_id,
            task=async_task,
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

    async def drain(self, *, wait: bool, exec_id: str) -> AsyncIterator[Event[Any]]:
        """Check for completed background tasks, optionally waiting."""
        if not self._tasks:
            return

        if wait:
            pending = {pt.task for pt in self._tasks.values()}
            await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

        completed_ids = [tid for tid, pt in self._tasks.items() if pt.task.done()]

        for tid in completed_ids:
            pt = self._tasks.pop(tid)
            try:
                result = pt.task.result()
                failed = isinstance(result, ToolErrorInfo)
            except Exception as exc:
                result = exc
                failed = True

            status = "failed" if failed else "completed"
            notification = InputMessageItem.from_text(
                f"[Background tool '{pt.tool_name}' {status}"
                f" (id: {pt.task_id})]\n\n{result}",
                role="user",
            )
            self._memory.update([notification])

            # Mark record as DELIVERED — notification reached memory
            if self.store and self.session_id:
                key = f"task/{self.session_id}/{pt.task_id}"
                existing = await self.store.load(key)
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
                    await self.store.save(key, record.model_dump_json().encode())

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
                source=self._agent_name,
                exec_id=exec_id,
                data=notification,
            )

    # --- Cancel ---

    async def cancel_all(self) -> None:
        """Cancel all pending background tasks and wait for cleanup."""
        if self.store and self.session_id:
            for pt in self._tasks.values():
                key = f"task/{self.session_id}/{pt.task_id}"
                existing = await self.store.load(key)
                if existing:
                    record = TaskRecord.model_validate_json(existing)
                    record = record.model_copy(
                        update={
                            "status": TaskStatus.CANCELLED,
                            "error": "Cancelled: agent loop terminated",
                            "updated_at": datetime.now(UTC),
                        }
                    )
                    await self.store.save(key, record.model_dump_json().encode())

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
        if not self.store or not self.session_id:
            return

        keys = await self.store.list_keys(f"task/{self.session_id}/")
        for key in keys:
            data = await self.store.load(key)
            if data is None:
                continue

            record = TaskRecord.model_validate_json(data)

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
                    f"[Background tool '{record.tool_name}' was interrupted "
                    f"(id: {record.task_id}) and did not complete]",
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
                    f"[Background tool '{record.tool_name}' completed "
                    f"(id: {record.task_id})]\n\n{record.result}",
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
            await self.store.save(record.store_key, record.model_dump_json().encode())

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
        """Re-spawn a child task from its checkpoint if possible."""
        if not record.child_session_id or not ctx or not exec_id:
            return False
        if not self.store:
            return False

        tool = self._tools.get(record.tool_name) if self._tools else None
        if not isinstance(tool, ProcessorTool) or not tool.resumable:
            return False

        child_proc = tool.processor.copy()
        child_proc.configure_session(record.child_session_id, self.store)

        async_task: asyncio.Task[Any] = asyncio.create_task(
            child_proc.resume(ctx=ctx, exec_id=exec_id)
        )
        self._tasks[record.task_id] = PendingTask(
            task_id=record.task_id,
            tool_name=record.tool_name,
            exec_id=exec_id,
            tool_call_id=record.tool_call_id,
            task=async_task,
        )
        logger.info(
            "Re-spawned child task %s (%s) from session %s",
            record.task_id,
            record.tool_name,
            record.child_session_id,
        )
        return True
