import asyncio
import logging
from typing import Any, Generic, Protocol, TypeVar

from .agent_message import AgentMessage
from .run_context import CtxT, RunContextWrapper
from .typing.io import AgentID, AgentState

logger = logging.getLogger(__name__)


_MH_PayloadT = TypeVar("_MH_PayloadT", contravariant=True)  # noqa: PLC0105
_MH_StateT = TypeVar("_MH_StateT", bound=AgentState, contravariant=True)  # noqa: PLC0105


class MessageHandler(Protocol[_MH_PayloadT, _MH_StateT, CtxT]):
    async def __call__(
        self,
        message: AgentMessage[_MH_PayloadT, _MH_StateT],
        ctx: RunContextWrapper[CtxT] | None,
        **kwargs: Any,
    ) -> None: ...


class AgentMessagePool(Generic[CtxT]):
    def __init__(self) -> None:
        self._queues: dict[AgentID, asyncio.Queue[AgentMessage[Any, AgentState]]] = {}
        self._message_handlers: dict[
            AgentID, MessageHandler[Any, AgentState, CtxT]
        ] = {}
        self._tasks: dict[AgentID, asyncio.Task[None]] = {}

    async def post(self, message: AgentMessage[Any, AgentState]) -> None:
        for recipient_id in message.recipient_ids:
            queue = self._queues.setdefault(recipient_id, asyncio.Queue())
            await queue.put(message)

    def register_message_handler(
        self,
        agent_id: AgentID,
        handler: MessageHandler[Any, AgentState, CtxT],
        ctx: RunContextWrapper[CtxT] | None = None,
        **run_kwargs: Any,
    ) -> None:
        self._message_handlers[agent_id] = handler
        self._queues.setdefault(agent_id, asyncio.Queue())
        if agent_id not in self._tasks:
            self._tasks[agent_id] = asyncio.create_task(
                self._process_messages(agent_id, ctx=ctx, **run_kwargs)
            )

    async def _process_messages(
        self,
        agent_id: AgentID,
        ctx: RunContextWrapper[CtxT] | None = None,
        **run_kwargs: Any,
    ) -> None:
        queue = self._queues[agent_id]
        while True:
            try:
                message = await queue.get()
                handler = self._message_handlers.get(agent_id)
                if handler is None:
                    break

                try:
                    await self._message_handlers[agent_id](
                        message, ctx=ctx, **run_kwargs
                    )
                except Exception:
                    logger.exception(f"Error handling message for {agent_id}")

                queue.task_done()

            except Exception:
                logger.exception(f"Unexpected error in processing loop for {agent_id}")

    async def unregister_message_handler(self, agent_id: AgentID) -> None:
        if task := self._tasks.get(agent_id):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.debug(f"{agent_id} exited")

        self._tasks.pop(agent_id, None)
        self._queues.pop(agent_id, None)
        self._message_handlers.pop(agent_id, None)

    async def stop_all(self) -> None:
        for agent_id in list(self._tasks):
            await self.unregister_message_handler(agent_id)
