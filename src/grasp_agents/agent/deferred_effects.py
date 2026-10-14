from collections.abc import Awaitable, Callable, Collection
from dataclasses import dataclass


@dataclass(frozen=True, eq=False)
class DeferredEffect:
    """A tool's durable side effect, held for the checkpoint that persists its call."""

    call_id: str | None
    run: Callable[[], Awaitable[None]]


class DeferredEffects:
    """
    The agent's outbox: tool side effects waiting for the checkpoint that
    persists their calls. Tools add to it through
    :meth:`AgentContext.defer_effect`; the agent runs it inside each
    checkpoint's :meth:`~grasp_agents.durability.CheckpointStore.transaction`.

    A failing effect aborts its checkpoint and stays pending, so every later
    checkpoint retries it. Use :meth:`discard` to give up on one.
    """

    def __init__(self) -> None:
        self._pending: list[DeferredEffect] = []

    @property
    def pending(self) -> tuple[DeferredEffect, ...]:
        return tuple(self._pending)

    def add(self, effect: DeferredEffect) -> None:
        self._pending.append(effect)

    async def run(self, *, retire_each: bool) -> int:
        """
        Run the pending effects in registration order; returns how many now
        await :meth:`commit`.

        ``retire_each`` drops each effect as soon as it succeeds — for a store
        without transactions, which cannot undo it, so a later failure in the
        same checkpoint must not run it again. Otherwise the batch stays
        pending until the checkpoint lands: a failure anywhere in it rolled
        their writes back too, and the next checkpoint retries them together.
        """
        batch = list(self._pending)
        for effect in batch:
            try:
                await effect.run()
            except Exception as exc:
                exc.add_note(
                    f"Deferred effect of tool call {effect.call_id!r} failed; it "
                    "stays pending and is retried at the next checkpoint. Drop it "
                    f"with agent.agent_ctx.effects.discard({effect.call_id!r})."
                )
                raise
            if retire_each:
                self._pending.remove(effect)
        return 0 if retire_each else len(batch)

    def commit(self, count: int) -> None:
        """Drop the ``count`` effects :meth:`run` left pending, once durable."""
        del self._pending[:count]

    def retain(self, call_ids: Collection[str]) -> None:
        """Drop effects whose tool call is not in ``call_ids``; unscoped ones stay."""
        self._pending[:] = [
            e for e in self._pending if e.call_id is None or e.call_id in call_ids
        ]

    def discard(self, call_id: str | None) -> int:
        """Drop the pending effects of one tool call (``None``: unscoped ones)."""
        before = len(self._pending)
        self._pending[:] = [e for e in self._pending if e.call_id != call_id]
        return before - len(self._pending)
