"""Tests pinning where ``on_after_generate`` fires when the policy executor
runs out of turns and falls back to ``_force_generate_final_answer_stream``.

The contract under test:
- ``_force_generate_final_answer_stream`` is a pure event producer; consumed in
  isolation, it must NOT invoke the hook.
- ``execute_stream`` is the orchestrator; after consuming the force-generate
  stream it MUST fire ``on_after_generate`` exactly once, with the gen message
  produced by the force-generate phase and the current ``num_turns``.
- Events emitted by the force-generate stream must be forwarded to the outer
  consumer of ``execute_stream``.
- The hook must fire AFTER all force-generate events have been delivered, not
  interleaved with them.
"""

import sys
import unittest
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pydantic import BaseModel

from grasp_agents.llm_agent_memory import LLMAgentMemory
from grasp_agents.llm_policy_executor import LLMPolicyExecutor
from grasp_agents.run_context import RunContext
from grasp_agents.typing.events import (
    Event,
    GenMessageEvent,
    UserMessageEvent,
)
from grasp_agents.typing.message import AssistantMessage
from grasp_agents.typing.tool import BaseTool, ToolChoice


# ---------------------- Stubs ----------------------


class _StubLLM:
    """Stand-in for the ``LLM`` argument. Never invoked because the test
    subclass overrides ``generate_message_stream`` to bypass the LLM entirely.
    """

    model_name: str = "stub-model"


class _DummyToolIn(BaseModel):
    pass


class _DummyTool(BaseTool[_DummyToolIn, None, Any]):
    name: str = "dummy_tool"
    description: str = "Makes ``executor.tools`` non-empty so the loop runs."

    async def run(
        self,
        inp: _DummyToolIn,
        *,
        ctx: RunContext[Any] | None = None,
        call_id: str | None = None,
    ) -> None:
        return None


class _StubExecutor(LLMPolicyExecutor[Any]):
    """Bypasses the LLM by yielding canned ``GenMessageEvent``s in the same
    order they appear in ``canned_messages``. Updates memory the same way the
    real ``generate_message_stream`` does, so ``get_final_answer`` works.
    """

    def __init__(
        self,
        *args: Any,
        canned_messages: list[AssistantMessage],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._canned_messages = list(canned_messages)
        self._gen_count = 0

    async def generate_message_stream(  # type: ignore[override]
        self,
        *,
        tool_choice: ToolChoice | None = None,
        ctx: RunContext[Any],
        call_id: str,
        extra_llm_settings: dict[str, Any],
    ) -> AsyncIterator[Event[Any]]:
        msg = self._canned_messages[self._gen_count]
        self._gen_count += 1
        self.memory.update([msg])
        yield GenMessageEvent(src_name=self.agent_name, call_id=call_id, data=msg)


# ---------------------- Test helpers ----------------------


def _make_executor(
    *,
    max_turns: int,
    canned_messages: list[AssistantMessage],
    with_tools: bool = True,
) -> _StubExecutor:
    tools = [_DummyTool()] if with_tools else None
    return _StubExecutor(
        agent_name="test_agent",
        llm=_StubLLM(),  # type: ignore[arg-type]
        memory=LLMAgentMemory(),
        tools=tools,
        max_turns=max_turns,
        canned_messages=canned_messages,
    )


def _msg(content: str) -> AssistantMessage:
    return AssistantMessage(content=content)


class _HookRecorder:
    """Records every invocation of ``on_after_generate_impl`` together with the
    list of events that have already been delivered to the outer consumer at
    hook time. The latter is what lets the ordering test prove the hook fires
    AFTER all preceding events have been yielded.
    """

    def __init__(self, observed_events: list[Event[Any]]) -> None:
        self.calls: list[dict[str, Any]] = []
        self._observed_events = observed_events

    async def __call__(
        self,
        gen_message: AssistantMessage,
        *,
        ctx: RunContext[Any],
        call_id: str,
        num_turns: int,
    ) -> None:
        self.calls.append(
            {
                "gen_message": gen_message,
                "num_turns": num_turns,
                "call_id": call_id,
                "events_seen_so_far": list(self._observed_events),
            }
        )


# ---------------------- Tests ----------------------


class TestForceGenerateAfterGenerateHook(unittest.IsolatedAsyncioTestCase):
    async def test_force_generate_stream_does_not_fire_hook(self) -> None:
        # Pure-producer invariant: calling the force-generate stream directly
        # must not invoke the hook. The hook is the orchestrator's
        # responsibility, not the producer's.
        executor = _make_executor(
            max_turns=1, canned_messages=[_msg("forced")]
        )
        recorder = _HookRecorder(observed_events=[])
        executor.on_after_generate_impl = recorder  # type: ignore[method-assign]

        ctx: RunContext[Any] = RunContext()
        events: list[Event[Any]] = []
        async for event in executor._force_generate_final_answer_stream(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            ctx=ctx, call_id="cid", extra_llm_settings={}
        ):
            events.append(event)

        self.assertEqual(recorder.calls, [])
        # And the producer still emits its expected events.
        self.assertTrue(any(isinstance(e, UserMessageEvent) for e in events))
        self.assertTrue(any(isinstance(e, GenMessageEvent) for e in events))

    async def test_execute_stream_fires_hook_after_force_generate(self) -> None:
        # With max_turns=0 the executor: (1) generates once, (2) enters the
        # loop, sees turns >= max_turns, (3) runs the force-generate path and
        # returns. The hook must fire after each generation — twice total —
        # and the second call must carry the *force-generate's* gen message.
        first = _msg("first turn")
        forced = _msg("forced final answer")
        executor = _make_executor(
            max_turns=0, canned_messages=[first, forced]
        )

        observed: list[Event[Any]] = []
        recorder = _HookRecorder(observed_events=observed)
        executor.on_after_generate_impl = recorder  # type: ignore[method-assign]

        ctx: RunContext[Any] = RunContext()
        async for event in executor.execute_stream(ctx=ctx, call_id="cid"):
            observed.append(event)

        self.assertEqual(len(recorder.calls), 2)
        first_call, second_call = recorder.calls

        self.assertEqual(first_call["gen_message"], first)
        self.assertEqual(first_call["num_turns"], 0)
        self.assertEqual(first_call["call_id"], "cid")

        self.assertEqual(second_call["gen_message"], forced)
        self.assertEqual(second_call["num_turns"], 0)
        self.assertEqual(second_call["call_id"], "cid")

    async def test_force_generate_events_are_forwarded(self) -> None:
        # Events from the force-generate stream (synthetic user prompt + the
        # gen message) must reach the outer consumer of ``execute_stream``.
        first = _msg("first turn")
        forced = _msg("forced final answer")
        executor = _make_executor(
            max_turns=0, canned_messages=[first, forced]
        )

        ctx: RunContext[Any] = RunContext()
        events: list[Event[Any]] = []
        async for event in executor.execute_stream(ctx=ctx, call_id="cid"):
            events.append(event)

        gen_events = [e for e in events if isinstance(e, GenMessageEvent)]
        user_events = [e for e in events if isinstance(e, UserMessageEvent)]

        self.assertEqual(len(gen_events), 2)
        self.assertEqual(gen_events[0].data, first)
        self.assertEqual(gen_events[1].data, forced)

        self.assertEqual(len(user_events), 1)
        self.assertIn(
            "Exceeded the maximum number of turns",
            str(user_events[0].data.content),
        )

    async def test_hook_fires_after_all_force_generate_events_yielded(
        self,
    ) -> None:
        # Ordering invariant: when the hook fires, every event from the
        # just-completed generation must already be visible to the consumer.
        # Proving this rules out any "hook runs mid-stream" regression.
        first = _msg("first turn")
        forced = _msg("forced final answer")
        executor = _make_executor(
            max_turns=0, canned_messages=[first, forced]
        )

        observed: list[Event[Any]] = []
        recorder = _HookRecorder(observed_events=observed)
        executor.on_after_generate_impl = recorder  # type: ignore[method-assign]

        ctx: RunContext[Any] = RunContext()
        async for event in executor.execute_stream(ctx=ctx, call_id="cid"):
            observed.append(event)

        # First hook fires after the first generation: 1 GenMessageEvent
        # has been delivered.
        first_seen = recorder.calls[0]["events_seen_so_far"]
        self.assertEqual(len(first_seen), 1)
        self.assertIsInstance(first_seen[0], GenMessageEvent)

        # Second hook fires after the force-generate: by then the consumer
        # has seen all three events in order — Gen, User, Gen.
        second_seen = recorder.calls[1]["events_seen_so_far"]
        self.assertEqual(len(second_seen), 3)
        self.assertIsInstance(second_seen[0], GenMessageEvent)
        self.assertIsInstance(second_seen[1], UserMessageEvent)
        self.assertIsInstance(second_seen[2], GenMessageEvent)


if __name__ == "__main__":
    unittest.main()
