"""Tests pinning when ``on_before_generate`` and ``on_after_generate`` fire
during ``_force_generate_final_answer_stream`` and how those firings are
observed end-to-end through ``execute_stream``.

Producer contract under test:
- ``_force_generate_final_answer_stream`` MUST fire ``on_before_generate``
  exactly once, AFTER the synthetic user-prompt event has been yielded and
  BEFORE any ``GenMessageEvent`` is yielded.
- It MUST fire ``on_after_generate`` exactly once, AFTER every event from
  the underlying generation has been yielded, with the produced
  ``gen_message`` and the ``num_turns`` it was called with.
- Mutations the before-hook makes to its ``extra_llm_settings`` argument
  MUST flow into ``generate_message_stream`` — that is the contract
  subclasses (e.g. usage trackers, per-call parameter injectors) rely on.

Orchestrator contract under test:
- ``execute_stream``, when ``max_turns`` is exhausted, surfaces both
  hook firings to subscribers: a before+after pair for the first generation,
  and a before+after pair for the force-generate. Prior to PRO-2052 the
  force-generate's pair was missing, which undercounted usage.
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
    subclass overrides ``generate_message_stream``."""

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
    """Bypasses the LLM by yielding canned ``GenMessageEvent``s. Also captures
    the ``extra_llm_settings`` dict it was called with so the before-hook
    propagation test can verify mutations made by the hook reach the LLM call.
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
        self.observed_settings: list[dict[str, Any]] = []

    async def generate_message_stream(  # type: ignore[override]
        self,
        *,
        tool_choice: ToolChoice | None = None,
        ctx: RunContext[Any],
        call_id: str,
        extra_llm_settings: dict[str, Any],
    ) -> AsyncIterator[Event[Any]]:
        # Snapshot what the LLM call would see, then behave like the real
        # ``generate_message_stream``: emit one GenMessageEvent and update
        # memory so ``get_final_answer`` works.
        self.observed_settings.append(dict(extra_llm_settings))
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


class _AfterHookRecorder:
    """Records every invocation of ``on_after_generate_impl`` together with
    the events delivered to the consumer at the moment the hook fired.
    The latter pins ordering — proving the hook fires AFTER its preceding
    events have been yielded.
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


class _BeforeHookRecorder:
    """Records every invocation of ``on_before_generate_impl``. Captures both
    a snapshot of ``extra_llm_settings`` (so later mutations don't bleed into
    the record) and the live reference (so the propagation test can verify
    the hook receives the same dict the LLM call will see).
    """

    def __init__(self, observed_events: list[Event[Any]]) -> None:
        self.calls: list[dict[str, Any]] = []
        self._observed_events = observed_events

    async def __call__(
        self,
        *,
        ctx: RunContext[Any],
        call_id: str,
        num_turns: int,
        extra_llm_settings: dict[str, Any],
    ) -> None:
        self.calls.append(
            {
                "extra_llm_settings": dict(extra_llm_settings),
                "extra_llm_settings_ref": extra_llm_settings,
                "num_turns": num_turns,
                "call_id": call_id,
                "events_seen_so_far": list(self._observed_events),
            }
        )


# ---------------------- After-generate hook tests ----------------------


class TestForceGenerateAfterHook(unittest.IsolatedAsyncioTestCase):
    async def test_producer_fires_after_hook_with_correct_args(self) -> None:
        # Producer-in-isolation contract: ``on_after_generate`` fires exactly
        # once, carrying the produced gen_message and the ``num_turns`` the
        # caller passed in.
        forced = _msg("forced")
        executor = _make_executor(max_turns=1, canned_messages=[forced])

        events: list[Event[Any]] = []
        recorder = _AfterHookRecorder(observed_events=events)
        executor.on_after_generate_impl = recorder  # type: ignore[method-assign]

        ctx: RunContext[Any] = RunContext()
        async for event in executor._force_generate_final_answer_stream(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            ctx=ctx, call_id="cid", extra_llm_settings={}, num_turns=7
        ):
            events.append(event)

        self.assertEqual(len(recorder.calls), 1)
        call = recorder.calls[0]
        self.assertEqual(call["gen_message"], forced)
        self.assertEqual(call["num_turns"], 7)
        self.assertEqual(call["call_id"], "cid")

    async def test_execute_stream_fires_after_hook_for_force_generated_message(
        self,
    ) -> None:
        # End-to-end: with ``max_turns=0`` the executor (1) generates once,
        # (2) hits the force-generate branch. After-hook must fire TWICE,
        # the second call carrying the force-generated message.
        first = _msg("first turn")
        forced = _msg("forced final answer")
        executor = _make_executor(max_turns=0, canned_messages=[first, forced])

        observed: list[Event[Any]] = []
        recorder = _AfterHookRecorder(observed_events=observed)
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
        # Events produced inside the force-generate stream (synthetic user
        # prompt + the final gen message) must reach ``execute_stream``'s
        # outer consumer.
        first = _msg("first turn")
        forced = _msg("forced final answer")
        executor = _make_executor(max_turns=0, canned_messages=[first, forced])

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

    async def test_after_hook_fires_after_all_force_generate_events_yielded(
        self,
    ) -> None:
        # Ordering invariant: by the time the after-hook fires, every event
        # from the just-completed generation must already be visible to the
        # outer consumer. Rules out a "hook runs mid-stream" regression.
        first = _msg("first turn")
        forced = _msg("forced final answer")
        executor = _make_executor(max_turns=0, canned_messages=[first, forced])

        observed: list[Event[Any]] = []
        recorder = _AfterHookRecorder(observed_events=observed)
        executor.on_after_generate_impl = recorder  # type: ignore[method-assign]

        ctx: RunContext[Any] = RunContext()
        async for event in executor.execute_stream(ctx=ctx, call_id="cid"):
            observed.append(event)

        # First after-hook (regular path): one GenMessageEvent has been
        # delivered to the consumer.
        first_seen = recorder.calls[0]["events_seen_so_far"]
        self.assertEqual(len(first_seen), 1)
        self.assertIsInstance(first_seen[0], GenMessageEvent)

        # Second after-hook (force-generate path) fires from inside the
        # producer, after every one of its events has been forwarded —
        # Gen, User, Gen.
        second_seen = recorder.calls[1]["events_seen_so_far"]
        self.assertEqual(len(second_seen), 3)
        self.assertIsInstance(second_seen[0], GenMessageEvent)
        self.assertIsInstance(second_seen[1], UserMessageEvent)
        self.assertIsInstance(second_seen[2], GenMessageEvent)


# ---------------------- Before-generate hook tests ----------------------


class TestForceGenerateBeforeHook(unittest.IsolatedAsyncioTestCase):
    async def test_producer_fires_before_hook_with_correct_args(self) -> None:
        # Producer-in-isolation contract: ``on_before_generate`` fires
        # exactly once with the caller-supplied ``num_turns`` / ``call_id``,
        # and is handed the deep-copied settings dict (NOT the caller's
        # original — mutations must not leak back).
        executor = _make_executor(max_turns=1, canned_messages=[_msg("forced")])
        events: list[Event[Any]] = []
        recorder = _BeforeHookRecorder(observed_events=events)
        executor.on_before_generate_impl = recorder  # type: ignore[method-assign]

        original_settings: dict[str, Any] = {"temperature": 0.5}
        ctx: RunContext[Any] = RunContext()
        async for event in executor._force_generate_final_answer_stream(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            ctx=ctx,
            call_id="cid",
            extra_llm_settings=original_settings,
            num_turns=4,
        ):
            events.append(event)

        self.assertEqual(len(recorder.calls), 1)
        call = recorder.calls[0]
        self.assertEqual(call["num_turns"], 4)
        self.assertEqual(call["call_id"], "cid")
        self.assertEqual(call["extra_llm_settings"], {"temperature": 0.5})
        # Deep-copy boundary: the dict the hook sees is not the caller's.
        self.assertIsNot(call["extra_llm_settings_ref"], original_settings)

    async def test_before_hook_fires_after_user_event_before_gen_event(
        self,
    ) -> None:
        # Ordering inside the producer: synthetic UserMessageEvent first,
        # THEN before-hook, THEN GenMessageEvent. Pins the hook between the
        # prompt-injection and the LLM call.
        executor = _make_executor(max_turns=1, canned_messages=[_msg("forced")])
        events: list[Event[Any]] = []
        recorder = _BeforeHookRecorder(observed_events=events)
        executor.on_before_generate_impl = recorder  # type: ignore[method-assign]

        ctx: RunContext[Any] = RunContext()
        async for event in executor._force_generate_final_answer_stream(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            ctx=ctx, call_id="cid", extra_llm_settings={}, num_turns=0
        ):
            events.append(event)

        # When the before-hook fired, exactly one UserMessageEvent had been
        # delivered, and zero GenMessageEvents.
        seen = recorder.calls[0]["events_seen_so_far"]
        self.assertEqual(len(seen), 1)
        self.assertIsInstance(seen[0], UserMessageEvent)
        # And the producer ultimately yielded the gen message AFTER the hook.
        gen_events = [e for e in events if isinstance(e, GenMessageEvent)]
        self.assertEqual(len(gen_events), 1)

    async def test_before_hook_mutations_propagate_to_generate_message_stream(
        self,
    ) -> None:
        # Subclass contract: the before-hook receives a mutable settings dict
        # and any keys it adds must reach ``generate_message_stream``. This
        # is how per-call LLM parameter injection is wired.
        executor = _make_executor(max_turns=1, canned_messages=[_msg("forced")])

        async def _injecting_hook(
            *,
            ctx: RunContext[Any],
            call_id: str,
            num_turns: int,
            extra_llm_settings: dict[str, Any],
        ) -> None:
            extra_llm_settings["injected"] = "by_before_hook"

        executor.on_before_generate_impl = _injecting_hook  # type: ignore[method-assign]

        ctx: RunContext[Any] = RunContext()
        async for _ in executor._force_generate_final_answer_stream(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            ctx=ctx, call_id="cid", extra_llm_settings={}, num_turns=0
        ):
            pass

        self.assertEqual(len(executor.observed_settings), 1)
        self.assertEqual(
            executor.observed_settings[0].get("injected"), "by_before_hook"
        )

    async def test_execute_stream_fires_before_hook_for_force_generate_path(
        self,
    ) -> None:
        # End-to-end: with ``max_turns=0`` the before-hook must fire for both
        # the first generation and the force-generate. Prior to the fix, the
        # force-generate's before-hook was missing — usage trackers that key
        # off it never observed the forced generation's settings.
        first = _msg("first turn")
        forced = _msg("forced final answer")
        executor = _make_executor(max_turns=0, canned_messages=[first, forced])

        observed: list[Event[Any]] = []
        recorder = _BeforeHookRecorder(observed_events=observed)
        executor.on_before_generate_impl = recorder  # type: ignore[method-assign]

        ctx: RunContext[Any] = RunContext()
        async for event in executor.execute_stream(ctx=ctx, call_id="cid"):
            observed.append(event)

        self.assertEqual(len(recorder.calls), 2)
        first_call, second_call = recorder.calls

        # First before-hook (regular first generation): nothing yielded yet.
        self.assertEqual(first_call["num_turns"], 0)
        self.assertEqual(first_call["call_id"], "cid")
        self.assertEqual(len(first_call["events_seen_so_far"]), 0)

        # Second before-hook (force-generate path) fires AFTER the first
        # GenMessageEvent and the synthetic UserMessageEvent have reached
        # the consumer, and BEFORE the forced GenMessageEvent.
        self.assertEqual(second_call["num_turns"], 0)
        self.assertEqual(second_call["call_id"], "cid")
        seen = second_call["events_seen_so_far"]
        self.assertEqual(len(seen), 2)
        self.assertIsInstance(seen[0], GenMessageEvent)
        self.assertIsInstance(seen[1], UserMessageEvent)


if __name__ == "__main__":
    unittest.main()
