"""
Resident-agent spike: one communicator runs its own loop off a session inbox,
consuming peer messages and its own background-task completions across turns
WITHOUT terminating on a final answer. Residency is set by attaching the agent's
session inbox to its loop — ``decide_next_step`` then never stops while the inbox
is open, so the loop ends only when its task is cancelled from outside. A resident
starts parked with no seed input: its first turn is the first inbox message. Lone/
triggered agents are unaffected (no inbox attached → original single-answer
behavior).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from typing import Any

import pytest

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.durability import InMemoryCheckpointStore
from grasp_agents.inbox import AgentInbox
from grasp_agents.mailbox import CheckpointMailboxTransport
from grasp_agents.session_context import SessionContext
from grasp_agents.tools.function_tool import function_tool
from grasp_agents.types.items import InputMessageItem
from grasp_agents.types.message import TeamMessage
from tests._helpers import MockLLM, _text_response, _tool_call_response

pytestmark = pytest.mark.asyncio


async def _until(pred: Callable[[], bool]) -> None:
    """Poll ``pred`` until true, bounded to ~3s so the resident loop can't hang."""
    for _ in range(300):
        if pred():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met within timeout")


async def test_resident_consumes_multiple_messages_in_one_run() -> None:
    agent = LLMAgent[Any, Any, None](
        name="curator",
        llm=MockLLM(
            responses_queue=[_text_response("reply 1"), _text_response("reply 2")]
        ),
        ctx=SessionContext[None](),
    )
    # Attaching an inbox makes the agent resident: the seedless run_stream then
    # drives the loop off this inbox instead of terminating on a final answer.
    inbox = AgentInbox(recipient="curator")
    agent.inbox = inbox

    run = asyncio.create_task(_collect(agent))
    await inbox.post(
        TeamMessage.from_text(sender="user", to="curator", text="first task")
    )
    try:
        await _until(lambda: agent.llm.call_count == 1)

        # The loop is still the SAME resident run — feed it a second message.
        await inbox.post(
            TeamMessage.from_text(sender="user", to="curator", text="second task")
        )
        await _until(lambda: agent.llm.call_count == 2)

        # Two activations, one run: it did not terminate on the first answer.
        assert not run.done()
    finally:
        run.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run
        await agent.aclose()

    texts = [
        part.text
        for m in agent.transcript.messages
        if isinstance(m, InputMessageItem)
        for part in m.content_parts
        if getattr(part, "text", None) is not None
    ]
    # Each peer message is rendered as a user turn ("Message from <sender>: ...").
    assert any("first task" in t for t in texts), texts
    assert any("second task" in t for t in texts), texts
    assert agent.llm.call_count == 2


async def test_resident_survives_past_max_turns() -> None:
    # A resident must NOT fall out of the loop on the turn budget — it runs until
    # cancelled. With max_turns=2 and 4 messages (cumulative turns > 2), the pre-fix
    # `while turn <= max_turns` exited and tripped the final-answer assert.
    agent = LLMAgent[Any, Any, None](
        name="curator",
        llm=MockLLM(responses_queue=[_text_response(f"reply {i}") for i in range(4)]),
        max_turns=2,
        ctx=SessionContext[None](),
    )
    inbox = AgentInbox(recipient="curator")
    agent.inbox = inbox

    run = asyncio.create_task(_collect(agent))
    try:
        for i in range(4):
            await inbox.post(
                TeamMessage.from_text(sender="user", to="curator", text=f"m{i}")
            )
            await _until(lambda i=i: agent.llm.call_count == i + 1)
        # Cumulative turns now exceed max_turns, yet the loop is still alive.
        assert not run.done()
    finally:
        run.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run
        await agent.aclose()

    assert agent.llm.call_count == 4


async def test_resident_force_finalizes_runaway_message_and_continues() -> None:
    # A resident stuck calling a tool on ONE message must not spin forever: at its
    # per-message turn budget it force-generates a reply and moves on. The run
    # itself never ends, and the next message gets a fresh budget.
    @function_tool
    async def spin(x: str) -> str:
        """A tool the model keeps calling."""
        del x
        return "again"

    agent = LLMAgent[Any, Any, None](
        name="curator",
        llm=MockLLM(
            responses_queue=[
                _tool_call_response("spin", '{"x":"a"}', "t1"),
                _tool_call_response("spin", '{"x":"b"}', "t2"),
                _tool_call_response("spin", '{"x":"c"}', "t3"),
                _text_response("forced reply for m0"),  # force-generated at budget
                _text_response("reply for m1"),  # the next message, fresh budget
            ]
        ),
        tools=[spin],
        max_turns=2,
        ctx=SessionContext[None](),
    )
    inbox = AgentInbox(recipient="curator")
    agent.inbox = inbox

    run = asyncio.create_task(_collect(agent))
    await inbox.post(TeamMessage.from_text(sender="user", to="curator", text="m0"))
    try:
        # m0: two productive tool turns, a third over-budget turn, then the forced
        # final answer — four LLM calls — after which the loop parks (does not end).
        await _until(lambda: agent.llm.call_count == 4)
        assert not run.done()

        # A fresh message is handled normally on its own per-message budget.
        await inbox.post(TeamMessage.from_text(sender="user", to="curator", text="m1"))
        await _until(lambda: agent.llm.call_count == 5)
        assert not run.done()
    finally:
        run.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run
        await agent.aclose()

    transcript_text = str(agent.transcript.messages)
    assert "forced reply for m0" in transcript_text, transcript_text
    assert "reply for m1" in transcript_text, transcript_text


async def test_bg_completion_while_idle_wakes_resident_loop() -> None:
    release = asyncio.Event()

    @function_tool(auto_background_at=0, blocks_final_answer=False)
    async def slow_job(text: str) -> str:
        """A slow background job that finishes only once released."""
        await release.wait()
        return f"job done: {text}"

    agent = LLMAgent[Any, Any, None](
        name="curator",
        llm=MockLLM(
            responses_queue=[
                _tool_call_response("slow_job", '{"text":"vault"}', "tc1"),
                _text_response("started; idling"),
                _text_response("completion handled"),
            ]
        ),
        tools=[slow_job],
        ctx=SessionContext[None](),
    )
    inbox = AgentInbox(recipient="curator")
    agent.inbox = inbox

    run = asyncio.create_task(_collect(agent))
    await inbox.post(
        TeamMessage.from_text(
            sender="user", to="curator", text="stash this in the vault"
        )
    )
    try:
        # Round 1: tool call (call 1) then "started; idling" (call 2). The loop is
        # now idle, parked on the inbox while the backgrounded job runs.
        await _until(lambda: agent.llm.call_count == 2)
        assert agent.background_tasks.has_live_tasks

        # Release the job: its completion (NOT a peer message) must wake the loop
        # so the next turn's drain delivers the <task_notification>.
        release.set()
        await _until(lambda: agent.llm.call_count == 3)
    finally:
        run.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run
        await agent.aclose()

    transcript_text = str(agent.transcript.messages)
    assert "job done: vault" in transcript_text, transcript_text
    assert "<task_notification>" in transcript_text


async def test_resident_reply_durable_and_message_released() -> None:
    # A resident's reply turn is checkpointed (so it survives a restart) and the
    # peer message it consumed is released only once that is durable: acked (gone
    # from the inbox) and recorded processed (so a re-delivery would be deduped).
    # A plain reply turn would otherwise never checkpoint, losing the reply.
    store = InMemoryCheckpointStore()
    transport = CheckpointMailboxTransport(store, session_key="s")
    msg = TeamMessage.from_text(sender="user", to="curator", text="task one")

    agent = LLMAgent[Any, Any, None](
        name="curator",
        llm=MockLLM(responses_queue=[_text_response("reply one")]),
        ctx=SessionContext[None](state=None, checkpoint_store=store),
    )
    agent.inbox = AgentInbox(transport=transport, recipient="curator")

    run = asyncio.create_task(_collect(agent))
    await transport.post(msg)
    try:
        await _until(lambda: agent.llm.call_count == 1)
        # The message is released only after the turn-boundary checkpoint lands.
        for _ in range(300):
            if not await transport.has_pending("curator"):
                break
            await asyncio.sleep(0.01)
        assert not await transport.has_pending("curator")
        assert await transport.was_processed("curator", msg.message_id)
    finally:
        run.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run
        await agent.aclose()

    # The reply is durable: a fresh agent on the same store reloads the turn.
    resumed = LLMAgent[Any, Any, None](
        name="curator",
        llm=MockLLM(responses_queue=[]),
        ctx=SessionContext[None](state=None, checkpoint_store=store),
    )
    checkpoint = await resumed.load_checkpoint()
    assert checkpoint is not None
    blob = str(resumed.transcript.messages)
    assert "task one" in blob
    assert "reply one" in blob


async def test_resident_message_released_on_absorption_not_at_reply() -> None:
    # Ack-on-absorption: a drained message is released at the *first* checkpoint
    # that persists it — the first tool turn — NOT held until the reply. Crash
    # safety then rests on the log (see the resume test below), so the message need
    # not linger in the mailbox across the whole handling.
    store = InMemoryCheckpointStore()
    transport = CheckpointMailboxTransport(store, session_key="s")
    r1 = asyncio.Event()
    r2 = asyncio.Event()

    @function_tool
    async def step1(text: str) -> str:
        """First step; completes when released."""
        await r1.wait()
        return f"one: {text}"

    @function_tool
    async def step2(text: str) -> str:
        """Second step; completes when released."""
        await r2.wait()
        return f"two: {text}"

    msg = TeamMessage.from_text(sender="user", to="curator", text="go")
    agent = LLMAgent[Any, Any, None](
        name="curator",
        llm=MockLLM(
            responses_queue=[
                _tool_call_response("step1", '{"text": "a"}', "tc1"),
                _tool_call_response("step2", '{"text": "b"}', "tc2"),
                _text_response("done"),
            ]
        ),
        tools=[step1, step2],
        ctx=SessionContext[None](state=None, checkpoint_store=store),
    )
    agent.inbox = AgentInbox(transport=transport, recipient="curator")

    run = asyncio.create_task(_collect(agent))
    await transport.post(msg)
    try:
        # First tool turn issued, tool still blocking → not yet checkpointed, so the
        # message is still leased (pending, not processed).
        await _until(lambda: agent.llm.call_count == 1)
        assert await transport.has_pending("curator")
        assert not await transport.was_processed("curator", msg.message_id)

        # Release step1: its result is checkpointed, releasing the message NOW — at
        # the first tool turn, while step2 still blocks and the reply (call 3) is
        # not yet produced.
        r1.set()
        await _until(lambda: agent.llm.call_count == 2)  # step2 issued, blocking
        for _ in range(300):
            if await transport.was_processed("curator", msg.message_id):
                break
            await asyncio.sleep(0.01)
        assert await transport.was_processed("curator", msg.message_id)
        assert not await transport.has_pending("curator")
        assert agent.llm.call_count == 2  # released before the reply, not at it
    finally:
        r2.set()
        run.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run
        await agent.aclose()


async def test_resident_resume_continues_owed_tool_results() -> None:
    # The strand fix: a message acked at its tool turn is safe because resume reads
    # "owes a response" off the log. A fresh agent whose restored transcript ends in
    # tool results (the reply was never produced) GENERATES the reply — it does not
    # park, re-take the (already-acked) message, or re-run the tool.
    store = InMemoryCheckpointStore()
    transport = CheckpointMailboxTransport(store, session_key="s")
    r_reply = asyncio.Event()
    msg = TeamMessage.from_text(sender="user", to="curator", text="go")

    @function_tool
    async def work(text: str) -> str:
        """Do the work (runs once, on the first agent)."""
        return f"worked: {text}"

    @function_tool
    async def block_reply(text: str) -> str:
        """Stand-in reply turn; blocks so the run can be 'crashed' before it."""
        await r_reply.wait()
        return text

    agent1 = LLMAgent[Any, Any, None](
        name="curator",
        llm=MockLLM(
            responses_queue=[
                _tool_call_response("work", '{"text": "a"}', "tc1"),
                _tool_call_response("block_reply", '{"text": "b"}', "tc2"),
            ]
        ),
        tools=[work, block_reply],
        ctx=SessionContext[None](state=None, checkpoint_store=store),
    )
    agent1.inbox = AgentInbox(transport=transport, recipient="curator")

    run = asyncio.create_task(_collect(agent1))
    await transport.post(msg)
    try:
        # work() ran and was checkpointed (message acked); the run then parks in
        # block_reply — the post-tool, pre-reply state we want to 'crash' from.
        await _until(lambda: agent1.llm.call_count == 2)
        for _ in range(300):
            if await transport.was_processed("curator", msg.message_id):
                break
            await asyncio.sleep(0.01)
        assert await transport.was_processed("curator", msg.message_id)
    finally:
        run.cancel()  # 'crash' before the reply is produced
        with contextlib.suppress(asyncio.CancelledError):
            await run
        await agent1.aclose()

    # Resume a fresh agent on the same store: the restored tail (work's result)
    # owes a response, so it generates the reply with no new inbox message.
    agent2 = LLMAgent[Any, Any, None](
        name="curator",
        llm=MockLLM(responses_queue=[_text_response("final reply")]),
        tools=[work, block_reply],
        ctx=SessionContext[None](state=None, checkpoint_store=store),
    )
    agent2.inbox = AgentInbox(transport=transport, recipient="curator")

    run2 = asyncio.create_task(_collect(agent2))
    try:
        await _until(lambda: agent2.llm.call_count == 1)
    finally:
        run2.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run2
        await agent2.aclose()

    assert agent2.llm.call_count == 1  # just the reply — no re-take, no tool re-run
    blob = str(agent2.transcript.messages)
    assert "final reply" in blob
    assert "worked: a" in blob  # the tool result carried over, not re-executed
    assert not await transport.has_pending("curator")  # message stayed acked


async def test_rewind_unleases_inbox_message() -> None:
    # A transcript rewind (rollback / failed-run revert, both via
    # ``_restore_session``) must drop the leased inbox message: it was never acked,
    # so it stays in the mailbox and the resident re-takes it on the next run —
    # rather than wedging behind the one-at-a-time lease on a message the rewound
    # transcript no longer reflects.
    agent = LLMAgent[Any, Any, None](
        name="curator",
        llm=MockLLM(responses_queue=[]),
        ctx=SessionContext[None](state=None),
    )
    inbox = AgentInbox(recipient="curator")
    agent.inbox = inbox
    msg = TeamMessage.from_text(sender="user", to="curator", text="x")
    await inbox.post(msg)

    taken = await inbox.take()
    assert taken is not None
    # While leased, the one-at-a-time gate hides it from a second take.
    assert await inbox.take() is None

    # The chokepoint both rollback and the failed-run revert pass through.
    agent._restore_session(agent._snapshot_session())  # pyright: ignore[reportPrivateUsage]

    # Un-leased: the still-unacked message is takeable again.
    retaken = await inbox.take()
    assert retaken is not None
    assert retaken.message_id == msg.message_id


async def _collect(agent: LLMAgent[Any, Any, None]) -> None:
    # A resident starts parked with no seed input; its first turn is the first
    # inbox message. (Lone/triggered agents are unaffected — no inbox attached.)
    async for _event in agent.run_stream():
        pass
