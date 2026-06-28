"""
End-to-end ``AgentTeam`` behavior with mocked LLMs: a two-member exchange runs
to quiescence, and a hop budget stops a team with mail still pending.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.agent_team.agent_team import AgentTeam
from grasp_agents.agent_team.message import TeamMessage
from grasp_agents.agent_team.sources import run_interval_source
from grasp_agents.agent_team.transport import (
    CheckpointMailboxTransport,
    MessageTransport,
)
from grasp_agents.durability import InMemoryCheckpointStore
from grasp_agents.file_backend.local import LocalFileBackend
from grasp_agents.processors.processor import Processor
from grasp_agents.run_context import RunContext
from grasp_agents.types.response import Response
from tests._helpers import MockLLM, _text_response, _tool_call_response


class ForwardProcessor(Processor[Any, Any, None]):
    """A non-agent processor member: forwards its input to a fixed recipient."""

    async def _process(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[Any] | None = None,
        exec_id: str,
        step: int | None = None,
    ) -> list[Any]:
        del chat_inputs, exec_id, step
        return list(in_args or [])


def _ctx(tmp_path: Path) -> RunContext[None]:
    return RunContext[None](
        state=None, file_backend=LocalFileBackend(allowed_roots=[tmp_path])
    )


def _agent(name: str, responses: list[Response]) -> LLMAgent[Any, Any, None]:
    return LLMAgent[Any, Any, None](name=name, llm=MockLLM(responses_queue=responses))


def _send(to: str, message: str, call_id: str) -> Response:
    return _tool_call_response(
        "SendMessage", f'{{"to": "{to}", "message": "{message}"}}', call_id
    )


class InMemoryTransport(MessageTransport):
    """A non-file transport, to prove a custom transport is used end-to-end."""

    def __init__(self) -> None:
        self.boxes: dict[str, list[TeamMessage]] = {}

    async def _deposit(self, message: TeamMessage) -> None:
        self.boxes.setdefault(message.recipients[0], []).append(message)

    async def fetch_next(self, recipient: str) -> TeamMessage | None:
        box = self.boxes.get(recipient)
        return box[0] if box else None

    async def ack(self, recipient: str, message_ids: list[str]) -> None:
        self.boxes[recipient] = [
            m for m in self.boxes.get(recipient, []) if m.message_id not in message_ids
        ]

    async def has_mail(self, recipient: str) -> bool:
        return bool(self.boxes.get(recipient))


class FailingLLM(MockLLM):
    """An LLM whose generation always raises, to exercise member failure."""

    async def _generate_response_once(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_two_member_ping_pong_runs_to_quiescence(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    alice = _agent(
        "alice",
        [
            _send("bob", "ping", "c1"),
            _text_response("alice: sent ping"),
            _text_response("alice: got the reply, done"),
        ],
    )
    bob = _agent(
        "bob",
        [
            _send("alice", "pong", "c2"),
            _text_response("bob: sent pong"),
        ],
    )
    team = AgentTeam([alice, bob], entry="alice", ctx=ctx)

    result = await team.run("kick off")

    assert result.stop_reason == "quiesced"
    assert result.activations == 3
    delivered = {(m.sender, m.recipient, m.text) for m in result.messages}
    assert ("user", "alice", "kick off") in delivered
    assert ("alice", "bob", "ping") in delivered
    assert ("bob", "alice", "pong") in delivered
    # alice ran twice (2 turns + 1 turn), bob once (2 turns).
    assert alice.llm.call_count == 3
    assert bob.llm.call_count == 2


@pytest.mark.asyncio
async def test_single_member_answers_and_quiesces(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    solo = _agent("solo", [_text_response("the answer")])
    team = AgentTeam([solo], ctx=ctx)

    result = await team.run("question?")

    assert result.stop_reason == "quiesced"
    assert result.activations == 1
    assert solo.llm.call_count == 1
    assert [(m.sender, m.recipient) for m in result.messages] == [("user", "solo")]


@pytest.mark.asyncio
async def test_multiple_pending_messages_deliver_one_per_turn(tmp_path: Path) -> None:
    # Two unrelated messages waiting for one member must activate it TWICE (one
    # group per turn), never merge into a single concatenated turn.
    ctx = _ctx(tmp_path)
    solo = _agent("solo", [_text_response("ack 1"), _text_response("ack 2")])
    transport = InMemoryTransport()
    team = AgentTeam([solo], ctx=ctx, transport=transport)

    await transport.send(
        TeamMessage.of_text(sender="user", to="solo", text="first")
    )
    await transport.send(
        TeamMessage.of_text(sender="user", to="solo", text="second")
    )

    result = await team.run()  # no kickoff; drive the two pending messages

    assert result.stop_reason == "quiesced"
    assert result.activations == 2  # one per message, not one merged turn
    assert solo.llm.call_count == 2
    assert [m.text for m in result.messages if m.recipient == "solo"] == [
        "first",
        "second",
    ]


@pytest.mark.asyncio
async def test_hop_budget_stops_with_pending_mail(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    alice = _agent(
        "alice", [_send("bob", "ping", "c1"), _text_response("alice done")]
    )
    # bob would reply, but the budget stops the team before it is ever activated.
    bob = _agent("bob", [_send("alice", "pong", "c2"), _text_response("bob done")])
    team = AgentTeam([alice, bob], entry="alice", ctx=ctx, max_hops=1)

    result = await team.run("go")

    assert result.stop_reason == "hop_budget_exhausted"
    assert result.activations == 1
    assert bob.llm.call_count == 0


@pytest.mark.asyncio
async def test_duplicate_member_names_rejected(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    a1 = _agent("dup", [_text_response("x")])
    a2 = _agent("dup", [_text_response("y")])
    with pytest.raises(ValueError, match="Duplicate member names"):
        AgentTeam([a1, a2], ctx=ctx)


@pytest.mark.asyncio
async def test_custom_transport_is_used_by_members() -> None:
    transport = InMemoryTransport()
    # No file_backend on ctx: if a member's SendMessage fell back to the file
    # default, its send would fail and bob would never be activated. So bob
    # running proves the custom transport reaches the members' tool, not just
    # the coordinator.
    ctx = RunContext[None](state=None)
    alice = _agent("alice", [_send("bob", "ping", "c1"), _text_response("alice done")])
    bob = _agent("bob", [_text_response("bob got it")])
    team = AgentTeam([alice, bob], entry="alice", ctx=ctx, transport=transport)

    result = await team.run("kick off")

    assert result.stop_reason == "quiesced"
    assert result.activations == 2
    assert bob.llm.call_count == 1
    delivered = {(m.sender, m.recipient, m.text) for m in result.messages}
    assert ("alice", "bob", "ping") in delivered


@pytest.mark.asyncio
async def test_member_failure_reports_member_error(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    solo = LLMAgent[Any, Any, None](name="solo", llm=FailingLLM())
    team = AgentTeam([solo], ctx=ctx)

    result = await team.run("do it")

    assert result.stop_reason == "member_error"
    assert result.activations == 1


@pytest.mark.asyncio
async def test_team_without_file_backend_uses_in_memory() -> None:
    # No file_backend and no explicit transport: the team provisions one shared
    # in-memory transport, so a single-process team needs zero file wiring.
    ctx = RunContext[None](state=None)
    alice = _agent("alice", [_send("bob", "hi", "c1"), _text_response("alice done")])
    bob = _agent("bob", [_text_response("bob got it")])
    team = AgentTeam([alice, bob], entry="alice", ctx=ctx)

    result = await team.run("go")

    assert result.stop_reason == "quiesced"
    assert result.activations == 2
    assert bob.llm.call_count == 1


@pytest.mark.asyncio
async def test_team_over_checkpoint_transport() -> None:
    # A team running on the durable CheckpointStore-backed mailbox (the same
    # substrate background tasks persist through).
    transport = CheckpointMailboxTransport(
        InMemoryCheckpointStore(), session_key="team"
    )
    ctx = RunContext[None](state=None)
    alice = _agent("alice", [_send("bob", "ping", "c1"), _text_response("alice done")])
    bob = _agent("bob", [_text_response("bob got it")])
    team = AgentTeam([alice, bob], entry="alice", ctx=ctx, transport=transport)

    result = await team.run("kick off")

    assert result.stop_reason == "quiesced"
    assert result.activations == 2
    assert bob.llm.call_count == 1
    delivered = {(m.sender, m.recipient, m.text) for m in result.messages}
    assert ("alice", "bob", "ping") in delivered


@pytest.mark.asyncio
async def test_processor_member_routes_to_agent(tmp_path: Path) -> None:
    # A non-agent Processor member consumes a message and hands its output off to
    # an agent member by name (recipients / select_recipients routing).
    ctx = _ctx(tmp_path)
    router = ForwardProcessor(name="router", recipients=["writer"])
    writer = _agent("writer", [_text_response("writer done")])
    team = AgentTeam([router, writer], entry="router", ctx=ctx)

    result = await team.run("hello")

    assert result.stop_reason == "quiesced"
    assert result.activations == 2  # router, then writer
    assert writer.llm.call_count == 1
    assert any(
        m.sender == "router" and m.recipient == "writer" for m in result.messages
    )


@pytest.mark.asyncio
async def test_agent_worker_with_recipients_hands_off(tmp_path: Path) -> None:
    # An LLMAgent given static recipients is a worker, not a communicator: it gets
    # no SendMessage tool and hands its final answer off by name, like a processor.
    ctx = _ctx(tmp_path)
    worker = LLMAgent[Any, Any, None](
        name="worker", llm=MockLLM(responses_queue=[_text_response("did the work")])
    )
    worker.recipients = ["sink"]
    sink = _agent("sink", [_text_response("sink done")])
    team = AgentTeam([worker, sink], entry="worker", ctx=ctx)

    result = await team.run("go")

    assert result.stop_reason == "quiesced"
    assert result.activations == 2
    assert sink.llm.call_count == 1
    assert "SendMessage" not in worker.tools  # worker is hand-off, not messaging
    assert any(
        m.sender == "worker" and m.recipient == "sink" for m in result.messages
    )


def test_member_unknown_recipient_rejected(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    router = ForwardProcessor(name="router", recipients=["ghost"])
    writer = _agent("writer", [_text_response("x")])
    with pytest.raises(ValueError, match="unknown recipient"):
        AgentTeam([router, writer], entry="router", ctx=ctx)


async def _until(pred: Callable[[], bool]) -> None:
    """Poll ``pred`` until true, bounded to ~3s so a daemon test can't hang."""
    for _ in range(300):
        if pred():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met within timeout")


async def _drain_daemon(team: AgentTeam[None]) -> None:
    async for _ in team.run_stream(daemon=True, poll_interval=0.01):
        pass


@pytest.mark.asyncio
async def test_daemon_keeps_running_past_quiescence(tmp_path: Path) -> None:
    # A daemon run does not stop at quiescence: after the first message is handled
    # (a bounded run would end here), a later-injected message is still picked up.
    ctx = _ctx(tmp_path)
    solo = _agent("solo", [_text_response("a1"), _text_response("a2")])
    transport = InMemoryTransport()
    team = AgentTeam([solo], ctx=ctx, transport=transport)

    consumer = asyncio.create_task(_drain_daemon(team))
    try:
        await transport.send(
            TeamMessage.of_text(sender="user", to="solo", text="first")
        )
        await _until(lambda: solo.llm.call_count == 1)  # first handled; team idle
        await transport.send(
            TeamMessage.of_text(sender="user", to="solo", text="second")
        )
        await _until(lambda: solo.llm.call_count == 2)  # daemon picked up the second
    finally:
        consumer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await consumer


@pytest.mark.asyncio
async def test_interval_source_feeds_daemon(tmp_path: Path) -> None:
    # An external interval source wakes a daemon team with messages it then handles.
    ctx = _ctx(tmp_path)
    solo = _agent("solo", [_text_response("a1"), _text_response("a2")])
    transport = InMemoryTransport()
    team = AgentTeam([solo], ctx=ctx, transport=transport)

    consumer = asyncio.create_task(_drain_daemon(team))
    source = asyncio.create_task(
        run_interval_source(
            transport,
            lambda: TeamMessage.of_text(sender="src", to="solo", text="tick"),
            interval=0.01,
            count=2,
        )
    )
    try:
        await source  # sends its two ticks
        await _until(lambda: solo.llm.call_count == 2)  # daemon handled both
    finally:
        consumer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await consumer
