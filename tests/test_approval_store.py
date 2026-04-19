"""
Tests for the async in-process tool-call approval store: scoped
allowlists, pre-approval, pending list, persistence, timeout,
clear_session, and agent-loop integration via `build_store_approval`.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)
from pydantic import BaseModel

import grasp_agents
from grasp_agents.agent.agent_loop import AgentLoop, ResponseCapture
from grasp_agents.agent.approval_store import (
    ApprovalAllow,
    ApprovalDecision,
    ApprovalDeny,
    ApprovalScope,
    ApprovalStore,
    InMemoryApprovalStore,
    PendingApproval,
    build_store_approval,
)
from grasp_agents.agent.llm_agent_memory import LLMAgentMemory
from grasp_agents.llm.llm import LLM
from grasp_agents.run_context import RunContext
from grasp_agents.types.content import OutputMessageText
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
    OutputMessageItem,
)
from grasp_agents.types.llm_events import (
    LlmEvent,
    OutputItemAdded,
    OutputItemDone,
    ResponseCompleted,
    ResponseCreated,
)
from grasp_agents.types.response import Response, ResponseUsage
from grasp_agents.types.tool import BaseTool

# ---------- Infrastructure (shared with prior approval tests) ----------


def _make_usage() -> ResponseUsage:
    return ResponseUsage(
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )


def _text_response(text: str) -> Response:
    return Response(
        model="mock",
        output_items=[
            OutputMessageItem(
                content_parts=[OutputMessageText(text=text)],
                status="completed",
            )
        ],
        usage_with_cost=_make_usage(),
    )


def _tool_call_response(
    calls: Sequence[tuple[str, str, str]],
) -> Response:
    return Response(
        model="mock",
        output_items=[
            FunctionToolCallItem(call_id=call_id, name=name, arguments=args)
            for name, args, call_id in calls
        ],
        usage_with_cost=_make_usage(),
    )


@dataclass(frozen=True)
class MockLLM(LLM):
    responses_queue: list[Response] = field(default_factory=list)

    def __post_init__(self):
        object.__setattr__(self, "_call_count", 0)

    async def _generate_response_once(
        self,
        input: Sequence[Any],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        response_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra_llm_settings: Any,
    ) -> Response:
        count = self._call_count  # type: ignore[attr-defined]
        object.__setattr__(self, "_call_count", count + 1)
        assert self.responses_queue, "MockLLM: no more responses"
        return self.responses_queue.pop(0)

    async def _generate_response_stream_once(
        self,
        input: Sequence[Any],
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        response_schema: Any | None = None,
        tool_choice: Any | None = None,
        **extra_llm_settings: Any,
    ) -> AsyncIterator[LlmEvent]:
        response = await self._generate_response_once(
            input,
            tools=tools,
            response_schema=response_schema,
            tool_choice=tool_choice,
            **extra_llm_settings,
        )
        seq = 0
        seq += 1
        yield ResponseCreated(response=response, sequence_number=seq)  # type: ignore[arg-type]
        for idx, item in enumerate(response.output):
            seq += 1
            yield OutputItemAdded(item=item, output_index=idx, sequence_number=seq)
            seq += 1
            yield OutputItemDone(item=item, output_index=idx, sequence_number=seq)
        seq += 1
        yield ResponseCompleted(response=response, sequence_number=seq)  # type: ignore[arg-type]


class EchoInput(BaseModel):
    text: str


class EchoTool(BaseTool[EchoInput, Any, Any]):
    def __init__(self) -> None:
        super().__init__(name="echo", description="Echoes input")
        _invocations[self.name] = []

    async def _run(self, inp: EchoInput, *, ctx: Any = None, **kwargs: Any) -> str:
        _invocations[self.name].append(inp.text)
        return f"echo: {inp.text}"


class DeleteTool(BaseTool[EchoInput, Any, Any]):
    def __init__(self) -> None:
        super().__init__(name="delete_file", description="Deletes a file")
        _invocations[self.name] = []

    async def _run(self, inp: EchoInput, *, ctx: Any = None, **kwargs: Any) -> str:
        _invocations[self.name].append(inp.text)
        return f"deleted {inp.text}"


_invocations: dict[str, list[str]] = {}


def _make_executor(
    responses: list[Response],
    *,
    tools: list[BaseTool[Any, Any, Any]] | None = None,
    max_turns: int = 10,
) -> tuple[AgentLoop[None], LLMAgentMemory, MockLLM]:
    llm = MockLLM(model_name="mock", responses_queue=responses)
    memory = LLMAgentMemory()
    memory.reset(instructions="sys")
    memory.update([InputMessageItem.from_text("go", role="user")])

    executor = AgentLoop[None](
        agent_name="test",
        llm=llm,
        memory=memory,
        tools=tools,
        max_turns=max_turns,
        stream_llm_responses=False,
    )
    executor.final_answer_extractor = (
        lambda *, ctx, exec_id, response=None, **kw: response.output_text
        if response and not response.tool_call_items
        else None
    )
    return executor, memory, llm


async def _drain(executor: AgentLoop[None], ctx: RunContext[None]) -> Response:
    stream = ResponseCapture(executor.execute_stream(ctx=ctx, exec_id="t"))
    async for _ in stream:
        pass
    assert stream.response is not None
    return stream.response


async def _drain_with_resolver(
    executor: AgentLoop[None],
    ctx: RunContext[None],
    resolver: Any,
) -> Response:
    """Run the executor and the resolver coroutine concurrently."""
    drain_task = asyncio.create_task(_drain(executor, ctx))
    resolve_task = asyncio.create_task(resolver)
    await asyncio.gather(drain_task, resolve_task)
    return drain_task.result()


# ---------- Store unit tests (no agent loop) ----------


class TestStoreBasics:
    """InMemoryApprovalStore basic correctness."""

    @pytest.mark.asyncio
    async def test_submit_and_resolve(self):
        store = InMemoryApprovalStore()
        pending = PendingApproval(
            session_key="s1",
            call_id="c1",
            tool_name="echo",
            arguments="{}",
            approval_key="echo",
        )

        fut = await store.submit_pending(pending)
        assert not fut.done()

        resolved = await store.resolve("s1", "c1", ApprovalAllow())
        assert resolved is True
        assert fut.done()
        assert isinstance(fut.result(), ApprovalAllow)

    @pytest.mark.asyncio
    async def test_resolve_unknown_returns_false(self):
        store = InMemoryApprovalStore()
        # Nothing submitted for (s1, c1)
        resolved = await store.resolve("s1", "c1", ApprovalAllow())
        assert resolved is False

    @pytest.mark.asyncio
    async def test_list_pending(self):
        store = InMemoryApprovalStore()
        for cid in ("c1", "c2"):
            await store.submit_pending(
                PendingApproval(
                    session_key="s1",
                    call_id=cid,
                    tool_name="echo",
                    arguments="{}",
                    approval_key="echo",
                )
            )

        pending = await store.list_pending("s1")
        assert {p.call_id for p in pending} == {"c1", "c2"}

        # Resolving removes from pending
        await store.resolve("s1", "c1", ApprovalAllow())
        pending = await store.list_pending("s1")
        assert {p.call_id for p in pending} == {"c2"}

    @pytest.mark.asyncio
    async def test_scope_session_adds_to_session_allowlist(self):
        store = InMemoryApprovalStore()
        await store.submit_pending(
            PendingApproval(
                session_key="s1",
                call_id="c1",
                tool_name="echo",
                arguments="{}",
                approval_key="echo-key",
            )
        )
        await store.resolve(
            "s1", "c1", ApprovalAllow(scope=ApprovalScope.SESSION)
        )

        # Same approval_key in same session → pre-approved
        assert await store.is_pre_approved("echo-key", session_key="s1")
        # Different session → NOT pre-approved
        assert not await store.is_pre_approved("echo-key", session_key="s2")

    @pytest.mark.asyncio
    async def test_scope_always_adds_to_persistent_allowlist(self):
        store = InMemoryApprovalStore()
        await store.submit_pending(
            PendingApproval(
                session_key="s1",
                call_id="c1",
                tool_name="echo",
                arguments="{}",
                approval_key="echo-key",
            )
        )
        await store.resolve(
            "s1", "c1", ApprovalAllow(scope=ApprovalScope.ALWAYS)
        )

        # Any session → pre-approved
        assert await store.is_pre_approved("echo-key", session_key="s1")
        assert await store.is_pre_approved("echo-key", session_key="s2")

    @pytest.mark.asyncio
    async def test_scope_once_does_not_persist(self):
        store = InMemoryApprovalStore()
        await store.submit_pending(
            PendingApproval(
                session_key="s1",
                call_id="c1",
                tool_name="echo",
                arguments="{}",
                approval_key="echo-key",
            )
        )
        await store.resolve(
            "s1", "c1", ApprovalAllow(scope=ApprovalScope.ONCE)
        )

        # Nothing remembered
        assert not await store.is_pre_approved("echo-key", session_key="s1")

    @pytest.mark.asyncio
    async def test_deny_does_not_add_to_allowlist(self):
        store = InMemoryApprovalStore()
        await store.submit_pending(
            PendingApproval(
                session_key="s1",
                call_id="c1",
                tool_name="echo",
                arguments="{}",
                approval_key="echo-key",
            )
        )
        await store.resolve("s1", "c1", ApprovalDeny(reason="nope"))

        assert not await store.is_pre_approved("echo-key", session_key="s1")

    @pytest.mark.asyncio
    async def test_clear_session_cancels_pending(self):
        store = InMemoryApprovalStore()
        fut = await store.submit_pending(
            PendingApproval(
                session_key="s1",
                call_id="c1",
                tool_name="echo",
                arguments="{}",
                approval_key="echo",
            )
        )

        await store.clear_session("s1")

        # The future was cancelled; awaiting it raises CancelledError
        assert fut.cancelled()
        # Session allowlist wiped
        await store.add_session("s1", "echo")
        await store.clear_session("s1")
        assert not await store.is_pre_approved("echo", session_key="s1")


class TestPersistence:
    """File-backed persistence of the `always` allowlist."""

    @pytest.mark.asyncio
    async def test_always_scope_writes_to_disk(self, tmp_path: Path):
        path = tmp_path / "approvals.json"
        store = InMemoryApprovalStore(persist_path=path)
        await store.submit_pending(
            PendingApproval(
                session_key="s1",
                call_id="c1",
                tool_name="echo",
                arguments="{}",
                approval_key="echo-key",
            )
        )
        await store.resolve(
            "s1", "c1", ApprovalAllow(scope=ApprovalScope.ALWAYS)
        )

        assert path.is_file()
        data = json.loads(path.read_text())
        assert data["always_approved"] == ["echo-key"]

    @pytest.mark.asyncio
    async def test_persisted_allowlist_loads_on_init(self, tmp_path: Path):
        path = tmp_path / "approvals.json"
        path.write_text(
            json.dumps({"always_approved": ["preloaded-key"]})
        )

        store = InMemoryApprovalStore(persist_path=path)
        assert await store.is_pre_approved(
            "preloaded-key", session_key="any"
        )

    @pytest.mark.asyncio
    async def test_malformed_persist_file_is_ignored(self, tmp_path: Path):
        path = tmp_path / "approvals.json"
        path.write_text("{not valid json")

        # Must not raise — malformed file just starts empty
        store = InMemoryApprovalStore(persist_path=path)
        assert not await store.is_pre_approved("x", session_key="s1")


# ---------- Integration with BeforeToolHook / agent loop ----------


async def _auto_resolve(
    store: ApprovalStore,
    session_key: str,
    *,
    decide: Any,
    stop_event: asyncio.Event,
) -> None:
    """
    Poll ``list_pending`` and resolve each with ``decide(pending)``
    until ``stop_event`` is set. Used in integration tests as a
    stand-in for a UI.
    """
    while not stop_event.is_set():
        pending = await store.list_pending(session_key)
        for p in pending:
            await store.resolve(p.session_key, p.call_id, decide(p))
        await asyncio.sleep(0.005)


class TestApprovalGate:
    """`build_store_approval` end-to-end, with the store on RunContext."""

    @pytest.mark.asyncio
    async def test_allow_runs_tool(self):
        _invocations.clear()
        responses = [
            _tool_call_response([("echo", '{"text":"hi"}', "tc1")]),
            _text_response("done"),
        ]
        executor, _, _ = _make_executor(responses, tools=[EchoTool()])
        store = InMemoryApprovalStore()

        executor.before_tool_hook = build_store_approval()  # type: ignore[assignment]

        async def resolver() -> None:
            for _ in range(100):
                if await store.list_pending("s1"):
                    break
                await asyncio.sleep(0.005)
            await store.resolve("s1", "tc1", ApprovalAllow())

        ctx = RunContext[None](
            approval_store=store, session_key="s1"
        )
        await _drain_with_resolver(executor, ctx, resolver())
        assert _invocations["echo"] == ["hi"]

    @pytest.mark.asyncio
    async def test_no_store_on_ctx_allows_all(self):
        """When ctx.approval_store is None the gate is a no-op."""
        _invocations.clear()
        responses = [
            _tool_call_response([("echo", '{"text":"hi"}', "tc1")]),
            _text_response("done"),
        ]
        executor, _, _ = _make_executor(responses, tools=[EchoTool()])

        executor.before_tool_hook = build_store_approval()  # type: ignore[assignment]

        # ctx with no approval_store → tool runs normally
        ctx = RunContext[None]()
        await _drain(executor, ctx)
        assert _invocations["echo"] == ["hi"]

    @pytest.mark.asyncio
    async def test_deny_skips_tool(self):
        _invocations.clear()
        responses = [
            _tool_call_response([("echo", '{"text":"hi"}', "tc1")]),
            _text_response("done"),
        ]
        executor, memory, _ = _make_executor(responses, tools=[EchoTool()])
        store = InMemoryApprovalStore()

        executor.before_tool_hook = build_store_approval()  # type: ignore[assignment]

        async def resolver() -> None:
            for _ in range(100):
                if await store.list_pending("s1"):
                    break
                await asyncio.sleep(0.005)
            await store.resolve(
                "s1", "tc1", ApprovalDeny(reason="policy block")
            )

        ctx = RunContext[None](
            approval_store=store, session_key="s1"
        )
        await _drain_with_resolver(executor, ctx, resolver())

        assert _invocations["echo"] == []
        outs = [
            m for m in memory.messages if isinstance(m, FunctionToolOutputItem)
        ]
        assert len(outs) == 1
        assert "policy block" in outs[0].text

    @pytest.mark.asyncio
    async def test_pre_approval_short_circuits(self):
        """Pre-approved key never lands in the pending queue."""
        _invocations.clear()
        responses = [
            _tool_call_response([("echo", '{"text":"ok"}', "tc1")]),
            _text_response("done"),
        ]
        executor, _, _ = _make_executor(responses, tools=[EchoTool()])
        store = InMemoryApprovalStore()
        await store.add_persistent("echo")

        executor.before_tool_hook = build_store_approval()  # type: ignore[assignment]

        ctx = RunContext[None](
            approval_store=store, session_key="s1"
        )
        await _drain(executor, ctx)

        assert _invocations["echo"] == ["ok"]
        # Never reached the queue
        assert await store.list_pending("s1") == []

    @pytest.mark.asyncio
    async def test_session_scope_caches_across_calls(self):
        """First call prompts + gets SESSION allow; second is pre-approved."""
        _invocations.clear()
        responses = [
            _tool_call_response([("echo", '{"text":"a"}', "tc1")]),
            _tool_call_response([("echo", '{"text":"b"}', "tc2")]),
            _text_response("done"),
        ]
        executor, _, _ = _make_executor(responses, tools=[EchoTool()])
        store = InMemoryApprovalStore()

        stop = asyncio.Event()
        resolved_ids: list[str] = []

        def decide(p: PendingApproval) -> ApprovalDecision:
            resolved_ids.append(p.call_id)
            return ApprovalAllow(scope=ApprovalScope.SESSION)

        executor.before_tool_hook = build_store_approval()  # type: ignore[assignment]

        ctx = RunContext[None](
            approval_store=store, session_key="s1"
        )
        resolver = asyncio.create_task(
            _auto_resolve(store, "s1", decide=decide, stop_event=stop)
        )
        try:
            await _drain(executor, ctx)
        finally:
            stop.set()
            await resolver

        assert _invocations["echo"] == ["a", "b"]
        # Only the first call ever landed in the queue — the second was
        # pre-approved by the session allowlist.
        assert resolved_ids == ["tc1"]

    @pytest.mark.asyncio
    async def test_session_isolation(self):
        """Two sessions on the same store do not share session state."""
        _invocations.clear()
        store = InMemoryApprovalStore()

        stop = asyncio.Event()
        resolved_sessions: list[str] = []

        def decide(p: PendingApproval) -> ApprovalDecision:
            resolved_sessions.append(p.session_key)
            return ApprovalAllow(scope=ApprovalScope.SESSION)

        async def run_user(session: str, call_id: str, text: str) -> None:
            executor, _, _ = _make_executor(
                [
                    _tool_call_response([("echo", f'{{"text":"{text}"}}', call_id)]),
                    _text_response("done"),
                ],
                tools=[EchoTool()],
            )
            executor.before_tool_hook = build_store_approval()  # type: ignore[assignment]
            resolver = asyncio.create_task(
                _auto_resolve(store, session, decide=decide, stop_event=stop)
            )
            try:
                await _drain(
                    executor,
                    RunContext[None](
                        approval_store=store, session_key=session
                    ),
                )
            finally:
                # Leave the resolver running across runs — we only stop at the end.
                pass

            # Stopping per-run would race; instead cancel after each run.
            resolver.cancel()
            with pytest.raises(asyncio.CancelledError):
                await resolver

        await run_user("user-A", "tcA", "a")
        await run_user("user-B", "tcB", "b")

        # Each user prompted independently (no cross-session pre-approval)
        assert resolved_sessions == ["user-A", "user-B"]

    @pytest.mark.asyncio
    async def test_tool_name_filter_bypasses_gate(self):
        _invocations.clear()
        responses = [
            _tool_call_response(
                [
                    ("echo", '{"text":"safe"}', "tc1"),
                    ("delete_file", '{"text":"risky"}', "tc2"),
                ]
            ),
            _text_response("done"),
        ]
        executor, _, _ = _make_executor(
            responses, tools=[EchoTool(), DeleteTool()]
        )
        store = InMemoryApprovalStore()

        async def resolver() -> None:
            for _ in range(100):
                if await store.list_pending("s1"):
                    break
                await asyncio.sleep(0.005)
            pending = await store.list_pending("s1")
            assert [p.tool_name for p in pending] == ["delete_file"]
            await store.resolve("s1", "tc2", ApprovalAllow())

        executor.before_tool_hook = build_store_approval(  # type: ignore[assignment]
            tool_names={"delete_file"}
        )

        ctx = RunContext[None](
            approval_store=store, session_key="s1"
        )
        await _drain_with_resolver(executor, ctx, resolver())

        assert _invocations["echo"] == ["safe"]
        assert _invocations["delete_file"] == ["risky"]

    @pytest.mark.asyncio
    async def test_timeout_denies(self):
        _invocations.clear()
        responses = [
            _tool_call_response([("echo", '{"text":"hi"}', "tc1")]),
            _text_response("done"),
        ]
        executor, memory, _ = _make_executor(responses, tools=[EchoTool()])
        store = InMemoryApprovalStore()

        executor.before_tool_hook = build_store_approval(  # type: ignore[assignment]
            timeout=0.05
        )

        ctx = RunContext[None](
            approval_store=store, session_key="s1"
        )
        await _drain(executor, ctx)

        assert _invocations["echo"] == []
        outs = [
            m for m in memory.messages if isinstance(m, FunctionToolOutputItem)
        ]
        assert "timed out" in outs[0].text

    @pytest.mark.asyncio
    async def test_custom_approval_key_fn(self):
        """
        Custom key_fn lets identical-args calls across turns share
        the same allowlist entry.
        """
        _invocations.clear()
        responses = [
            _tool_call_response([("echo", '{"text":"safe"}', "tc1")]),
            _tool_call_response([("echo", '{"text":"safe"}', "tc2")]),
            _text_response("done"),
        ]
        executor, _, _ = _make_executor(responses, tools=[EchoTool()])
        store = InMemoryApprovalStore()

        stop = asyncio.Event()
        resolved_ids: list[str] = []

        def decide(p: PendingApproval) -> ApprovalDecision:
            resolved_ids.append(p.call_id)
            return ApprovalAllow(scope=ApprovalScope.SESSION)

        executor.before_tool_hook = build_store_approval(  # type: ignore[assignment]
            approval_key_fn=lambda c: f"{c.name}:{c.arguments}"
        )

        ctx = RunContext[None](
            approval_store=store, session_key="s1"
        )
        resolver = asyncio.create_task(
            _auto_resolve(store, "s1", decide=decide, stop_event=stop)
        )
        try:
            await _drain(executor, ctx)
        finally:
            stop.set()
            await resolver

        assert _invocations["echo"] == ["safe", "safe"]
        # Turn 0 resolves tc1 with SESSION scope; turn 1's tc2 has
        # identical args, so its allowlist key matches and it's
        # pre-approved — never enters the pending queue.
        assert resolved_ids == ["tc1"]


class TestProtocolConformance:
    """InMemoryApprovalStore satisfies the ApprovalStore protocol."""

    def test_in_memory_is_approval_store(self):
        store: ApprovalStore = InMemoryApprovalStore()
        assert store is not None


def test_approval_store_public_api():
    """All approval-store names exported from the top-level package."""
    assert grasp_agents.ApprovalScope is ApprovalScope
    assert grasp_agents.ApprovalAllow is ApprovalAllow
    assert grasp_agents.ApprovalDeny is ApprovalDeny
    assert grasp_agents.ApprovalDecision is ApprovalDecision
    assert grasp_agents.PendingApproval is PendingApproval
    assert grasp_agents.ApprovalStore is ApprovalStore
    assert grasp_agents.InMemoryApprovalStore is InMemoryApprovalStore
    assert grasp_agents.build_store_approval is build_store_approval
