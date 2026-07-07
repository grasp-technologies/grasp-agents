"""
Tests for AgentLoop callback-slot hooks.

Verifies that:
- before_generate / after_generate hooks fire on every turn with correct args
- before_tool / after_tool hooks fire during OBSERVE phase
- after_tool receives the tool_messages produced by execution
- hooks are skipped (no error) when set to None
- before_generate can mutate extra_llm_settings (e.g. inject tool_choice)
- hook ordering matches RL phases: PRE-ACT → ACT → JUDGE → OBSERVE
"""

from typing import Any

import pytest
from pydantic import BaseModel

from grasp_agents.agent.agent_loop import AgentLoop, ResponseCapture
from grasp_agents.agent.llm_agent_transcript import LLMAgentTranscript
from grasp_agents.session_context import SessionContext
from grasp_agents.tools.base import BaseTool
from grasp_agents.types.items import (
    FunctionToolOutputItem,
    InputMessageItem,
)
from grasp_agents.types.response import Response
from tests._helpers import (
    MockLLM,
    _make_agent_loop,
    _text_response,
    _tool_call_response,
)

# ---------- Infrastructure ----------


class EchoInput(BaseModel):
    text: str


class EchoTool(BaseTool[EchoInput, Any, Any]):
    def __init__(self) -> None:
        super().__init__(name="echo", description="Echoes input")

    async def _run(self, inp: EchoInput, *, ctx: Any = None, **kwargs: Any) -> str:
        return f"echo: {inp.text}"


def _make_executor(
    responses: list[Response],
    *,
    tools: list[BaseTool[Any, Any, Any]] | None = None,
    max_turns: int = 10,
    ctx: SessionContext[None] | None = None,
) -> tuple[AgentLoop[None], LLMAgentTranscript, MockLLM]:
    llm = MockLLM(model_name="mock", responses_queue=responses)
    memory = LLMAgentTranscript()
    memory.messages = [InputMessageItem.from_text("sys", role="system")]
    memory.update([InputMessageItem.from_text("go", role="user")])

    ctx = ctx if ctx is not None else SessionContext[None](state=None)
    executor = _make_agent_loop(
        agent_name="test",
        llm=llm,
        transcript=memory,
        tools=tools,
        ctx=ctx,
        max_turns=max_turns,
        stream_llm=False,
    )
    return executor, memory, llm


async def _drain(executor: AgentLoop[None], ctx: SessionContext[None]) -> Response:
    # The bound ctx on the executor is what's used internally; the ctx
    # arg here is kept in the test signature to mark intent and so the
    # bound ctx can be reused via the executor if needed.
    del ctx
    stream = ResponseCapture(executor.execute_stream(exec_id="t"))
    async for _ in stream:
        pass
    assert stream.response is not None
    return stream.response


# ---------- Tests ----------


class TestBeforeAfterLlmHooks:
    """Verify before/after generate hooks fire with correct arguments."""

    @pytest.mark.asyncio
    async def test_hooks_fire_on_every_turn(self):
        """
        Two-turn flow: tool_call → final_answer.
        Both hooks should fire twice (once per LLM call).
        """
        responses = [
            _tool_call_response("echo", '{"text":"hi"}', "tc1"),
            _text_response("done"),
        ]
        executor, _, _ = _make_executor(responses, tools=[EchoTool()])

        # Set a final answer checker that stops on text-only responses
        executor.final_answer_extractor = lambda *, exec_id, response=None, **kw: (
            response.output_text if response and not response.tool_call_items else None
        )

        before_calls: list[dict[str, Any]] = []
        after_calls: list[dict[str, Any]] = []

        async def before_hook(*, exec_id, turn, extra_llm_settings):
            before_calls.append(
                {
                    "turn": turn,
                    "exec_id": exec_id,
                    "settings_keys": list(extra_llm_settings.keys()),
                }
            )

        async def after_hook(response, *, exec_id, turn):
            after_calls.append(
                {
                    "turn": turn,
                    "exec_id": exec_id,
                    "has_tool_calls": bool(response.tool_call_items),
                    "text": response.output_text,
                }
            )

        executor.before_llm_hooks = [before_hook]  # type: ignore[assignment]
        executor.after_llm_hooks = [after_hook]  # type: ignore[assignment]

        ctx = SessionContext[None]()
        await _drain(executor, ctx)

        assert len(before_calls) == 2
        assert len(after_calls) == 2

        # Turn 0: tool call
        assert before_calls[0]["turn"] == 0
        assert after_calls[0]["has_tool_calls"] is True

        # Turn 1: final answer
        assert before_calls[1]["turn"] == 1
        assert after_calls[1]["text"] == "done"

    @pytest.mark.asyncio
    async def test_before_generate_can_mutate_settings(self):
        """
        before_generate can inject tool_choice into extra_llm_settings.
        The loop should pick it up and pass it to query_llm.
        """
        responses = [_text_response("ok")]
        executor, _, _ = _make_executor(responses)

        injected_tool_choice = None

        # Monkey-patch query_llm to capture tool_choice
        original_gen = executor.query_llm

        async def capturing_gen(*, tool_choice=None, exec_id, extra_llm_settings):
            nonlocal injected_tool_choice
            injected_tool_choice = tool_choice
            async for event in original_gen(
                tool_choice=tool_choice,
                exec_id=exec_id,
                extra_llm_settings=extra_llm_settings,
            ):
                yield event

        executor.query_llm = capturing_gen  # type: ignore[assignment]

        async def inject_tool_choice(*, exec_id, turn, extra_llm_settings):
            extra_llm_settings["tool_choice"] = "required"

        executor.before_llm_hooks = [inject_tool_choice]  # type: ignore[assignment]

        ctx = SessionContext[None]()
        await _drain(executor, ctx)

        assert injected_tool_choice == "required"

    @pytest.mark.asyncio
    async def test_no_hooks_set_runs_cleanly(self):
        """Executor runs without errors when no hooks are set."""
        responses = [_text_response("works")]
        executor, _, _ = _make_executor(responses)

        assert executor.before_llm_hooks == []
        assert executor.after_llm_hooks == []

        ctx = SessionContext[None]()
        await _drain(executor, ctx)
        assert executor.final_answer == "works"


class TestBeforeAfterToolHooks:
    """Verify tool lifecycle hooks fire during the OBSERVE phase."""

    @pytest.mark.asyncio
    async def test_hooks_receive_correct_tool_data(self):
        """
        before_tool gets the tool calls; after_tool gets both
        tool calls and the resulting tool messages.
        """
        responses = [
            _tool_call_response("echo", '{"text":"hello"}', "tc1"),
            _text_response("done"),
        ]
        executor, _, _ = _make_executor(responses, tools=[EchoTool()])
        executor.final_answer_extractor = lambda *, exec_id, response=None, **kw: (
            response.output_text if response and not response.tool_call_items else None
        )

        before_data: dict[str, Any] = {}
        after_data: dict[str, Any] = {}

        async def before_hook(*, tool_calls, ctx, exec_id):
            before_data["num_calls"] = len(tool_calls)
            before_data["tool_names"] = [tc.name for tc in tool_calls]
            before_data["call_ids"] = [tc.call_id for tc in tool_calls]

        async def after_hook(*, tool_calls, tool_messages, exec_id):
            after_data["num_calls"] = len(tool_calls)
            after_data["num_messages"] = len(tool_messages)
            after_data["message_types"] = [type(m).__name__ for m in tool_messages]
            # Verify tool output content
            for msg in tool_messages:
                if isinstance(msg, FunctionToolOutputItem):
                    after_data["output"] = msg.output

        executor.before_tool_hooks = [before_hook]  # type: ignore[assignment]
        executor.after_tool_hooks = [after_hook]  # type: ignore[assignment]

        ctx = SessionContext[None]()
        await _drain(executor, ctx)

        # before_tool assertions
        assert before_data["num_calls"] == 1
        assert before_data["tool_names"] == ["echo"]
        assert before_data["call_ids"] == ["tc1"]

        # after_tool assertions
        assert after_data["num_calls"] == 1
        assert after_data["num_messages"] == 1
        assert after_data["message_types"] == ["FunctionToolOutputItem"]
        assert "echo: hello" in after_data["output"]

    @pytest.mark.asyncio
    async def test_hooks_fire_every_tool_round(self):
        """
        Multi-tool-round flow: 2 tool calls → final answer.
        Tool hooks should fire twice (once per OBSERVE phase).
        """
        responses = [
            _tool_call_response("echo", '{"text":"a"}', "tc1"),
            _tool_call_response("echo", '{"text":"b"}', "tc2"),
            _text_response("all done"),
        ]
        executor, _, _ = _make_executor(responses, tools=[EchoTool()])
        executor.final_answer_extractor = lambda *, exec_id, response=None, **kw: (
            response.output_text if response and not response.tool_call_items else None
        )

        before_count = 0
        after_count = 0

        async def before_hook(*, tool_calls, ctx, exec_id):
            nonlocal before_count
            before_count += 1

        async def after_hook(*, tool_calls, tool_messages, exec_id):
            nonlocal after_count
            after_count += 1

        executor.before_tool_hooks = [before_hook]  # type: ignore[assignment]
        executor.after_tool_hooks = [after_hook]  # type: ignore[assignment]

        ctx = SessionContext[None]()
        await _drain(executor, ctx)

        assert before_count == 2
        assert after_count == 2

    @pytest.mark.asyncio
    async def test_tool_hooks_not_called_on_text_response(self):
        """Tool hooks should NOT fire when the LLM produces a text-only response."""
        responses = [_text_response("no tools")]
        executor, _, _ = _make_executor(responses, tools=[EchoTool()])
        executor.final_answer_extractor = lambda *, exec_id, response=None, **kw: (
            response.output_text if response and not response.tool_call_items else None
        )

        called = False

        async def should_not_fire(*, tool_calls, ctx, exec_id, **kw):
            nonlocal called
            called = True

        executor.before_tool_hooks = [should_not_fire]  # type: ignore[assignment]
        executor.after_tool_hooks = [should_not_fire]  # type: ignore[assignment]

        ctx = SessionContext[None]()
        await _drain(executor, ctx)

        assert called is False


class TestHookOrdering:
    """Verify that hooks fire in the expected RL phase order."""

    @pytest.mark.asyncio
    async def test_phase_order_with_tool_call(self):
        """
        Full cycle: PRE-ACT → ACT → JUDGE → OBSERVE → PRE-ACT → ACT → JUDGE.
        Hook firing order should be:
        before_generate(0) → after_generate(0) → before_tool → after_tool →
        before_generate(1) → after_generate(1)
        """
        responses = [
            _tool_call_response("echo", '{"text":"x"}', "tc1"),
            _text_response("final"),
        ]
        executor, _, _ = _make_executor(responses, tools=[EchoTool()])
        executor.final_answer_extractor = lambda *, exec_id, response=None, **kw: (
            response.output_text if response and not response.tool_call_items else None
        )

        trace: list[str] = []

        async def bg(*, exec_id, turn, extra_llm_settings):
            trace.append(f"before_gen:{turn}")

        async def ag(response, *, exec_id, turn):
            trace.append(f"after_gen:{turn}")

        async def btc(*, tool_calls, ctx, exec_id):
            trace.append("before_tool")

        async def atc(*, tool_calls, tool_messages, exec_id):
            trace.append("after_tool")

        executor.before_llm_hooks = [bg]  # type: ignore[assignment]
        executor.after_llm_hooks = [ag]  # type: ignore[assignment]
        executor.before_tool_hooks = [btc]  # type: ignore[assignment]
        executor.after_tool_hooks = [atc]  # type: ignore[assignment]

        ctx = SessionContext[None]()
        await _drain(executor, ctx)

        assert trace == [
            "before_gen:0",
            "after_gen:0",
            "before_tool",
            "after_tool",
            "before_gen:1",
            "after_gen:1",
        ]

    @pytest.mark.asyncio
    async def test_max_turns_fires_before_llm_on_forced_answer_too(self):
        """
        When max_turns is hit, the loop generates a forced final answer.
        The before-LLM hook fires for the tool-call turn (turn 0) AND for
        the forced final-answer call — the forced call carries the longest
        transcript of the run, so context-trimming hooks must see it.
        """
        responses = [
            _tool_call_response("echo", '{"text":"x"}', "tc1"),
            _text_response("forced"),
        ]
        executor, _, _ = _make_executor(responses, tools=[EchoTool()], max_turns=0)

        gen_turns: list[int] = []

        async def bg(*, exec_id, turn, extra_llm_settings):
            gen_turns.append(turn)

        executor.before_llm_hooks = [bg]  # type: ignore[assignment]

        ctx = SessionContext[None]()
        await _drain(executor, ctx)

        # Turn 0's regular ACT call, then the forced final-answer call.
        assert gen_turns == [0, 0]


class TestToolInputConverter:
    """Verify per-tool input converters can override fields before execution."""

    @pytest.mark.asyncio
    async def test_converter_overrides_field(self):
        """
        MCP use case: converter overrides a field the LLM generated
        with a value from context state.
        """

        class SearchInput(BaseModel):
            query: str
            api_key: str = ""

        received_inputs: list[SearchInput] = []

        class SearchTool(BaseTool[SearchInput, Any, str]):
            def __init__(self) -> None:
                super().__init__(name="search", description="Search")

            async def _run(
                self, inp: SearchInput, *, ctx: Any = None, **kwargs: Any
            ) -> str:
                received_inputs.append(inp)
                return f"results for {inp.query}"

        responses = [
            _tool_call_response(
                "search", '{"query":"python","api_key":"wrong"}', "tc1"
            ),
            _text_response("done"),
        ]
        executor, _, _ = _make_executor(responses, tools=[SearchTool()])
        executor.final_answer_extractor = lambda *, exec_id, response=None, **kw: (
            response.output_text if response and not response.tool_call_items else None
        )

        async def override_api_key(llm_args, *, exec_id):
            return llm_args.model_copy(update={"api_key": "secret-key"})

        executor.tool_input_converters["search"] = override_api_key  # type: ignore[assignment]

        ctx = SessionContext[str](state="unused")
        await _drain(executor, ctx)

        assert len(received_inputs) == 1
        assert received_inputs[0].query == "python"
        assert received_inputs[0].api_key == "secret-key"

    @pytest.mark.asyncio
    async def test_no_converter_passes_validated_input(self):
        """Without a converter, the tool receives the validated LLM args directly."""
        received_inputs: list[EchoInput] = []
        original_run = EchoTool._run

        class CapturingEchoTool(EchoTool):
            async def _run(
                self, inp: EchoInput, *, ctx: Any = None, **kwargs: Any
            ) -> str:  # type: ignore[override]
                received_inputs.append(inp)
                return await original_run(self, inp, **kwargs)

        responses = [
            _tool_call_response("echo", '{"text":"hello"}', "tc1"),
            _text_response("done"),
        ]
        executor, _, _ = _make_executor(responses, tools=[CapturingEchoTool()])
        executor.final_answer_extractor = lambda *, exec_id, response=None, **kw: (
            response.output_text if response and not response.tool_call_items else None
        )

        ctx = SessionContext[None]()
        await _drain(executor, ctx)

        assert len(received_inputs) == 1
        assert received_inputs[0].text == "hello"


class TestLlmInType:
    """Verify llm_in_type controls validation and schema generation."""

    @pytest.mark.asyncio
    async def test_llm_in_type_with_converter(self):
        """
        When llm_in_type is a reduced schema, _convert_tool_input validates
        against it, and the converter bridges to the full in_type.
        """
        from grasp_agents.utils.schema import exclude_fields

        class FullInput(BaseModel):
            query: str
            api_key: str

        LlmInput = exclude_fields(FullInput, {"api_key"})

        received_inputs: list[FullInput] = []

        class SearchTool(BaseTool[FullInput, Any, str]):
            def __init__(self) -> None:
                super().__init__(name="search", description="Search")
                self.llm_in_type = LlmInput

            async def _run(
                self, inp: FullInput, *, ctx: Any = None, **kwargs: Any
            ) -> str:
                received_inputs.append(inp)
                return f"results for {inp.query}"

        # LLM generates only `query` — no api_key
        responses = [
            _tool_call_response("search", '{"query":"python"}', "tc1"),
            _text_response("done"),
        ]
        executor, _, _ = _make_executor(responses, tools=[SearchTool()])
        executor.final_answer_extractor = lambda *, exec_id, response=None, **kw: (
            response.output_text if response and not response.tool_call_items else None
        )

        async def inject_api_key(llm_args, *, exec_id):
            return FullInput(query=llm_args.query, api_key="secret-key")

        executor.tool_input_converters["search"] = inject_api_key  # type: ignore[assignment]

        ctx = SessionContext[str](state="unused")
        await _drain(executor, ctx)

        assert len(received_inputs) == 1
        assert received_inputs[0].query == "python"
        assert received_inputs[0].api_key == "secret-key"

    @pytest.mark.asyncio
    async def test_llm_in_type_defaults_to_in_type(self):
        """Without override, llm_in_type is the same as in_type."""
        tool = EchoTool()
        assert tool.llm_in_type is tool.in_type

    @pytest.mark.asyncio
    async def test_llm_in_type_rejects_extra_fields(self):
        """
        When llm_in_type is reduced, validation should reject fields
        that aren't in the LLM schema (the LLM shouldn't generate them).
        """
        from grasp_agents.utils.schema import exclude_fields

        class FullInput(BaseModel):
            query: str
            api_key: str

        LlmInput = exclude_fields(FullInput, {"api_key"})

        class SearchTool(BaseTool[FullInput, Any, None]):
            def __init__(self) -> None:
                super().__init__(name="search", description="Search")
                self.llm_in_type = LlmInput

            async def _run(self, inp: FullInput, **kwargs: Any) -> str:
                return "ok"

        # LLM generates both fields — api_key should fail validation
        # against the reduced LlmInput schema (strict by default in Pydantic)
        responses = [
            _tool_call_response("search", '{"query":"python","api_key":"bad"}', "tc1"),
            _text_response("done"),
        ]
        executor, _, _ = _make_executor(responses, tools=[SearchTool()])
        executor.final_answer_extractor = lambda *, exec_id, response=None, **kw: (
            response.output_text if response and not response.tool_call_items else None
        )

        # Pydantic v2 ignores extra fields by default, so this should
        # succeed — the extra api_key is silently dropped.
        ctx = SessionContext[None]()
        await _drain(executor, ctx)


class TestToolOutputConverter:
    """Verify per-tool output converters intercept tool result formatting."""

    @pytest.mark.asyncio
    async def test_custom_tool_output_converter(self):
        """
        A per-tool converter wraps tool output in XML tags.
        The converted output should appear in memory.
        """
        responses = [
            _tool_call_response("echo", '{"text":"world"}', "tc1"),
            _text_response("done"),
        ]
        executor, memory, _ = _make_executor(responses, tools=[EchoTool()])
        executor.final_answer_extractor = lambda *, exec_id, response=None, **kw: (
            response.output_text if response and not response.tool_call_items else None
        )

        async def xml_converter(tool_output, *, exec_id):
            return f"<result>{tool_output}</result>"

        executor.tool_output_converters["echo"] = xml_converter  # type: ignore[assignment]

        ctx = SessionContext[None]()
        await _drain(executor, ctx)

        tool_outputs = [
            m for m in memory.messages if isinstance(m, FunctionToolOutputItem)
        ]
        assert len(tool_outputs) == 1
        assert "<result>echo: world</result>" in tool_outputs[0].text


class TestStackedLoopHooks:
    """Multiple hooks of the same kind run, in registration order (A1)."""

    @pytest.mark.asyncio
    async def test_before_and_after_llm_hooks_all_fire_in_order(self):
        responses = [_text_response("ok")]
        executor, _, _ = _make_executor(responses)

        order: list[str] = []

        async def before1(*, exec_id, turn, extra_llm_settings):
            order.append("before1")

        async def before2(*, exec_id, turn, extra_llm_settings):
            order.append("before2")

        async def after1(response, *, exec_id, turn):
            order.append("after1")

        async def after2(response, *, exec_id, turn):
            order.append("after2")

        executor.before_llm_hooks = [before1, before2]  # type: ignore[list-item]
        executor.after_llm_hooks = [after1, after2]  # type: ignore[list-item]

        ctx = SessionContext[None]()
        await _drain(executor, ctx)

        assert order == ["before1", "before2", "after1", "after2"]

    @pytest.mark.asyncio
    async def test_after_tool_hooks_all_fire_in_order(self):
        responses = [
            _tool_call_response("echo", '{"text":"hi"}', "tc1"),
            _text_response("done"),
        ]
        executor, _, _ = _make_executor(responses, tools=[EchoTool()])
        executor.final_answer_extractor = lambda *, exec_id, response=None, **kw: (
            response.output_text if response and not response.tool_call_items else None
        )

        order: list[str] = []

        async def after1(*, tool_calls, tool_messages, exec_id):
            order.append("after1")

        async def after2(*, tool_calls, tool_messages, exec_id):
            order.append("after2")

        executor.after_tool_hooks = [after1, after2]  # type: ignore[list-item]

        ctx = SessionContext[None]()
        await _drain(executor, ctx)

        assert order == ["after1", "after2"]
