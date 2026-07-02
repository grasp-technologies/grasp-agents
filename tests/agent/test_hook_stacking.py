"""
Hook composition (stacking) at the LLMAgent registration layer.

The observer / builder hooks accumulate instead of replacing: registering a
second ``@agent.add_*`` of the same kind appends it, and a subclass ``*_impl``
override is registered first so it runs before decorator-added hooks. The
single-decision hooks (``output_parser``, ``final_answer_extractor``) stay
single-slot — a later registration replaces.

Loop-dispatch ordering (that the loop actually invokes every stacked hook) is
covered in ``test_agent_loop_hooks`` and ``test_before_tool_decision``; here we
assert the registration contract.
"""

from typing import Any

from grasp_agents import LLMAgent
from tests._helpers import MockLLM


def _make_agent() -> LLMAgent[Any, Any, Any]:
    return LLMAgent[Any, Any, Any](
        name="stack_test",
        llm=MockLLM(model_name="mock", responses_queue=[]),
    )


# ---------- Observer hooks append (run-all), not replace ----------


class TestObserverHooksStack:
    def test_after_llm_hooks_append_in_order(self) -> None:
        agent = _make_agent()

        async def h1(response: Any, *, exec_id: str, turn: int) -> None: ...

        async def h2(response: Any, *, exec_id: str, turn: int) -> None: ...

        agent.add_after_llm_hook(h1)
        agent.add_after_llm_hook(h2)

        assert agent._loop.after_llm_hooks == [h1, h2]

    def test_before_llm_hooks_append_in_order(self) -> None:
        agent = _make_agent()

        async def h1(
            *, exec_id: str, turn: int, extra_llm_settings: dict[str, Any]
        ) -> None: ...

        async def h2(
            *, exec_id: str, turn: int, extra_llm_settings: dict[str, Any]
        ) -> None: ...

        agent.add_before_llm_hook(h1)
        agent.add_before_llm_hook(h2)

        assert agent._loop.before_llm_hooks == [h1, h2]

    def test_before_and_after_tool_hooks_append(self) -> None:
        agent = _make_agent()

        async def bt(*, tool_calls: Any, ctx: Any, exec_id: str) -> None: ...

        async def at(*, tool_calls: Any, tool_messages: Any, exec_id: str) -> None: ...

        agent.add_before_tool_hook(bt)
        agent.add_before_tool_hook(bt)
        agent.add_after_tool_hook(at)

        assert agent._loop.before_tool_hooks == [bt, bt]
        assert agent._loop.after_tool_hooks == [at]


# ---------- Subclass override registered before decorator hooks ----------


class TestSubclassOverrideOrdering:
    def test_subclass_before_llm_impl_runs_first(self) -> None:
        class _Sub(LLMAgent[Any, Any, Any]):
            async def on_before_llm_impl(
                self, *, exec_id: str, turn: int, extra_llm_settings: dict[str, Any]
            ) -> None: ...

        agent = _Sub(name="sub", llm=MockLLM(model_name="mock", responses_queue=[]))

        # The subclass override is appended during __init__…
        assert len(agent._loop.before_llm_hooks) == 1
        assert (
            agent._loop.before_llm_hooks[0].__func__  # type: ignore[attr-defined]
            is _Sub.on_before_llm_impl
        )

        # …and a decorator-registered hook runs after it.
        async def later(
            *, exec_id: str, turn: int, extra_llm_settings: dict[str, Any]
        ) -> None: ...

        agent.add_before_llm_hook(later)
        assert agent._loop.before_llm_hooks[1] is later


# ---------- Single-decision hooks still replace (last wins) ----------


class TestSingleSlotHooksReplace:
    def test_output_parser_replaces(self) -> None:
        agent = _make_agent()

        def p1(final_answer: str, *, in_args: Any = None, exec_id: str) -> Any:
            return 1

        def p2(final_answer: str, *, in_args: Any = None, exec_id: str) -> Any:
            return 2

        agent.add_output_parser(p1)
        agent.add_output_parser(p2)

        assert agent.parse_output_impl is p2

    def test_final_answer_extractor_replaces(self) -> None:
        agent = _make_agent()

        def f1(*, exec_id: str, **kwargs: Any) -> str | None:
            return None

        def f2(*, exec_id: str, **kwargs: Any) -> str | None:
            return None

        agent.add_final_answer_extractor(f1)
        agent.add_final_answer_extractor(f2)

        assert agent._loop.final_answer_extractor is f2

    def test_initial_context_builder_replaces(self) -> None:
        agent = _make_agent()

        async def b1(messages: Any, *, exec_id: str) -> Any:
            return messages

        async def b2(messages: Any, *, exec_id: str) -> Any:
            return messages

        agent.add_initial_context_builder(b1)
        agent.add_initial_context_builder(b2)

        assert agent._prompt_builder.initial_context_builder is b2
