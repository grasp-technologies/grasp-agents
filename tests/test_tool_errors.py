"""Tests for tool error handling: timeout, on_error hook, concurrent tool failure isolation."""

import asyncio
from typing import Any

import pytest
from pydantic import BaseModel

from grasp_agents.run_context import RunContext
from grasp_agents.types.events import ToolErrorEvent, ToolErrorInfo, ToolOutputEvent
from grasp_agents.types.tool import BaseTool, ToolProgressCallback

# ---------- Test tools ----------


class AddInput(BaseModel):
    a: int
    b: int


class SucceedingTool(BaseTool[AddInput, int, Any]):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="add", description="Adds two numbers", **kwargs)

    async def _run(
        self,
        inp: AddInput,
        *,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
    ) -> int:
        return inp.a + inp.b


class FailingTool(BaseTool[AddInput, int, Any]):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="failing", description="Always fails", **kwargs)

    async def _run(
        self,
        inp: AddInput,
        *,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
    ) -> int:
        msg = "Intentional failure"
        raise RuntimeError(msg)


class SlowTool(BaseTool[AddInput, int, Any]):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="slow", description="Takes a long time", **kwargs)

    async def _run(
        self,
        inp: AddInput,
        *,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
    ) -> int:
        await asyncio.sleep(10)
        return inp.a + inp.b


class CustomErrorTool(BaseTool[AddInput, str, Any]):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name="custom_error", description="Custom error handler", **kwargs
        )

    async def _run(
        self,
        inp: AddInput,
        *,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
    ) -> str:
        msg = "something broke"
        raise ValueError(msg)

    def _on_error_impl(self, error: Exception) -> ToolErrorInfo:  # type: ignore[override]
        return ToolErrorInfo(
            tool_name=self.name,
            error=f"Recovered from: {error}",
        )


class ReraisingTool(BaseTool[AddInput, int, Any]):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="reraising", description="Re-raises errors", **kwargs)

    async def _run(
        self,
        inp: AddInput,
        *,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
    ) -> int:
        msg = "critical failure"
        raise RuntimeError(msg)

    def _on_error_impl(self, error: Exception) -> ToolErrorInfo:
        raise error


# ---------- Tests ----------


class TestToolErrorHandling:
    @pytest.mark.asyncio
    async def test_successful_tool_call(self) -> None:
        tool = SucceedingTool()
        result = await tool(a=2, b=3)
        assert result == 5

    @pytest.mark.asyncio
    async def test_failing_tool_returns_error_info(self) -> None:
        """Tool failure returns ToolErrorInfo instead of crashing."""
        tool = FailingTool()
        result = await tool(a=1, b=2)
        assert isinstance(result, ToolErrorInfo)
        assert result.tool_name == "failing"
        assert "Intentional failure" in result.error

    @pytest.mark.asyncio
    async def test_timeout_returns_error_info(self) -> None:
        """Tool with timeout returns ToolErrorInfo."""
        tool = SlowTool(timeout=0.01)
        result = await tool(a=1, b=2)
        assert isinstance(result, ToolErrorInfo)
        assert result.timed_out is True
        assert "Timed out" in result.error
        assert "0.01s" in result.error

    @pytest.mark.asyncio
    async def test_no_timeout_waits_normally(self) -> None:
        """Tool without timeout doesn't time out."""
        tool = SucceedingTool(timeout=None)
        result = await tool(a=2, b=3)
        assert result == 5

    @pytest.mark.asyncio
    async def test_custom_on_error(self) -> None:
        """Custom _on_error_impl override returns custom ToolErrorInfo."""
        tool = CustomErrorTool()
        result = await tool(a=1, b=2)
        assert isinstance(result, ToolErrorInfo)
        assert "Recovered from: something broke" in result.error

    @pytest.mark.asyncio
    async def test_on_error_can_reraise(self) -> None:
        """on_error can re-raise to let the error propagate."""
        tool = ReraisingTool()
        with pytest.raises(RuntimeError, match="critical failure"):
            await tool(a=1, b=2)


class TestToolErrorsInConcurrency:
    @pytest.mark.asyncio
    async def test_one_failure_doesnt_kill_others(self) -> None:
        """Multiple concurrent tools: one fails, others succeed."""
        good = SucceedingTool()
        bad = FailingTool()

        results = await asyncio.gather(
            good(a=1, b=2),
            bad(a=1, b=2),
            good(a=3, b=4),
        )

        assert results[0] == 3
        assert isinstance(results[1], ToolErrorInfo)
        assert results[2] == 7

    @pytest.mark.asyncio
    async def test_timeout_doesnt_kill_others(self) -> None:
        """Timed-out tool doesn't affect other concurrent tools."""
        fast = SucceedingTool()
        slow = SlowTool(timeout=0.01)

        results = await asyncio.gather(
            fast(a=1, b=2),
            slow(a=1, b=2),
        )

        assert results[0] == 3
        assert isinstance(results[1], ToolErrorInfo)
        assert results[1].timed_out is True


class TestToolErrorsInStreaming:
    @pytest.mark.asyncio
    async def test_run_stream_catches_errors(self) -> None:
        """run_stream also catches errors via _run_stream_with_timeout."""
        tool = FailingTool()
        events = [e async for e in tool.run_stream(AddInput(a=1, b=2))]

        assert len(events) == 1
        assert isinstance(events[0], ToolErrorEvent)
        assert isinstance(events[0].data, ToolErrorInfo)
        assert "Intentional failure" in events[0].data.error

    @pytest.mark.asyncio
    async def test_run_stream_catches_timeout(self) -> None:
        """run_stream handles timeout errors."""
        tool = SlowTool(timeout=0.01)
        events = [e async for e in tool.run_stream(AddInput(a=1, b=2))]

        assert len(events) == 1
        assert isinstance(events[0], ToolErrorEvent)
        assert isinstance(events[0].data, ToolErrorInfo)
        assert "Timed out" in events[0].data.error
