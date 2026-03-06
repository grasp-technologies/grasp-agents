"""Tests for UsageTracker.update."""

import pytest
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)

from grasp_agents.types.content import OutputTextContentPart
from grasp_agents.types.items import OutputMessageItem
from grasp_agents.types.response import Response, ResponseUsage
from grasp_agents.usage_tracker import UsageTracker


def _make_response_usage(
    input_tokens: int = 100,
    output_tokens: int = 50,
    reasoning_tokens: int = 0,
    cached_tokens: int = 0,
) -> ResponseUsage:
    return ResponseUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=cached_tokens),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=reasoning_tokens),
    )


def _make_response(
    usage: ResponseUsage | None = None,
    text: str = "Hello",
) -> Response:
    return Response(
        model="test-model",
        output_items=[
            OutputMessageItem(
                content_parts=[OutputTextContentPart(text=text)],
                status="completed",
            )
        ],
        usage_with_cost=usage,
    )


class TestUpdateFromResponse:
    def test_basic_usage_tracking(self):
        """Basic token counting from a response."""
        tracker = UsageTracker()
        usage = _make_response_usage(input_tokens=100, output_tokens=50)
        response = _make_response(usage=usage)

        tracker.update("agent_a", [response])

        agent_usage = tracker.usages["agent_a"]
        assert agent_usage.input_tokens == 100
        assert agent_usage.output_tokens == 50

    def test_reasoning_tokens_separated(self):
        """Reasoning tokens are tracked in output_tokens_details."""
        tracker = UsageTracker()
        usage = _make_response_usage(
            input_tokens=200, output_tokens=100, reasoning_tokens=30
        )
        response = _make_response(usage=usage)

        tracker.update("agent_a", [response])

        agent_usage = tracker.usages["agent_a"]
        assert agent_usage.input_tokens == 200
        assert agent_usage.output_tokens == 100
        assert agent_usage.output_tokens_details.reasoning_tokens == 30

    def test_cached_tokens_tracked(self):
        """Cached tokens are tracked from input_tokens_details."""
        tracker = UsageTracker()
        usage = _make_response_usage(
            input_tokens=500, output_tokens=100, cached_tokens=200
        )
        response = _make_response(usage=usage)

        tracker.update("agent_a", [response])

        assert tracker.usages["agent_a"].input_tokens_details.cached_tokens == 200

    def test_no_usage_is_noop(self):
        """Response with no usage data is silently skipped."""
        tracker = UsageTracker()
        response = _make_response(usage=None)

        tracker.update("agent_a", [response])

        assert "agent_a" not in tracker.usages

    def test_accumulates_across_responses(self):
        """Multiple responses for the same agent accumulate tokens."""
        tracker = UsageTracker()

        r1 = _make_response(usage=_make_response_usage(100, 50))
        r2 = _make_response(usage=_make_response_usage(200, 80))

        tracker.update("agent_a", [r1])
        tracker.update("agent_a", [r2])

        agent_usage = tracker.usages["agent_a"]
        assert agent_usage.input_tokens == 300
        assert agent_usage.output_tokens == 130

    def test_multiple_agents(self):
        """Different agents track independently."""
        tracker = UsageTracker()

        r1 = _make_response(usage=_make_response_usage(100, 50))
        r2 = _make_response(usage=_make_response_usage(200, 80))

        tracker.update("agent_a", [r1])
        tracker.update("agent_b", [r2])

        assert tracker.usages["agent_a"].input_tokens == 100
        assert tracker.usages["agent_b"].input_tokens == 200

    def test_total_usage(self):
        """total_usage aggregates across all agents."""
        tracker = UsageTracker()

        r1 = _make_response(usage=_make_response_usage(100, 50))
        r2 = _make_response(usage=_make_response_usage(200, 80))

        tracker.update("agent_a", [r1])
        tracker.update("agent_b", [r2])

        total = tracker.total_usage
        assert total.input_tokens == 300
        assert total.output_tokens == 130


class TestUpdateFromResponseWithCosts:
    def test_cost_calculation_with_model(self):
        """Cost is calculated when model_name matches costs_dict."""
        tracker = UsageTracker()
        # Override costs_dict with a known model
        tracker.costs_dict = {
            "test-model": {
                "input": 3.0,  # per million tokens
                "output": 15.0,
            }
        }

        usage = _make_response_usage(input_tokens=1_000_000, output_tokens=100_000)
        response = _make_response(usage=usage)

        tracker.update("agent_a", [response], model_name="test-model")

        agent_usage = tracker.usages["agent_a"]
        assert agent_usage.cost is not None
        # input: 1M * 3.0 / 1M = 3.0
        # output: 100K * 15.0 / 1M = 1.5
        assert abs(agent_usage.cost - 4.5) < 0.001

    def test_no_cost_without_model(self):
        """No cost is calculated when model_name is None."""
        tracker = UsageTracker()
        tracker.costs_dict = {"test-model": {"input": 3.0, "output": 15.0}}

        usage = _make_response_usage(input_tokens=1000, output_tokens=500)
        response = _make_response(usage=usage)

        tracker.update("agent_a", [response])

        assert tracker.usages["agent_a"].cost is None

    def test_no_cost_unknown_model(self):
        """No cost when model not in costs_dict."""
        tracker = UsageTracker()
        tracker.costs_dict = {"other-model": {"input": 3.0, "output": 15.0}}

        usage = _make_response_usage(input_tokens=1000, output_tokens=500)
        response = _make_response(usage=usage)

        tracker.update("agent_a", [response], model_name="test-model")

        assert tracker.usages["agent_a"].cost is None
