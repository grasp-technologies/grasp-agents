"""Tests for model_info facade: graceful fallback for unknown models."""

from grasp_agents.model_info import count_tokens, get_context_window, get_model_capabilities


class TestModelInfo:

    def test_unknown_model_gets_permissive_defaults(self) -> None:
        """Model not in LiteLLM database → all capabilities True, no limits."""
        caps = get_model_capabilities("totally-fake-model-xyz-999")
        assert caps.function_calling is True
        assert caps.vision is True
        assert caps.reasoning is True
        assert caps.max_input_tokens is None
        assert caps.max_output_tokens is None

    def test_count_tokens_unknown_model_still_works(self) -> None:
        """Unknown model → falls back to default tokenizer, doesn't crash."""
        result = count_tokens("totally-fake-model-xyz-999", text="hello world")
        assert result >= 0  # LiteLLM uses a default tokenizer

    def test_context_window_unknown_model_returns_none(self) -> None:
        """Unknown model → None, not an exception."""
        result = get_context_window("totally-fake-model-xyz-999")
        assert result is None
