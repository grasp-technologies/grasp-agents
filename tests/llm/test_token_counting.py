"""Pre-generation token counting of the model-facing view (llm.token_counting)."""

import pytest

from grasp_agents.llm.token_counting import count_input_tokens
from grasp_agents.types.content import InputFile, InputImage, InputText
from grasp_agents.types.items import FunctionToolOutputItem, ReasoningItem


def _result(text: str) -> FunctionToolOutputItem:
    return FunctionToolOutputItem.from_tool_result(call_id="c1", output=text)


def test_empty_view_is_zero() -> None:
    assert count_input_tokens("mock", []) == 0


def test_counts_images_not_just_text() -> None:
    text_only = count_input_tokens("mock", [_result("hello")])
    with_image = count_input_tokens(
        "mock",
        [
            FunctionToolOutputItem(
                call_id="c1",
                output_parts=[
                    InputText(text="hello"),
                    InputImage(image_url="https://example.com/x.png"),
                ],
            )
        ],
    )
    assert with_image > text_only  # the image is counted, not ignored


def test_counts_files_by_base64_size() -> None:
    text_only = count_input_tokens("mock", [_result("see file")])
    with_file = count_input_tokens(
        "mock",
        [
            FunctionToolOutputItem(
                call_id="c1",
                output_parts=[
                    InputText(text="see file"),
                    InputFile(file_data="A" * 4000),
                ],
            )
        ],
    )
    # 4000-char base64 ≈ 3000 bytes ≈ 750 tokens added on top of the text.
    assert with_file - text_only == 750


def test_counts_reasoning_encrypted_content() -> None:
    # Encrypted reasoning has no countable text (only a short summary), but it is
    # replayed to the provider and can dominate the context — size it from the
    # blob rather than counting it as zero.
    baseline = count_input_tokens("mock", [_result("done")])
    with_reasoning = count_input_tokens(
        "mock", [_result("done"), ReasoningItem(encrypted_content="A" * 4000)]
    )
    # 4000-char base64 blob ≈ 3000 bytes ≈ 750 tokens, like an inline file.
    assert with_reasoning - baseline == 750


def test_falls_back_to_chars_without_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No tokenizer available (count_tokens returns 0) → chars-per-token estimate.
    monkeypatch.setattr(
        "grasp_agents.llm.token_counting.count_tokens", lambda *a, **k: 0
    )
    text = "A" * 400
    assert count_input_tokens("x", [_result(text)]) == len(text) // 4
