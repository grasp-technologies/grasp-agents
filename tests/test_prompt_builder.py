"""
Tests for PromptBuilder input content resolution.

Verifies the 4-level resolution order:
  1. InputContentBuilder hook (highest priority)
  2. InputRenderable.to_input_parts() on the input model
  3. in_prompt template with BaseModel fields (text-only)
  4. JSON fallback

Also tests:
- InputRenderable with multimodal content (images + text)
- in_prompt template formatting (no image support)
- BaseModel without in_prompt → JSON serialization
- Non-BaseModel inputs → JSON fallback
- InputContentBuilder overrides InputRenderable
- chat_inputs path (separate from in_args)
- None input type
"""

from typing import Any

import pytest
from pydantic import BaseModel, Field

from grasp_agents.agent.prompt_builder import PromptBuilder
from grasp_agents.run_context import RunContext
from grasp_agents.types.content import (
    Content,
    InputImage,
    InputPart,
    InputRenderable,
    InputText,
)

# ---------- Test models ----------


class SimpleInput(BaseModel):
    query: str
    context: str


class RenderableInput(BaseModel):
    """Input model that implements InputRenderable."""

    title: str
    body: str

    def to_input_parts(self) -> str:
        return f"<TITLE>{self.title}</TITLE>\n<BODY>{self.body}</BODY>"


class MultimodalRenderableInput(BaseModel):
    """Input model with images that renders itself."""

    description: str
    image_url: str

    def to_input_parts(self) -> list[InputPart]:
        return [
            InputImage.from_url(self.image_url),
            InputText(text=f"<DESC>{self.description}</DESC>"),
        ]


class NestedInput(BaseModel):
    """Input with nested objects to test JSON serialization."""

    name: str
    tags: list[str]
    metadata: dict[str, int] = Field(default_factory=dict)


class ExcludedFieldInput(BaseModel):
    """Input with an excluded field."""

    visible: str
    hidden: str = Field(exclude=True)


# ---------- Helpers ----------


def _make_builder(
    in_type: type,
    *,
    in_prompt: str | None = None,
    sys_prompt: str | None = None,
) -> PromptBuilder[Any, None]:
    return PromptBuilder[in_type, None](  # type: ignore[type-arg]
        agent_name="test_agent",
        sys_prompt=sys_prompt,
        in_prompt=in_prompt,
    )


def _ctx() -> RunContext[None]:
    return RunContext[None]()


# ---------- Tests ----------


class TestInputRenderableResolution:
    """Test resolution level 2: InputRenderable.to_input_parts()."""

    def test_renderable_model_uses_to_input_parts(self):
        builder = _make_builder(RenderableInput)
        inp = RenderableInput(title="Hello", body="World")

        content = builder.build_input_content(in_args=inp, ctx=_ctx(), exec_id="c1")

        assert len(content.parts) == 1
        text = content.parts[0]
        assert isinstance(text, InputText)
        assert "<TITLE>Hello</TITLE>" in text.text
        assert "<BODY>World</BODY>" in text.text

    def test_multimodal_renderable_produces_image_and_text(self):
        builder = _make_builder(MultimodalRenderableInput)
        inp = MultimodalRenderableInput(
            description="A cat", image_url="https://example.com/cat.jpg"
        )

        content = builder.build_input_content(in_args=inp, ctx=_ctx(), exec_id="c1")

        assert len(content.parts) == 2
        assert isinstance(content.parts[0], InputImage)
        assert content.parts[0].image_url == "https://example.com/cat.jpg"
        assert isinstance(content.parts[1], InputText)
        assert "<DESC>A cat</DESC>" in content.parts[1].text

    def test_renderable_bypasses_in_prompt_template(self):
        """Even if in_prompt is set, InputRenderable takes priority."""
        builder = _make_builder(
            RenderableInput,
            in_prompt="TEMPLATE: {title} - {body}",
        )
        inp = RenderableInput(title="Hello", body="World")

        content = builder.build_input_content(in_args=inp, ctx=_ctx(), exec_id="c1")

        # Should use to_input_parts(), NOT the template
        text = content.parts[0]
        assert isinstance(text, InputText)
        assert "<TITLE>Hello</TITLE>" in text.text
        assert "TEMPLATE:" not in text.text

    def test_protocol_is_runtime_checkable(self):
        assert isinstance(RenderableInput(title="a", body="b"), InputRenderable)
        assert not isinstance(SimpleInput(query="a", context="b"), InputRenderable)


class TestInputContentBuilderHookPriority:
    """Test resolution level 1: InputContentBuilder hook overrides everything."""

    def test_hook_overrides_renderable(self):
        """InputContentBuilder takes priority over InputRenderable."""
        builder = _make_builder(RenderableInput)

        def custom_builder(in_args, *, ctx, exec_id):
            return Content.from_text(f"CUSTOM: {in_args.title}")

        builder.input_content_builder = custom_builder  # type: ignore[assignment]

        inp = RenderableInput(title="Hello", body="World")
        content = builder.build_input_content(in_args=inp, ctx=_ctx(), exec_id="c1")

        text = content.parts[0]
        assert isinstance(text, InputText)
        assert text.text == "CUSTOM: Hello"

    def test_hook_overrides_in_prompt_template(self):
        builder = _make_builder(
            SimpleInput,
            in_prompt="<Q>{query}</Q>\n<C>{context}</C>",
        )

        def custom_builder(in_args, *, ctx, exec_id):
            return Content.from_text(f"OVERRIDE: {in_args.query}")

        builder.input_content_builder = custom_builder  # type: ignore[assignment]

        inp = SimpleInput(query="test", context="ctx")
        content = builder.build_input_content(in_args=inp, ctx=_ctx(), exec_id="c1")

        assert content.parts[0].text == "OVERRIDE: test"  # type: ignore[union-attr]


class TestInPromptTemplate:
    """Test resolution level 3: in_prompt template with BaseModel fields."""

    def test_text_only_template_formatting(self):
        builder = _make_builder(
            SimpleInput,
            in_prompt="<QUERY>{query}</QUERY>\n<CONTEXT>{context}</CONTEXT>",
        )
        inp = SimpleInput(query="What is AI?", context="CS course")

        content = builder.build_input_content(in_args=inp, ctx=_ctx(), exec_id="c1")

        assert len(content.parts) == 1
        text = content.parts[0]
        assert isinstance(text, InputText)
        assert "<QUERY>What is AI?</QUERY>" in text.text
        assert "<CONTEXT>CS course</CONTEXT>" in text.text

    def test_template_with_nested_fields_serializes_to_json(self):
        """Non-primitive fields are JSON-serialized before template substitution."""
        builder = _make_builder(
            NestedInput,
            in_prompt="Name: {name}\nTags: {tags}",
        )
        inp = NestedInput(name="test", tags=["a", "b", "c"])

        content = builder.build_input_content(in_args=inp, ctx=_ctx(), exec_id="c1")

        text = content.parts[0].text  # type: ignore[union-attr]
        assert "Name: test" in text
        # tags should be JSON-serialized
        assert '"a"' in text
        assert '"b"' in text

    def test_template_respects_excluded_fields(self):
        builder = _make_builder(
            ExcludedFieldInput,
            in_prompt="Visible: {visible}",
        )
        inp = ExcludedFieldInput(visible="shown", hidden="secret")

        content = builder.build_input_content(in_args=inp, ctx=_ctx(), exec_id="c1")

        text = content.parts[0].text  # type: ignore[union-attr]
        assert "shown" in text
        assert "secret" not in text


class TestJSONFallback:
    """Test resolution level 4: JSON serialization fallback."""

    def test_basemodel_without_in_prompt_gives_json(self):
        builder = _make_builder(SimpleInput)
        inp = SimpleInput(query="hello", context="world")

        content = builder.build_input_content(in_args=inp, ctx=_ctx(), exec_id="c1")

        text = content.parts[0].text  # type: ignore[union-attr]
        assert '"query"' in text
        assert '"hello"' in text
        assert '"context"' in text

    def test_string_input_gives_json(self):
        builder = _make_builder(str)

        content = builder.build_input_content(
            in_args="raw string", ctx=_ctx(), exec_id="c1"
        )

        text = content.parts[0].text  # type: ignore[union-attr]
        assert "raw string" in text

    def test_int_input_gives_json(self):
        builder = _make_builder(int)

        content = builder.build_input_content(in_args=42, ctx=_ctx(), exec_id="c1")

        text = content.parts[0].text  # type: ignore[union-attr]
        assert "42" in text


class TestNoneInputType:
    """Test behavior when InT is None."""

    def test_none_input_type_with_none_args(self):
        builder = _make_builder(type(None))

        # Should not raise — None is valid when InT is type(None)
        content = builder.build_input_content(in_args=None, ctx=_ctx(), exec_id="c1")
        assert content is not None

    def test_non_none_input_type_with_none_args_raises(self):
        builder = _make_builder(str)

        with pytest.raises(Exception, match="input arguments must be provided"):
            builder.build_input_content(in_args=None, ctx=_ctx(), exec_id="c1")


class TestBuildInputMessage:
    """Test the build_input_message entry point."""

    def test_chat_inputs_string(self):
        builder = _make_builder(type(None))
        msg = builder.build_input_message(
            chat_inputs="Hello there", exec_id="c1", ctx=_ctx()
        )
        assert msg is not None
        assert msg.role == "user"

    def test_chat_inputs_with_image(self):
        builder = _make_builder(type(None))
        img = InputImage.from_url("https://example.com/img.jpg")

        msg = builder.build_input_message(
            chat_inputs=["Look at this:", img], exec_id="c1", ctx=_ctx()
        )
        assert msg is not None
        assert len(msg.content_parts) == 2
        assert isinstance(msg.content_parts[0], InputText)
        assert isinstance(msg.content_parts[1], InputImage)

    def test_chat_inputs_and_in_args_raises(self):
        builder = _make_builder(str)

        with pytest.raises(Exception, match="Cannot use both"):
            builder.build_input_message(
                chat_inputs="hi",
                in_args="also hi",
                exec_id="c1",
                ctx=_ctx(),
            )

    def test_in_args_produces_user_message(self):
        builder = _make_builder(
            SimpleInput,
            in_prompt="Q: {query}",
        )
        msg = builder.build_input_message(
            in_args=SimpleInput(query="test", context="c"),
            exec_id="c1",
            ctx=_ctx(),
        )
        assert msg is not None
        assert msg.role == "user"


class TestSystemPromptBuilder:
    """Test system prompt hook."""

    @pytest.mark.anyio
    async def test_default_returns_sys_prompt(self):
        builder = _make_builder(str, sys_prompt="Be helpful.")
        result = await builder.build_system_prompt(ctx=_ctx(), exec_id="c1")
        assert result == "Be helpful."

    @pytest.mark.anyio
    async def test_hook_overrides_sys_prompt(self):
        builder = _make_builder(str, sys_prompt="Original.")

        def custom_sys(*, ctx, exec_id):
            return "Dynamic prompt"

        builder.system_prompt_builder = custom_sys  # type: ignore[assignment]
        result = await builder.build_system_prompt(ctx=_ctx(), exec_id="c1")
        assert result == "Dynamic prompt"

    @pytest.mark.anyio
    async def test_no_sys_prompt_returns_none(self):
        builder = _make_builder(str)
        result = await builder.build_system_prompt(ctx=_ctx(), exec_id="c1")
        assert result is None
