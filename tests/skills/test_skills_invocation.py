"""Tests for SkillRegistry.render_invocation and the slash-command parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from grasp_agents.skills import (
    ParsedSlashCommand,
    Skill,
    SkillFrontmatter,
    SkillNotFoundError,
    SkillRegistry,
    parse_named_args,
    parse_slash_command,
)


def _skill(name: str, body: str = "Body.") -> Skill:
    fm = SkillFrontmatter.model_validate({"name": name, "description": "x"})
    return Skill(frontmatter=fm, body=body, path=Path(f"/skills/{name}/SKILL.md"))


# ---------- render_invocation ----------


class TestRenderInvocation:
    def test_no_args_no_template(self) -> None:
        reg = SkillRegistry([_skill("alpha", body="Static body.")])
        text = reg.render_invocation("alpha")
        assert "[SYSTEM: user invoked skill alpha]" in text
        assert "Static body." in text

    def test_wrap_false_returns_bare_body(self) -> None:
        reg = SkillRegistry([_skill("alpha", body="Static body.")])
        text = reg.render_invocation("alpha", wrap=False)
        assert "[SYSTEM:" not in text
        assert text.strip() == "Static body."

    def test_arguments_substitution(self) -> None:
        reg = SkillRegistry(
            [_skill("alpha", body="Run with: $ARGUMENTS")]
        )
        text = reg.render_invocation("alpha", args="hello world")
        assert "Run with: hello world" in text

    def test_arguments_substitution_with_mapping_includes_full_string(self) -> None:
        reg = SkillRegistry([_skill("alpha", body="Args: $ARGUMENTS")])
        text = reg.render_invocation("alpha", args={"q": "transformers", "n": "5"})
        # When args is a mapping, $ARGUMENTS becomes the joined "key=value" string.
        assert "q=transformers" in text
        assert "n=5" in text

    def test_named_arg_substitution(self) -> None:
        reg = SkillRegistry(
            [_skill("alpha", body="Query: $QUERY (limit $LIMIT)")]
        )
        text = reg.render_invocation(
            "alpha", args={"QUERY": "transformers", "LIMIT": "5"}
        )
        assert "Query: transformers (limit 5)" in text

    def test_named_arg_missing_keeps_placeholder(self) -> None:
        reg = SkillRegistry(
            [_skill("alpha", body="Query: $QUERY answer: $UNSET")]
        )
        text = reg.render_invocation("alpha", args={"QUERY": "transformers"})
        assert "Query: transformers" in text
        assert "$UNSET" in text  # unmatched placeholder preserved verbatim

    def test_named_arg_with_str_args_does_nothing(self) -> None:
        reg = SkillRegistry(
            [_skill("alpha", body="Args: $ARGUMENTS, query: $QUERY")]
        )
        text = reg.render_invocation("alpha", args="bare string")
        assert "Args: bare string" in text
        assert "$QUERY" in text  # mapping not provided, placeholder kept

    def test_unknown_skill_raises(self) -> None:
        reg = SkillRegistry()
        with pytest.raises(SkillNotFoundError):
            reg.render_invocation("nope")


# ---------- parse_slash_command ----------


class TestParseSlashCommand:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("/foo", ParsedSlashCommand(name="foo", args="")),
            ("/foo bar", ParsedSlashCommand(name="foo", args="bar")),
            ("/foo bar baz", ParsedSlashCommand(name="foo", args="bar baz")),
            ("/foo-bar baz", ParsedSlashCommand(name="foo-bar", args="baz")),
            ("/a1-b-c2 q", ParsedSlashCommand(name="a1-b-c2", args="q")),
            (
                "  /foo   args  here  ",
                ParsedSlashCommand(name="foo", args="args  here"),
            ),
            (
                "/foo --query=transformers --limit=5",
                ParsedSlashCommand(
                    name="foo", args="--query=transformers --limit=5"
                ),
            ),
        ],
    )
    def test_valid(
        self, text: str, expected: ParsedSlashCommand
    ) -> None:
        assert parse_slash_command(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "  ",
            "no slash here",
            "/",  # bare slash, no name
            "/-foo",  # leading hyphen disallowed
            "/foo-",  # trailing hyphen disallowed
            "/foo--bar",  # consecutive hyphens
            "/Foo",  # uppercase disallowed
            "/foo_bar",  # underscore not in agentskills.io spec
            "//foo",  # double slash
            "  /  foo",  # space between slash and name
        ],
    )
    def test_invalid(self, text: str) -> None:
        assert parse_slash_command(text) is None

    def test_long_name_rejected(self) -> None:
        long_name = "a" * 65
        assert parse_slash_command(f"/{long_name}") is None


class TestParseNamedArgs:
    def test_simple(self) -> None:
        assert parse_named_args("--key=value") == {"key": "value"}

    def test_multiple(self) -> None:
        assert parse_named_args("--query=transformers --limit=5") == {
            "query": "transformers",
            "limit": "5",
        }

    def test_bare_flag(self) -> None:
        assert parse_named_args("--verbose") == {"verbose": ""}

    def test_ignores_positional(self) -> None:
        assert parse_named_args("hello world --key=value extra") == {"key": "value"}

    def test_empty(self) -> None:
        assert parse_named_args("") == {}

    def test_hyphenated_keys(self) -> None:
        assert parse_named_args("--max-results=10") == {"max-results": "10"}


# ---------- End-to-end: parse + render ----------


class TestSlashEndToEnd:
    def test_typical_flow(self) -> None:
        reg = SkillRegistry([_skill("arxiv-search", body="Query: $ARGUMENTS")])
        parsed = parse_slash_command("/arxiv-search transformers")
        assert parsed is not None
        text = reg.render_invocation(parsed.name, args=parsed.args)
        assert "[SYSTEM: user invoked skill arxiv-search]" in text
        assert "Query: transformers" in text

    def test_typical_flow_named(self) -> None:
        reg = SkillRegistry(
            [_skill("alpha", body="q=$QUERY n=$LIMIT")]
        )
        parsed = parse_slash_command("/alpha --QUERY=transformers --LIMIT=5")
        assert parsed is not None
        named = parse_named_args(parsed.args)
        text = reg.render_invocation(parsed.name, args=named)
        assert "q=transformers n=5" in text
