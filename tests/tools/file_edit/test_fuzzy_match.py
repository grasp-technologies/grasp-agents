"""
Unit tests for :mod:`grasp_agents.tools.file_edit.fuzzy_match`.

Covers each of the 9 strategies end-to-end, fallthrough ordering,
extended Unicode coverage (OpenClaw merge), quote preservation, and
the ``replace_all`` / uniqueness / refusal semantics.
"""

from __future__ import annotations

import pytest

from grasp_agents.tools.file_edit.fuzzy_match import (
    UNICODE_MAP,
    apply_replacements,
    fuzzy_find,
    fuzzy_find_and_replace,
    preserve_quote_style,
)

# ---------------------------------------------------------------------------
# Strategy 1: exact
# ---------------------------------------------------------------------------


def test_exact_single_match() -> None:
    content = "def foo():\n    return 1\n"
    out, count, strategy, err = fuzzy_find_and_replace(content, "return 1", "return 2")
    assert err is None
    assert count == 1
    assert strategy == "exact"
    assert "return 2" in out
    assert "return 1" not in out


def test_exact_multiple_needs_replace_all() -> None:
    content = "x = 1\ny = 1\nz = 1\n"
    _, count, _, err = fuzzy_find_and_replace(content, "= 1", "= 2")
    assert count == 0
    assert err is not None
    assert "3 matches" in err


def test_exact_replace_all() -> None:
    content = "x = 1\ny = 1\nz = 1\n"
    out, count, strategy, err = fuzzy_find_and_replace(
        content, "= 1", "= 2", replace_all=True
    )
    assert err is None
    assert count == 3
    assert strategy == "exact"
    assert out == "x = 2\ny = 2\nz = 2\n"


# ---------------------------------------------------------------------------
# Strategy 2: line_trimmed
# ---------------------------------------------------------------------------


def test_line_trimmed_strips_trailing_whitespace() -> None:
    # Content has trailing spaces the model's pattern omits.
    content = "def foo():   \n    return 1   \n"
    pattern = "def foo():\n    return 1"
    out, count, strategy, err = fuzzy_find_and_replace(content, pattern, "def bar():")
    assert err is None
    assert count == 1
    assert strategy == "line_trimmed"
    assert "def bar():" in out


# ---------------------------------------------------------------------------
# Strategy 3: whitespace_normalized
# ---------------------------------------------------------------------------


def test_whitespace_normalized_collapses_spaces() -> None:
    content = "x  =   1\n"  # extra spaces in content
    pattern = "x = 1"  # single spaces in pattern
    out, count, strategy, err = fuzzy_find_and_replace(content, pattern, "x = 42")
    assert err is None
    assert count == 1
    assert strategy == "whitespace_normalized"
    assert "x = 42" in out


# ---------------------------------------------------------------------------
# Strategy 4: indentation_flexible
# ---------------------------------------------------------------------------


def test_indentation_drift_handled() -> None:
    """
    ``line_trimmed`` (strips both ends) subsumes ``indentation_flexible``
    (lstrip only) for pure leading-whitespace drift. Both are in the chain
    but the earlier one fires first. Test documents this reality.
    """
    # Pattern unindented; content indented.
    content = "class A:\n    def foo(self):\n        return 1\n"
    pattern = "def foo(self):\nreturn 1"
    out, count, strategy, err = fuzzy_find_and_replace(
        content, pattern, "def foo(self):\nreturn 2"
    )
    assert err is None
    assert count == 1
    # line_trimmed runs before indentation_flexible and strips both ends,
    # so it wins whenever lstrip-only would have worked.
    assert strategy in {
        "line_trimmed",
        "indentation_flexible",
        "block_anchor",
        "context_aware",
    }
    assert "return 2" in out


# ---------------------------------------------------------------------------
# Strategy 5: escape_normalized
# ---------------------------------------------------------------------------


def test_escape_normalized_converts_literal_newlines() -> None:
    # Actual newlines in content; model sent literal backslash-n.
    content = "line1\nline2\nline3\n"
    pattern = "line1\\nline2"  # \\n is literal backslash + 'n'
    out, count, strategy, err = fuzzy_find_and_replace(
        content, pattern, "line1\nrewritten"
    )
    assert err is None
    assert count == 1
    assert strategy == "escape_normalized"
    assert "rewritten" in out


def test_escape_normalized_skipped_when_no_escapes() -> None:
    # Pattern has no \\n / \\t / \\r; escape_normalized must short-circuit
    # so exact or later strategies get a fair shot.
    content = "hello world\n"
    pattern = "hello world"
    _, _, strategy, err = fuzzy_find_and_replace(content, pattern, "goodbye world")
    assert err is None
    # Should match via exact, not escape_normalized.
    assert strategy == "exact"


# ---------------------------------------------------------------------------
# Strategy 6: trimmed_boundary
# ---------------------------------------------------------------------------


def test_trimmed_boundary_handles_boundary_whitespace() -> None:
    # Model adds spurious whitespace to first/last lines only.
    content = "def foo():\n    x = 1\n    return x\n"
    pattern = "   def foo():   \n    x = 1\n   return x   "
    out, _, strategy, err = fuzzy_find_and_replace(
        content, pattern, "def bar():\n    x = 1\n    return x"
    )
    assert err is None
    # Could fall through multiple strategies; assert it matched somehow.
    assert strategy is not None
    assert "def bar()" in out


# ---------------------------------------------------------------------------
# Strategy 7: unicode_normalized + UNICODE_MAP coverage
# ---------------------------------------------------------------------------


def test_unicode_smart_quotes_matched() -> None:
    # File has curly quotes; model sent straight.
    content = "print(\u201chello\u201d)\n"
    pattern = 'print("hello")'
    out, count, strategy, err = fuzzy_find_and_replace(content, pattern, 'print("bye")')
    assert err is None
    assert count == 1
    assert strategy == "unicode_normalized"
    # Default semantic: substring replaced verbatim (no quote preservation
    # applied at the convenience wrapper; that's EditTool's job).
    assert 'print("bye")' in out


def test_unicode_em_dash_expansion() -> None:
    # Hermes behavior: em-dash normalizes to '--'.
    content = "flag \u2014 value\n"  # file has real em-dash
    pattern = "flag -- value"  # model sent plain ASCII
    _, count, strategy, err = fuzzy_find_and_replace(content, pattern, "flag -- v2")
    assert err is None
    assert count == 1
    assert strategy == "unicode_normalized"


def test_unicode_non_breaking_space_matched() -> None:
    # File has NBSP; pattern has regular space.
    content = "x\u00a0=\u00a01\n"
    pattern = "x = 1"
    out, _, strategy, err = fuzzy_find_and_replace(content, pattern, "x = 2")
    assert err is None
    assert strategy == "unicode_normalized"
    assert "x = 2" in out


def test_unicode_openclaw_exotic_space() -> None:
    # U+3000 (ideographic space) — in OpenClaw's list, not in Hermes's.
    content = "a\u3000b\n"
    pattern = "a b"
    _, _, strategy, err = fuzzy_find_and_replace(content, pattern, "a c")
    assert err is None
    assert strategy == "unicode_normalized"


def test_unicode_openclaw_figure_dash() -> None:
    # U+2012 figure dash — in OpenClaw's list, not originally in Hermes's.
    content = "phone: 555\u20121234\n"
    pattern = "phone: 555-1234"
    _, _, strategy, err = fuzzy_find_and_replace(content, pattern, "phone: 555-5678")
    assert err is None
    assert strategy == "unicode_normalized"


@pytest.mark.parametrize("char", sorted(UNICODE_MAP.keys()))
def test_unicode_map_characters_all_matchable(char: str) -> None:
    """Every ``UNICODE_MAP`` key participates in ``_strategy_unicode_normalized``."""
    repl = UNICODE_MAP[char]
    content = f"X{char}Y\n"
    pattern = f"X{repl}Y"
    _, _, strategy, err = fuzzy_find_and_replace(content, pattern, "XZY")
    assert err is None, f"Failed to match char U+{ord(char):04X}: {err}"
    assert strategy == "unicode_normalized"


# ---------------------------------------------------------------------------
# Strategy 8: block_anchor
# ---------------------------------------------------------------------------


def test_block_anchor_matches_first_and_last_lines() -> None:
    # Middle text differs slightly but first+last anchor exactly.
    content = (
        "def foo():\n"
        "    # this is the real comment\n"
        "    x = compute_value()\n"
        "    return x\n"
    )
    # Model sends anchors matching first+last, middle paraphrased above the
    # 0.50 single-candidate threshold.
    pattern = (
        "def foo():\n"
        "    # this is the real comment\n"
        "    x = compute_val()\n"
        "    return x"
    )
    out, _, strategy, err = fuzzy_find_and_replace(
        content,
        pattern,
        "def foo():\n    return 42",
    )
    # Earlier strategies shouldn't match; block_anchor is the one with
    # middle-similarity tolerance.
    assert err is None
    assert strategy in {"block_anchor", "context_aware"}
    assert "return 42" in out


# ---------------------------------------------------------------------------
# Strategy 9: context_aware
# ---------------------------------------------------------------------------


def test_context_aware_similar_block() -> None:
    content = "def foo():\n    a = 1\n    b = 2\n    c = 3\n    return a + b + c\n"
    # All lines are high-similarity to the pattern but not byte-identical,
    # and first/last line don't anchor (so block_anchor doesn't fire).
    pattern = "def fou():\n    a = 1\n    b = 2\n    c = 3\n    return a + b + d\n"
    _, _, strategy, err = fuzzy_find_and_replace(content, pattern, "replaced block\n")
    # Should match somewhere in the fuzzy tail — context_aware, or possibly
    # block_anchor if first/last happen to align under normalization.
    assert err is None
    assert strategy in {"context_aware", "block_anchor"}


# ---------------------------------------------------------------------------
# Fallthrough ordering — earlier strategy beats later
# ---------------------------------------------------------------------------


def test_exact_beats_line_trimmed() -> None:
    # Both would match, but exact must win.
    content = "hello\n"
    pattern = "hello"
    _, _, strategy, err = fuzzy_find_and_replace(content, pattern, "world")
    assert err is None
    assert strategy == "exact"


def test_line_trimmed_beats_whitespace_normalized() -> None:
    # Need trailing whitespace *inside* the match window so exact substring
    # search can't see the block — a single-line pattern with trailing
    # spaces would still be a substring of the content.
    content = "line1\nline2  \nline3\n"
    pattern = "line1\nline2\nline3"
    _, _, strategy, err = fuzzy_find_and_replace(content, pattern, "x")
    assert err is None
    # Exact fails (line2 has trailing spaces); line_trimmed normalizes both
    # sides' per-line whitespace and matches block-at-a-time.
    assert strategy == "line_trimmed"


# ---------------------------------------------------------------------------
# No-match / refusal semantics
# ---------------------------------------------------------------------------


def test_no_match_returns_error() -> None:
    content = "nothing\n"
    out, count, strategy, err = fuzzy_find_and_replace(
        content, "this-is-not-present", "x"
    )
    assert count == 0
    assert strategy is None
    assert err is not None
    assert "Could not find" in err
    assert out == content  # unchanged


def test_empty_old_string_refused() -> None:
    _, count, _, err = fuzzy_find_and_replace("content", "", "anything")
    assert count == 0
    assert err is not None
    assert "empty" in err


def test_identical_old_and_new_refused() -> None:
    _, count, _, err = fuzzy_find_and_replace("abc def", "def", "def")
    assert count == 0
    assert err is not None
    assert "identical" in err


# ---------------------------------------------------------------------------
# apply_replacements — right-to-left ordering with length changes
# ---------------------------------------------------------------------------


def test_apply_replacements_right_to_left() -> None:
    content = "aXbXcXd"
    matches = [(1, 2), (3, 4), (5, 6)]
    # new_string is longer than 'X' — the right-to-left order is essential.
    result = apply_replacements(content, matches, "YYY")
    assert result == "aYYYbYYYcYYYd"


def test_apply_replacements_unordered_matches() -> None:
    # Matches supplied in arbitrary order still produce the right result.
    content = "aXbXc"
    matches = [(3, 4), (1, 2)]  # out of order
    result = apply_replacements(content, matches, "YYY")
    assert result == "aYYYbYYYc"


# ---------------------------------------------------------------------------
# fuzzy_find — direct-use API
# ---------------------------------------------------------------------------


def test_fuzzy_find_returns_match_positions() -> None:
    content = "hello world hello"
    matches, strategy, err = fuzzy_find(content, "hello")
    assert err is None
    assert strategy == "exact"
    assert matches == [(0, 5), (12, 17)]


def test_fuzzy_find_empty_pattern() -> None:
    matches, strategy, err = fuzzy_find("content", "")
    assert matches == []
    assert strategy is None
    assert err == "old_string cannot be empty"


# ---------------------------------------------------------------------------
# preserve_quote_style
# ---------------------------------------------------------------------------


def test_preserve_quote_style_straight_to_curly_double() -> None:
    # File used curly; model sent straight. Rewrite straight → curly.
    original = "print(\u201chello\u201d)"
    new = 'print("world")'
    out = preserve_quote_style(original, new)
    # Both " characters rewritten to the first curly seen (left double).
    assert "\u201c" in out
    assert '"' not in out


def test_preserve_quote_style_straight_to_curly_single() -> None:
    original = "name = \u2018alice\u2019"
    new = "name = 'bob'"
    out = preserve_quote_style(original, new)
    assert "\u2018" in out
    assert "'" not in out


def test_preserve_quote_style_no_change_when_original_uses_straight() -> None:
    original = 'x = "hello"'
    new = 'y = "world"'
    out = preserve_quote_style(original, new)
    assert out == new  # no transformation


def test_preserve_quote_style_no_change_when_new_has_no_straight() -> None:
    # Original has curly, new has no straight quotes at all — nothing to do.
    original = "\u201chi\u201d"
    new = "plain text"
    out = preserve_quote_style(original, new)
    assert out == new


def test_preserve_quote_style_mixed_quotes_independent() -> None:
    # Original has curly single only. Only ' should be rewritten; " stays.
    original = "she said \u2018hi\u2019"
    new = "x = \"value\" and 'other'"
    out = preserve_quote_style(original, new)
    assert "\u2018" in out  # straight ' rewritten
    assert '"' in out  # straight " preserved


def test_preserve_quote_style_empty_inputs() -> None:
    assert preserve_quote_style("", 'x = "y"') == 'x = "y"'
    assert not preserve_quote_style("\u201chi\u201d", "")
