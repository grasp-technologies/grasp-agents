r"""
Fuzzy find-and-replace used by ``Edit`` — a chain of matching strategies,
tried in order until one returns matches.

LLM-generated ``old_string`` blocks drift in many small ways: stripped
trailing whitespace, swapped tabs for spaces, smart quotes from Markdown,
escaped ``\n`` literals that should be real newlines. Each drift class has
a targeted normalizer. Strict strategies run first so a deterministic exact
match isn't bypassed by a fuzzier one; loose similarity-based strategies
run last and only fire when everything tighter fails.

The strategies, in order:

1. ``exact`` — ``str.find`` substring.
2. ``line_trimmed`` — strip each line, then compare line-blocks.
3. ``whitespace_normalized`` — collapse runs of spaces/tabs to a single space.
4. ``indentation_flexible`` — ``lstrip`` each line.
5. ``escape_normalized`` — convert literal ``\n`` / ``\t`` / ``\r`` to the
   real characters (common when the model double-escapes).
6. ``trimmed_boundary`` — trim only the first and last line.
7. ``unicode_normalized`` — apply ``UNICODE_MAP`` to both sides and retry
   exact + line-trimmed on the normalized text; map positions back to
   original offsets (em-dash → ``--`` expands by one character).
8. ``block_anchor`` — require first and last lines to match exactly; score
   the middle with ``SequenceMatcher.ratio`` (threshold 0.50 when the anchor
   is unique, 0.70 when multiple anchors match).
9. ``context_aware`` — per-line ``SequenceMatcher`` ≥ 0.80 on at least 50%
   of lines. Loosest.

Each strategy returns ``(start, end)`` half-open ranges in the original
content. The caller enforces ``replace_all`` / uniqueness and applies the
replacements.
"""

from __future__ import annotations

import operator
import re
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

# =============================================================================
# Unicode normalization map
# =============================================================================
# Smart quotes, exotic dashes, ellipsis, and non-breaking/exotic spaces all
# normalized to their ASCII equivalents. Em-dash maps to ``--`` rather than
# ``-`` so patterns like ``--verbose`` match when the file has a real em-dash
# around the flag; the position-mapping code handles the one-char expansion.
# Non-breaking spaces are the most common real-world cause of failed matches
# (they paste in from browsers and are invisible in most editors).
UNICODE_MAP: dict[str, str] = {
    # Double quotes → "
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u201f": '"',
    # Single quotes → '
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    # Em-dash expands to two hyphens so CLI-flag idioms (``--verbose``) match.
    "\u2014": "--",
    # Other dashes → -
    "\u2010": "-",  # hyphen
    "\u2011": "-",  # non-breaking hyphen
    "\u2012": "-",  # figure dash
    "\u2013": "-",  # en-dash
    "\u2015": "-",  # horizontal bar
    "\u2212": "-",  # minus sign
    # Ellipsis
    "\u2026": "...",
    # Spaces → " "
    "\u00a0": " ",  # non-breaking
    "\u2002": " ",  # en space
    "\u2003": " ",  # em space
    "\u2004": " ",  # three-per-em
    "\u2005": " ",  # four-per-em
    "\u2006": " ",  # six-per-em
    "\u2007": " ",  # figure space
    "\u2008": " ",  # punctuation space
    "\u2009": " ",  # thin space
    "\u200a": " ",  # hair space
    "\u202f": " ",  # narrow no-break
    "\u205f": " ",  # medium mathematical
    "\u3000": " ",  # ideographic
}


def _unicode_normalize(text: str) -> str:
    """Replace every key in ``UNICODE_MAP`` with its value, left-to-right."""
    for char, repl in UNICODE_MAP.items():
        text = text.replace(char, repl)
    return text


# =============================================================================
# Public API
# =============================================================================


def fuzzy_find(
    content: str,
    old_string: str,
) -> tuple[list[tuple[int, int]], str | None, str | None]:
    """
    Run the 9-strategy chain; return the first strategy's matches.

    Returns ``(matches, strategy_name, error)``:

    * ``matches`` is a list of half-open ``(start, end)`` ranges, empty on
      failure. On success, the list has at least one element.
    * ``strategy_name`` identifies which strategy produced the matches,
      or ``None`` on failure.
    * ``error`` is a human-readable message when matching is impossible
      (empty ``old_string``, etc.); ``None`` on success.

    This is the primary find API. Callers (like ``EditTool``) use it to
    read match positions without performing replacement — needed for
    per-match quote preservation and ambiguity handling.
    """
    if not old_string:
        return [], None, "old_string cannot be empty"

    strategies: list[tuple[str, Callable[[str, str], list[tuple[int, int]]]]] = [
        ("exact", _strategy_exact),
        ("line_trimmed", _strategy_line_trimmed),
        ("whitespace_normalized", _strategy_whitespace_normalized),
        ("indentation_flexible", _strategy_indentation_flexible),
        ("escape_normalized", _strategy_escape_normalized),
        ("trimmed_boundary", _strategy_trimmed_boundary),
        ("unicode_normalized", _strategy_unicode_normalized),
        ("block_anchor", _strategy_block_anchor),
        ("context_aware", _strategy_context_aware),
    ]

    for strategy_name, strategy_fn in strategies:
        matches = strategy_fn(content, old_string)
        if matches:
            return matches, strategy_name, None

    return [], None, "Could not find a match for old_string in the file"


def apply_replacements(
    content: str,
    matches: list[tuple[int, int]],
    new_string: str,
) -> str:
    """
    Replace each ``(start, end)`` in ``matches`` with ``new_string``.

    Processing from right to left preserves offset validity for earlier
    matches — critical when ``len(new_string) != end - start``.
    """
    sorted_matches = sorted(matches, key=operator.itemgetter(0), reverse=True)
    result = content
    for start, end in sorted_matches:
        result = result[:start] + new_string + result[end:]
    return result


def fuzzy_find_and_replace(
    content: str,
    old_string: str,
    new_string: str,
    *,
    replace_all: bool = False,
) -> tuple[str, int, str | None, str | None]:
    """
    Convenience wrapper: find + replace in one call.

    Returns ``(new_content, match_count, strategy, error)``:

    * Success: ``(modified_content, count, strategy_name, None)``.
    * Failure: ``(original_content, 0, None, error_message)``.

    ``EditTool`` uses the lower-level ``fuzzy_find`` + ``apply_replacements``
    directly so it can apply ``preserve_quote_style`` between them. This
    wrapper is kept for tests and simple external callers.
    """
    if old_string == new_string:
        return content, 0, None, "old_string and new_string are identical"

    matches, strategy_name, error = fuzzy_find(content, old_string)
    if error is not None:
        return content, 0, None, error
    if not matches:
        return content, 0, None, "Could not find a match for old_string in the file"

    if len(matches) > 1 and not replace_all:
        return (
            content,
            0,
            None,
            (
                f"Found {len(matches)} matches for old_string "
                f"(strategy: {strategy_name}). "
                "Provide more context to make it unique, or pass replace_all=True."
            ),
        )

    new_content = apply_replacements(content, matches, new_string)
    return new_content, len(matches), strategy_name, None


def preserve_quote_style(original_slice: str, new_string: str) -> str:
    """
    Rewrite quotes in ``new_string`` to match the style in ``original_slice``.

    Used after a fuzzy match won via Unicode normalization: the file uses
    curly quotes, the model wrote straight, and the replacement would
    otherwise flip the file's quote convention.

    Rules per quote family:

    * **Double quotes** — if the original has both ``\u201c`` (left) and
      ``\u201d`` (right), toggle between them in ``new_string`` so open/close
      pairs alternate, starting with ``\u201c``. If only one variant is
      present, use it for all straight ``"``. If only exotic variants
      (``\u201e`` / ``\u201f``) are present, use the first seen.
    * **Single quotes** — no pair tracking (apostrophes are ambiguous); use
      the first curly variant seen in the original.
    * **No-op** if the original has no curly variants or ``new_string`` has
      no straight quotes.
    """
    if not original_slice or not new_string:
        return new_string

    result = new_string

    # Double quotes — pair-aware when both sides of a pair exist.
    if '"' in result:
        has_left = "\u201c" in original_slice
        has_right = "\u201d" in original_slice
        if has_left and has_right:
            result = _toggle_pair_quotes(result, '"', "\u201c", "\u201d")
        elif has_left:
            result = result.replace('"', "\u201c")
        elif has_right:
            result = result.replace('"', "\u201d")
        else:
            for c in ("\u201e", "\u201f"):
                if c in original_slice:
                    result = result.replace('"', c)
                    break

    # Single quotes — no pair tracking (apostrophes would break it); use the
    # first curly variant we see in the original.
    if "'" in result:
        for c in ("\u2018", "\u2019", "\u201a", "\u201b"):
            if c in original_slice:
                result = result.replace("'", c)
                break

    return result


def _toggle_pair_quotes(s: str, straight: str, open_ch: str, close_ch: str) -> str:
    """Substitute straight quotes with alternating ``open``/``close`` curlies."""
    parts: list[str] = []
    open_next = True
    for ch in s:
        if ch == straight:
            parts.append(open_ch if open_next else close_ch)
            open_next = not open_next
        else:
            parts.append(ch)
    return "".join(parts)


# =============================================================================
# Strategies
# =============================================================================


def _strategy_exact(content: str, pattern: str) -> list[tuple[int, int]]:
    """1. Exact substring match."""
    matches: list[tuple[int, int]] = []
    start = 0
    while True:
        pos = content.find(pattern, start)
        if pos == -1:
            break
        matches.append((pos, pos + len(pattern)))
        start = pos + 1
    return matches


def _strategy_line_trimmed(content: str, pattern: str) -> list[tuple[int, int]]:
    """2. Per-line ``strip`` on both sides, block-compare."""
    pattern_lines = [line.strip() for line in pattern.split("\n")]
    pattern_normalized = "\n".join(pattern_lines)

    content_lines = content.split("\n")
    content_normalized_lines = [line.strip() for line in content_lines]

    return _find_block_matches(
        content_lines,
        content_normalized_lines,
        pattern_normalized,
        len(content),
    )


def _strategy_whitespace_normalized(
    content: str, pattern: str
) -> list[tuple[int, int]]:
    r"""3. Collapse runs of ``[ \t]+`` to a single space (newlines preserved)."""

    def normalize(s: str) -> str:
        return re.sub(r"[ \t]+", " ", s)

    pattern_normalized = normalize(pattern)
    content_normalized = normalize(content)

    norm_matches = _strategy_exact(content_normalized, pattern_normalized)
    if not norm_matches:
        return []

    return _map_whitespace_positions(content, content_normalized, norm_matches)


def _strategy_indentation_flexible(content: str, pattern: str) -> list[tuple[int, int]]:
    """4. Strip leading whitespace per line, then block-compare."""
    content_lines = content.split("\n")
    content_stripped_lines = [line.lstrip() for line in content_lines]
    pattern_stripped = "\n".join(line.lstrip() for line in pattern.split("\n"))

    return _find_block_matches(
        content_lines,
        content_stripped_lines,
        pattern_stripped,
        len(content),
    )


def _strategy_escape_normalized(content: str, pattern: str) -> list[tuple[int, int]]:
    r"""
    5. Convert literal ``\n`` / ``\t`` / ``\r`` to real characters.

    Catches the case where the model sends ``"foo\\nbar"`` thinking it's
    JSON-escaping a newline — which becomes a literal backslash-n on the
    Python side. Skips when the pattern has no escape sequences (would
    duplicate the ``exact`` strategy).
    """

    def unescape(s: str) -> str:
        return s.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")

    pattern_unescaped = unescape(pattern)
    if pattern_unescaped == pattern:
        return []

    return _strategy_exact(content, pattern_unescaped)


def _strategy_trimmed_boundary(content: str, pattern: str) -> list[tuple[int, int]]:
    """
    6. Trim whitespace from only the first and last lines of the pattern.

    Useful when the model's ``old_string`` has extra leading/trailing whitespace
    on the block boundaries but the interior lines are byte-exact.
    """
    pattern_lines = pattern.split("\n")
    if not pattern_lines:
        return []

    pattern_lines[0] = pattern_lines[0].strip()
    if len(pattern_lines) > 1:
        pattern_lines[-1] = pattern_lines[-1].strip()

    modified_pattern = "\n".join(pattern_lines)
    content_lines = content.split("\n")
    pattern_line_count = len(pattern_lines)

    matches: list[tuple[int, int]] = []
    for i in range(len(content_lines) - pattern_line_count + 1):
        block_lines = content_lines[i : i + pattern_line_count]
        check_lines = block_lines.copy()
        check_lines[0] = check_lines[0].strip()
        if len(check_lines) > 1:
            check_lines[-1] = check_lines[-1].strip()

        if "\n".join(check_lines) == modified_pattern:
            start_pos, end_pos = _calculate_line_positions(
                content_lines, i, i + pattern_line_count, len(content)
            )
            matches.append((start_pos, end_pos))

    return matches


def _build_orig_to_norm_map(original: str) -> list[int]:
    """
    Index each original char by its position in the ``UNICODE_MAP``-normalized string.

    Needed because some replacements expand (``\u2014`` → ``--`` adds a char,
    ellipsis adds two). Returned list has length ``len(original) + 1`` so the
    last entry is the final normalized length (sentinel).
    """
    result: list[int] = []
    norm_pos = 0
    for char in original:
        result.append(norm_pos)
        repl = UNICODE_MAP.get(char)
        norm_pos += len(repl) if repl is not None else 1
    result.append(norm_pos)
    return result


def _map_positions_norm_to_orig(
    orig_to_norm: list[int],
    norm_matches: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Invert ``orig_to_norm`` to map normalized ranges back to original offsets."""
    norm_to_orig_start: dict[int, int] = {}
    for orig_pos, norm_pos in enumerate(orig_to_norm[:-1]):
        if norm_pos not in norm_to_orig_start:
            norm_to_orig_start[norm_pos] = orig_pos

    results: list[tuple[int, int]] = []
    orig_len = len(orig_to_norm) - 1

    for norm_start, norm_end in norm_matches:
        if norm_start not in norm_to_orig_start:
            continue
        orig_start = norm_to_orig_start[norm_start]

        orig_end = orig_start
        while orig_end < orig_len and orig_to_norm[orig_end] < norm_end:
            orig_end += 1

        results.append((orig_start, orig_end))

    return results


def _strategy_unicode_normalized(content: str, pattern: str) -> list[tuple[int, int]]:
    """
    7. Apply ``UNICODE_MAP`` on both sides, then retry ``exact`` + ``line_trimmed``.

    Skips if neither side changes under normalization (would be identical to
    earlier strategies). Position mapping goes through ``_build_orig_to_norm_map``
    to handle expansions (em-dash adds a character, ellipsis adds two).
    """
    norm_pattern = _unicode_normalize(pattern)
    norm_content = _unicode_normalize(content)
    if norm_content == content and norm_pattern == pattern:
        return []

    norm_matches = _strategy_exact(norm_content, norm_pattern)
    if not norm_matches:
        # Also try line_trimmed on the normalized text — catches the combo
        # case of whitespace drift + curly quotes.
        norm_content_lines = norm_content.split("\n")
        norm_content_trimmed_lines = [line.strip() for line in norm_content_lines]
        pattern_trimmed = "\n".join(line.strip() for line in norm_pattern.split("\n"))
        norm_matches = _find_block_matches(
            norm_content_lines,
            norm_content_trimmed_lines,
            pattern_trimmed,
            len(norm_content),
        )

    if not norm_matches:
        return []

    orig_to_norm = _build_orig_to_norm_map(content)
    return _map_positions_norm_to_orig(orig_to_norm, norm_matches)


def _strategy_block_anchor(content: str, pattern: str) -> list[tuple[int, int]]:
    """
    8. Anchor on first + last line; compare middle via ``SequenceMatcher.ratio``.

    Threshold is 0.50 when a single anchor pair matches, 0.70 when multiple —
    the higher bar applies when ambiguity is possible. Don't loosen these:
    values below ~0.40 will happily match unrelated blocks.
    """
    norm_pattern = _unicode_normalize(pattern)
    norm_content = _unicode_normalize(content)

    pattern_lines = norm_pattern.split("\n")
    if len(pattern_lines) < 2:
        return []

    first_line = pattern_lines[0].strip()
    last_line = pattern_lines[-1].strip()

    norm_content_lines = norm_content.split("\n")
    orig_content_lines = content.split("\n")
    pattern_line_count = len(pattern_lines)

    potential_matches: list[int] = []
    for i in range(len(norm_content_lines) - pattern_line_count + 1):
        if (
            norm_content_lines[i].strip() == first_line
            and norm_content_lines[i + pattern_line_count - 1].strip() == last_line
        ):
            potential_matches.append(i)

    matches: list[tuple[int, int]] = []
    candidate_count = len(potential_matches)
    threshold = 0.50 if candidate_count == 1 else 0.70

    for i in potential_matches:
        if pattern_line_count <= 2:
            similarity = 1.0
        else:
            content_middle = "\n".join(
                norm_content_lines[i + 1 : i + pattern_line_count - 1]
            )
            pattern_middle = "\n".join(pattern_lines[1:-1])
            similarity = SequenceMatcher(None, content_middle, pattern_middle).ratio()

        if similarity >= threshold:
            start_pos, end_pos = _calculate_line_positions(
                orig_content_lines, i, i + pattern_line_count, len(content)
            )
            matches.append((start_pos, end_pos))

    return matches


def _strategy_context_aware(content: str, pattern: str) -> list[tuple[int, int]]:
    """
    9. Per-line ``SequenceMatcher`` ≥ 0.80; at least 50% of lines must clear it.

    Loosest strategy. Runs only if all prior exact / structural / normalized
    attempts failed. Don't lower the 0.80 per-line threshold — it's what keeps
    "``for x in items``" from matching "``if x in items``".
    """
    pattern_lines = pattern.split("\n")
    content_lines = content.split("\n")

    if not pattern_lines:
        return []

    matches: list[tuple[int, int]] = []
    pattern_line_count = len(pattern_lines)

    for i in range(len(content_lines) - pattern_line_count + 1):
        block_lines = content_lines[i : i + pattern_line_count]

        high_similarity_count = 0
        for p_line, c_line in zip(pattern_lines, block_lines, strict=False):
            sim = SequenceMatcher(None, p_line.strip(), c_line.strip()).ratio()
            if sim >= 0.80:
                high_similarity_count += 1

        if high_similarity_count >= len(pattern_lines) * 0.5:
            start_pos, end_pos = _calculate_line_positions(
                content_lines, i, i + pattern_line_count, len(content)
            )
            matches.append((start_pos, end_pos))

    return matches


# =============================================================================
# Helpers
# =============================================================================


def _calculate_line_positions(
    content_lines: list[str],
    start_line: int,
    end_line: int,
    content_length: int,
) -> tuple[int, int]:
    """Translate ``(start_line, end_line)`` to character offsets in the original."""
    start_pos = sum(len(line) + 1 for line in content_lines[:start_line])
    end_pos = sum(len(line) + 1 for line in content_lines[:end_line]) - 1
    end_pos = min(content_length, end_pos)
    return start_pos, end_pos


def _find_block_matches(
    content_lines: list[str],
    content_normalized_lines: list[str],
    pattern_normalized: str,
    content_length: int,
) -> list[tuple[int, int]]:
    """Scan the normalized content for a block equal to ``pattern_normalized``."""
    pattern_norm_lines = pattern_normalized.split("\n")
    num_pattern_lines = len(pattern_norm_lines)

    matches: list[tuple[int, int]] = []
    for i in range(len(content_normalized_lines) - num_pattern_lines + 1):
        block = "\n".join(content_normalized_lines[i : i + num_pattern_lines])
        if block == pattern_normalized:
            start_pos, end_pos = _calculate_line_positions(
                content_lines, i, i + num_pattern_lines, content_length
            )
            matches.append((start_pos, end_pos))
    return matches


def _map_whitespace_positions(
    original: str,
    normalized: str,
    normalized_matches: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """
    Map positions from whitespace-collapsed to original, best-effort.

    Only used by ``_strategy_whitespace_normalized`` — can't use
    ``_build_orig_to_norm_map`` because whitespace collapsing isn't
    per-character substitution. Walks both strings in parallel, tracking
    runs of whitespace that collapsed to a single space.
    """
    if not normalized_matches:
        return []

    orig_to_norm: list[int] = []
    orig_idx = 0
    norm_idx = 0

    while orig_idx < len(original) and norm_idx < len(normalized):
        if original[orig_idx] == normalized[norm_idx]:
            orig_to_norm.append(norm_idx)
            orig_idx += 1
            norm_idx += 1
        elif original[orig_idx] in " \t" and normalized[norm_idx] == " ":
            orig_to_norm.append(norm_idx)
            orig_idx += 1
            if orig_idx < len(original) and original[orig_idx] not in " \t":
                norm_idx += 1
        elif original[orig_idx] in " \t":
            orig_to_norm.append(norm_idx)
            orig_idx += 1
        else:
            orig_to_norm.append(norm_idx)
            orig_idx += 1

    while orig_idx < len(original):
        orig_to_norm.append(len(normalized))
        orig_idx += 1

    norm_to_orig_start: dict[int, int] = {}
    norm_to_orig_end: dict[int, int] = {}
    for orig_pos, norm_pos in enumerate(orig_to_norm):
        if norm_pos not in norm_to_orig_start:
            norm_to_orig_start[norm_pos] = orig_pos
        norm_to_orig_end[norm_pos] = orig_pos

    original_matches: list[tuple[int, int]] = []
    for norm_start, norm_end in normalized_matches:
        if norm_start in norm_to_orig_start:
            orig_start = norm_to_orig_start[norm_start]
        else:
            orig_start = min(i for i, n in enumerate(orig_to_norm) if n >= norm_start)

        if norm_end - 1 in norm_to_orig_end:
            orig_end = norm_to_orig_end[norm_end - 1] + 1
        else:
            orig_end = orig_start + (norm_end - norm_start)

        while orig_end < len(original) and original[orig_end] in " \t":
            orig_end += 1

        original_matches.append((orig_start, min(orig_end, len(original))))

    return original_matches
