"""
Pure-Python grep over ``(path, text)`` pairs.

Backends that hold file text themselves (:class:`InMemoryFileBackend`, a
database-backed backend after its SQL prefilter) await
:func:`grep_texts_async` in their ``grep`` method and return its
:class:`GrepRawResult`. The result is shaped exactly like the
ripgrep-driven local backend's, so the ``Grep`` tool renders it the same
way:

* ``files_with_matches``: ``files`` lists matching paths; ``num_matches``
  equals the number of files.
* ``count``: ``counts`` holds ``(path, matching_lines)``.
* ``content``: ``lines`` holds ``f"{path}:{line_no}:{text}"`` for matching
  lines and ``f"{path}-{line_no}-{text}"`` for context lines (no line
  number when ``show_line_numbers`` is False). Overlapping context windows
  are merged; groups are not separated by ``--``.

Files are ordered by path components, like ``rg --sort path``. Paths whose
components under ``root`` start with a dot are skipped unless
``include_hidden`` is set, and text containing a NUL byte is treated as
binary and skipped. The regex is the ``regex`` module's ``re``-compatible
flavor with ``MULTILINE`` (plus ``DOTALL`` in multiline mode) rather than
ripgrep's Rust flavor.

Python regexes can backtrack for a very long time (``(x+x+)+y``, or
``.*a.*b`` on a long line), so every call runs under a time budget
(``time_budget``, :data:`DEFAULT_GREP_TIME_BUDGET` seconds by default) and
the engine releases the GIL while it matches. When the budget runs out the
result keeps the files searched so far and says so in ``notes``.
:func:`grep_texts` is synchronous; async code calls :func:`grep_texts_async`,
which runs it in a worker thread so the event loop keeps running.

Globs follow ripgrep with one difference: a glob containing ``/`` matches
the path relative to ``root`` (ripgrep matches it relative to the process
working directory). A glob without ``/`` matches the file name at any
depth, ``!`` negates, ``**`` spans directories and ``{a,b}`` alternates (up
to :data:`~.glob_match.MAX_BRACE_EXPANSIONS` alternatives).
"""

from __future__ import annotations

import asyncio
import fnmatch
import time
from typing import TYPE_CHECKING

import regex

from .base import GrepOutputMode, GrepRawResult
from .glob_match import expand_braces, match_segments

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

DEFAULT_GREP_TIME_BUDGET = 5.0
"""Seconds one :func:`grep_texts` call may spend searching."""

MAX_PATTERN_LENGTH = 1_000
"""Longest regex, and longest glob, :func:`grep_texts` accepts (characters)."""

# A subset of ``rg --type-list``.
# cspell:ignore jspx sarif jpynb mdwn tcsh
FILE_TYPE_GLOBS: dict[str, tuple[str, ...]] = {
    "c": ("*.[chH]", "*.[chH].in", "*.cats"),
    "config": ("*.cfg", "*.conf", "*.config", "*.ini"),
    "cpp": ("*.[ChH]", "*.[ch]pp", "*.[ch]xx", "*.cc", "*.hh", "*.inl"),
    "css": ("*.css", "*.scss"),
    "csv": ("*.csv",),
    "go": ("*.go",),
    "html": ("*.ejs", "*.htm", "*.html"),
    "java": ("*.java", "*.jsp", "*.jspx", "*.properties"),
    "js": ("*.cjs", "*.js", "*.jsx", "*.mjs", "*.vue"),
    "json": ("*.json", "*.sarif", "composer.lock"),
    "jsonl": ("*.jsonl",),
    "jupyter": ("*.ipynb", "*.jpynb"),
    "kotlin": ("*.kt", "*.kts"),
    "lua": ("*.lua",),
    "markdown": ("*.markdown", "*.md", "*.mdown", "*.mdwn", "*.mdx", "*.mkd"),
    "md": ("*.markdown", "*.md", "*.mdown", "*.mdwn", "*.mdx", "*.mkd"),
    "php": ("*.php", "*.phtml"),
    "py": ("*.py", "*.pyi"),
    "r": ("*.R", "*.Rmd", "*.Rnw", "*.r"),
    "rst": ("*.rst",),
    "ruby": ("*.gemspec", "*.rb", "*.rbw", "Gemfile", "Rakefile"),
    "rust": ("*.rs",),
    "scala": ("*.sbt", "*.scala"),
    "sh": ("*.bash", "*.csh", "*.ksh", "*.sh", "*.tcsh", "*.zsh"),
    "sql": ("*.psql", "*.sql"),
    "swift": ("*.swift",),
    "tex": ("*.bib", "*.cls", "*.ltx", "*.sty", "*.tex"),
    "toml": ("*.toml", "Cargo.lock"),
    "ts": ("*.cts", "*.mts", "*.ts", "*.tsx"),
    "txt": ("*.txt",),
    "xml": ("*.xml", "*.xsd", "*.xsl", "*.xslt", "*.xhtml"),
    "yaml": ("*.yaml", "*.yml"),
}


def grep_texts(
    files: Iterable[tuple[Path, str]],
    pattern: str,
    *,
    root: Path,
    glob: str | None = None,
    file_type: str | None = None,
    case_insensitive: bool = False,
    multiline: bool = False,
    output_mode: GrepOutputMode = "files_with_matches",
    show_line_numbers: bool = True,
    before_context: int | None = None,
    after_context: int | None = None,
    context: int | None = None,
    include_hidden: bool = False,
    time_budget: float = DEFAULT_GREP_TIME_BUDGET,
) -> GrepRawResult:
    """
    Search ``files`` for ``pattern`` the way the local ``rg`` backend does.

    ``files`` may include paths outside ``root``; they are ignored, so a
    caller can pass a coarse candidate set. ``root`` may be a directory or
    a single file. The other keyword arguments mirror
    :meth:`FileBackend.grep`.

    ``time_budget`` is the time in seconds the whole call may take. When it
    runs out, the result covers the files searched completely so far and
    ``notes`` says where the search stopped. This function blocks for up to
    that long; call :func:`grep_texts_async` from async code.

    Raises:
        ValueError: ``pattern`` is not a valid regex, ``pattern`` or ``glob``
            is longer than :data:`MAX_PATTERN_LENGTH`, ``glob`` has too many
            brace alternatives, or ``file_type`` is not a known type.

    """
    deadline = time.monotonic() + time_budget
    _check_length("Pattern", pattern)
    if glob is not None:
        _check_length("Glob", glob)
    compiled = _compile(pattern, case_insensitive=case_insensitive, multiline=multiline)

    type_globs: tuple[str, ...] | None = None
    if file_type is not None:
        type_globs = FILE_TYPE_GLOBS.get(file_type)
        if type_globs is None:
            raise ValueError(f"unrecognized file type: {file_type}")

    if context is not None:
        before = after = context
    else:
        before = before_context or 0
        after = after_context or 0

    glob_filter = _GlobFilter(glob) if glob is not None else None
    selected: list[tuple[Path, str]] = []
    for path, text in files:
        rel = _relative_parts(path, root)
        if rel is None:
            continue
        if rel and not include_hidden and any(part.startswith(".") for part in rel):
            continue
        if glob_filter is not None and rel and not glob_filter.selects(rel):
            continue
        if type_globs is not None and not any(
            fnmatch.fnmatchcase(path.name, g) for g in type_globs
        ):
            continue
        if "\x00" in text:
            continue
        selected.append((path, text))
    selected.sort(key=lambda pair: pair[0].parts)

    found_files: list[Path] = []
    counts: list[tuple[Path, int]] = []
    rendered: list[str] = []
    notes: list[str] = []
    num_matches = 0
    num_files_matched = 0
    for searched, (path, text) in enumerate(selected):
        lines = _split_lines(text)
        try:
            spans = _match_spans(
                compiled,
                text,
                lines,
                multiline=multiline,
                first_only=output_mode == "files_with_matches",
                deadline=deadline,
            )
        except TimeoutError:
            notes.append(
                f"The search stopped after {time_budget:g}s in {path}, so it covers "
                f"only {searched} of {len(selected)} files. Try a simpler pattern "
                "(nested or repeated wildcards such as `(a+)+` or `.*a.*b` are "
                "slow on long lines) or a narrower path or glob."
            )
            break
        if not spans:
            continue
        num_files_matched += 1
        hits = len(spans) if multiline else sum(end - start + 1 for start, end in spans)
        if output_mode == "files_with_matches":
            found_files.append(path)
            num_matches += 1
        elif output_mode == "count":
            counts.append((path, hits))
            num_matches += hits
        else:
            rendered.extend(
                _render_content(
                    path,
                    lines,
                    spans,
                    before=before,
                    after=after,
                    show_line_numbers=show_line_numbers,
                )
            )
            num_matches += hits
    return GrepRawResult(
        files=found_files,
        counts=counts,
        lines=rendered,
        num_matches=num_matches,
        num_files_matched=num_files_matched,
        notes=notes,
    )


async def grep_texts_async(
    files: Iterable[tuple[Path, str]],
    pattern: str,
    *,
    root: Path,
    glob: str | None = None,
    file_type: str | None = None,
    case_insensitive: bool = False,
    multiline: bool = False,
    output_mode: GrepOutputMode = "files_with_matches",
    show_line_numbers: bool = True,
    before_context: int | None = None,
    after_context: int | None = None,
    context: int | None = None,
    include_hidden: bool = False,
    time_budget: float = DEFAULT_GREP_TIME_BUDGET,
) -> GrepRawResult:
    """
    Run :func:`grep_texts` in a worker thread, so the event loop keeps running.

    Takes the same arguments. ``files`` is consumed in the worker thread:
    pass a list, or a generator over data nothing changes meanwhile. A
    cancelled caller stops waiting at once, but the thread runs on until the
    search finishes or ``time_budget`` runs out.
    """
    return await asyncio.to_thread(
        grep_texts,
        files,
        pattern,
        root=root,
        glob=glob,
        file_type=file_type,
        case_insensitive=case_insensitive,
        multiline=multiline,
        output_mode=output_mode,
        show_line_numbers=show_line_numbers,
        before_context=before_context,
        after_context=after_context,
        context=context,
        include_hidden=include_hidden,
        time_budget=time_budget,
    )


def _check_length(kind: str, value: str) -> None:
    if len(value) > MAX_PATTERN_LENGTH:
        raise ValueError(
            f"{kind} is {len(value):,} characters, over the limit of "
            f"{MAX_PATTERN_LENGTH:,}. Search for something shorter and more specific."
        )


def _compile(
    pattern: str, *, case_insensitive: bool, multiline: bool
) -> regex.Pattern[str]:
    flags = regex.MULTILINE
    if case_insensitive:
        flags |= regex.IGNORECASE
    if multiline:
        flags |= regex.DOTALL
    try:
        return regex.compile(pattern, flags)
    except regex.error as exc:
        raise ValueError(f"Invalid regex {pattern!r}: {exc}") from exc


def _remaining(deadline: float) -> float:
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError
    return left


def _relative_parts(path: Path, root: Path) -> tuple[str, ...] | None:
    if path == root:
        return ()
    try:
        return path.relative_to(root).parts
    except ValueError:
        return None


def _split_lines(text: str) -> list[str]:
    if not text:
        return []
    lines = text.split("\n")
    if text.endswith("\n"):
        lines.pop()
    return lines


def _match_spans(
    compiled: regex.Pattern[str],
    text: str,
    lines: list[str],
    *,
    multiline: bool,
    first_only: bool,
    deadline: float,
) -> list[tuple[int, int]]:
    """
    0-based inclusive line spans that contain a match.

    Line mode yields one single-line span per matching line. Multiline mode
    yields one span per match, with overlapping or touching spans merged.
    ``first_only`` stops at the first span. Once ``deadline`` (a
    :func:`time.monotonic` value) has passed, :class:`TimeoutError`
    propagates.
    """
    spans: list[tuple[int, int]] = []
    if not multiline:
        for i, line in enumerate(lines):
            if compiled.search(line, concurrent=True, timeout=_remaining(deadline)):
                spans.append((i, i))
                if first_only:
                    break
        return spans

    starts: list[int] = []
    offset = 0
    for line in lines:
        starts.append(offset)
        offset += len(line) + 1
    if not starts:
        return spans

    def line_of(pos: int) -> int:
        lo, hi = 0, len(starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if starts[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo

    for m in compiled.finditer(text, concurrent=True, timeout=_remaining(deadline)):
        first = line_of(m.start())
        last = line_of(max(m.start(), m.end() - 1))
        if spans and first <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], last))
        else:
            spans.append((first, last))
        if first_only:
            break
    return spans


def _render_content(
    path: Path,
    lines: list[str],
    spans: list[tuple[int, int]],
    *,
    before: int,
    after: int,
    show_line_numbers: bool,
) -> list[str]:
    match_lines: set[int] = set()
    span_at: dict[int, tuple[int, int]] = {}
    for start, end in spans:
        span_at[start] = (start, end)
        match_lines.update(range(start, end + 1))

    shown: set[int] = set(match_lines)
    for start, end in spans:
        shown.update(range(max(0, start - before), start))
        shown.update(range(end + 1, min(len(lines), end + 1 + after)))

    out: list[str] = []
    i = 0
    ordered = sorted(shown)
    while i < len(ordered):
        n = ordered[i]
        if n in span_at:
            start, end = span_at[n]
            text = "\n".join(lines[start : end + 1])
            out.append(_render_line(path, start, text, ":", show_line_numbers))
            while i < len(ordered) and ordered[i] <= end:
                i += 1
            continue
        if n in match_lines:
            i += 1
            continue
        out.append(_render_line(path, n, lines[n], "-", show_line_numbers))
        i += 1
    return out


def _render_line(
    path: Path, index: int, text: str, sep: str, show_line_numbers: bool
) -> str:
    if show_line_numbers:
        return f"{path}{sep}{index + 1}{sep}{text}"
    return f"{path}{sep}{text}"


class _GlobFilter:
    """A ripgrep-style ``--glob``, parsed once per search."""

    def __init__(self, glob: str) -> None:
        self._negated = glob.startswith("!")
        self._names: list[str] = []
        self._paths: list[list[str]] = []
        for pattern in expand_braces(glob[1:] if self._negated else glob):
            if "/" in pattern.rstrip("/"):
                self._paths.append(pattern.strip("/").split("/"))
            else:
                self._names.append(pattern.rstrip("/"))

    def selects(self, rel: tuple[str, ...]) -> bool:
        if self._negated:
            return not self._hits(rel, match_dirs=True)
        return self._hits(rel, match_dirs=False)

    def _hits(self, rel: tuple[str, ...], *, match_dirs: bool) -> bool:
        candidates = rel if match_dirs else rel[-1:]
        if any(
            fnmatch.fnmatchcase(part, name)
            for name in self._names
            for part in candidates
        ):
            return True
        for parts in self._paths:
            if match_segments(parts, rel):
                return True
            if match_dirs and any(
                match_segments(parts, rel[:k]) for k in range(1, len(rel))
            ):
                return True
        return False


__all__ = [
    "DEFAULT_GREP_TIME_BUDGET",
    "FILE_TYPE_GLOBS",
    "MAX_PATTERN_LENGTH",
    "grep_texts",
    "grep_texts_async",
]
