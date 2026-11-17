import importlib
import sys
import textwrap
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from grasp_agents.evals import FunctionScorer, Score, ScoreContext, ScoreReason, scorer
from grasp_agents.evals._util import (
    CODE_SCHEME,
    UNKNOWN_SOURCE,
    _parsed_text,
    code_hash,
    code_scheme,
    identity_hash,
)
from grasp_agents.evals.scorer import normalize_scores


def test_mapping_values_are_named_by_their_keys() -> None:
    scores = normalize_scores(
        "judge", {"accuracy": Score(name="draft", value=True), "tone": "warm"}
    )
    assert [s.name for s in scores] == ["accuracy", "tone"]
    assert all(s.scorer == "judge" for s in scores)


def test_non_finite_values_are_unscored() -> None:
    (from_scalar,) = normalize_scores("j", float("nan"))
    (from_score,) = normalize_scores("j", Score(name="q", value=float("inf")))
    for score in (from_scalar, from_score):
        assert score.value is None
        assert score.reason == ScoreReason.NON_FINITE_VALUE


def test_one_output_cannot_repeat_a_score_name() -> None:
    with pytest.raises(ValueError, match="several scores named"):
        normalize_scores("j", [Score(name="q", value=1.0), Score(name="q", value=0.5)])


def test_none_means_not_applicable() -> None:
    assert normalize_scores("j", None) == []


# --- Identity ---


@pytest.fixture
def modules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> "ModuleWriter":
    monkeypatch.syspath_prepend(str(tmp_path))
    return ModuleWriter(tmp_path)


class ModuleWriter:
    """Writes modules to a directory on ``sys.path`` and imports them."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self, name: str, source: str) -> ModuleType:
        (self.root / f"{name}.py").write_text(textwrap.dedent(source))
        sys.modules.pop(name, None)
        return importlib.import_module(name)


def test_docstrings_comments_and_layout_leave_the_code_hash(
    modules: ModuleWriter,
) -> None:
    plain = modules.load(
        "score_plain",
        """
        def exact(ctx):
            return ctx.output == ctx.reference
        """,
    )
    documented = modules.load(
        "score_documented",
        '''
        def exact(ctx):
            """Whether the output is the reference."""
            # Compare as stored.
            return (
                ctx.output == ctx.reference
            )
        ''',
    )
    changed = modules.load(
        "score_changed",
        """
        def exact(ctx):
            return ctx.output != ctx.reference
        """,
    )
    assert code_hash(plain.exact) == code_hash(documented.exact)
    assert code_hash(plain.exact) != code_hash(changed.exact)
    assert code_scheme(code_hash(plain.exact)) == CODE_SCHEME


def test_helpers_and_constants_a_scorer_uses_are_hashed(modules: ModuleWriter) -> None:
    modules.load("helpers_v1", "def norm(text):\n    return text.strip()\n")
    modules.load(
        "helpers_v1_doc",
        'def norm(text):\n    """Trim."""\n    return text.strip()\n',
    )
    modules.load("helpers_v2", "def norm(text):\n    return text.strip().lower()\n")
    body = """
        {imports}
        PROMPT = {prompt!r}

        def exact(ctx):
            return {call}(ctx.output) == ctx.reference and bool(PROMPT)
        """

    def scorer_with(name: str, imports: str, call: str, prompt: str = "v1") -> Any:
        source = body.format(imports=imports, call=call, prompt=prompt)
        return modules.load(name, source).exact

    imported = scorer_with("s_imported", "from helpers_v1 import norm", "norm")
    documented = scorer_with("s_documented", "from helpers_v1_doc import norm", "norm")
    edited = scorer_with("s_edited", "from helpers_v2 import norm", "norm")
    reprompted = scorer_with(
        "s_reprompted", "from helpers_v1 import norm", "norm", prompt="v2"
    )
    via_module = scorer_with("s_module", "import helpers_v1 as h", "h.norm")
    via_module_v2 = scorer_with("s_module_v2", "import helpers_v2 as h", "h.norm")
    assert code_hash(imported) == code_hash(documented)
    assert code_hash(imported) != code_hash(edited)
    assert code_hash(imported) != code_hash(reprompted)
    assert code_hash(via_module) != code_hash(via_module_v2)


def test_declared_identity_parts_are_part_of_the_fingerprint(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.md"
    prompt.write_text("Grade strictly.")

    def judge(ctx: ScoreContext[Any, Any, Any]) -> bool:
        return bool(ctx.output)

    plain = FunctionScorer(judge).describe()
    declared = FunctionScorer(judge, identity=[prompt, "model-a"]).describe()
    assert plain.fingerprint is None
    assert declared.fingerprint is not None
    prompt.write_text("Grade leniently.")
    assert FunctionScorer(judge, identity=[prompt, "model-a"]).describe() != declared
    assert (
        FunctionScorer(judge, identity=[prompt, "model-b"]).describe().fingerprint
        != FunctionScorer(judge, identity=[prompt, "model-a"]).describe().fingerprint
    )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        FunctionScorer(judge, identity=[tmp_path / "missing.md"]).describe()


def test_lower_is_better_names_the_scores() -> None:
    def defects(ctx: ScoreContext[Any, Any, Any]) -> float:
        return 0.0

    assert FunctionScorer(defects, lower_is_better=True).lower_is_better == {"defects"}
    assert scorer(lower_is_better={"a", "b"})(defects).lower_is_better == {"a", "b"}
    assert not FunctionScorer(defects).lower_is_better


def _hashes(
    tmp_path: Path,
    files: dict[str, str],
    values: tuple[str, str],
    attr: str = "score",
) -> list[str | None]:
    """``code_hash(main.<attr>)`` with ``$X`` in ``files`` set to each value."""
    names = {
        rel.removesuffix(".py").removesuffix("/__init__").replace("/", ".")
        for rel in files
    }
    found: list[str | None] = []
    for tag, value in zip("ab", values, strict=True):
        root = tmp_path / tag
        for rel, text in files.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(textwrap.dedent(text).replace("$X", value))
        sys.path.insert(0, str(root))
        try:
            found.append(code_hash(getattr(importlib.import_module("main"), attr)))
        finally:
            sys.path.remove(str(root))
            for name in names:
                sys.modules.pop(name, None)
    return found


_DEPENDENCIES: dict[str, tuple[dict[str, str], tuple[str, str], str]] = {
    "constant list": (
        {"main.py": "WORDS = [$X]\ndef score(ctx):\n    return ctx in WORDS\n"},
        ('"yes"', '"yes", "ok"'),
        "score",
    ),
    "constant dict": (
        {"main.py": 'RUBRIC = {"a": $X}\ndef score(ctx):\n    return RUBRIC[ctx]\n'},
        ("1", "2"),
        "score",
    ),
    "compiled pattern": (
        {
            "main.py": """
                import re
                PAT = re.compile($X)
                def score(ctx):
                    return bool(PAT.search(ctx))
                """
        },
        ('r"\\d+"', 'r"\\w+"'),
        "score",
    ),
    "class attribute": (
        {
            "main.py": """
                PROMPT = $X
                class Judge:
                    prompt = PROMPT
                    def score(self, ctx):
                        return self.prompt in ctx
                """
        },
        ('"strict"', '"lenient"'),
        "Judge",
    ),
    "method default": (
        {
            "main.py": """
                LIMIT = $X
                class Judge:
                    def score(self, ctx, limit=LIMIT):
                        return ctx > limit
                """
        },
        ("0.5", "0.9"),
        "Judge",
    ),
    "inherited method": (
        {
            "base.py": """
                class Base:
                    def score(self, ctx):
                        return ctx == $X
                """,
            "main.py": "from base import Base\nclass Judge(Base):\n    pass\n",
        },
        ("1", "2"),
        "Judge",
    ),
    "dotted module path": (
        {
            "pkg/__init__.py": "",
            "pkg/sub.py": "def norm(x):\n    return x + $X\n",
            "main.py": """
                import pkg.sub
                def score(ctx):
                    return pkg.sub.norm(ctx)
                """,
        },
        ("1", "2"),
        "score",
    ),
    "instance settings": (
        {
            "main.py": """
                class Checker:
                    def __init__(self, words):
                        self.words = words
                    def __call__(self, x):
                        return x in self.words
                CHECK = Checker(words=($X,))
                def score(ctx):
                    return CHECK(ctx)
                """
        },
        ('"a"', '"a", "b"'),
        "score",
    ),
    "wrapping decorator": (
        {
            "wrappers.py": """
                import functools
                def normalized(fn):
                    @functools.wraps(fn)
                    def wrapper(ctx):
                        return fn(ctx.$X())
                    return wrapper
                """,
            "main.py": """
                from wrappers import normalized
                @normalized
                def score(ctx):
                    return ctx == "a"
                """,
        },
        ("lower", "strip"),
        "score",
    ),
    "cached helper": (
        {
            "main.py": """
                import functools
                @functools.cache
                def rubric():
                    return $X
                def score(ctx):
                    return rubric() in ctx
                """
        },
        ('"strict"', '"lenient"'),
        "score",
    ),
    "helper first reached deep": (
        {
            "helper.py": "def check(x):\n    return x == $X\n",
            "main.py": """
                from helper import check
                def score(ctx):
                    return a1(ctx) and check(ctx)
                def a1(x): return a2(x)
                def a2(x): return a3(x)
                def a3(x): return a4(x)
                def a4(x): return check(x)
                """,
        },
        ("1", "2"),
        "score",
    ),
}


@pytest.mark.parametrize("case", list(_DEPENDENCIES))
def test_a_changed_dependency_changes_the_code_hash(tmp_path: Path, case: str) -> None:
    files, values, attr = _DEPENDENCIES[case]
    first, second = _hashes(tmp_path, files, values, attr)
    assert first is not None
    assert second is not None
    assert first != second


def test_library_callables_count_by_name_and_state_by_type(tmp_path: Path) -> None:
    files = {
        "main.py": """
            import json
            cache = {}
            def make(normalize=str.strip, dump=json.dumps):
                def inner(ctx):
                    cache[ctx] = dump(normalize(ctx))
                    return cache[ctx] $X "a"
                return inner
            score = make()
            """
    }
    first, second = _hashes(tmp_path, files, ("==", "!="))
    assert first is not None
    assert first != second
    sys.path.insert(0, str(tmp_path / "a"))
    try:
        main = importlib.import_module("main")
        before = code_hash(main.score)
        main.score(" x ")
        assert code_hash(main.score) == before
    finally:
        sys.path.remove(str(tmp_path / "a"))
        sys.modules.pop("main", None)


def test_code_without_source_never_matches() -> None:
    namespace: dict[str, Any] = {}
    exec("def judge(ctx):\n    return True\n", namespace)  # ruff: ignore[exec-builtin]
    info = FunctionScorer(namespace["judge"]).describe()
    assert info.source == UNKNOWN_SOURCE
    assert not info.matches(info)
    with pytest.raises(ValueError, match="source is unavailable"):
        identity_hash([namespace["judge"]])
    assert identity_hash([len]) != identity_hash([str.strip])


def test_directory_identities_ignore_caches_and_hidden_files(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "judge.md").write_text("Grade strictly.")
    before = identity_hash([prompts])
    (prompts / "__pycache__").mkdir()
    (prompts / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"\x00")
    (prompts / ".DS_Store").write_bytes(b"\x00")
    assert identity_hash([prompts]) == before
    (prompts / "judge.md").write_text("Grade leniently.")
    assert identity_hash([prompts]) != before


def test_parsed_code_leaves_out_empty_fields() -> None:
    # Python versions differ in which empty AST fields they spell out.
    text = _parsed_text("def f(x):\n    return x\n")
    assert "type_params" not in text
    assert "decorator_list" not in text


def test_library_scorers_record_no_source() -> None:
    from grasp_agents.evals.validation import LabelAgreement

    assert LabelAgreement(scores=["q"]).describe().source is None
