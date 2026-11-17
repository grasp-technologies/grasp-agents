import importlib
import sys
import textwrap
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from grasp_agents.evals import FunctionScorer, Score, ScoreContext, ScoreReason, scorer
from grasp_agents.evals._util import CODE_SCHEME, code_hash, code_scheme
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

    assert FunctionScorer(defects, lower_is_better=True).lower_is_better == {
        "defects"
    }
    assert scorer(lower_is_better={"a", "b"})(defects).lower_is_better == {"a", "b"}
    assert not FunctionScorer(defects).lower_is_better
