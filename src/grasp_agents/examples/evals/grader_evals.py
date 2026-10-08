"""
Demo: evaluating a short-answer grader built from grasp-agents processors.

The system under test is a two-step ``SequentialWorkflow``: an analyzer that
checks which of the reference answer's key points a student's answer covers,
and a writer that turns the analysis into a verdict and feedback. ``v1`` is a
naive first attempt; ``v2`` adds stemming, synonyms and negation handling.
The writer flips borderline verdicts at random (``noise``) to behave like a
sampled model, so repetitions and pass^k are meaningful.

Everything runs offline. ``llm_grader()`` builds the same component as an
``LLMAgent`` for running against a real model instead.

Run from the repo root, e.g.::

    python -m grasp_agents.evals run \\
        src/grasp_agents/examples/evals/grader_evals.py:grader_v1 --split dev
"""

import random
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from grasp_agents.evals import (
    EvalContext,
    Evaluation,
    Example,
    FunctionPairwiseJudge,
    PairwiseContext,
    PairwiseVerdict,
    PassHatK,
    PassRate,
    ProcessorTask,
    Score,
    scorer,
)
from grasp_agents.evals.metrics import Measure, Percentile
from grasp_agents.processors.processor import Processor
from grasp_agents.types.events import Event, ProcPayloadOutEvent
from grasp_agents.workflow.sequential_workflow import SequentialWorkflow

DATA = Path(__file__).parent / "data" / "short_answers.jsonl"

type Verdict = Literal["correct", "partial", "incorrect"]

# --- Data ---


class Submission(BaseModel):
    question: str
    reference_answer: str
    student_answer: str


class TeacherGrade(BaseModel):
    verdict: Verdict
    # The mistake or omission that good feedback must name, if any.
    key_issue: str | None = None


class Analysis(BaseModel):
    key_points: list[str]
    covered: list[str]
    contradicted: list[str] = Field(default_factory=list)


class Grade(BaseModel):
    verdict: Verdict
    feedback: str


# --- The system under test ---

_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the "
    "their them they this to was were which with into than then also only "
    "because through around just what does each more about there these those".split()
)
_SYNONYMS = {
    "sun": "sunlight",
    "h2o": "water",
    "co2": "carbon",
    "pull": "attract",
    "pulls": "attract",
    "things": "objects",
    "push": "pumps",
    "pushes": "pumps",
    "bigger": "greater",
    "larger": "greater",
    "12": "twelve",
}
_NEGATIONS = frozenset({"not", "no", "never", "doesn't", "don't", "isn't", "aren't"})


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _stem(word: str) -> str:
    if word.endswith("'s"):
        return word[:-2]
    if len(word) > 4 and word.endswith(("ches", "shes", "sses", "xes")):
        return word[:-2]
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _normalize(word: str) -> str:
    return _stem(_SYNONYMS.get(word, word))


def _key_points(reference: str) -> list[str]:
    seen: dict[str, None] = {}
    for word in _tokens(reference):
        if len(word) > 3 and word not in _STOPWORDS:
            seen.setdefault(word, None)
    return list(seen)


class Analyzer(Processor[Submission, Analysis, None]):
    """Which key points of the reference answer the student's answer covers."""

    def __init__(self, name: str = "analyzer", *, version: str = "v1") -> None:
        super().__init__(name=name)
        self.version = version

    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[Submission] | None = None,
        exec_id: str,
        step: int | None = None,
    ) -> AsyncIterator[Event[Any]]:
        for submission in in_args or []:
            yield ProcPayloadOutEvent(
                data=self._analyze(submission), source=self.name, exec_id=exec_id
            )

    def _analyze(self, submission: Submission) -> Analysis:
        points = _key_points(submission.reference_answer)
        words = _tokens(submission.student_answer)
        if self.version == "v1":
            present = set(words)
            covered = [p for p in points if p in present]
            return Analysis(key_points=points, covered=covered)
        normalized = [_normalize(w) for w in words]
        covered: list[str] = []
        contradicted: list[str] = []
        for point in points:
            target = _normalize(point)
            for i, word in enumerate(normalized):
                if word == target:
                    window = words[max(0, i - 3) : i]
                    if _NEGATIONS.intersection(window):
                        contradicted.append(point)
                    else:
                        covered.append(point)
                    break
        return Analysis(key_points=points, covered=covered, contradicted=contradicted)


class FeedbackWriter(Processor[Analysis, Grade, None]):
    """Turns an analysis into a verdict and feedback."""

    def __init__(
        self, name: str = "writer", *, version: str = "v1", noise: float = 0.15
    ) -> None:
        super().__init__(name=name)
        self.version = version
        self.noise = noise

    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[Analysis] | None = None,
        exec_id: str,
        step: int | None = None,
    ) -> AsyncIterator[Event[Any]]:
        for analysis in in_args or []:
            yield ProcPayloadOutEvent(
                data=self._write(analysis), source=self.name, exec_id=exec_id
            )

    def _write(self, analysis: Analysis) -> Grade:
        share = len(analysis.covered) / max(len(analysis.key_points), 1)
        missing = [p for p in analysis.key_points if p not in analysis.covered]
        verdict: Verdict
        if analysis.contradicted:
            verdict = "incorrect"
        elif share >= 0.75:
            verdict = "correct"
        elif share >= 0.3:
            verdict = "partial"
        else:
            verdict = "incorrect"
        # A sampled model is not perfectly consistent on borderline answers.
        if 0.3 <= share < 0.9 and random.random() < self.noise:  # noqa: S311
            verdict = "partial" if verdict != "partial" else "correct"
        if self.version == "v1":
            feedback = {
                "correct": "Good job!",
                "partial": "Some points are missing.",
                "incorrect": "This is not right, please review the material.",
            }[verdict]
        elif analysis.contradicted:
            feedback = f"Careful: {', '.join(analysis.contradicted)} is stated wrongly."
        elif missing and verdict != "correct":
            feedback = f"You missed: {', '.join(missing[:3])}."
        else:
            feedback = "Correct and complete."
        return Grade(verdict=verdict, feedback=feedback)


def build_grader(
    version: str = "v1", *, noise: float = 0.15
) -> Processor[Submission, Grade, None]:
    return SequentialWorkflow[Submission, Grade, None](
        name="grader",
        subprocs=[
            Analyzer(version=version),
            FeedbackWriter(version=version, noise=noise),
        ],
    )


def llm_grader(llm: Any) -> Processor[Submission, Grade, None]:
    """
    The same component as an LLM agent (pass any grasp-agents ``LLM``, built
    with ``apply_output_schema_via_provider=True`` for JSON output).
    """
    from grasp_agents.agent.llm_agent import LLMAgent  # noqa: PLC0415

    return LLMAgent[Submission, Grade, None](
        name="grader",
        llm=llm,
        sys_prompt=(
            "You grade a student's short answer against the reference answer. "
            "Verdict: correct (all key points), partial (some), incorrect (none, or "
            "a key point stated wrongly). Feedback: one sentence naming what is "
            "missing or wrong."
        ),
    )


# --- Scorers ---

type Ctx = EvalContext[Submission, Grade, TeacherGrade]


@scorer(version="1")
def agrees_with_teacher(ctx: Ctx) -> Score:
    assert ctx.reference is not None
    return Score(
        name="agrees_with_teacher",
        value=ctx.output.verdict == ctx.reference.verdict,
        explanation=f"teacher: {ctx.reference.verdict}, grader: {ctx.output.verdict}",
    )


@scorer(version="1")
def names_key_issue(ctx: Ctx) -> Score | None:
    """Does the feedback name what is wrong or missing? N/A when nothing is."""
    if ctx.reference is None or ctx.reference.key_issue is None:
        return None
    issue = ctx.reference.key_issue
    hit = issue.lower() in ctx.output.feedback.lower()
    return Score(
        name="names_key_issue",
        value=hit,
        explanation=f"expected '{issue}' in: {ctx.output.feedback}",
    )


@scorer(version="1")
def feedback_concise(ctx: Ctx) -> bool:
    return len(ctx.output.feedback) <= 160


@scorer(name="feedback_quality", version="1")
def feedback_quality_v1(ctx: Ctx) -> Score:
    """
    A lenient stand-in for an LLM judge (code, so it is recorded as such):
    any non-trivial feedback passes.
    """
    ok = len(ctx.output.feedback.split()) >= 3
    return Score(name="feedback_quality", value=ok, explanation=ctx.output.feedback)


@scorer(name="feedback_quality", version="2")
def feedback_quality_v2(ctx: Ctx) -> Score:
    """A stricter stand-in judge: feedback on an imperfect answer must be specific."""
    assert ctx.reference is not None
    if ctx.reference.verdict == "correct":
        return Score(
            name="feedback_quality", value=True, explanation="answer was correct"
        )
    words = set(_tokens(ctx.output.feedback))
    points = set(_key_points(ctx.input.reference_answer))
    specific = bool(words & points)
    return Score(
        name="feedback_quality",
        value=specific,
        explanation=f"mentions a key point: {specific} — {ctx.output.feedback}",
    )


def _specificity(
    ctx: PairwiseContext[Submission, Any, TeacherGrade], grade: Any
) -> int:
    feedback = grade["feedback"] if isinstance(grade, dict) else grade.feedback
    return len(set(_tokens(feedback)) & set(_key_points(ctx.input.reference_answer)))


def _more_specific(
    ctx: PairwiseContext[Submission, Any, TeacherGrade],
) -> PairwiseVerdict:
    first, second = _specificity(ctx, ctx.first), _specificity(ctx, ctx.second)
    if first == second:
        return PairwiseVerdict(winner="tie")
    return PairwiseVerdict(
        winner="first" if first > second else "second",
        explanation=f"key points named: {first} vs {second}",
    )


specific_feedback_judge = FunctionPairwiseJudge(
    _more_specific, name="specificity", annotator="CODE"
)


# --- Dataset checks ---


def key_issue_implies_imperfect(example: Example[Any, Any]) -> str | None:
    grade = example.reference
    if grade is not None and grade.key_issue and grade.verdict == "correct":
        return "a correct answer cannot have a key issue"
    return None


# --- Evaluations ---

_METRICS = [
    PassRate("agrees_with_teacher"),
    PassHatK("agrees_with_teacher", 3),
    PassRate("names_key_issue"),
    PassRate("feedback_concise"),
    PassRate("feedback_quality"),
    Percentile(Measure.DURATION, 95),
]


def _grader_evaluation(
    version: str, quality: Any, *, grader: Any = None, name: str = "short-answer-grader"
) -> Evaluation:
    return Evaluation(
        name=name,
        description="Does the grader agree with teachers and explain mistakes?",
        task=lambda: ProcessorTask(grader or build_grader(version), version=version),
        dataset=DATA,
        input_type=Submission,
        reference_type=TeacherGrade,
        scorers=[agrees_with_teacher, names_key_issue, feedback_concise, quality],
        metrics=_METRICS,
        repetitions=3,
        group_by=["difficulty"],
        sealed_splits=["test"],
        dataset_checks=[key_issue_implies_imperfect],
        tags=["demo"],
    )


grader_v1 = _grader_evaluation("v1", feedback_quality_v1)
grader_v2 = _grader_evaluation("v2", feedback_quality_v1)
# Same task as v2, judged by the stricter feedback scorer — use it to
# rescore a stored run: ``grasp-evals rescore <run> --spec ...:grader_v2_strict``.
grader_v2_strict = _grader_evaluation("v2", feedback_quality_v2)


def llm_grader_evaluation(llm: Any) -> Evaluation:
    """The same evaluation with the grader replaced by an LLM agent on ``llm``."""
    return _grader_evaluation(
        "llm",
        feedback_quality_v1,
        grader=llm_grader(llm),
        name="short-answer-grader-llm",
    )
