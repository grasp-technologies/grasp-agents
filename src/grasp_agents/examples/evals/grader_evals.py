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
    JudgedOutput,
    PairwiseContext,
    PairwiseVerdict,
    PassHatK,
    PassRate,
    Perturbation,
    ProcessorScorer,
    ProcessorTask,
    Score,
    TraceQuery,
    ValidationGate,
    judge_probes,
    judge_validation,
    scorer,
)
from grasp_agents.evals.metrics import Measure, Percentile
from grasp_agents.processors.processor import Processor
from grasp_agents.types.events import Event, ProcPayloadOutEvent
from grasp_agents.workflow.sequential_workflow import SequentialWorkflow

DATA = Path(__file__).parent / "data" / "short_answers.jsonl"
# Teachers' labels of graders' feedback: was it specific enough to act on?
LABELS = Path(__file__).parent / "data" / "feedback_labels.jsonl"
# The Phoenix project the grader is traced into in production (production.py).
PRODUCTION_PROJECT = "short-answer-grader"

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
        super().__init__(name=name, version=version)

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
        super().__init__(name=name, version=version)
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
        version=version,
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


# --- A judge as a processor ---

type Judged = JudgedOutput[Submission, Grade, TeacherGrade]


class FeedbackVerdict(BaseModel):
    specific: bool
    explanation: str


class FeedbackJudge(Processor[Judged, FeedbackVerdict, None]):
    """
    An offline stand-in for an LLM judge of feedback quality. ``v1`` passes
    any feedback of a few words; ``v2`` wants feedback on an imperfect answer
    to name a key point. Like a sampled model it is not perfectly consistent:
    it flips some verdicts at random (``noise``).
    """

    def __init__(
        self, name: str = "feedback_judge", *, version: str = "v1", noise: float = 0.1
    ) -> None:
        super().__init__(name=name, version=version)
        self.noise = noise

    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[Judged] | None = None,
        exec_id: str,
        step: int | None = None,
    ) -> AsyncIterator[Event[Any]]:
        for item in in_args or []:
            yield ProcPayloadOutEvent(
                data=self._judge(item), source=self.name, exec_id=exec_id
            )

    def _judge(self, item: Judged) -> FeedbackVerdict:
        feedback = item.output.feedback
        if self.version == "v1":
            specific = len(feedback.split()) >= 3
            explanation = f"{len(feedback.split())} words"
        elif item.reference is not None and item.reference.verdict == "correct":
            specific, explanation = True, "the answer was correct"
        else:
            named = set(_tokens(feedback)) & set(
                _key_points(item.input.reference_answer)
            )
            specific = bool(named)
            explanation = f"names {sorted(named)}" if named else "names no key point"
        if random.random() < self.noise:  # noqa: S311
            specific = not specific
            explanation += " (on a second reading: the opposite)"
        return FeedbackVerdict(specific=specific, explanation=explanation)


def _quality(verdict: FeedbackVerdict) -> Score:
    return Score(
        name="feedback_quality", value=verdict.specific, explanation=verdict.explanation
    )


def feedback_judge(version: str) -> ProcessorScorer[Any, Any, Any, Any, Any]:
    """The feedback-quality judge, version ``v1`` or ``v2``."""
    return ProcessorScorer(
        FeedbackJudge(version=version),
        name="feedback_quality",
        version=version,
        to_scores=_quality,
    )


def llm_feedback_judge(llm: Any) -> ProcessorScorer[Any, Any, Any, Any, Any]:
    """The same judge as an LLM agent (``apply_output_schema_via_provider=True``)."""
    from grasp_agents.agent.llm_agent import LLMAgent  # noqa: PLC0415

    agent = LLMAgent[Judged, FeedbackVerdict, None](
        name="feedback_judge",
        llm=llm,
        sys_prompt=(
            "You review a grader's feedback on a student's short answer. The input "
            "holds the question, the reference answer, the student's answer, the "
            "grade and the teacher's grade. specific: true when the feedback tells "
            "the student exactly what is missing or wrong (or, for a correct "
            "answer, confirms it), false when it is generic or misleading. "
            "explanation: one sentence."
        ),
    )
    return ProcessorScorer(
        agent, name="feedback_quality", version="llm-1", to_scores=_quality
    )


# Perturbations of graded outputs: what a feedback judge must notice, and what
# it must not.
_PADDING = " Keep up the effort, and ask if anything is unclear."
PROBES = [
    Perturbation(
        "generic",
        lambda item: (
            item.output.model_copy(update={"feedback": "Please review the material."})
            if item.reference is not None and item.reference.verdict != "correct"
            else None
        ),
        expect="lower",
    ),
    Perturbation(
        "padded",
        lambda item: item.output.model_copy(
            update={"feedback": item.output.feedback + _PADDING}
        ),
        expect="same",
    ),
    Perturbation(
        "uppercase",
        lambda item: item.output.model_copy(
            update={"feedback": item.output.feedback.upper()}
        ),
        expect="same",
    ),
]


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
    version: str,
    quality: Any,
    *,
    grader: Any = None,
    name: str = "short-answer-grader",
    validation_gates: Any = None,
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
        validation_gates=validation_gates or {},
        tags=["demo"],
    )


grader_v1 = _grader_evaluation("v1", feedback_quality_v1)
grader_v2 = _grader_evaluation("v2", feedback_quality_v1)
# Same task as v2, judged by the stricter feedback scorer — use it to
# rescore a stored run: ``grasp-evals rescore <run> --spec ...:grader_v2_strict``.
grader_v2_strict = _grader_evaluation("v2", feedback_quality_v2)


# Validating the feedback judge against teachers' labels (dev to iterate, the
# sealed test split once at the end), and probing it with changed outputs.
_JUDGED_TYPES: dict[str, Any] = {
    "input_type": Submission,
    "output_type": Grade,
    "reference_type": TeacherGrade,
}
judge_v1_validation = judge_validation(
    feedback_judge("v1"), LABELS, repetitions=3, **_JUDGED_TYPES
)
judge_v2_validation = judge_validation(
    feedback_judge("v2"), LABELS, repetitions=3, **_JUDGED_TYPES
)
# The labels' test split stays sealed in probe runs too.
judge_v1_probes = judge_probes(
    feedback_judge("v1"), LABELS, PROBES, sealed_splits=["test"], **_JUDGED_TYPES
)
judge_v2_probes = judge_probes(
    feedback_judge("v2"), LABELS, PROBES, sealed_splits=["test"], **_JUDGED_TYPES
)

# v2 graded by the processor judge, which must have passed validation first.
grader_v2_judged = _grader_evaluation(
    "v2",
    feedback_judge("v2"),
    validation_gates={
        "feedback_quality": ValidationGate(min_kappa=0.2, labels="feedback_labels")
    },
)


def llm_grader_evaluation(llm: Any) -> Evaluation:
    """The same evaluation with the grader replaced by an LLM agent on ``llm``."""
    return _grader_evaluation(
        "llm",
        feedback_quality_v1,
        grader=llm_grader(llm),
        name="short-answer-grader-llm",
    )


# The grader in production, scored online: its traced runs are read from
# Phoenix, judged by the same scorers (those that need no teacher grade),
# and the scores written back onto the traces. The judge must have passed
# validation, and its pass rate is reported corrected for its errors. Students'
# requests are correlated, so intervals are clustered on the session.
grader_online = Evaluation(
    name="short-answer-grader-online",
    description="Is the grader's feedback in production concise and specific?",
    input_type=Submission,
    output_type=Grade,
    scorers=[feedback_concise, feedback_judge("v2")],
    metrics=[PassRate("feedback_concise"), PassRate("feedback_quality")],
    cluster_by="session_id",
    group_by=["version"],
    validation_gates={
        "feedback_quality": ValidationGate(min_kappa=0.2, labels="feedback_labels")
    },
    traces=TraceQuery(
        project=PRODUCTION_PROJECT,
        processor="grader",
        # A real deployment waits longer for spans to be exported (an hour by
        # default); the demo reads them seconds after they are made.
        completion_buffer_s=5,
    ),
    tags=["demo"],
)
