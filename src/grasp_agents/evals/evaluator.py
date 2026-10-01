import inspect
import math
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, overload

from grasp_agents.session_context import SessionContext
from grasp_agents.types.events import Event

from ._util import qualified_name
from .types import ComponentInfo, Example, Score, Trial, Usage

type ScalarScore = bool | int | float | str
type EvaluatorOutput = (
    Score | Sequence[Score] | ScalarScore | Mapping[str, ScalarScore | Score] | None
)
"""
What an evaluator may return. A scalar becomes one score named after the
evaluator (``bool`` → pass/fail, number → score, ``str`` → label); a mapping
names several scores; ``None`` means "not applicable" (nothing is recorded).
"""


@dataclass(frozen=True)
class EvalContext[InT, OutT, RefT]:
    """Everything an evaluator may look at for one trial."""

    example: Example[InT, RefT]
    output: OutT
    trial: Trial
    # The task's event stream (messages, tool calls, responses) without token
    # deltas. Empty for imported trials; generic ``Event`` objects whose
    # ``data`` is plain JSON for trials reloaded from disk.
    events: Sequence[Event[Any]] = field(default_factory=tuple[Event[Any], ...])
    # The trial's session after the task finished — the *outcome* for tasks
    # whose effect is a ``state`` mutation. ``None`` when the task did not run
    # in this process (rescoring, imported traces).
    session: SessionContext[Any] | None = None
    # Usage reported by evaluators that call models (see ``record_usage``).
    usage: list[Usage] = field(default_factory=list[Usage])

    def record_usage(self, usage: Usage) -> None:
        """Attribute an evaluator's own model usage (cost) to this trial."""
        self.usage.append(usage)

    @property
    def input(self) -> InT:
        return self.example.input

    @property
    def reference(self) -> RefT | None:
        return self.example.reference

    @property
    def metadata(self) -> dict[str, Any]:
        return self.example.metadata


class Evaluator[InT, OutT, RefT](ABC):
    """
    A task-specific measurement instrument producing per-example scores.

    Set ``version`` and bump it whenever the evaluator's semantics change
    (prompt, rubric, model, thresholds): it is recorded with every run, and a
    run is only comparable with runs scored by the same versions. Return
    configuration that affects judgments from :meth:`config` so it is
    recorded too.
    """

    name: str = ""
    version: str = "1"
    # Who produces the judgments; recorded with the run and sent to Phoenix.
    annotator: Literal["CODE", "LLM", "HUMAN"] = "CODE"
    # Also evaluate trials whose task raised (e.g. transcript checks of
    # failed runs). Off by default: scores of failed trials are missing.
    evaluates_errors: bool = False

    def __init__(self, name: str | None = None, version: str | None = None) -> None:
        if name is not None:
            self.name = name
        if version is not None:
            self.version = version
        if not self.name:
            self.name = _snake_case(type(self).__name__)

    def config(self) -> dict[str, Any]:
        return {}

    def describe(self) -> ComponentInfo:
        return ComponentInfo(
            name=self.name,
            kind=qualified_name(type(self)),
            version=self.version,
            config=self.config(),
            annotator=self.annotator,
        )

    @abstractmethod
    def evaluate(
        self, ctx: EvalContext[InT, OutT, RefT]
    ) -> EvaluatorOutput | Awaitable[EvaluatorOutput]: ...


type EvaluatorFn[InT, OutT, RefT] = Callable[
    [EvalContext[InT, OutT, RefT]], EvaluatorOutput | Awaitable[EvaluatorOutput]
]


class FunctionEvaluator[InT, OutT, RefT](Evaluator[InT, OutT, RefT]):
    def __init__(
        self,
        fn: EvaluatorFn[InT, OutT, RefT],
        *,
        name: str | None = None,
        version: str = "1",
        config: Mapping[str, Any] | None = None,
        evaluates_errors: bool = False,
        annotator: Literal["CODE", "LLM", "HUMAN"] = "CODE",
    ) -> None:
        super().__init__(name=name or fn.__name__, version=version)
        self._fn = fn
        self._config = dict(config or {})
        self.evaluates_errors = evaluates_errors
        self.annotator = annotator

    def config(self) -> dict[str, Any]:
        return dict(self._config)

    def describe(self) -> ComponentInfo:
        info = super().describe()
        info.kind = qualified_name(self._fn)
        return info

    def evaluate(
        self, ctx: EvalContext[InT, OutT, RefT]
    ) -> EvaluatorOutput | Awaitable[EvaluatorOutput]:
        return self._fn(ctx)


@overload
def evaluator[InT, OutT, RefT](
    fn: EvaluatorFn[InT, OutT, RefT], /
) -> FunctionEvaluator[InT, OutT, RefT]: ...
@overload
def evaluator(
    *,
    name: str | None = None,
    version: str = "1",
    config: Mapping[str, Any] | None = None,
    evaluates_errors: bool = False,
    annotator: Literal["CODE", "LLM", "HUMAN"] = "CODE",
) -> Callable[[EvaluatorFn[Any, Any, Any]], FunctionEvaluator[Any, Any, Any]]: ...
def evaluator(
    fn: EvaluatorFn[Any, Any, Any] | None = None,
    /,
    *,
    name: str | None = None,
    version: str = "1",
    config: Mapping[str, Any] | None = None,
    evaluates_errors: bool = False,
    annotator: Literal["CODE", "LLM", "HUMAN"] = "CODE",
) -> (
    FunctionEvaluator[Any, Any, Any]
    | Callable[[EvaluatorFn[Any, Any, Any]], FunctionEvaluator[Any, Any, Any]]
):
    """
    Turn a function of :class:`EvalContext` into an evaluator::

        @evaluator(version="2")
        def exact_match(ctx: EvalContext[str, str, str]) -> bool:
            return ctx.output == ctx.reference
    """

    def wrap(f: EvaluatorFn[Any, Any, Any]) -> FunctionEvaluator[Any, Any, Any]:
        return FunctionEvaluator(
            f,
            name=name,
            version=version,
            config=config,
            evaluates_errors=evaluates_errors,
            annotator=annotator,
        )

    return wrap(fn) if fn is not None else wrap


async def run_evaluator(
    evaluator: Evaluator[Any, Any, Any], ctx: EvalContext[Any, Any, Any]
) -> list[Score]:
    result = evaluator.evaluate(ctx)
    if inspect.isawaitable(result):
        result = await result
    return normalize_scores(evaluator.name, result)


def normalize_scores(evaluator_name: str, output: EvaluatorOutput) -> list[Score]:
    if output is None:
        return []
    if isinstance(output, Score):
        scores = [output]
    elif isinstance(output, bool | int | float | str):
        scores = [_scalar_score(evaluator_name, output)]
    elif isinstance(output, Mapping):
        scores = [
            value.model_copy(update={"name": key})
            if isinstance(value, Score)
            else _scalar_score(key, value)
            for key, value in output.items()
        ]
    else:
        scores = list(output)
        for score in scores:
            if not isinstance(score, Score):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise TypeError(
                    f"Evaluator {evaluator_name!r} returned an unsupported value "
                    f"{score!r}; expected a Score, scalar, mapping or None"
                )
    return [
        s
        if s.evaluator is not None
        else s.model_copy(update={"evaluator": evaluator_name})
        for s in scores
    ]


def _scalar_score(name: str, value: ScalarScore) -> Score:
    if isinstance(value, bool | str):
        return Score(name=name, value=value)
    number = float(value)
    if not math.isfinite(number):
        return Score.unscored(name, reason="non_finite_value")
    return Score(name=name, value=number)


def _snake_case(name: str) -> str:
    out: list[str] = []
    for i, ch in enumerate(name):
        if ch.isupper() and i and not name[i - 1].isupper():
            out.append("_")
        out.append(ch.lower())
    return "".join(out)
