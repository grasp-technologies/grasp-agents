import asyncio
import contextvars
import inspect
import math
import threading
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast, overload

from grasp_agents.session_context import SessionContext
from grasp_agents.types.events import Event

from ._util import code_hash, qualified_name
from .types import ComponentInfo, Example, Score, ScoreReason, Trial, Usage

type ScalarScore = bool | int | float | str
type ScorerOutput = (
    Score | Sequence[Score] | ScalarScore | Mapping[str, ScalarScore | Score] | None
)
"""
What a scorer may return. A scalar becomes one score named after the
scorer (``bool`` → pass/fail, number → score, ``str`` → label); a mapping
names several scores; ``None`` means "not applicable" (nothing is recorded).
"""


@dataclass(frozen=True)
class EvalContext[InT, OutT, RefT]:
    """Everything a scorer may look at for one trial."""

    example: Example[InT, RefT]
    output: OutT
    trial: Trial
    # The task's event stream (messages, tool calls, responses) without token
    # deltas. Empty for imported trials and when the run does not capture
    # events (``capture_events=False``); generic ``Event`` objects whose
    # ``data`` is plain JSON for trials reloaded from disk.
    events: Sequence[Event[Any]] = field(default_factory=tuple[Event[Any], ...])
    # The trial's session after the task finished — the *outcome* for tasks
    # whose effect is a ``state`` mutation. ``None`` when the task did not run
    # in this process (rescoring, imported traces).
    session: SessionContext[Any] | None = None
    # Usage reported by scorers that call models (see ``record_usage``).
    usage: list[Usage] = field(default_factory=list[Usage])
    # Models those calls used, per agent.
    models: dict[str, list[str]] = field(default_factory=dict[str, list[str]])

    def record_usage(
        self, usage: Usage, *, models: Mapping[str, Sequence[str]] | None = None
    ) -> None:
        """
        Attribute a scorer's own model usage (cost) to this trial, and
        the models it used per agent (recorded with the trial's models).
        """
        self.usage.append(usage)
        merge_models(self.models, models or {})

    @property
    def input(self) -> InT:
        return self.example.input

    @property
    def reference(self) -> RefT | None:
        return self.example.reference

    @property
    def metadata(self) -> dict[str, Any]:
        return self.example.metadata


class Scorer[InT, OutT, RefT](ABC):
    """
    A task-specific measurement instrument producing per-example scores.

    Set ``version`` and bump it whenever the scorer's semantics change
    (prompt, rubric, model, thresholds): it is recorded with every run, and a
    run is only comparable with runs scored by the same versions. Return
    configuration that affects judgments from :meth:`config` so it is
    recorded too.

    ``evaluate`` may be sync or async. Sync scorers run in a worker
    thread, so a slow one (a blocking HTTP judge, a subprocess) does not
    stall concurrent trials; keep them thread-safe.
    """

    name: str = ""
    version: str = "1"
    # Who produces the judgments; recorded with the run and sent to Phoenix.
    annotator: Literal["CODE", "LLM", "HUMAN"] = "CODE"
    # Also evaluate trials whose task raised (e.g. transcript checks of
    # failed runs). Off by default: scores of failed trials are missing.
    scores_errors: bool = False

    def __init__(self, name: str | None = None, version: str | None = None) -> None:
        if name is not None:
            self.name = name
        if version is not None:
            self.version = version
        if not self.name:
            self.name = snake_case(type(self).__name__)

    def config(self) -> dict[str, Any]:
        return {}

    def describe(self) -> ComponentInfo:
        return ComponentInfo(
            name=self.name,
            kind=qualified_name(type(self)),
            version=self.version,
            config=self.config(),
            annotator=self.annotator,
            source=code_hash(type(self)),
        )

    @abstractmethod
    def score(
        self, ctx: EvalContext[InT, OutT, RefT]
    ) -> ScorerOutput | Awaitable[ScorerOutput]: ...


type ScorerFn[InT, OutT, RefT] = Callable[
    [EvalContext[InT, OutT, RefT]], ScorerOutput | Awaitable[ScorerOutput]
]


class FunctionScorer[InT, OutT, RefT](Scorer[InT, OutT, RefT]):
    def __init__(
        self,
        fn: ScorerFn[InT, OutT, RefT],
        *,
        name: str | None = None,
        version: str = "1",
        config: Mapping[str, Any] | None = None,
        scores_errors: bool = False,
        annotator: Literal["CODE", "LLM", "HUMAN"] = "CODE",
    ) -> None:
        super().__init__(name=name or fn.__name__, version=version)
        self._fn = fn
        self._config = dict(config or {})
        self.scores_errors = scores_errors
        self.annotator = annotator

    def config(self) -> dict[str, Any]:
        return dict(self._config)

    def describe(self) -> ComponentInfo:
        info = super().describe()
        info.kind = qualified_name(self._fn)
        info.source = code_hash(self._fn)
        return info

    @property
    def fn(self) -> ScorerFn[InT, OutT, RefT]:
        return self._fn

    def score(
        self, ctx: EvalContext[InT, OutT, RefT]
    ) -> ScorerOutput | Awaitable[ScorerOutput]:
        return self._fn(ctx)


class ScorerDecorator(Protocol):
    def __call__[InT, OutT, RefT](
        self, fn: ScorerFn[InT, OutT, RefT], /
    ) -> FunctionScorer[InT, OutT, RefT]: ...


@overload
def scorer[InT, OutT, RefT](
    fn: ScorerFn[InT, OutT, RefT], /
) -> FunctionScorer[InT, OutT, RefT]: ...


@overload
def scorer(
    *,
    name: str | None = None,
    version: str = "1",
    config: Mapping[str, Any] | None = None,
    scores_errors: bool = False,
    annotator: Literal["CODE", "LLM", "HUMAN"] = "CODE",
) -> ScorerDecorator: ...


def scorer(
    fn: ScorerFn[Any, Any, Any] | None = None,
    /,
    *,
    name: str | None = None,
    version: str = "1",
    config: Mapping[str, Any] | None = None,
    scores_errors: bool = False,
    annotator: Literal["CODE", "LLM", "HUMAN"] = "CODE",
) -> FunctionScorer[Any, Any, Any] | ScorerDecorator:
    """
    Turn a function of :class:`EvalContext` into a scorer::

        @scorer(version="2")
        def exact_match(ctx: EvalContext[str, str, str]) -> bool:
            return ctx.output == ctx.reference
    """

    def wrap(f: ScorerFn[Any, Any, Any]) -> FunctionScorer[Any, Any, Any]:
        return FunctionScorer(
            f,
            name=name,
            version=version,
            config=config,
            scores_errors=scores_errors,
            annotator=annotator,
        )

    if fn is not None:
        return wrap(fn)
    return cast("ScorerDecorator", wrap)


def _is_async(scorer: Scorer[Any, Any, Any]) -> bool:
    if isinstance(scorer, FunctionScorer):
        return inspect.iscoroutinefunction(scorer.fn)
    return inspect.iscoroutinefunction(scorer.score)


async def run_scorer(
    scorer: Scorer[Any, Any, Any],
    ctx: EvalContext[Any, Any, Any],
    *,
    timeout_s: float | None = None,
) -> list[Score]:
    """
    Run one scorer on one trial. Sync scorers run in a thread of their
    own; on timeout the score is abandoned (a thread cannot be interrupted)
    and the thread finishes in the background without delaying the exit.
    """
    async with asyncio.timeout(timeout_s):
        result = await call_off_loop(scorer.score, ctx, is_async=_is_async(scorer))
    return normalize_scores(scorer.name, result)


async def call_off_loop[A, R](
    fn: Callable[[A], R | Awaitable[R]], arg: A, *, is_async: bool
) -> R:
    """
    Call user code without blocking the event loop: ``fn(arg)`` is awaited
    when ``is_async``, otherwise it runs in a thread of its own.
    """
    result = fn(arg) if is_async else await _in_thread(fn, arg)
    if inspect.isawaitable(result):
        return await result
    return cast("R", result)


async def _in_thread[T](fn: Callable[[Any], T], arg: Any) -> T:
    # A daemon thread rather than the loop's executor: the executor is joined
    # when the loop shuts down, so one call that never returns would hang the
    # program, and a few would exhaust the pool.
    loop = asyncio.get_running_loop()
    future: asyncio.Future[T] = loop.create_future()
    context = contextvars.copy_context()

    def settle(setter: Callable[[Any], None], value: Any) -> None:
        def apply() -> None:
            if not future.done():
                setter(value)

        try:
            loop.call_soon_threadsafe(apply)
        except RuntimeError:  # the loop has closed
            pass

    def work() -> None:
        try:
            result = context.run(fn, arg)
        except BaseException as exc:
            settle(future.set_exception, exc)
        else:
            settle(future.set_result, result)

    threading.Thread(target=work, name="grasp-evals-sync-call", daemon=True).start()
    return await future


def normalize_scores(scorer_name: str, output: ScorerOutput) -> list[Score]:
    if output is None:
        return []

    if isinstance(output, Score):
        scores = [output]

    elif isinstance(output, bool | int | float | str):
        scores = [_scalar_score(scorer_name, output)]

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
                    f"Scorer {scorer_name!r} returned an unsupported value "
                    f"{score!r}; expected a Score, scalar, mapping or None"
                )

    names = [s.name for s in scores]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(
            f"Scorer {scorer_name!r} returned several scores named {duplicates}"
        )

    return [_finite(s, scorer_name) for s in scores]


def _finite(score: Score, scorer_name: str) -> Score:
    update: dict[str, Any] = {}
    if score.scorer is None:
        update["scorer"] = scorer_name

    value = score.value
    if isinstance(value, float) and not math.isfinite(value):
        update["value"] = None
        update["reason"] = ScoreReason.NON_FINITE_VALUE

    return score.model_copy(update=update) if update else score


def _scalar_score(name: str, value: ScalarScore) -> Score:
    if isinstance(value, bool | str):
        return Score(name=name, value=value)

    number = float(value)
    if not math.isfinite(number):
        return Score.unscored(name, reason=ScoreReason.NON_FINITE_VALUE)

    return Score(name=name, value=number)


async def run_all_or_cancel[T](calls: Iterable[Coroutine[Any, Any, T]]) -> list[T]:
    """
    Await ``calls`` together; when one fails, the others are cancelled and
    awaited (so their usage is still recorded) and its error is raised.
    """
    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(call) for call in calls]
    except BaseExceptionGroup as failures:
        errors = [
            exc
            for exc in failures.exceptions
            if not isinstance(exc, asyncio.CancelledError)
        ]
        raise (errors or list(failures.exceptions))[0] from None
    return [task.result() for task in tasks]


def merge_models(
    into: dict[str, list[str]], models: Mapping[str, Sequence[str]]
) -> None:
    """Add ``models`` (per agent) to ``into``, keeping each name once."""
    for agent, names in models.items():
        known = into.setdefault(agent, [])
        known.extend(name for name in names if name not in known)


def snake_case(name: str) -> str:
    out: list[str] = []
    for i, ch in enumerate(name):
        if ch.isupper() and i and not name[i - 1].isupper():
            out.append("_")
        out.append(ch.lower())
    return "".join(out)
