"""
Small-sample statistics for eval results, without numpy/scipy.

Conventions follow "Adding Error Bars to Evals" (Miller, 2024): the standard
error of a mean uses the CLT, related items are handled with clustered
standard errors, and two systems evaluated on the same examples are compared
on paired differences.
"""

import math
import random
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from statistics import NormalDist

_NORMAL = NormalDist()


@dataclass(frozen=True)
class Estimate:
    value: float
    n: int
    stderr: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None


@dataclass(frozen=True)
class PairedEstimate:
    n: int
    base_mean: float
    candidate_mean: float
    diff: float
    stderr: float | None
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    # Smallest true difference detectable at 80% power with this SE.
    mde: float | None
    test: str


# --- Distributions ---


def _betacf(a: float, b: float, x: float) -> float:
    # Continued fraction for the regularized incomplete beta (Lentz's method).
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf(t: float, df: float) -> float:
    """Survival function P(T > t) of Student's t."""
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return tail if t >= 0 else 1.0 - tail


def t_ppf(p: float, df: float) -> float:
    """Quantile of Student's t (bisection on the exact CDF)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if math.isclose(p, 0.5):
        return 0.0
    if p < 0.5:
        return -t_ppf(1.0 - p, df)
    lo, hi = 0.0, 1.0
    while t_sf(hi, df) > 1.0 - p:
        hi *= 2.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_sf(mid, df) > 1.0 - p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    return (lo + hi) / 2.0


def critical_value(confidence: float, df: float | None = None) -> float:
    """Two-sided critical value: normal when ``df`` is None, else Student's t."""
    q = 0.5 + confidence / 2.0
    return _NORMAL.inv_cdf(q) if df is None else t_ppf(q, df)


# --- Point estimates ---


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def mean_estimate(
    values: Sequence[float],
    *,
    clusters: Sequence[Hashable] | None = None,
    confidence: float = 0.95,
) -> Estimate:
    """
    Mean with CLT standard error and a t-based confidence interval.

    With ``clusters`` (one key per value) the clustered standard error is used:
    items in a cluster are not independent, so the naive SE understates the
    uncertainty — often by a large factor.
    """
    n = len(values)
    if n == 0:
        return Estimate(value=math.nan, n=0)
    mean = _mean(values)
    if n == 1:
        return Estimate(value=mean, n=1)
    if clusters is not None:
        if len(clusters) != n:
            raise ValueError("clusters must align with values")
        sums: dict[Hashable, float] = {}
        for value, key in zip(values, clusters, strict=True):
            sums[key] = sums.get(key, 0.0) + (value - mean)
        n_clusters = len(sums)
        se = math.sqrt(math.fsum(s * s for s in sums.values())) / n
        df = float(max(n_clusters - 1, 1))
    else:
        variance = math.fsum((v - mean) ** 2 for v in values) / (n - 1)
        se = math.sqrt(variance / n)
        df = float(n - 1)
    half = critical_value(confidence, df) * se
    return Estimate(value=mean, n=n, stderr=se, ci_low=mean - half, ci_high=mean + half)


def proportion_estimate(
    successes: int, n: int, *, confidence: float = 0.95
) -> Estimate:
    """Proportion with the Wilson score interval (well-behaved near 0 and 1)."""
    if n == 0:
        return Estimate(value=math.nan, n=0)
    p = successes / n
    z = critical_value(confidence)
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return Estimate(
        value=p,
        n=n,
        stderr=math.sqrt(p * (1 - p) / n),
        ci_low=max(0.0, centre - half),
        ci_high=min(1.0, centre + half),
    )


def percentile(values: Sequence[float], q: float) -> float:
    """``q``-th percentile (0-100) with linear interpolation."""
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (q / 100.0) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_estimate(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> Estimate:
    """Percentile-bootstrap interval for an arbitrary statistic (e.g. p95)."""
    n = len(values)
    if n == 0:
        return Estimate(value=math.nan, n=0)
    point = statistic(values)
    if n == 1:
        return Estimate(value=point, n=1)
    rng = random.Random(seed)  # noqa: S311
    resampled = sorted(statistic(rng.choices(values, k=n)) for _ in range(n_resamples))
    alpha = (1.0 - confidence) / 2.0
    mean = _mean(resampled)
    se = math.sqrt(math.fsum((v - mean) ** 2 for v in resampled) / (n_resamples - 1))
    return Estimate(
        value=point,
        n=n,
        stderr=se,
        ci_low=percentile(resampled, 100.0 * alpha),
        ci_high=percentile(resampled, 100.0 * (1.0 - alpha)),
    )


# --- Paired comparison ---


def binomial_two_sided(k: int, n: int) -> float:
    """Exact two-sided p-value of k successes out of n under p = 0.5."""
    if n == 0:
        return 1.0
    tail = math.fsum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / 2.0**n
    return min(1.0, 2.0 * tail)


def mcnemar_exact(base_only: int, candidate_only: int) -> float:
    """Exact McNemar p-value from the discordant pair counts."""
    return binomial_two_sided(base_only, base_only + candidate_only)


def minimum_detectable_effect(
    stderr: float, *, confidence: float = 0.95, power: float = 0.8
) -> float:
    return (critical_value(confidence) + _NORMAL.inv_cdf(power)) * stderr


def paired_difference(
    base: Sequence[float],
    candidate: Sequence[float],
    *,
    confidence: float = 0.95,
) -> PairedEstimate:
    """
    Compare two systems on the same examples via per-example differences.

    Binary values (all 0/1) use the exact McNemar test on discordant pairs;
    anything else a paired t-test. The interval is always on the mean paired
    difference (candidate - base).
    """
    if len(base) != len(candidate):
        raise ValueError("base and candidate must be paired")
    n = len(base)
    if n == 0:
        return PairedEstimate(
            n=0,
            base_mean=math.nan,
            candidate_mean=math.nan,
            diff=math.nan,
            stderr=None,
            ci_low=None,
            ci_high=None,
            p_value=None,
            mde=None,
            test="none",
        )
    diffs = [c - b for b, c in zip(base, candidate, strict=True)]
    estimate = mean_estimate(diffs, confidence=confidence)
    binary = all(v in {0.0, 1.0} for v in (*base, *candidate))
    p_value: float | None
    if binary:
        base_only = sum(1 for b, c in zip(base, candidate, strict=True) if b > c)
        candidate_only = sum(1 for b, c in zip(base, candidate, strict=True) if c > b)
        p_value = mcnemar_exact(base_only, candidate_only)
        test = "mcnemar"
    elif estimate.stderr is None:
        p_value, test = None, "paired_t"
    elif estimate.stderr <= 0.0:
        no_change = math.isclose(estimate.value, 0.0, abs_tol=1e-12)
        p_value, test = (1.0 if no_change else 0.0), "paired_t"
    else:
        t = estimate.value / estimate.stderr
        p_value = min(1.0, 2.0 * t_sf(abs(t), n - 1))
        test = "paired_t"
    return PairedEstimate(
        n=n,
        base_mean=_mean(base),
        candidate_mean=_mean(candidate),
        diff=estimate.value,
        stderr=estimate.stderr,
        ci_low=estimate.ci_low,
        ci_high=estimate.ci_high,
        p_value=p_value,
        mde=(
            None
            if estimate.stderr is None
            else minimum_detectable_effect(estimate.stderr, confidence=confidence)
        ),
        test=test,
    )


# --- Agreement (judge validation) ---


def cohens_kappa(pairs: Sequence[tuple[Hashable, Hashable]]) -> float | None:
    """Cohen's κ between two raters' labels (``None`` when undefined)."""
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(1 for a, b in pairs if a == b) / n
    left: dict[Hashable, int] = {}
    right: dict[Hashable, int] = {}
    for a, b in pairs:
        left[a] = left.get(a, 0) + 1
        right[b] = right.get(b, 0) + 1
    expected = sum(left[k] * right.get(k, 0) for k in left) / (n * n)
    if expected >= 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def corrected_prevalence(observed: float, tpr: float, tnr: float) -> float | None:
    """
    True pass rate implied by an imperfect binary judge (Rogan-Gladen):
    ``(observed + TNR - 1) / (TPR + TNR - 1)``, clipped to [0, 1].
    """
    denominator = tpr + tnr - 1.0
    if denominator <= 0.0:
        return None
    return min(1.0, max(0.0, (observed + tnr - 1.0) / denominator))
