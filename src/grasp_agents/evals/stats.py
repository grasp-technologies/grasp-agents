"""
Small-sample statistics for eval results, without numpy/scipy.

Conventions follow "Adding Error Bars to Evals" (Miller, 2024): the standard
error of a mean uses the CLT, related items are handled with clustered
standard errors (with the usual G/(G-1) small-sample correction, CR1), and
two systems evaluated on the same examples are compared on paired
differences.
"""

import math
import random
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from statistics import NormalDist

_NORMAL = NormalDist()
# Standard errors this small relative to the mean are rounding noise: the
# values are identical and no t-statistic is defined.
_DEGENERATE = 1e-12


@dataclass(frozen=True)
class Estimate:
    value: float
    n: int
    stderr: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    # Degrees of freedom behind the interval (n - 1, or clusters - 1).
    df: float | None = None
    # Independent units behind the estimate: values, or clusters when
    # clustered.
    units: int = 0


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

    With ``clusters`` (one key per value) the clustered standard error (CR1)
    is used: items in a cluster are not independent, so the naive SE
    understates the uncertainty — often by a large factor. With fewer than
    two clusters there is no standard error.
    """
    n = len(values)
    if n == 0:
        return Estimate(value=math.nan, n=0)

    mean = _mean(values)

    if clusters is not None:
        if len(clusters) != n:
            raise ValueError("clusters must align with values")

        sums: dict[Hashable, float] = {}
        for value, key in zip(values, clusters, strict=True):
            sums[key] = sums.get(key, 0.0) + (value - mean)

        n_clusters = len(sums)
        if n_clusters < 2:
            return Estimate(value=mean, n=n, units=n_clusters)

        correction = n_clusters / (n_clusters - 1)
        se = math.sqrt(correction * math.fsum(s * s for s in sums.values())) / n
        df = float(n_clusters - 1)
        units = n_clusters

    else:
        if n == 1:
            return Estimate(value=mean, n=1, units=1)

        variance = math.fsum((v - mean) ** 2 for v in values) / (n - 1)
        se = math.sqrt(variance / n)
        df = float(n - 1)
        units = n

    half = critical_value(confidence, df) * se

    return Estimate(
        value=mean,
        n=n,
        stderr=se,
        ci_low=mean - half,
        ci_high=mean + half,
        df=df,
        units=units,
    )


def wilson_interval(p: float, n: float, *, critical: float) -> tuple[float, float]:
    """Wilson score interval for a share ``p`` of ``n`` (possibly effective) units."""
    if n <= 0:
        return 0.0, 1.0

    z2 = critical * critical
    denominator = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denominator
    half = critical * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    low = 0.0 if p <= 0.0 else max(0.0, centre - half)
    high = 1.0 if p >= 1.0 else min(1.0, centre + half)

    return low, high


def proportion_estimate(
    successes: int, n: int, *, confidence: float = 0.95
) -> Estimate:
    """Proportion with the Wilson score interval (well-behaved near 0 and 1)."""
    if n == 0:
        return Estimate(value=math.nan, n=0)

    p = successes / n
    low, high = wilson_interval(p, n, critical=critical_value(confidence))

    return Estimate(
        value=p,
        n=n,
        stderr=math.sqrt(p * (1 - p) / n),
        ci_low=low,
        ci_high=high,
        units=n,
    )


def bounded_mean_estimate(
    values: Sequence[float],
    *,
    clusters: Sequence[Hashable] | None = None,
    observations: int | None = None,
    confidence: float = 0.95,
) -> Estimate:
    """
    Mean of per-unit shares in [0, 1] (pass rates over repetitions, pass@k
    estimates, win preferences) with a Wilson interval on the effective
    sample size ``p(1-p)/SE²`` (from the clustered SE when clustered), at
    most ``observations`` — the pass/fail outcomes behind the values, e.g.
    every repetition (by default one per value).

    Unlike a t-interval it stays inside [0, 1] and does not collapse when
    every unit passes; there, and whenever the SE is degenerate, each unit
    (cluster) counts once, since repetitions alone cannot show how much the
    units differ.
    """
    estimate = mean_estimate(values, clusters=clusters, confidence=confidence)
    if estimate.n == 0:
        return estimate

    p = min(1.0, max(0.0, estimate.value))
    se = estimate.stderr
    saturated = p <= 0.0 or p >= 1.0
    if se is None or saturated or se <= _DEGENERATE:
        n_eff = float(max(estimate.units, 1))
    else:
        cap = observations if observations is not None else estimate.n
        n_eff = min(p * (1.0 - p) / (se * se), float(cap))

    low, high = wilson_interval(
        p, n_eff, critical=critical_value(confidence, estimate.df)
    )

    return Estimate(
        value=estimate.value,
        n=estimate.n,
        stderr=se,
        ci_low=low,
        ci_high=high,
        df=estimate.df,
        units=estimate.units,
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
    clusters: Sequence[Hashable] | None = None,
    n_resamples: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> Estimate:
    """
    Percentile-bootstrap interval for an arbitrary statistic (e.g. p95).
    With ``clusters`` whole clusters are resampled, keeping correlated values
    (repetitions of one example) together.
    """
    if n_resamples < 2:
        raise ValueError("n_resamples must be >= 2")

    n = len(values)
    if n == 0:
        return Estimate(value=math.nan, n=0)

    point = statistic(values)
    groups: list[list[float]]
    if clusters is None:
        groups = [[v] for v in values]

    else:
        if len(clusters) != n:
            raise ValueError("clusters must align with values")

        by_key: dict[Hashable, list[float]] = {}
        for value, key in zip(values, clusters, strict=True):
            by_key.setdefault(key, []).append(value)
        groups = list(by_key.values())

    if len(groups) < 2:
        return Estimate(value=point, n=n, units=len(groups))

    rng = random.Random(seed)  # ruff: ignore[suspicious-non-cryptographic-random-usage]
    resampled = sorted(
        statistic([v for g in rng.choices(groups, k=len(groups)) for v in g])
        for _ in range(n_resamples)
    )
    alpha = (1.0 - confidence) / 2.0
    mean = _mean(resampled)
    se = math.sqrt(math.fsum((v - mean) ** 2 for v in resampled) / (n_resamples - 1))

    return Estimate(
        value=point,
        n=n,
        stderr=se,
        ci_low=percentile(resampled, 100.0 * alpha),
        ci_high=percentile(resampled, 100.0 * (1.0 - alpha)),
        units=len(groups),
    )


# --- Tests and paired comparison ---


def binomial_two_sided(k: int, n: int) -> float:
    """Exact two-sided p-value of k successes out of n under p = 0.5."""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(k, n - k) + 1))
    # Exact integer arithmetic: the counts overflow floats beyond n ≈ 1000.
    return min(1.0, 2 * tail / (1 << n))


def mcnemar_exact(base_only: int, candidate_only: int) -> float:
    """Exact McNemar p-value from the discordant pair counts."""
    return binomial_two_sided(base_only, base_only + candidate_only)


def sign_test(differences: Sequence[float]) -> float:
    """Exact sign test of the non-zero differences against a zero median."""
    positive = sum(1 for d in differences if d > 0)
    negative = sum(1 for d in differences if d < 0)
    return binomial_two_sided(positive, positive + negative)


def holm_adjust(p_values: Sequence[float | None]) -> list[float | None]:
    """Holm step-down adjusted p-values (family-wise error control)."""
    present = sorted(
        (p, i) for i, p in enumerate(p_values) if p is not None and not math.isnan(p)
    )
    m = len(present)
    adjusted: list[float | None] = [None] * len(p_values)
    running = 0.0
    for rank, (p, index) in enumerate(present):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[index] = running
    return adjusted


def minimum_detectable_effect(
    stderr: float,
    *,
    df: float | None = None,
    confidence: float = 0.95,
    power: float = 0.8,
) -> float:
    """
    Smallest true difference a two-sided test at ``confidence`` detects with
    probability ``power``: ``(t_{1-alpha/2} + t_power) · SE`` (normal quantiles when
    ``df`` is None).
    """
    tail = _NORMAL.inv_cdf(power) if df is None else t_ppf(power, df)
    return (critical_value(confidence, df) + tail) * stderr


def newcombe_paired_interval(
    both: int,
    base_only: int,
    candidate_only: int,
    neither: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Newcombe's hybrid score interval (method 10) for the difference of two
    paired proportions, candidate - base.
    """
    n = both + base_only + candidate_only + neither
    if n == 0:
        return -1.0, 1.0

    z = critical_value(confidence)
    p_candidate = (both + candidate_only) / n
    p_base = (both + base_only) / n
    l1, u1 = wilson_interval(p_candidate, n, critical=z)
    l2, u2 = wilson_interval(p_base, n, critical=z)
    margins = (
        (both + base_only)
        * (candidate_only + neither)
        * (both + candidate_only)
        * (base_only + neither)
    )
    phi = (
        0.0
        if margins == 0
        else (both * neither - base_only * candidate_only) / math.sqrt(margins)
    )
    diff = p_candidate - p_base
    lower = math.sqrt(
        max(
            0.0,
            (p_candidate - l1) ** 2
            - 2 * phi * (p_candidate - l1) * (u2 - p_base)
            + (u2 - p_base) ** 2,
        )
    )
    upper = math.sqrt(
        max(
            0.0,
            (u1 - p_candidate) ** 2
            - 2 * phi * (u1 - p_candidate) * (p_base - l2)
            + (p_base - l2) ** 2,
        )
    )
    return max(-1.0, diff - lower), min(1.0, diff + upper)


def paired_difference(
    base: Sequence[float],
    candidate: Sequence[float],
    *,
    clusters: Sequence[Hashable] | None = None,
    confidence: float = 0.95,
    power: float = 0.8,
) -> PairedEstimate:
    """
    Compare two systems on the same examples via per-example differences
    (candidate - base).

    Binary values (all 0/1) use the exact McNemar test with Newcombe's
    interval; anything else a paired t-test, on clustered standard errors
    when ``clusters`` maps each pair to a cluster. When every difference is
    identical (no variance) the exact sign test is used instead.
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
    estimate = mean_estimate(diffs, clusters=clusters, confidence=confidence)
    se = estimate.stderr
    degenerate = se is not None and se <= _DEGENERATE * max(1.0, abs(estimate.value))
    binary = clusters is None and all(v in {0.0, 1.0} for v in (*base, *candidate))
    ci_low, ci_high = estimate.ci_low, estimate.ci_high
    p_value: float | None

    if binary:
        pairs = list(zip(base, candidate, strict=True))
        base_only = sum(1 for b, c in pairs if b > c)
        candidate_only = sum(1 for b, c in pairs if c > b)
        both = sum(1 for b, c in pairs if b > 0.5 and c > 0.5)
        p_value = mcnemar_exact(base_only, candidate_only)
        ci_low, ci_high = newcombe_paired_interval(
            both,
            base_only,
            candidate_only,
            n - both - base_only - candidate_only,
            confidence=confidence,
        )
        test = "mcnemar"

    elif se is None:
        p_value, test = None, "none"

    elif degenerate:
        p_value, test = sign_test(diffs), "sign"

    else:
        t = estimate.value / se
        df = estimate.df if estimate.df is not None else float(n - 1)
        p_value = min(1.0, 2.0 * t_sf(abs(t), df))
        test = "paired_t" if clusters is None else "clustered_paired_t"

    return PairedEstimate(
        n=n,
        base_mean=_mean(base),
        candidate_mean=_mean(candidate),
        diff=estimate.value,
        stderr=se,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        mde=(
            None
            if se is None or degenerate
            else minimum_detectable_effect(
                se, df=estimate.df, confidence=confidence, power=power
            )
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
