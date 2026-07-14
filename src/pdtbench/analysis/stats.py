"""The statistics the design commits to, and nothing more.

The plan is explicit that n = 30 paired windows is power-limited: the standard error of
an episode Sharpe estimated from 90 daily bars is around 1.6-1.7 annualized units. So
everything here reports an interval, the model-vs-model claim is a *paired* test on the
same windows, and `power_note()` exists to put the achieved precision on the slide rather
than let a reader discover it.

Nothing here does a one-sided test or a p-value fish. If an effect needs that to show up,
it is not there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

DEFAULT_B = 10_000


@dataclass(frozen=True)
class Interval:
    point: float
    lo: float
    hi: float
    n: int

    def __str__(self) -> str:
        return f"{self.point:+.2f} [{self.lo:+.2f}, {self.hi:+.2f}]"

    @property
    def width(self) -> float:
        return self.hi - self.lo

    @property
    def excludes_zero(self) -> bool:
        return (self.lo > 0) or (self.hi < 0)


def bootstrap_ci(
    values, stat=np.median, B: int = DEFAULT_B, alpha: float = 0.05, seed: int = 0
) -> Interval:
    """Percentile bootstrap over episodes. `stat` defaults to the median, because the
    ranking metric is a median so a single lucky window cannot carry a model (D2)."""
    x = np.asarray(list(values), dtype=np.float64)
    if len(x) == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    if len(x) == 1:
        v = float(stat(x))
        return Interval(v, v, v, 1)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(x), size=(B, len(x)))
    boot = stat(x[draws], axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(float(stat(x)), float(lo), float(hi), len(x))


def paired_diff_ci(
    a, b, stat=np.median, B: int = DEFAULT_B, alpha: float = 0.05, seed: int = 0
) -> Interval:
    """Bootstrap the *paired* difference. Resamples pairs, not the two arms separately —
    the whole value of the paired design is that window difficulty cancels."""
    d = np.asarray(list(a), dtype=np.float64) - np.asarray(list(b), dtype=np.float64)
    return bootstrap_ci(d, stat=stat, B=B, alpha=alpha, seed=seed)


@dataclass(frozen=True)
class TestResult:
    name: str
    statistic: float
    pvalue: float
    n: int

    def __str__(self) -> str:
        return f"{self.name} W={self.statistic:.1f} p={self.pvalue:.3f} (n={self.n})"


def wilcoxon(a, b) -> TestResult:
    """Paired signed-rank — the model-vs-model test the design commits to (D5). Every
    model sees identical windows, so pairing is available and refusing to use it would
    throw away most of the power we have."""
    a = np.asarray(list(a), dtype=np.float64)
    b = np.asarray(list(b), dtype=np.float64)
    if len(a) != len(b):
        raise ValueError("wilcoxon needs paired samples")
    if len(a) < 2 or np.allclose(a, b):
        return TestResult("wilcoxon", float("nan"), 1.0, len(a))
    r = sps.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return TestResult("wilcoxon", float(r.statistic), float(r.pvalue), len(a))


def mann_whitney(a, b) -> TestResult:
    """Unpaired. Kept only because the design names it, and demoted to descriptive: the
    twin comparison is paired by construction, so Wilcoxon is the right test and this one
    throws the pairing away (D9)."""
    a = np.asarray(list(a), dtype=np.float64)
    b = np.asarray(list(b), dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return TestResult("mann_whitney", float("nan"), 1.0, min(len(a), len(b)))
    r = sps.mannwhitneyu(a, b, alternative="two-sided")
    return TestResult("mann_whitney", float(r.statistic), float(r.pvalue), min(len(a), len(b)))


@dataclass(frozen=True)
class Slope:
    slope: float
    intercept: float
    lo: float
    hi: float
    pvalue: float
    r: float
    n: int

    def __str__(self) -> str:
        return f"{self.slope:+.4f} [{self.lo:+.4f}, {self.hi:+.4f}] p={self.pvalue:.3f}"

    @property
    def significant(self) -> bool:
        """Purely statistical: the interval clears zero. Necessary, not sufficient — see
        `matters`."""
        return (self.lo > 0) or (self.hi < 0)

    def matters(self, min_effect: float) -> bool:
        """Statistically distinguishable from zero **and** big enough to care about.

        A t-test on a series with almost no residual variance will happily certify an
        effect of 1e-07 at p = 0.013. That happened here: the buy-and-hold baseline's
        excess slope came back "significant" at -1.0e-07 Sharpe per episode, which is
        seven orders of magnitude below anything a person could trade, and it would have
        been reported as a finding.

        Every claim this module makes therefore has to clear an effect-size floor as well
        as a p-value. Significance without magnitude is noise with a certificate, and this
        is the audience least likely to let that pass.
        """
        return self.significant and abs(self.slope) >= min_effect


def ols_slope(x, y, alpha: float = 0.05) -> Slope:
    """Least squares with a t-interval on the slope. Used for two things: whether an
    agent's excess Sharpe rises with episode index (learning), and whether its real-vs-twin
    edge rises with how identifiable the window is (memorization)."""
    x = np.asarray(list(x), dtype=np.float64)
    y = np.asarray(list(y), dtype=np.float64)
    n = len(x)
    if n < 3 or np.std(x) == 0:
        # No variation in the regressor: there is nothing to regress against, and saying so
        # is more honest than a number. Callers check `isfinite(slope)`.
        return Slope(float("nan"), float("nan"), float("nan"), float("nan"), 1.0, float("nan"), n)
    if np.std(y) == 0:
        # A perfectly flat series has slope exactly zero with no fit uncertainty. scipy
        # returns NaN here; a NaN dressed up as a p-value is worse than the truth.
        return Slope(0.0, float(y[0]), 0.0, 0.0, 1.0, 0.0, n)

    res = sps.linregress(x, y)
    tcrit = sps.t.ppf(1 - alpha / 2, df=n - 2)
    half = tcrit * res.stderr
    return Slope(
        slope=float(res.slope),
        intercept=float(res.intercept),
        lo=float(res.slope - half),
        hi=float(res.slope + half),
        pvalue=float(res.pvalue),
        r=float(res.rvalue),
        n=n,
    )


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Interval:
    """Wilson score interval for a proportion. Used for probe accuracy against the 12.5%
    chance line, where n is small enough that a normal-approximation interval would run
    off the end of [0, 1] and look ridiculous."""
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    z = sps.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half), n)


def binomial_p(k: int, n: int, chance: float) -> float:
    """One-sided exact binomial: is accuracy above chance? The only one-sided test here,
    and it is one-sided because 'the model identifies the series *worse* than chance' is
    not a hypothesis anyone holds."""
    if n == 0:
        return 1.0
    return float(sps.binomtest(k, n, chance, alternative="greater").pvalue)


def power_note(episode_sharpe_se: float = 1.65, n: int = 30) -> str:
    """The sentence that goes on the slide before someone else says it.

    An episode Sharpe estimated from 89 daily returns carries a standard error of roughly
    1.6-1.7 annualized units. Owning that up front is cheaper than being handed it.
    """
    se_median = 1.253 * episode_sharpe_se / math.sqrt(n)  # SE of a median, normal approx
    return (
        f"n = {n} paired windows. A single episode's Sharpe carries SE ~ {episode_sharpe_se:.2f} "
        f"annualized units off 89 daily returns, so the median carries SE ~ {se_median:.2f} "
        f"and a 95% interval spans ~ +/-{1.96 * se_median:.2f}. This design is power-limited "
        "by construction: intervals will overlap, and the paired Wilcoxon on identical "
        "windows is where what power we have actually lives."
    )
