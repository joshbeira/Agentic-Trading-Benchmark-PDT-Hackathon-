"""The trading scoreboard.

Ranked by **median vol-floored Sharpe** — median so that one lucky window cannot carry a
model, floored so that an agent cannot manufacture a ratio by refusing to deploy capital
(D2). Reported as distributions with bootstrap intervals, never as a single number.

Turnover, fees, drawdown, and time-in-market sit in the same table as the Sharpe rather
than being folded into it. A model that scalps itself to death or never leaves cash is
supposed to be *visible*, not penalized by a fudge factor someone chose. The reader
convicts it; the metric does not.

Two things this module refuses to do:

**Pool the regimes for a buy-and-hold claim.** The window sample is 10 bull / 10 bear /
10 chop by design (D12), which forces the pooled buy-and-hold median to roughly zero *by
construction*. "We beat buy-and-hold on the pooled sample" would therefore be an artifact
of our own sampling, so the comparison is only ever stated within-regime, and the pooled
row says so.

**Compare two models unpaired.** Every model sees identical windows, so the paired
Wilcoxon on the same 30 windows is available, and throwing that away for an unpaired test
would discard most of the little power this design has.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import stats as S
from .loader import Episode, Run

REGIMES = ("bull", "bear", "chop")

#: The ceiling on time_in_market is 88/90, not 1.0: tick 0 is always flat (the first fill
#: lands at open_1) and tick 89 is always flat (post-liquidation). Buy-and-hold sits here.
TIME_IN_MARKET_CEILING = 88 / 90


@dataclass
class Row:
    agent_id: str
    kind: str
    track: str
    regime: str | None  # None = pooled
    n: int
    sharpe: S.Interval
    sharpe_raw_median: float
    total_return: float
    max_drawdown: float
    turnover: float
    fees_cents: float
    time_in_market: float
    n_trades: float
    pct_floor_binding: float
    sharpes: list[float] = field(default_factory=list)


def _row(agent_id, kind, track, regime, eps: list[Episode], seed: int = 0) -> Row:
    sh = [e.metrics["sharpe_floored"] for e in eps]
    med = lambda k: float(np.median([e.metrics[k] for e in eps])) if eps else float("nan")  # noqa: E731
    return Row(
        agent_id=agent_id,
        kind=kind,
        track=track,
        regime=regime,
        n=len(eps),
        sharpe=S.bootstrap_ci(sh, seed=seed),
        sharpe_raw_median=med("sharpe_raw"),
        total_return=med("total_return"),
        max_drawdown=med("max_drawdown"),
        turnover=med("turnover"),
        fees_cents=med("fees_paid_cents"),
        time_in_market=med("time_in_market"),
        n_trades=med("n_trades"),
        pct_floor_binding=(
            float(np.mean([e.metrics["vol_floor_binding"] for e in eps])) if eps else float("nan")
        ),
        sharpes=sh,
    )


def build(run: Run, track: str, seed: int = 0) -> list[Row]:
    """Pooled rows, one per agent, ranked by median floored Sharpe."""
    rows = [
        _row(a, run.agent_kind(a), track, None, run.select(agent_id=a, track=track), seed)
        for a in run.agents
    ]
    rows = [r for r in rows if r.n]
    return sorted(rows, key=lambda r: r.sharpe.point, reverse=True)


def by_regime(run: Run, track: str, seed: int = 0) -> dict[str, list[Row]]:
    """The primary trading result (D12). Ten episodes per cell — small, and the interval
    says so."""
    out = {}
    for regime in REGIMES:
        rows = [
            _row(a, run.agent_kind(a), track, regime,
                 run.select(agent_id=a, track=track, regime=regime), seed)
            for a in run.agents
        ]
        out[regime] = sorted([r for r in rows if r.n], key=lambda r: r.sharpe.point, reverse=True)
    return out


@dataclass
class Comparison:
    a: str
    b: str
    track: str
    n_pairs: int
    median_diff: S.Interval
    test: S.TestResult

    @property
    def verdict(self) -> str:
        if self.test.pvalue < 0.05 and self.median_diff.excludes_zero:
            winner = self.a if self.median_diff.point > 0 else self.b
            return f"{winner} ahead (p={self.test.pvalue:.3f})"
        return "not separated at this sample size"


def compare(run: Run, a: str, b: str, track: str, seed: int = 0) -> Comparison:
    """Paired, on the identical windows both models saw (D5)."""
    ea = {e.window_id: e for e in run.select(agent_id=a, track=track)}
    eb = {e.window_id: e for e in run.select(agent_id=b, track=track)}
    shared = [w for w in ea if w in eb]

    xa = [ea[w].metrics["sharpe_floored"] for w in shared]
    xb = [eb[w].metrics["sharpe_floored"] for w in shared]
    return Comparison(
        a=a, b=b, track=track, n_pairs=len(shared),
        median_diff=S.paired_diff_ci(xa, xb, seed=seed),
        test=S.wilcoxon(xa, xb),
    )


def vs_buy_and_hold(run: Run, agent_id: str, track: str, bh_id: str = "buy_and_hold",
                    seed: int = 0) -> dict[str, Comparison]:
    """Within-regime only, and only within-regime.

    Pooling would compare against a buy-and-hold median that our own 10/10/10 regime
    balance pins near zero. That is not a finding about the agent; it is a finding about
    our sampling (D12).
    """
    if bh_id not in run.agents:
        return {}
    out = {}
    for regime in REGIMES:
        ea = {e.window_id: e for e in run.select(agent_id=agent_id, track=track, regime=regime)}
        eb = {e.window_id: e for e in run.select(agent_id=bh_id, track=track, regime=regime)}
        shared = [w for w in ea if w in eb]
        if not shared:
            continue
        xa = [ea[w].metrics["sharpe_floored"] for w in shared]
        xb = [eb[w].metrics["sharpe_floored"] for w in shared]
        out[regime] = Comparison(
            a=agent_id, b=bh_id, track=track, n_pairs=len(shared),
            median_diff=S.paired_diff_ci(xa, xb, seed=seed),
            test=S.wilcoxon(xa, xb),
        )
    return out


# ------------------------------------------------------------------------ rendering


def render(rows: list[Row], title: str) -> str:
    out = [title, "=" * len(title), ""]
    out.append(
        f"  {'agent':<14} {'kind':<9} {'n':>3}  {'median Sharpe (95% CI)':<24} "
        f"{'raw':>6} {'ret':>7} {'maxDD':>7} {'turn':>6} {'fees':>7} {'in mkt':>7} "
        f"{'trades':>6} {'floor':>6}"
    )
    out.append("  " + "-" * 116)
    for r in rows:
        ci = f"{r.sharpe.point:+.2f} [{r.sharpe.lo:+.2f}, {r.sharpe.hi:+.2f}]"
        out.append(
            f"  {r.agent_id:<14} {r.kind:<9} {r.n:>3}  {ci:<24} "
            f"{r.sharpe_raw_median:>6.2f} {r.total_return:>+6.1%} {r.max_drawdown:>+6.1%} "
            f"{r.turnover:>6.1f} ${r.fees_cents / 100:>6.0f} {r.time_in_market:>6.1%} "
            f"{r.n_trades:>6.0f} {r.pct_floor_binding:>5.0%}"
        )
    out.append("")
    out.append(f"  time-in-market ceiling is {TIME_IN_MARKET_CEILING:.1%}, not 100% "
               "(tick 0 pre-fill and tick 89 post-liquidation are flat by construction)")
    return "\n".join(out)
