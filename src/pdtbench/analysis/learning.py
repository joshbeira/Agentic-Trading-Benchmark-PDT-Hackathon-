"""Learning curves — and the baseline trace that makes them mean anything.

Weights are frozen. Nothing here is policy improvement. What the design tests is whether
an agent, handed its own performance summary and allowed a 500-token note into its next
episode, converts that feedback into better behavior (D11). Plot episode Sharpe against
episode index and you get a curve. The question is whether the curve is *the agent*.

It usually is not. Episode index also indexes *which window* — every agent walks the same
fixed presentation order — so a rising curve can mean nothing more than "the later windows
were easier". The regimes interleave bull/bear/chop precisely so this cannot become a
monotone ramp, but it still leaves a sawtooth of difficulty with period 3 sitting under
every curve.

The fix is the reason baselines are in the run at all. Buy-and-hold, flat, random, and the
SMA crossover **cannot learn**. Whatever their Sharpe does across episode index is pure
window difficulty. Subtract that trace and what remains is the only honest candidate for
learning:

    excess(i) = sharpe_agent(i) - median over baselines of sharpe(i)

A positive slope on `excess` is evidence. A positive slope on the raw curve is a mood.

The no-memory LLM ablation arm, if present, is deliberately *not* part of the trace — it is
a comparison arm, not a difficulty probe (see schemas/analysis_artifacts.md Q2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import stats as S
from .loader import Run

#: Sharpe per episode. Over a 30-episode lane this compounds to ~0.15 Sharpe — roughly the
#: smallest improvement anyone would bother to describe as learning. Below it, a p-value on
#: a near-zero slope is certifying rounding noise.
MIN_LEARNING_SLOPE = 0.005


@dataclass
class Curve:
    agent_id: str
    track: str
    memory: str
    index: list[int]
    sharpe: list[float]
    baseline_trace: list[float]
    excess: list[float]
    raw_slope: S.Slope
    excess_slope: S.Slope
    first_half: float = float("nan")
    second_half: float = float("nan")

    @property
    def learned(self) -> bool:
        """Only the excess slope can say this — and only if it clears zero *and* is big
        enough to matter. Over a 30-episode lane, MIN_LEARNING_SLOPE compounds to about
        0.15 Sharpe; below that we are reporting rounding noise with a p-value attached."""
        return self.excess_slope.matters(MIN_LEARNING_SLOPE) and self.excess_slope.slope > 0

    @property
    def verdict(self) -> str:
        if self.memory == "none":
            return "no memory (control arm)"
        if self.learned:
            return f"learning: excess Sharpe +{self.excess_slope.slope:.3f}/episode"
        if self.excess_slope.matters(MIN_LEARNING_SLOPE):
            return f"DEGRADING: excess Sharpe {self.excess_slope.slope:+.3f}/episode"
        if self.excess_slope.significant:
            return (f"a slope of {self.excess_slope.slope:+.4f}/episode clears zero but is "
                    f"below the {MIN_LEARNING_SLOPE} effect floor — not worth claiming")
        return "no detectable learning at this sample size"


def trace_baselines(run: Run, track: str) -> list[str]:
    """Which baselines actually carry a difficulty signal.

    **A baseline whose score does not depend on the window cannot tell you anything about
    the window.** The `flat` baseline never trades, so its Sharpe is exactly 0 on every
    window in every regime — it has zero variance across the lane and zero information
    about difficulty. Including it in the median does not merely add nothing; it actively
    drags the trace toward zero and leaves a residual trend in the excess, which then reads
    as learning. That is not hypothetical: it showed up as a spurious, *statistically
    significant* +0.004/episode slope for an agent with no learning planted at all.

    So the trace is built only from baselines that take market exposure and therefore
    respond to the window.
    """
    out = []
    for a in run.baselines:
        sh = [e.metrics["sharpe_floored"] for e in run.lane(a, track)]
        if len(sh) > 1 and float(np.std(sh)) > 1e-9:
            out.append(a)
    return out or run.baselines  # degenerate run: better a poor trace than none


def baseline_trace(run: Run, track: str) -> dict[int, float]:
    """Window difficulty, as seen by strategies that cannot learn.

    Median across the informative baselines at each episode index. Median rather than mean
    because the baselines are wildly heterogeneous — a churner posts -1.4 Sharpe while
    buy-and-hold posts +0.3 — and a mean would let the worst baseline drag the trace.
    """
    baselines = trace_baselines(run, track)
    trace: dict[int, float] = {}
    if not baselines:
        return trace

    by_index: dict[int, list[float]] = {}
    for a in baselines:
        for e in run.lane(a, track):
            by_index.setdefault(e.episode_index, []).append(e.metrics["sharpe_floored"])
    for i, vals in by_index.items():
        trace[i] = float(np.median(vals))
    return trace


def curve(run: Run, agent_id: str, track: str) -> Curve:
    lane = run.lane(agent_id, track)
    trace = baseline_trace(run, track)

    idx = [e.episode_index for e in lane]
    sh = [e.metrics["sharpe_floored"] for e in lane]
    base = [trace.get(i, 0.0) for i in idx]
    excess = [s - b for s, b in zip(sh, base)]

    half = len(excess) // 2
    return Curve(
        agent_id=agent_id,
        track=track,
        memory=run.agent_memory(agent_id),
        index=idx,
        sharpe=sh,
        baseline_trace=base,
        excess=excess,
        raw_slope=S.ols_slope(idx, sh),
        excess_slope=S.ols_slope(idx, excess),
        first_half=float(np.median(excess[:half])) if half else float("nan"),
        second_half=float(np.median(excess[half:])) if half else float("nan"),
    )


def build(run: Run, track: str) -> list[Curve]:
    """Curves for every learning agent. Baselines get one too — theirs is the control, and
    a baseline whose *excess* slope is non-zero would mean the trace is broken."""
    return [curve(run, a, track) for a in run.agents]


def render(curves: list[Curve], track: str, run: Run | None = None) -> str:
    out = [f"LEARNING CURVES ({track} track)", "=" * (16 + len(track) + 8), ""]
    if run is not None:
        used = trace_baselines(run, track)
        dropped = [b for b in run.baselines if b not in used]
        out.append(f"  difficulty trace = median of {used}")
        if dropped:
            out.append(f"  excluded from the trace: {dropped} — zero variance across "
                       "windows, so they carry no difficulty signal")
        out.append("")
    out.append(
        f"  {'agent':<14} {'memory':<13} {'raw slope':<22} {'excess slope':<22} "
        f"{'1st half':>9} {'2nd half':>9}"
    )
    out.append("  " + "-" * 96)
    for c in curves:
        out.append(
            f"  {c.agent_id:<14} {c.memory:<13} {str(c.raw_slope):<22} "
            f"{str(c.excess_slope):<22} {c.first_half:>+9.2f} {c.second_half:>+9.2f}"
        )
    out.append("")
    out.append("  'excess' is Sharpe minus the median baseline Sharpe on the same window.")
    out.append("  The baselines cannot learn, so their trace IS window difficulty --")
    out.append("  a rising RAW curve may only mean the later windows were kinder.")
    out.append("  Only the excess slope is evidence, and only if its interval clears zero.")
    for c in curves:
        if c.memory != "none":
            out.append(f"    {c.agent_id}: {c.verdict}")
    return "\n".join(out)
