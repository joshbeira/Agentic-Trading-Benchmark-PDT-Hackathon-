"""Figures. Four of them, one per claim.

Distributions, not bars-of-a-single-number: the design promises to report spread and it
would be strange to then hand a quant a bar chart of medians. Every panel shows the
episodes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import leakage as LK  # noqa: E402
from . import learning as LC  # noqa: E402
from . import reliability as RL  # noqa: E402
from . import scoreboard as SB  # noqa: E402
from .loader import Run  # noqa: E402

BASE = "#8899a6"
HILITE = ["#2b6cb0", "#c05621", "#2f855a", "#6b46c1"]
plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
})


def _color(i: int, kind: str) -> str:
    return BASE if kind == "baseline" else HILITE[i % len(HILITE)]


def trading(run: Run, track: str, out: Path) -> Path:
    """Per-regime Sharpe distributions. The regime panels are the primary result (D12);
    the pooled panel exists mainly to show why it is not the headline."""
    panels = SB.by_regime(run, track)
    pooled = SB.build(run, track)
    order = [r.agent_id for r in pooled]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
    for ax, (name, rows) in zip(axes, [*panels.items(), ("pooled", pooled)]):
        rows = sorted(rows, key=lambda r: order.index(r.agent_id))
        for i, r in enumerate(rows):
            y = r.sharpes
            x = np.full(len(y), i) + np.random.default_rng(i).normal(0, 0.06, len(y))
            ax.scatter(x, y, s=14, alpha=0.5, color=_color(i, r.kind), zorder=2)
            ax.hlines(r.sharpe.point, i - 0.3, i + 0.3, color=_color(i, r.kind), lw=2.5, zorder=3)
            ax.vlines(i, r.sharpe.lo, r.sharpe.hi, color=_color(i, r.kind), lw=1.2, zorder=3)
        ax.axhline(0, color="#333", lw=0.8, zorder=1)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([r.agent_id for r in rows], rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{name}  (n={rows[0].n if rows else 0})", fontsize=10)
    axes[0].set_ylabel("episode Sharpe (vol-floored)")
    fig.suptitle(
        f"Trading scoreboard — {track} track. Bars = median with 95% bootstrap CI.\n"
        "Pooled buy-and-hold sits near zero by construction: the sample is 10/10/10 by "
        "regime, so the pooled comparison is an artifact of our sampling, not a result.",
        fontsize=9,
    )
    fig.tight_layout()
    return _save(fig, out)


def reliability_fig(run: Run, out: Path) -> Path:
    rows = RL.build(run)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))

    ids = [r.agent_id for r in rows]
    a1.barh(ids, [r.invalid_rate for r in rows],
            color=[_color(i, r.kind) for i, r in enumerate(rows)])
    a1.set_xlabel("invalid-call rate")
    a1.set_title("Interface reliability — no P&L attaches to any of this")

    a2.barh(ids, [r.forced_waits_per_episode for r in rows],
            color=[_color(i, r.kind) for i, r in enumerate(rows)])
    a2.set_xlabel("forced Waits per episode")
    a2.set_title("Anti-stall interventions")
    fig.tight_layout()
    return _save(fig, out)


def learning_fig(run: Run, track: str, out: Path) -> Path:
    """Raw curve on the left, excess on the right. The point of the pair is that the left
    panel is seductive and the right one is the evidence."""
    curves = LC.build(run, track)
    llms = [c for c in curves if c.memory != "none"]
    trace = LC.baseline_trace(run, track)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    idx = sorted(trace)
    a1.plot(idx, [trace[i] for i in idx], color=BASE, lw=2, marker="o", ms=3,
            label="baseline trace (window difficulty)")
    for i, c in enumerate(llms):
        a1.plot(c.index, c.sharpe, color=HILITE[i % len(HILITE)], lw=1.5, marker="o",
                ms=3, alpha=0.85, label=c.agent_id)
    a1.axhline(0, color="#333", lw=0.8)
    a1.set_xlabel("episode index (memory lane)")
    a1.set_ylabel("episode Sharpe")
    a1.set_title("Raw — indistinguishable from window difficulty")
    a1.legend(fontsize=7)

    for i, c in enumerate(llms):
        col = HILITE[i % len(HILITE)]
        a2.plot(c.index, c.excess, color=col, lw=1.2, marker="o", ms=3, alpha=0.5)
        fit = np.poly1d([c.excess_slope.slope, c.excess_slope.intercept])
        star = "*" if c.excess_slope.significant else ""
        a2.plot(c.index, fit(c.index), color=col, lw=2.5,
                label=f"{c.agent_id}: {c.excess_slope.slope:+.3f}/ep{star}")
    a2.axhline(0, color="#333", lw=0.8)
    a2.set_xlabel("episode index (memory lane)")
    a2.set_ylabel("excess Sharpe over baseline trace")
    a2.set_title("Excess — the only honest evidence of learning")
    a2.legend(fontsize=7)

    fig.suptitle(f"Learning curves — {track} track. Weights are frozen; this is in-context "
                 "learning via a 500-token note (D11).", fontsize=9)
    fig.tight_layout()
    return _save(fig, out)


def leakage_fig(results: list[LK.LeakageResult], out: Path) -> Path:
    """The headline. Left: does it identify? Right: does identification pay?"""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))

    x = np.arange(len(results))
    w = 0.35
    chance = results[0].chance if results else 0.125
    a1.bar(x - w / 2, [r.accuracy_real.point for r in results], w, label="real windows",
           color=HILITE[0])
    a1.bar(x + w / 2, [r.accuracy_twin.point for r in results], w,
           label="twins (false-positive rate)", color=BASE)
    a1.errorbar(x - w / 2, [r.accuracy_real.point for r in results],
                yerr=[[r.accuracy_real.point - r.accuracy_real.lo for r in results],
                      [r.accuracy_real.hi - r.accuracy_real.point for r in results]],
                fmt="none", ecolor="#333", lw=1)
    a1.axhline(chance, color="#c53030", ls="--", lw=1.2, label=f"chance ({chance:.1%})")
    a1.set_xticks(x)
    a1.set_xticklabels([r.agent_id for r in results])
    a1.set_ylabel("probe accuracy")
    a1.set_title("Can it name the series?\nThe gap to the twin bar is the only real signal")
    a1.legend(fontsize=7)

    for i, r in enumerate(results):
        col = HILITE[i % len(HILITE)]
        ident = [p.identifiability for p in r.points]
        edge = [p.edge for p in r.points]
        a2.scatter(ident, edge, s=26, alpha=0.6, color=col, label=r.agent_id)
        if r.slope.n >= 3 and np.isfinite(r.slope.slope):
            xs = np.linspace(min(ident), max(ident), 10)
            star = "*" if r.slope.significant else ""
            a2.plot(xs, r.slope.slope * xs + r.slope.intercept, color=col, lw=2,
                    label=f"slope {r.slope.slope:+.2f}{star}")
    a2.axhline(0, color="#333", lw=0.8)
    a2.axvline(0, color="#333", lw=0.8)
    a2.set_xlabel("calibrated identifiability  (p_real − p_twin)")
    a2.set_ylabel("Sharpe edge  (real − twin)")
    a2.set_title("Does recognition pay?\nSlope > 0 ⇒ memorization converted into P&L")
    a2.legend(fontsize=7)

    fig.suptitle("Leakage experiment — each point is one window. The twin is matched to its "
                 "source on return, volatility, and regime; only path identity differs.",
                 fontsize=9)
    fig.tight_layout()
    return _save(fig, out)


def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
