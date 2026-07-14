#!/usr/bin/env python
"""Generate a SYNTHETIC run with known planted effects, to exercise the analytics.

This is a development tool. Nothing it produces is a result — the Sharpes are planted, the
probe is a stub, and no model was ever called. It exists so the whole analytics layer can
be driven end to end and checked against effects whose true size we chose.

    python scripts/make_fixture_run.py
    python scripts/analyze.py runs/_fixture_demo

What is planted:
  flagship  +0.5 Sharpe of genuine skill over window difficulty, genuine in-context
            learning at +0.02/episode, AND a real leak — it recognizes some real windows
            far better than others, and its real-vs-twin edge concentrates in exactly the
            ones it can name (slope 1.5, plus noise). The analysis must convict it.
  cheap     no skill edge, no learning, and it "identifies" real windows and twins alike
            at 40% — five times chance — because it is reading volatility texture rather
            than recalling a path. The analysis must CLEAR it. This is the harder of the
            two cases and the reason the twin arm exists.

The recovered leak slope comes back a little under the planted 1.5, and that is expected:
per-window identifiability is quantized to 1/n_reps, which is measurement error in the
regressor, which biases an OLS slope toward zero. The headline understates leakage rather
than inflating it.
"""

from __future__ import annotations

import shutil
import sys

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pdtbench.analysis import leakage as LK  # noqa: E402
from tests import fixtures as F  # noqa: E402

RUN_DIR = Path("runs/_fixture_demo")
N_REPS = 5

# planted ground truth --------------------------------------------------------
SKILL = {"flagship": 0.5, "cheap": 0.0}
LEAK_BETA = {"flagship": 1.5, "cheap": 0.0}  # Sharpe edge per unit identifiability
BASELINE_OFFSET = {"buy_and_hold": 0.1, "flat": 0.0, "random_5pct": -0.1, "sma_10_50": -0.2}


def main() -> int:
    if RUN_DIR.exists():
        shutil.rmtree(RUN_DIR)

    windows = F.window_specs()
    agents = [
        F.agent_spec("flagship"),
        F.agent_spec("cheap"),
        *[F.agent_spec(b, kind="baseline", memory="none") for b in BASELINE_OFFSET],
    ]

    rng = np.random.default_rng(3)

    # flagship recognizes some windows far better than others -- famous crises are nameable
    # from shape alone, a quiet 2013 chop window is not. Accuracy is quantized to 1/n_reps
    # by construction, which is the real resolution limit of the identifiability axis.
    TWIN_FLOOR = 0.2  # what texture alone buys you on a series never seen
    ident = {
        w["window_id"]: float(np.round(rng.uniform(0.2, 1.0) * N_REPS) / N_REPS)
        for w in windows
    }
    accuracy = {
        "flagship": {
            **{("real", w["window_id"]): ident[w["window_id"]] for w in windows},
            **{("twin", w["window_id"]): TWIN_FLOOR for w in windows},
        },
        # cheap "identifies" real and twin equally: it is reading volatility texture, not
        # recalling a path, and the twin arm is what stops us convicting it for that.
        "cheap": {
            **{("real", w["window_id"]): 0.5 for w in windows},
            **{("twin", w["window_id"]): 0.5 for w in windows},
        },
    }
    # Real edges are noisy: an agent that exploits what it remembers still has bad days.
    noise = {w["window_id"]: float(rng.normal(0, 0.20)) for w in windows}

    def sharpe_fn(agent, track, w, i):
        base = w["difficulty"] + 0.03 * i  # windows get mildly easier: a trap for the curve
        if agent["kind"] == "baseline":
            return base + BASELINE_OFFSET[agent["id"]]
        edge = 0.0
        if track == "real":
            leak = LEAK_BETA[agent["id"]] * (ident[w["window_id"]] - TWIN_FLOOR)
            edge = SKILL[agent["id"]] + leak + (noise[w["window_id"]] if leak else 0.0)
        # flagship also genuinely learns: +0.02 Sharpe per episode on top of everything
        learn = 0.02 * i if agent["id"] == "flagship" else 0.0
        return base + edge + learn

    F.make_run(RUN_DIR, agents=agents, windows=windows, sharpe_fn=sharpe_fn,
               run_id="_fixture_demo",
               # the flat baseline must actually look flat: never in the market, no fees,
               # Sharpe exactly 0 -- it is the metric's fixed point and the scoreboard
               # should show it as one
               exposure_fn=lambda a: 0.0 if a["id"] == "flat" else 1.0)

    # --- probe files ---------------------------------------------------------
    for agent_id, acc in accuracy.items():
        for w in windows:
            truth = {"ticker": w["source_ticker"], "period": w["period"]}
            options = LK.build_options(w["window_id"], truth, F.TICKERS, F.YEARS)
            for track in ("real", "twin"):
                qs = LK.build_questions(w["window_id"], track, truth, options, N_REPS)
                responses = [
                    LK.ask(q, [], _stub(acc[(track, w["window_id"])], truth, track))
                    for q in qs
                ]
                s = LK.score(agent_id, responses)
                LK.write_probe_file(
                    RUN_DIR / "probe" / f"{agent_id}__{track}__{w['window_id']}.json",
                    agent_id, "_fixture_demo", responses, s, n_options=8,
                )

    print("=" * 74)
    print("SYNTHETIC RUN — NOT A RESULT. Sharpes are planted; no model was called.")
    print("=" * 74)
    print(f"  wrote {RUN_DIR}")
    print(f"  {len(agents)} agents x 2 tracks x {len(windows)} windows = "
          f"{len(agents) * 2 * len(windows)} episodes")
    print()
    print("  planted, for the analysis to recover:")
    print("    flagship  skill +0.50 | learning +0.02/ep | leak slope 1.50  -> must CONVICT")
    print("    cheap     skill  0.00 | learning  0.00    | leak slope 0.00  -> must CLEAR,")
    print("              despite identifying real windows at 40% -- 5x chance -- because it")
    print("              identifies the twins at 40% too. Texture, not recall.")
    print()
    print("  (the recovered slope lands under 1.50: identifiability is quantized to 1/n_reps,")
    print("   and measurement error in the regressor attenuates an OLS slope toward zero)")
    print()
    print(f"  now run:  python scripts/analyze.py {RUN_DIR}")
    return 0


def _stub(acc: float, truth: dict, track: str):
    """Deterministic: correct on rep r iff r < round(acc * n_reps)."""

    def fn(bars, options, context):
        right = context["rep"] < round(acc * N_REPS)
        out = {}
        for axis in ("ticker", "period"):
            if right:
                out[axis] = truth[axis]
            else:
                wrong = [o for o in options[axis] if o != truth[axis]]
                out[axis] = wrong[context["rep"] % len(wrong)]
        return out

    return fn


if __name__ == "__main__":
    raise SystemExit(main())
