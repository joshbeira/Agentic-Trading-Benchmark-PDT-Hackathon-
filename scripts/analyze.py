#!/usr/bin/env python
"""Render every scoreboard and figure for a run.

    python scripts/analyze.py runs/<run_id>
    python scripts/analyze.py runs/<run_id> --figures figures/

Reads tick logs and nothing else. If it prints a number, that number came out of a JSONL
file in that directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdtbench.analysis import leakage as LK  # noqa: E402
from pdtbench.analysis import learning as LC  # noqa: E402
from pdtbench.analysis import reliability as RL  # noqa: E402
from pdtbench.analysis import report as RP  # noqa: E402
from pdtbench.analysis import scoreboard as SB  # noqa: E402
from pdtbench.analysis import stats as S  # noqa: E402
from pdtbench.analysis.loader import load_run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--figures", type=Path, default=Path("figures"))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    run = load_run(args.run_dir)
    print(f"run {run.run_id}: {len(run.episodes)} episodes, "
          f"{len(run.llms)} models, {len(run.baselines)} baselines\n")

    if run.drifted:
        print(f"  !! {len(run.drifted)} episodes whose cached metrics disagree with a")
        print("     recomputation from their own equity series. The engine and the")
        print("     specification have diverged; one of them is wrong.")
        for e in run.drifted[:3]:
            print(f"     {e.episode_id}: {e.metric_drift}")
        print()
    if run.excluded:
        print(f"  {len(run.excluded)} episodes excluded from the trading scoreboard "
              f"(status=agent_error): {[e.episode_id for e in run.excluded][:5]}")
        print("  They are counted on the reliability board.\n")

    for track in ("real", "twin"):
        if not run.select(track=track):
            continue
        print(SB.render(SB.build(run, track), f"TRADING SCOREBOARD — {track.upper()} TRACK"))
        print()
        for regime, rows in SB.by_regime(run, track).items():
            print(SB.render(rows, f"  regime: {regime}"))
            print()

    # --- model vs model, paired on identical windows -------------------------
    if len(run.llms) >= 2:
        print("MODEL vs MODEL (paired Wilcoxon on the identical 30 windows)")
        print("=" * 60)
        a, b = run.llms[0], run.llms[1]
        for track in ("real", "twin"):
            c = SB.compare(run, a, b, track)
            print(f"  {track:<5} {c.a} − {c.b}: median diff {c.median_diff}  "
                  f"{c.test}  →  {c.verdict}")
        print()

    print(RL.render(RL.build(run)))
    print()

    for track in ("real", "twin"):
        if run.select(track=track):
            print(LC.render(LC.build(run, track), track, run))
            print()

    # --- leakage -------------------------------------------------------------
    probe_dir = args.run_dir / "probe"
    results = []
    if probe_dir.is_dir():
        probes = LK.load_probe_dir(probe_dir)
        for agent in run.llms:
            if any(p.agent_id == agent for p in probes):
                res = LK.analyze(run, agent, probes)
                results.append(res)
                print(LK.render(res))
                print()
    else:
        print("LEAKAGE PROBE: no probe/ directory — run the probe harness first.\n")

    print("POWER")
    print("=====")
    print(" ", S.power_note())
    print()

    if not args.no_figures:
        out = args.figures
        made = [RP.trading(run, "real", out / "trading_real.png"),
                RP.reliability_fig(run, out / "reliability.png"),
                RP.learning_fig(run, "real", out / "learning_real.png")]
        if run.select(track="twin"):
            made.append(RP.trading(run, "twin", out / "trading_twin.png"))
        if results:
            made.append(RP.leakage_fig(results, out / "leakage.png"))
        print("figures:")
        for p in made:
            print(f"  {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
