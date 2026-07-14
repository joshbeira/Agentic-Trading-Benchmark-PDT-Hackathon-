#!/usr/bin/env python
"""Engine verification report.

Runs every scripted policy across every window on both tracks, and checks the four
things the benchmark's credibility rests on — the equity identity, the absence of
look-ahead, conformance to the tick-log contract, and the ability to regenerate every
reported number from the log alone. Prints counts, not adjectives.

    python scripts/verify_engine.py

**Every check here reads the JSONL and the pinned parquet, never the live engine's
state.** That is not fastidiousness. The previous version of section 1 did this:

    cash = eq - int(round(sh * closes[t] * 100))
    if eq != cash + int(round(sh * closes[t] * 100)):   # i.e. `eq != eq`

It derived cash from the equity mark it was about to check, then asserted the identity
it had just used to derive it. It could not fail. Sabotage the engine's equity mark by
12,345 cents and it still printed PASS across 5,400 tick marks, while an honest
reconstruction flagged 89 of 90 ticks. A verifier that cannot fail is a decoration, and
this is the one script a sceptical reader would actually open.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pdtbench.config import DEFAULT as cfg, WINDOWS_DIR  # noqa: E402
from pdtbench.data.windows import load_episode, load_manifest  # noqa: E402
from pdtbench.engine import LookAheadError, TradingEnv  # noqa: E402
from pdtbench.engine.replay import _shares_after, load, replay  # noqa: E402
from pdtbench.schema import validate_episode  # noqa: E402
from tests import policies as P  # noqa: E402

POLICIES = {
    "flat": P.Flat,
    "buy_and_hold": P.BuyAndHold,
    "sma_10_50": P.SmaCrossover,
    "random_5pct": lambda: P.RandomTrader(seed=11),
    "churn": P.Churn,
    "dust": P.Dust,
}


def _audit_log(log: Path, series) -> tuple[int, int, list[str], list[str]]:
    """Re-derive the episode from its own bytes: cash and shares from the fill records,
    marked against the parquet. Returns (n_marks, n_fills, equity errors, fill errors).

    Nothing here reads `obs.portfolio.cash_cents` or `obs.portfolio.shares` to *build*
    the state it is checking — those are the numbers under test. They are compared
    against the reconstruction, not trusted as inputs.
    """
    meta, ticks, _ = load(log)
    opens = dict(zip(series["bar_idx"], series["open"]))
    closes = dict(zip(series["bar_idx"], series["close"]))
    last = meta["config"]["n_scored"] - 1

    landing: dict[int, list[dict]] = {}
    for rec in ticks:
        if rec.get("fill"):
            landing.setdefault(rec["fill"]["fill_tick"], []).append(rec["fill"])

    cash = meta["config"]["initial_capital_cents"]
    shares = 0.0
    bad_equity: list[str] = []
    bad_fill: list[str] = []
    n_marks = n_fills = 0

    for rec in ticks:
        t = rec["t"]
        for f in landing.get(t, []):
            cash += f["cash_delta_cents"]
            shares = _shares_after(shares, f)

        expect = cash + round(shares * closes[t] * 100)
        if rec["equity_cents"] != expect or rec["equity_cents"] <= 0:
            bad_equity.append(f"@{t}: logged {rec['equity_cents']} != rebuilt {expect}")
        n_marks += 1

        f = rec.get("fill")
        if f:
            n_fills += 1
            want = closes[last] if f["side"] == "liquidation" else opens.get(f["fill_tick"])
            if want is None or abs(f["fill_price"] - want) > 1e-9:
                bad_fill.append(f"@{t}: filled at {f['fill_price']}, expected {want}")
            if f["side"] != "liquidation" and f["fill_tick"] != t + 1:
                bad_fill.append(f"@{t}: fill_tick {f['fill_tick']} != {t + 1}")
            if f["side"] == "liquidation" and t != last:
                bad_fill.append(f"@{t}: liquidation off the terminal tick")

    return n_marks, n_fills, bad_equity, bad_fill


def main() -> int:
    man = load_manifest(WINDOWS_DIR)
    ids = man["presentation_order"]
    tracks = ("real", "twin")
    tmp = Path(tempfile.mkdtemp(prefix="pdtbench_verify_"))

    n_ep = n_marks = n_fills = 0
    bad_equity: list[str] = []
    bad_fill: list[str] = []
    schema_fail: list[str] = []
    replay_fail: list[str] = []
    check_counts: dict[str, int] = {}
    rows: dict[str, list[dict]] = {k: [] for k in POLICIES}

    print("running", len(POLICIES) * len(ids) * len(tracks), "episodes ...\n")

    for pname, make_policy in POLICIES.items():
        for track in tracks:
            for i, wid in enumerate(ids):
                series, spec = load_episode(WINDOWS_DIR, wid, track)
                log = tmp / f"{pname}__{track}__{wid}.jsonl"
                env = TradingEnv(
                    series, spec, cfg, log_path=log,
                    meta_extra={
                        "episode_id": f"{pname}__{track}__{wid}",
                        "episode_index": i,
                        "track": track,
                        "agent": {"id": pname, "kind": "baseline", "memory": "none"},
                        "dataset": {"dataset_sha256": man["dataset_sha256"]},
                    },
                )
                P.run(env, make_policy())
                env.close_log()
                n_ep += 1
                tag = f"{pname}/{track}/{wid}"

                # --- 1 & 2. equity and look-ahead, rebuilt from the log alone --
                m, fl, be, bf = _audit_log(log, series)
                n_marks += m
                n_fills += fl
                bad_equity += [f"{tag}{s}" for s in be]
                bad_fill += [f"{tag}{s}" for s in bf]

                # --- 3. the log conforms to the contract ----------------------
                rep = validate_episode(log)
                if not rep.ok:
                    schema_fail.append(f"{tag}: {rep.errors[:2]}")

                # --- 4. replay: rebuild the scoreboard row from the log alone --
                res = replay(log, WINDOWS_DIR)
                for k, v in res.checks.items():
                    check_counts[k] = check_counts.get(k, 0) + int(v)
                if not res.ok:
                    replay_fail.append(f"{tag}: {res.failures}")

                rows[pname].append({**env.summary, "regime": spec["regime"], "track": track})

    # ---------------------------------------------------------------- report
    print("=" * 78)
    print("1. EQUITY INVARIANT      equity == cash + shares x close, to the cent")
    print("=" * 78)
    print(f"   {n_marks:,} tick marks across {n_ep} episodes, with cash and shares")
    print("   rebuilt from the fill records alone -- not read back out of the log's")
    print("   own portfolio block, which is the number under test")
    print(f"   violations: {len(bad_equity)}   {'PASS' if not bad_equity else 'FAIL ' + str(bad_equity[:3])}")

    print()
    print("=" * 78)
    print("2. NO LOOK-AHEAD         a decision at t fills at open_{t+1}, never earlier")
    print("=" * 78)
    print(f"   {n_fills:,} fills re-priced against the pinned parquet")
    print(f"   mispriced: {len(bad_fill)}   {'PASS' if not bad_fill else 'FAIL ' + str(bad_fill[:3])}")
    print("   read-audit tripwire (an engine that peeks must raise):", end=" ")
    print("PASS" if _tripwire_fires() else "FAIL")

    print()
    print("=" * 78)
    print("3. SCHEMA CONFORMANCE    the log obeys the contract in schema.py")
    print("=" * 78)
    print(f"   {n_ep} episodes validated (ordering, tick shapes, enums, integer cents,")
    print("   forced-wait rules, no undeclared fields)")
    print(f"   rejected: {len(schema_fail)}   {'PASS' if not schema_fail else 'FAIL ' + str(schema_fail[:2])}")

    print()
    print("=" * 78)
    print("4. REPLAY REPRODUCIBILITY   every number rebuilt from the JSONL alone")
    print("=" * 78)
    for name, passed in sorted(check_counts.items()):
        status = "PASS" if passed == n_ep else f"FAIL ({n_ep - passed} episodes)"
        print(f"   {name:<28} {passed:>3}/{n_ep}  {status}")
    print(f"\n   episodes failing replay: {len(replay_fail)}")
    for f in replay_fail[:3]:
        print(f"     {f}")

    _economics(rows)
    print(f"\nlogs: {tmp}")
    return 1 if (bad_equity or bad_fill or schema_fail or replay_fail) else 0


def _tripwire_fires() -> bool:
    """The audit must fire when the observation path reads tomorrow."""
    series, spec = load_episode(WINDOWS_DIR, "w00", "real")
    env = TradingEnv(series, spec, cfg)
    original = TradingEnv._stats
    try:
        TradingEnv._stats = lambda self, t: (self.bars.close(t + 1), original(self, t))[1]
        try:
            P.run(env, P.Flat())
        except LookAheadError:
            return True
        return False
    finally:
        TradingEnv._stats = original


def _economics(rows: dict[str, list[dict]]) -> None:
    """Does the simulated world behave like a market? A sanity read, not a result."""
    print()
    print("=" * 78)
    print("ENGINE ECONOMICS (scripted policies -- a sanity read, not the benchmark)")
    print("=" * 78)
    print(f"   {'policy':<14} {'med Sharpe':>11} {'med ret':>9} {'med DD':>8} "
          f"{'turnover':>9} {'fees':>7} {'in mkt':>7} {'floor':>6}")
    for name, rs in rows.items():
        sh = np.median([r["sharpe_floored"] for r in rs])
        rt = np.median([r["total_return"] for r in rs])
        dd = np.median([r["max_drawdown"] for r in rs])
        to = np.median([r["turnover"] for r in rs])
        fe = np.median([r["fees_paid_cents"] for r in rs]) / 100
        im = np.median([r["time_in_market"] for r in rs])
        fl = np.mean([r["vol_floor_binding"] for r in rs])
        print(f"   {name:<14} {sh:>11.2f} {rt:>8.1%} {dd:>8.1%} {to:>9.1f} "
              f"${fe:>6.0f} {im:>6.0%} {fl:>5.0%}")

    print("\n   Reading this: flat scores exactly 0 by construction. Churn pays ~20bps a")
    print("   day and is destroyed -- friction is doing its job. Dust trades trip the")
    print("   vol floor 100% of the time, which is what stops a lucky 1% position from")
    print("   posting a Sharpe of +30.")


if __name__ == "__main__":
    raise SystemExit(main())
