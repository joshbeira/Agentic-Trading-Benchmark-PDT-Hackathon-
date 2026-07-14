"""Replay: re-derive an episode from its log and check that it holds together.

This is where "every number on the scoreboard can be regenerated from the logs"
stops being a slide bullet. Replay reads a JSONL file and the pinned price series,
and nothing else: it rebuilds cash, shares, the equity curve and every metric from
the recorded ticks and fills, re-prices every fill, and demands agreement to the cent.

If any of it disagrees, the episode is not admissible evidence.

**There is deliberately no `Config` parameter.** Accepting one was a real bug: the
metric block was recomputed with whatever `vol_floor_multiple` and
`trading_days_per_year` the *caller* happened to hold, not the ones stamped in the
episode's own `config` block. It passed only because every call site handed it the
same defaults the logs were written under — so on the day someone changed a default,
every archived episode would have failed replay, and on an episode where the vol floor
binds it recomputed a Sharpe off by 2x. Both free parameters of the ranking metric are
pinned in the log precisely so that a verifier reads them from there. Removing the
parameter is what makes reintroducing that bug impossible rather than merely unlikely.

Likewise, cash and shares are rebuilt **from the fill records** rather than read back
out of `obs.portfolio`. A verifier that trusts the numbers it is checking is not a
verifier: deriving cash from the equity it is supposed to explain is exactly how the
old `verify_engine.py` equity check degenerated into `eq != eq`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..hashing import hash_frame
from ..metrics import compute as compute_metrics
from ..schema import SHARE_DECIMALS

SERIES_COLS = ["bar_idx", "open", "high", "low", "close", "volume"]


@dataclass
class ReplayResult:
    episode_id: str
    ok: bool
    checks: dict[str, bool] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    recomputed: dict = field(default_factory=dict)
    cached: dict = field(default_factory=dict)

    def require(self, name: str, cond: bool, detail: str = "") -> None:
        self.checks[name] = bool(cond)
        if not cond:
            self.ok = False
            self.failures.append(f"{name}: {detail}" if detail else name)


def load(path: Path) -> tuple[dict, list[dict], dict]:
    recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not recs or recs[0].get("type") != "meta":
        raise ValueError("first record must be `meta`")
    if recs[-1].get("type") != "episode_end":
        raise ValueError("last record must be `episode_end`")
    ticks = [r for r in recs[1:-1] if r.get("type") == "tick"]
    return recs[0], ticks, recs[-1]


def _shares_after(shares: float, f: dict) -> float:
    """Apply one fill to a share count, mirroring the engine's own arithmetic.

    Rounding to `SHARE_DECIMALS` and clamping at zero is not cosmetic. Summing
    `shares_delta` naively leaves float dust (~3.5e-15) after a position is fully
    sold, so `shares > 0` stays true and `time_in_market` counts a flat tick as
    invested — which is how a fills-only reconstruction lands at 0.9889, above the
    88/90 = 0.978 ceiling the format documents as unreachable.
    """
    if f["side"] == "buy":
        return round(shares + f["shares_delta"], SHARE_DECIMALS)
    out = round(max(shares + f["shares_delta"], 0.0), SHARE_DECIMALS)
    return 0.0 if out <= 0 else out


def replay(path: Path, windows_dir: Path) -> ReplayResult:
    meta, ticks, end = load(path)
    res = ReplayResult(episode_id=meta.get("episode_id", path.stem), ok=True)

    # --- every parameter comes from the log, never from a default ------------
    cfg = meta["config"]
    n_scored = cfg["n_scored"]
    initial_capital = cfg["initial_capital_cents"]
    vol_floor_multiple = cfg["vol_floor_multiple"]
    trading_days = cfg["trading_days_per_year"]

    # --- the series the episode claims to have traded ------------------------
    w = meta["window"]
    series = pd.read_parquet(windows_dir / meta["track"] / f"{w['window_id']}.parquet")
    res.require(
        "series_hash_matches_log",
        hash_frame(series, SERIES_COLS) == w["series_sha256"],
        "the log was written against a different price series than the one on disk",
    )
    opens = dict(zip(series["bar_idx"], series["open"]))
    closes = dict(zip(series["bar_idx"], series["close"]))

    # --- structure -----------------------------------------------------------
    res.require("tick_count", len(ticks) == n_scored, f"{len(ticks)} != {n_scored}")
    res.require(
        "ticks_ascending",
        [t["t"] for t in ticks] == list(range(len(ticks))),
        "tick indices are not 0..n-1 in order",
    )
    if len(ticks) != n_scored:
        return res  # everything below indexes by tick; don't cascade noise

    last = n_scored - 1

    # A fill decided at tick t *lands* at fill_tick (t+1, or t for the terminal
    # liquidation), so index the fills by where they land, not where they were chosen.
    landing: dict[int, list[dict]] = {}
    for rec in ticks:
        f = rec.get("fill")
        if f:
            landing.setdefault(f["fill_tick"], []).append(f)

    cash = initial_capital
    shares = 0.0
    equity_series: list[int] = []
    shares_by_tick: list[float] = []
    fills: list[dict] = []
    friction_total = 0
    bad_equity: list[str] = []
    bad_fill: list[str] = []
    bad_time: list[str] = []

    for rec in ticks:
        t = rec["t"]

        # Apply whatever landed at this bar's open (or, at t == last, the liquidation
        # at this bar's close) *before* marking: that is the engine's own ordering.
        for f in landing.get(t, []):
            cash += f["cash_delta_cents"]
            shares = _shares_after(shares, f)
            friction_total += f["friction_cents"]
            fills.append(f)

        # --- equity, rebuilt from the fills alone ---------------------------
        expect = cash + round(shares * closes[t] * 100)
        if rec["equity_cents"] != expect:
            bad_equity.append(f"t={t}: logged {rec['equity_cents']} != rebuilt {expect}")

        p = rec["obs"]["portfolio"]
        if p["equity_cents"] != rec["equity_cents"]:
            bad_equity.append(f"t={t}: top-level equity != obs.portfolio.equity")
        if p["cash_cents"] != cash:
            bad_equity.append(f"t={t}: obs cash {p['cash_cents']} != rebuilt {cash}")
        if abs(p["shares"] - shares) > 10 ** -SHARE_DECIMALS / 2:
            bad_equity.append(f"t={t}: obs shares {p['shares']} != rebuilt {shares}")
        if p["cash_cents"] + p["position_value_cents"] != p["equity_cents"]:
            bad_equity.append(f"t={t}: equity != cash + position_value")

        equity_series.append(rec["equity_cents"])
        shares_by_tick.append(shares)

        # --- the clock moves exactly once, and only where it may -------------
        advanced = [c for c in rec["calls"] if c.get("advanced_time")]
        action = rec["action"]
        forced = bool(action and action.get("forced"))
        if rec["decision_point"]:
            if action is None:
                bad_time.append(f"t={t}: a decision tick must record an action")
            elif forced:
                # The engine took the turn away; the agent never made a call to credit.
                if advanced:
                    bad_time.append(f"t={t}: a forced action must have no advancing call")
                if action.get("tool") != "Wait":
                    bad_time.append(f"t={t}: a forced action must be a Wait")
            elif len(advanced) != 1:
                bad_time.append(f"t={t}: {len(advanced)} time-advancing calls (expected 1)")
            elif (
                action.get("tool") != advanced[0].get("tool")
                or action.get("args") != advanced[0].get("args")
            ):
                bad_time.append(f"t={t}: action is not the call that advanced the clock")
        else:
            if advanced:
                bad_time.append(f"t={t}: a non-decision tick has a time-advancing call")
            if action is not None:
                bad_time.append(f"t={t}: a non-decision tick must not record an action")

        # --- no look-ahead: a fill lands at the NEXT bar's open, at that price -
        f = rec.get("fill")
        if f:
            if f["side"] == "liquidation":
                if f["fill_tick"] != t or t != last:
                    bad_fill.append(f"t={t}: a liquidation may only land on tick {last}")
                elif abs(f["fill_price"] - closes[t]) > 1e-9:
                    bad_fill.append(f"t={t}: liquidation must price at close_{t}")
            elif f["fill_tick"] != t + 1:
                bad_fill.append(f"t={t}: fill_tick {f['fill_tick']} != {t + 1}")
            elif abs(f["fill_price"] - opens[t + 1]) > 1e-9:
                bad_fill.append(
                    f"t={t}: filled at {f['fill_price']} but open_{t + 1} is {opens[t + 1]}"
                )

    res.require("equity_invariant", not bad_equity, "; ".join(bad_equity[:3]))
    res.require("no_lookahead_fills", not bad_fill, "; ".join(bad_fill[:3]))
    res.require("time_advances_once", not bad_time, "; ".join(bad_time[:3]))

    # --- cash conservation ---------------------------------------------------
    flows = sum(f["cash_delta_cents"] for f in fills)
    res.require(
        "cash_conservation",
        ticks[-1]["obs"]["portfolio"]["cash_cents"] == initial_capital + flows,
        f"final cash {ticks[-1]['obs']['portfolio']['cash_cents']} != "
        f"initial {initial_capital} + flows {flows}",
    )
    res.require(
        "equity_series_matches_ticks",
        equity_series == end["equity_series_cents"],
        "episode_end.equity_series_cents disagrees with the per-tick marks",
    )

    # --- metrics regenerate, under THIS episode's parameters ------------------
    recomputed = compute_metrics(
        equity_series,
        fills,
        shares_by_tick,
        w["bh_daily_vol"],
        vol_floor_multiple=vol_floor_multiple,
        trading_days=trading_days,
    )
    cached = end["metrics"]
    res.recomputed, res.cached = recomputed, cached

    drift = {
        k: (cached.get(k), v)
        for k, v in recomputed.items()
        if not _close(cached.get(k), v)
    }
    res.require(
        "metrics_regenerate",
        not drift,
        "; ".join(f"{k}: cached={a!r} recomputed={b!r}" for k, (a, b) in list(drift.items())[:4]),
    )
    res.require(
        "fees_match_fills",
        cached.get("fees_paid_cents") == friction_total,
        f"cached fees {cached.get('fees_paid_cents')} != summed friction {friction_total}",
    )
    return res


def _close(a, b, tol: float = 1e-9) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    return a == b
