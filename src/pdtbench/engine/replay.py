"""Replay: re-derive an episode from its log and check that it holds together.

This is where "every number on the scoreboard can be regenerated from the logs"
stops being a slide bullet. Replay reads a JSONL file and nothing else, rebuilds
the equity curve and the metrics from the recorded ticks and fills, re-prices
every fill against the pinned price series, and demands agreement to the cent.

If any of it disagrees, the episode is not admissible evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..config import Config
from ..hashing import hash_frame
from . import metrics

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


def replay(path: Path, cfg: Config, windows_dir: Path) -> ReplayResult:
    meta, ticks, end = load(path)
    res = ReplayResult(episode_id=meta.get("episode_id", path.stem), ok=True)

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
    res.require("tick_count", len(ticks) == cfg.n_scored, f"{len(ticks)} != {cfg.n_scored}")
    res.require(
        "ticks_ascending",
        [t["t"] for t in ticks] == list(range(cfg.n_scored)),
        "tick indices are not 0..n-1 in order",
    )

    # --- per-tick invariants -------------------------------------------------
    cash = meta["config"]["initial_capital_cents"]
    equity_series: list[int] = []
    shares_by_tick: list[float] = []
    fills: list[dict] = []
    cash_flow = 0
    friction_total = 0
    bad_equity: list[str] = []
    bad_fill: list[str] = []
    bad_time: list[str] = []

    for rec in ticks:
        t = rec["t"]
        p = rec["obs"]["portfolio"]

        # equity invariant: equity == cash + shares marked at this bar's close
        expect = p["cash_cents"] + round(p["shares"] * closes[t] * 100)
        if p["equity_cents"] != expect:
            bad_equity.append(f"t={t}: logged {p['equity_cents']} != {expect}")
        if rec["equity_cents"] != p["equity_cents"]:
            bad_equity.append(f"t={t}: top-level equity != obs.portfolio.equity")

        equity_series.append(rec["equity_cents"])
        shares_by_tick.append(p["shares"])

        # exactly one clock-moving call on a decision tick, none on a skipped one
        advanced = [c for c in rec["calls"] if c.get("advanced_time")]
        forced = bool(rec["action"] and rec["action"].get("forced"))
        if rec["decision_point"]:
            if len(advanced) != 1 and not forced:
                bad_time.append(f"t={t}: {len(advanced)} time-advancing calls, forced={forced}")
        elif advanced:
            bad_time.append(f"t={t}: skipped tick has a time-advancing call")

        # no look-ahead: a fill lands at the NEXT bar's open, at that exact price
        f = rec["fill"]
        if f:
            fills.append(f)
            cash_flow += f["cash_delta_cents"]
            friction_total += f["friction_cents"]
            if f["side"] == "liquidation":
                if f["fill_tick"] != t or abs(f["fill_price"] - closes[t]) > 1e-9:
                    bad_fill.append(f"t={t}: liquidation must price at close_{t}")
            else:
                if f["fill_tick"] != t + 1:
                    bad_fill.append(f"t={t}: fill_tick {f['fill_tick']} != {t + 1}")
                elif abs(f["fill_price"] - opens[t + 1]) > 1e-9:
                    bad_fill.append(
                        f"t={t}: filled at {f['fill_price']} but open_{t + 1} is {opens[t + 1]}"
                    )

    res.require("equity_invariant", not bad_equity, "; ".join(bad_equity[:3]))
    res.require("no_lookahead_fills", not bad_fill, "; ".join(bad_fill[:3]))
    res.require("time_advances_once", not bad_time, "; ".join(bad_time[:3]))

    # --- cash conservation ---------------------------------------------------
    final_cash = ticks[-1]["obs"]["portfolio"]["cash_cents"]
    res.require(
        "cash_conservation",
        final_cash == cash + cash_flow,
        f"final cash {final_cash} != initial {cash} + flows {cash_flow}",
    )
    res.require(
        "equity_series_matches_ticks",
        equity_series == end["equity_series_cents"],
        "episode_end.equity_series_cents disagrees with the per-tick marks",
    )

    # --- metrics regenerate --------------------------------------------------
    recomputed = metrics.compute(
        equity_series, fills, shares_by_tick, w["bh_daily_vol"], cfg
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
