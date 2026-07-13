"""Window selection (D12).

Candidate 290-bar windows are scanned across the whole universe, labelled by
regime, filtered for hygiene, and 30 are drawn — 10 bull, 10 bear, 10 chop.

Three hygiene rules, each there for a reason:

  * **one window per ticker.** Two windows on the same name are not two draws.
  * **at most two windows per calendar quarter.** Distinct tickers are not
    distinct samples if they all cover March 2020. Without this the bear bucket
    collapses onto two or three crises and the effective n is nowhere near 10.
  * **a guard band around the regime thresholds.** A window whose return sits at
    exactly +8.0% would be free to flip its own label under 2dp rounding, which
    would make the label depend on the masking step. Skip that neighbourhood.

Selection is round-robin across regimes rather than regime-by-regime, so no one
bucket eats the good tickers first. The selection order *is* the presentation
order (D11): regimes interleave, so a learning curve cannot be an artifact of
seeing all ten bull windows first.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..hashing import hash_frame
from .masking import mask_series, scored_returns

REGIMES = ("bull", "bear", "chop")
REGIME_GUARD_BAND = 0.005  # keep windows this far from a threshold


@dataclass(frozen=True)
class WindowSpec:
    window_id: str
    ticker: str
    regime: str
    start_idx: int  # index of the first warmup bar in the ticker's series
    date_warmup_start: str
    date_scored_start: str
    date_scored_end: str
    total_return: float
    realized_vol_ann: float
    bh_daily_vol: float
    series_sha256: str

    def as_dict(self) -> dict:
        return asdict(self)


def classify(total_return: float, cfg: Config) -> str | None:
    """Regime label, or None inside the guard band around a threshold."""
    bull, bear = cfg.regime_bull_threshold, cfg.regime_bear_threshold
    if abs(total_return - bull) < REGIME_GUARD_BAND:
        return None
    if abs(total_return - bear) < REGIME_GUARD_BAND:
        return None
    if total_return >= bull:
        return "bull"
    if total_return <= bear:
        return "bear"
    return "chop"


def build_candidates(
    prices: pd.DataFrame, cfg: Config, extreme_moves: list[dict] | None = None
) -> pd.DataFrame:
    """Every hygienic 290-bar window in the universe, labelled by regime."""
    bad_dates: dict[str, set[str]] = {}
    for m in extreme_moves or []:
        bad_dates.setdefault(m["ticker"], set()).add(m["date"])

    rows = []
    for ticker, g in prices.groupby("ticker", sort=True):
        g = g.sort_values("date", kind="mergesort").reset_index(drop=True)
        n = len(g)
        if n < cfg.n_bars:
            continue
        close = g["close"].to_numpy(dtype=np.float64)
        dates = g["date"].dt.strftime("%Y-%m-%d").to_numpy()
        flagged = bad_dates.get(ticker, set())

        for s in range(0, n - cfg.n_bars + 1, cfg.candidate_stride):
            e = s + cfg.n_bars - 1  # inclusive index of the last scored bar
            scored_start = s + cfg.n_warmup
            if flagged and flagged.intersection(dates[s : e + 1]):
                continue  # a suspect bar anywhere in the window disqualifies it

            c = close[scored_start : e + 1]
            total_return = float(c[-1] / c[0] - 1.0)
            regime = classify(total_return, cfg)
            if regime is None:
                continue

            r = c[1:] / c[:-1] - 1.0
            rows.append(
                {
                    "ticker": ticker,
                    "start_idx": s,
                    "regime": regime,
                    "total_return": total_return,
                    "daily_vol": float(np.std(r, ddof=1)),
                    "date_scored_start": dates[scored_start],
                    "quarter": pd.Timestamp(dates[scored_start]).to_period("Q").strftime("%YQ%q"),
                }
            )

    return pd.DataFrame(rows)


def select_windows(candidates: pd.DataFrame, cfg: Config) -> list[dict]:
    """Draw 10 windows per regime under the hygiene constraints, round-robin."""
    rng = np.random.default_rng(cfg.master_seed)
    pools: dict[str, list[dict]] = {}
    for regime in REGIMES:
        pool = candidates[candidates["regime"] == regime].to_dict("records")
        order = rng.permutation(len(pool))
        pools[regime] = [pool[i] for i in order]

    used_tickers: set[str] = set()
    quarters: Counter[str] = Counter()
    cursors = {r: 0 for r in REGIMES}
    selected: list[dict] = []
    quarter_cap = cfg.max_windows_per_quarter
    relaxations: list[str] = []

    def take(regime: str, cap: int) -> dict | None:
        pool = pools[regime]
        i = cursors[regime]
        while i < len(pool):
            cand = pool[i]
            i += 1
            if cand["ticker"] in used_tickers:
                continue
            if quarters[cand["quarter"]] >= cap:
                continue
            cursors[regime] = i
            return cand
        cursors[regime] = i
        return None

    for _round in range(cfg.n_windows_per_regime):
        for regime in REGIMES:
            cand = take(regime, quarter_cap)
            if cand is None:
                # A regime that clusters in time (bear especially) can exhaust the
                # quarter cap. Relax it rather than fail, but say so out loud.
                cursors[regime] = 0
                cand = take(regime, quarter_cap + 2)
                if cand is None:
                    raise RuntimeError(
                        f"cannot fill regime '{regime}': "
                        f"{sum(1 for s in selected if s['regime'] == regime)} of "
                        f"{cfg.n_windows_per_regime} selected, pool exhausted"
                    )
                relaxations.append(f"{regime}@{cand['quarter']}")
            used_tickers.add(cand["ticker"])
            quarters[cand["quarter"]] += 1
            selected.append(cand)

    if relaxations:
        print(f"  [warn] quarter cap relaxed for: {', '.join(relaxations)}")

    # Selection order is presentation order (D11): bull, bear, chop, bull, ...
    for i, s in enumerate(selected):
        s["window_id"] = f"w{i:02d}"
    return selected


def cut_raw(prices: pd.DataFrame, cand: dict, cfg: Config) -> pd.DataFrame:
    """The 290 raw bars behind a window. The twin generator resamples these, so
    both tracks descend from exactly the same source bars."""
    g = (
        prices[prices["ticker"] == cand["ticker"]]
        .sort_values("date", kind="mergesort")
        .reset_index(drop=True)
    )
    s = int(cand["start_idx"])
    return g.iloc[s : s + cfg.n_bars].reset_index(drop=True)


def materialize(
    prices: pd.DataFrame, cand: dict, cfg: Config
) -> tuple[pd.DataFrame, WindowSpec]:
    """Cut the 290 raw bars, mask them, and derive the window's frozen properties."""
    raw = cut_raw(prices, cand, cfg)
    s = int(cand["start_idx"])
    presented = mask_series(raw, cfg)

    # Recomputed on the *presented* series, because that is what the agent trades
    # and what the equity curve is scored against. Rounding is inside the numbers.
    r = scored_returns(presented)
    total_return = float(np.prod(1.0 + r) - 1.0)
    bh_daily_vol = float(np.std(r, ddof=1))
    regime = classify(total_return, cfg)
    if regime != cand["regime"]:
        raise RuntimeError(
            f"regime flipped under masking for {cand['ticker']}@{s}: "
            f"{cand['regime']} -> {regime} (ret={total_return:.4f})"
        )

    dates = raw["date"].dt.strftime("%Y-%m-%d").tolist()
    spec = WindowSpec(
        window_id=cand["window_id"],
        ticker=cand["ticker"],
        regime=regime,
        start_idx=s,
        date_warmup_start=dates[0],
        date_scored_start=dates[cfg.n_warmup],
        date_scored_end=dates[-1],
        total_return=total_return,
        realized_vol_ann=float(bh_daily_vol * np.sqrt(cfg.trading_days_per_year)),
        bh_daily_vol=bh_daily_vol,
        series_sha256=hash_frame(presented, ["bar_idx", "open", "high", "low", "close", "volume"]),
    )
    return presented, spec


def load_manifest(windows_dir: Path) -> dict:
    return json.loads((windows_dir / "manifest.json").read_text())


def load_episode(windows_dir: Path, window_id: str, track: str) -> tuple[pd.DataFrame, dict]:
    """The presented bars and frozen metadata for one (window, track) pair.

    `track` is "real" or "twin"; a twin carries its source window's id, which is
    exactly what pairs them in the leakage analysis (D9).
    """
    manifest = load_manifest(windows_dir)
    if track not in ("real", "twin"):
        raise ValueError(f"track must be 'real' or 'twin', got {track!r}")
    specs = {s["window_id"]: s for s in manifest[track]}
    if window_id not in specs:
        raise KeyError(f"no window {window_id!r} on track {track!r}")
    series = pd.read_parquet(windows_dir / track / f"{window_id}.parquet")
    return series, specs[window_id]
