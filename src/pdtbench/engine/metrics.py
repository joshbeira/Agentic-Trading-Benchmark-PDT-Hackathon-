"""Scoreboard metrics.

Everything is a pure function of the equity series and the fills — the two things
the tick log records. That is not a stylistic choice: it is what makes "any number
on the scoreboard can be regenerated from the logs" a testable claim instead of a
slide bullet. The replay test recomputes this whole block from a log file and
demands bit-identical agreement with the values cached in it.

The headline is the **vol-floored Sharpe** (D2):

    sharpe = mean(r) / max( std(r), 0.25 * bh_daily_vol ) * sqrt(252)

Plain episode Sharpe is degenerate on a mostly-cash equity curve. An agent that
puts 1% of its capital to work for two lucky days has almost no return volatility
and posts a Sharpe around +30 — not because it traded well but because the
denominator collapsed. The floor scales the denominator to the volatility the
window actually offered (`bh_daily_vol`, a frozen property of the window,
identical for every agent), so under-deploying capital cannot manufacture a ratio.

The floor is deliberately *not* a minimum trade size. A size rule would still let
a 5%-of-equity lucky trade post a Sharpe of 8; this closes the exploit at the
metric, where it lives. An agent that never trades still scores exactly 0, and
there is no discontinuity at zero.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from ..config import Config


def daily_returns(equity_cents: Sequence[int]) -> np.ndarray:
    e = np.asarray(equity_cents, dtype=np.float64)
    if len(e) < 2:
        return np.zeros(0)
    if np.any(e[:-1] <= 0):
        raise ValueError("non-positive equity: the account was wiped out, which the "
                         "long-only + cash action space is supposed to make impossible")
    return e[1:] / e[:-1] - 1.0


def sharpe(returns: np.ndarray, bh_daily_vol: float, cfg: Config, floored: bool = True) -> float:
    """Annualized Sharpe, rf = 0. Zero volatility scores 0, never NaN."""
    if len(returns) < 2:
        return 0.0
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))

    if floored:
        denom = max(std, cfg.vol_floor_multiple * bh_daily_vol)
    else:
        denom = std

    if denom <= 0:
        return 0.0
    return mean / denom * math.sqrt(cfg.trading_days_per_year)


def max_drawdown(equity_cents: Sequence[int]) -> float:
    e = np.asarray(equity_cents, dtype=np.float64)
    if len(e) == 0:
        return 0.0
    peak = np.maximum.accumulate(e)
    return float(np.min(e / peak - 1.0))


def compute(
    equity_cents: Sequence[int],
    fills: Iterable[dict],
    shares_by_tick: Sequence[float],
    bh_daily_vol: float,
    cfg: Config,
) -> dict:
    """The full trading scoreboard row for one episode."""
    equity = list(equity_cents)
    fills = list(fills)
    r = daily_returns(equity)

    std = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    floor = cfg.vol_floor_multiple * bh_daily_vol

    e0 = equity[0]
    trades = [f for f in fills if f["side"] != "liquidation"]

    return {
        "total_return": equity[-1] / e0 - 1.0,
        "sharpe_floored": sharpe(r, bh_daily_vol, cfg, floored=True),
        "sharpe_raw": sharpe(r, bh_daily_vol, cfg, floored=False),
        "vol_floor_binding": bool(std < floor),
        "realized_vol_ann": std * math.sqrt(cfg.trading_days_per_year),
        "max_drawdown": max_drawdown(equity),
        "turnover": sum(f["gross_notional_cents"] for f in fills) / e0,
        "fees_paid_cents": int(sum(f["friction_cents"] for f in fills)),
        "time_in_market": float(np.mean([s > 0 for s in shares_by_tick])),
        "n_trades": len(trades),
    }
