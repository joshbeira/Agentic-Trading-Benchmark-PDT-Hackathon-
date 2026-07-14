"""Scoreboard metrics — the canonical implementation of the schema's definition table.

Deliberately free of any dependency on the engine or on a Config object. Metrics are
part of the *specification*, not an engine internal: the engine writes them into a log,
and the analytics layer recomputes them from that log to check the engine's arithmetic.
Both call this. If it lived inside the engine, the scoreboard would be grading the
engine with the engine's own homework.

Every function here is a pure function of the two things a tick log records — the equity
series and the fills. That is what makes "any number on the scoreboard can be regenerated
from the logs" a testable claim rather than a slide bullet.

The headline is the **vol-floored Sharpe** (D2):

    sharpe = mean(r) / max( std(r), vol_floor_multiple * bh_daily_vol ) * sqrt(252)

Plain episode Sharpe is degenerate on a mostly-cash equity curve. An agent that puts 1%
of its capital to work for two lucky days has almost no return volatility and posts a
Sharpe around +30 — not because it traded well, but because the denominator collapsed.
The floor scales the denominator to the volatility the window actually offered
(`bh_daily_vol`, a frozen property of the window, identical for every agent), so
under-deploying capital cannot manufacture a ratio. An agent that never trades still
scores exactly 0, with no discontinuity there.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

DEFAULT_VOL_FLOOR_MULTIPLE = 0.25
DEFAULT_TRADING_DAYS = 252


def daily_returns(equity_cents: Sequence[int]) -> np.ndarray:
    e = np.asarray(equity_cents, dtype=np.float64)
    if len(e) < 2:
        return np.zeros(0)
    if np.any(e[:-1] <= 0):
        raise ValueError(
            "non-positive equity: the account was wiped out, which a long-only "
            "cash-plus-one-asset action space is supposed to make impossible"
        )
    return e[1:] / e[:-1] - 1.0


def sharpe(
    returns: np.ndarray,
    bh_daily_vol: float,
    *,
    floored: bool = True,
    vol_floor_multiple: float = DEFAULT_VOL_FLOOR_MULTIPLE,
    trading_days: int = DEFAULT_TRADING_DAYS,
) -> float:
    """Annualized Sharpe, rf = 0. Zero volatility scores 0, never NaN."""
    if len(returns) < 2:
        return 0.0
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    denom = max(std, vol_floor_multiple * bh_daily_vol) if floored else std
    if denom <= 0:
        return 0.0
    return mean / denom * math.sqrt(trading_days)


def max_drawdown(equity_cents: Sequence[int]) -> float:
    e = np.asarray(equity_cents, dtype=np.float64)
    if len(e) == 0:
        return 0.0
    return float(np.min(e / np.maximum.accumulate(e) - 1.0))


def compute(
    equity_cents: Sequence[int],
    fills: Iterable[dict],
    shares_by_tick: Sequence[float],
    bh_daily_vol: float,
    *,
    vol_floor_multiple: float = DEFAULT_VOL_FLOOR_MULTIPLE,
    trading_days: int = DEFAULT_TRADING_DAYS,
) -> dict:
    """The full trading scoreboard row for one episode."""
    equity = list(equity_cents)
    fills = list(fills)
    r = daily_returns(equity)

    std = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    e0 = equity[0]
    kw = {"vol_floor_multiple": vol_floor_multiple, "trading_days": trading_days}

    return {
        "total_return": equity[-1] / e0 - 1.0,
        "sharpe_floored": sharpe(r, bh_daily_vol, floored=True, **kw),
        "sharpe_raw": sharpe(r, bh_daily_vol, floored=False, **kw),
        "vol_floor_binding": bool(std < vol_floor_multiple * bh_daily_vol),
        "realized_vol_ann": std * math.sqrt(trading_days),
        "max_drawdown": max_drawdown(equity),
        "turnover": sum(f["gross_notional_cents"] for f in fills) / e0,
        "fees_paid_cents": int(sum(f["friction_cents"] for f in fills)),
        "time_in_market": float(np.mean([s > 0 for s in shares_by_tick])),
        "n_trades": sum(1 for f in fills if f["side"] != "liquidation"),
    }
