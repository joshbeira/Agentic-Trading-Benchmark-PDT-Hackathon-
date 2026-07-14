"""Engine-side view of the metrics.

The implementation lives in `pdtbench.metrics`, which knows nothing about the engine.
Metrics are part of the specification — the engine writes them into a log and the
analytics layer recomputes them from that log to check the engine's arithmetic. If the
implementation lived here, the scoreboard would be grading the engine with the engine's
own homework.

This module is the thin adapter that supplies the parameters from a `Config`.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from ..config import Config
from ..metrics import daily_returns, max_drawdown
from ..metrics import compute as _compute
from ..metrics import sharpe as _sharpe

__all__ = ["compute", "sharpe", "daily_returns", "max_drawdown"]


def sharpe(returns, bh_daily_vol: float, cfg: Config, floored: bool = True) -> float:
    return _sharpe(
        returns,
        bh_daily_vol,
        floored=floored,
        vol_floor_multiple=cfg.vol_floor_multiple,
        trading_days=cfg.trading_days_per_year,
    )


def compute(
    equity_cents: Sequence[int],
    fills: Iterable[dict],
    shares_by_tick: Sequence[float],
    bh_daily_vol: float,
    cfg: Config,
) -> dict:
    return _compute(
        equity_cents,
        fills,
        shares_by_tick,
        bh_daily_vol,
        vol_floor_multiple=cfg.vol_floor_multiple,
        trading_days=cfg.trading_days_per_year,
    )
