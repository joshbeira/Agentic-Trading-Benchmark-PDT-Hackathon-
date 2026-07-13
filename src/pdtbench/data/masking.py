"""Anonymization — the one function both tracks pass through.

This is deliberately the *only* place a series is turned into what an agent sees,
and both the masked-real windows and the bootstrap twins go through it with the
same parameters. If real bars were masked here and synthetic bars were built
some other way, any difference in bar texture — decimal places, volume scaling,
the treatment of the first bar — would be a free tell for the leakage probe, and
the whole cross-track comparison (D9) would be measuring our formatting instead
of the model's memory.

What masking does:
  * strips dates (bars are addressed by integer index only)
  * rebases the price level so the first bar closes at 100
  * rounds prices to 2dp, then repairs OHLC bounds broken by that rounding
  * replaces volume with a multiple of its own trailing rolling average
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config

PRESENTED_COLUMNS = ["bar_idx", "open", "high", "low", "close", "volume"]


def mask_series(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Turn a raw OHLCV frame into the bars an agent is allowed to see.

    `raw` must hold exactly `cfg.n_bars` rows in chronological order, with
    columns open/high/low/close/volume. Returns a frame indexed by `bar_idx`,
    running from `-n_warmup` to `n_scored - 1`, so tick t and bar_idx t agree.
    """
    if len(raw) != cfg.n_bars:
        raise ValueError(f"expected {cfg.n_bars} bars, got {len(raw)}")

    price_cols = ["open", "high", "low", "close"]
    px = raw[price_cols].to_numpy(dtype=np.float64)
    vol = raw["volume"].to_numpy(dtype=np.float64)

    # Level is meaningless after anonymization; rebase so bar 0 closes at 100.
    px = px * (cfg.price_base / px[0, 3])
    px = np.round(px, cfg.price_decimals)

    # Rounding can push high below max(o,c) or low above min(o,c) by a cent.
    o, h, l, c = px[:, 0], px[:, 1], px[:, 2], px[:, 3]
    h = np.maximum.reduce([h, o, c, l])
    l = np.minimum.reduce([l, o, c, h])

    # Volume as a multiple of its own trailing average: carries the shape of a
    # volume spike without carrying the share count that would name the stock.
    v = pd.Series(vol)
    roll = v.rolling(cfg.volume_rolling_window, min_periods=1).mean()
    v_rel = np.round((v / roll).to_numpy(dtype=np.float64), cfg.volume_decimals)

    out = pd.DataFrame(
        {
            "bar_idx": np.arange(-cfg.n_warmup, cfg.n_scored, dtype=np.int64),
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v_rel,
        }
    )
    assert_presented_consistency(out)
    return out[PRESENTED_COLUMNS]


def assert_presented_consistency(df: pd.DataFrame) -> None:
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    if not np.all(h >= np.maximum(o, c) - 1e-9):
        raise AssertionError("high does not bound the bar")
    if not np.all(l <= np.minimum(o, c) + 1e-9):
        raise AssertionError("low does not floor the bar")
    if not np.all(h >= l - 1e-9):
        raise AssertionError("high below low")
    if not np.all(np.isfinite(df[["open", "high", "low", "close", "volume"]].to_numpy())):
        raise AssertionError("non-finite value in presented series")
    if not np.all(c > 0):
        raise AssertionError("non-positive close")


def scored_returns(df: pd.DataFrame) -> np.ndarray:
    """Close-to-close simple returns over the scored bars (89 of them).

    This is the series the vol floor is scaled against and the regime label is
    read from, so it is defined in exactly one place.
    """
    c = df.loc[df["bar_idx"] >= 0, "close"].to_numpy(dtype=np.float64)
    return c[1:] / c[:-1] - 1.0
