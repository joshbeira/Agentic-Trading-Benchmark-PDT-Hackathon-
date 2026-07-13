"""Data pipeline: masking, window hygiene, and the pinning hashes."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from pdtbench.data.fetch import load_dataset
from pdtbench.data.masking import assert_presented_consistency, scored_returns
from pdtbench.data.windows import REGIMES, classify, load_episode
from pdtbench.hashing import hash_frame

SERIES_COLS = ["bar_idx", "open", "high", "low", "close", "volume"]


def test_the_cached_dataset_still_hashes_to_what_it_claims(processed_dir):
    """load_dataset re-hashes the parquet and refuses a mismatch. If this fails, the
    committed data has drifted from the hash stamped into every episode log."""
    prices, meta = load_dataset(processed_dir)
    assert len(prices) == meta["n_rows"]
    assert prices["ticker"].nunique() == meta["n_tickers"]
    assert meta["auto_adjust"] is True


def test_thirty_windows_ten_per_regime(manifest):
    assert len(manifest["real"]) == 30
    assert Counter(s["regime"] for s in manifest["real"]) == {"bull": 10, "bear": 10, "chop": 10}


def test_window_hygiene(manifest):
    """One window per ticker, at most two per quarter. Distinct tickers are not
    distinct samples if they all cover March 2020 (D12)."""
    real = manifest["real"]

    tickers = [s["ticker"] for s in real]
    assert len(set(tickers)) == len(tickers), "a ticker was sampled twice"

    quarters = Counter(
        pd.Timestamp(s["date_scored_start"]).to_period("Q") for s in real
    )
    assert max(quarters.values()) <= 2, f"a quarter is over-sampled: {quarters.most_common(3)}"


def test_presentation_order_interleaves_regimes(manifest):
    """The learning curve must not be an artifact of seeing ten bulls in a row (D11)."""
    order = [
        next(s["regime"] for s in manifest["real"] if s["window_id"] == wid)
        for wid in manifest["presentation_order"]
    ]
    assert order[:6] == ["bull", "bear", "chop", "bull", "bear", "chop"]
    # no regime ever runs three deep
    assert not any(order[i] == order[i + 1] == order[i + 2] for i in range(len(order) - 2))


@pytest.mark.parametrize("track", ["real", "twin"])
def test_every_series_is_well_formed(windows_dir, manifest, cfg, track):
    for spec in manifest[track]:
        series, _ = load_episode(windows_dir, spec["window_id"], track)

        assert len(series) == cfg.n_bars == 290
        assert list(series["bar_idx"]) == list(range(-cfg.n_warmup, cfg.n_scored))
        assert_presented_consistency(series)

        # anonymized: base 100, 2dp, volume as a multiple of its own average
        assert series["close"].iloc[0] == pytest.approx(100.0)
        px = series[["open", "high", "low", "close"]].to_numpy()
        assert np.allclose(px, np.round(px, 2)), "prices leak more precision than a real quote"
        assert 0.5 < series["volume"].mean() < 2.0
        assert "date" not in series.columns and "ticker" not in series.columns


@pytest.mark.parametrize("track", ["real", "twin"])
def test_series_hashes_pin_the_data(windows_dir, manifest, track):
    """Every episode log stamps series_sha256. If a rebuild silently changed a bar,
    replay would still 'pass' against the new data unless this holds."""
    for spec in manifest[track]:
        series, _ = load_episode(windows_dir, spec["window_id"], track)
        assert hash_frame(series, SERIES_COLS) == spec["series_sha256"], spec["window_id"]


@pytest.mark.parametrize("track", ["real", "twin"])
def test_the_frozen_window_properties_are_the_ones_the_metric_will_use(
    windows_dir, manifest, cfg, track
):
    """bh_daily_vol is the vol-floor denominator (D2). It must be a property of the
    presented series, not of some upstream unrounded version of it."""
    for spec in manifest[track]:
        series, _ = load_episode(windows_dir, spec["window_id"], track)
        r = scored_returns(series)

        assert len(r) == 89
        assert spec["bh_daily_vol"] == pytest.approx(float(np.std(r, ddof=1)))
        assert spec["total_return"] == pytest.approx(float(np.prod(1 + r) - 1))
        assert classify(spec["total_return"], cfg) == spec["regime"]


def test_the_regime_guard_band_rejects_knife_edges(cfg):
    assert classify(0.20, cfg) == "bull"
    assert classify(-0.20, cfg) == "bear"
    assert classify(0.0, cfg) == "chop"
    assert classify(0.0801, cfg) is None, "a window this close to the line could flip on rounding"
    assert classify(-0.0799, cfg) is None
