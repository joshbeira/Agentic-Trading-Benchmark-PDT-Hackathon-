"""Fetch, validate, clean, and cache the price data.

Fetched once from Yahoo with `auto_adjust=True` so splits and dividends never
show up as artificial price jumps, then validated for OHLC consistency, cleaned,
written to Parquet, and pinned by a content hash. Everything downstream — window
selection, twins, every episode — runs off the cached Parquet. There is no live
API dependency at run time, and none on stage.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..hashing import hash_frame, hash_json

COLUMNS = ["ticker", "date", "open", "high", "low", "close", "volume"]
HASH_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "volume"]

# Adjusted OHLC from Yahoo can violate high >= max(open, close) by a hair, purely
# from adjustment rounding. Below this relative tolerance we clip the bar back
# into consistency; above it we treat the bar as corrupt and drop it.
REPAIR_TOL = 1e-4


@dataclass
class ValidationReport:
    n_raw: int = 0
    n_final: int = 0
    dropped_nan: int = 0
    dropped_nonpositive: int = 0
    dropped_zero_volume: int = 0
    dropped_ohlc_violation: int = 0
    repaired_ohlc: int = 0
    tickers_requested: int = 0
    tickers_kept: int = 0
    tickers_dropped: list[str] = field(default_factory=list)
    extreme_moves: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "n_raw": self.n_raw,
            "n_final": self.n_final,
            "dropped_nan": self.dropped_nan,
            "dropped_nonpositive": self.dropped_nonpositive,
            "dropped_zero_volume": self.dropped_zero_volume,
            "dropped_ohlc_violation": self.dropped_ohlc_violation,
            "repaired_ohlc": self.repaired_ohlc,
            "tickers_requested": self.tickers_requested,
            "tickers_kept": self.tickers_kept,
            "tickers_dropped": self.tickers_dropped,
            "n_extreme_moves": len(self.extreme_moves),
            "extreme_moves": self.extreme_moves[:20],
        }


def fetch_raw(
    tickers: list[str],
    start: str,
    end: str | None = None,
    retries: int = 3,
) -> pd.DataFrame:
    """Download daily bars for `tickers` and return them in long format.

    One batched call, retried with backoff. `auto_adjust=True` is yfinance's
    default now but is passed explicitly — it is a load-bearing assumption, not
    a default we want to inherit silently.
    """
    import yfinance as yf

    last_err: Exception | None = None
    raw = None
    for attempt in range(retries):
        try:
            raw = yf.download(
                list(tickers),
                start=start,
                end=end,
                interval="1d",
                auto_adjust=True,
                actions=False,
                group_by="ticker",
                threads=True,
                progress=False,
                timeout=30,
            )
            if raw is not None and not raw.empty:
                break
        except Exception as exc:  # noqa: BLE001 - network flake, retry
            last_err = exc
        time.sleep(2 * (attempt + 1))

    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data (last error: {last_err})")

    frames = []
    for ticker in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker not in raw.columns.get_level_values(0):
                continue
            sub = raw[ticker].copy()
        else:  # single-ticker download
            sub = raw.copy()
        sub = sub.rename(columns=str.lower)
        missing = {"open", "high", "low", "close", "volume"} - set(sub.columns)
        if missing:
            continue
        sub = sub[["open", "high", "low", "close", "volume"]]
        sub.insert(0, "ticker", ticker)
        sub = sub.reset_index().rename(columns={"Date": "date", "index": "date"})
        frames.append(sub)

    if not frames:
        raise RuntimeError("no ticker produced usable columns")

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df[COLUMNS]


def validate_and_clean(
    df: pd.DataFrame, min_bars: int, extreme_move: float = 0.5
) -> tuple[pd.DataFrame, ValidationReport]:
    """Enforce the invariants a bar has to satisfy before we are willing to trade it."""
    rep = ValidationReport()
    rep.n_raw = len(df)
    rep.tickers_requested = df["ticker"].nunique()

    df = df.copy()
    price_cols = ["open", "high", "low", "close"]

    before = len(df)
    df = df.dropna(subset=price_cols + ["volume"])
    rep.dropped_nan = before - len(df)

    before = len(df)
    df = df[(df[price_cols] > 0).all(axis=1)]
    rep.dropped_nonpositive = before - len(df)

    before = len(df)
    df = df[df["volume"] > 0]
    rep.dropped_zero_volume = before - len(df)

    # OHLC consistency: high must bound the bar, low must floor it.
    hi_should = df[price_cols].max(axis=1)
    lo_should = df[price_cols].min(axis=1)
    hi_err = (hi_should - df["high"]) / df["high"]
    lo_err = (df["low"] - lo_should) / df["low"]

    corrupt = (hi_err > REPAIR_TOL) | (lo_err > REPAIR_TOL)
    rep.dropped_ohlc_violation = int(corrupt.sum())
    df = df[~corrupt]

    hi_should = df[price_cols].max(axis=1)
    lo_should = df[price_cols].min(axis=1)
    needs_repair = (df["high"] < hi_should) | (df["low"] > lo_should)
    rep.repaired_ohlc = int(needs_repair.sum())
    df.loc[:, "high"] = np.maximum(df["high"].to_numpy(), hi_should.to_numpy())
    df.loc[:, "low"] = np.minimum(df["low"].to_numpy(), lo_should.to_numpy())

    df = df.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)

    # Flag implausible single-day moves. We do NOT drop them: dropping a row
    # would splice two non-adjacent bars together and fabricate a gap. Instead
    # they are recorded, and the window sampler refuses any window containing one.
    rets = df.groupby("ticker", sort=False)["close"].pct_change()
    flagged = df.loc[rets.abs() > extreme_move, ["ticker", "date", "close"]]
    rep.extreme_moves = [
        {"ticker": r.ticker, "date": r.date.date().isoformat(), "close": float(r.close)}
        for r in flagged.itertuples()
    ]

    counts = df.groupby("ticker", sort=False)["date"].count()
    keep = counts[counts >= min_bars].index
    rep.tickers_dropped = sorted(set(counts.index) - set(keep))
    df = df[df["ticker"].isin(keep)].reset_index(drop=True)

    rep.n_final = len(df)
    rep.tickers_kept = df["ticker"].nunique()
    return df, rep


def save_dataset(df: pd.DataFrame, cfg: Config, out_dir: Path, report: ValidationReport) -> dict:
    """Write Parquet + a sidecar meta with the content hash the tick log pins."""
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "prices.parquet"

    df = df.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)
    hashable = df.assign(date=df["date"].dt.strftime("%Y-%m-%d"))
    dataset_sha256 = hash_frame(hashable, HASH_COLUMNS)

    df.to_parquet(parquet_path, index=False)

    meta = {
        "dataset_sha256": dataset_sha256,
        "parquet_path": str(parquet_path.relative_to(cfg_repo_root(cfg))),
        "as_of": pd.Timestamp.utcnow().date().isoformat(),
        "start": cfg.data_start,
        "end": cfg.data_end,
        "auto_adjust": True,
        "source": "yfinance",
        "n_rows": int(len(df)),
        "n_tickers": int(df["ticker"].nunique()),
        "tickers": sorted(df["ticker"].unique().tolist()),
        "date_min": df["date"].min().date().isoformat(),
        "date_max": df["date"].max().date().isoformat(),
        "validation": report.as_dict(),
    }
    meta["meta_sha256"] = hash_json({k: v for k, v in meta.items() if k != "validation"})
    (out_dir / "prices.meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def cfg_repo_root(cfg: Config) -> Path:
    from ..config import REPO_ROOT

    return REPO_ROOT


def load_dataset(out_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Load the cached Parquet and verify it still hashes to what the meta claims."""
    df = pd.read_parquet(out_dir / "prices.parquet")
    meta = json.loads((out_dir / "prices.meta.json").read_text())
    hashable = df.assign(date=pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"))
    got = hash_frame(hashable, HASH_COLUMNS)
    if got != meta["dataset_sha256"]:
        raise RuntimeError(
            f"dataset hash mismatch: parquet={got} meta={meta['dataset_sha256']}"
        )
    return df, meta
