"""Twin generator (D6, D9) — moment-matched stationary block bootstrap.

A twin is the control in the leakage experiment, so it has to hold constant
*everything a non-memorizing agent could exploit* and vary only the one thing
under test: whether the path is a real, potentially-memorized one.

Two mechanisms get us there.

**Resample whole bars, from the source window's own bars.** What is drawn is a
tuple `(O,H,L,C,V)` expressed relative to the previous close — not a close-to-
close return. Resampling returns would force us to fabricate open/high/low/volume,
producing bars with a different texture from the real track and handing the
leakage probe a free tell. Drawing from the source *window* (not the ticker's
whole history) is what keeps a bear window's twin as violent as the bear window.

**Then match the realized moments.** This is the part that a naive bootstrap gets
wrong, and it is worth being precise about, because it nearly sank the design.
Block resampling draws *with replacement*, so a twin's realized 90-bar path is a
fresh draw: same expected drift and volatility as the source, but a total return
whose standard deviation is roughly sigma*sqrt(90) — about +/-16% in practice. In
the first build, a +54% META bull window produced a -19% bear twin. Comparing an
agent's Sharpe across that pair measures which path happened to trend, not which
path the model remembers, and that is precisely the confound the twin exists to
eliminate.

So twins are accepted by rejection: a draw is kept only if its scored segment
realizes the same total return (within a few points), the same volatility (within
a relative band), and the same regime label as its source — and its warmup
segment matches too, since the agent sees those 200 bars and forms its priors on
them. A twin is therefore **not an unbiased draw from the source's DGP**, and it
is not meant to be. It is a counterfactual path matched on opportunity, differing
in identity. That is what a control is.

What survives the bootstrap: dependence within ~20 bars, so volatility clustering
lives and vol timing stays a real edge (D7). What does not: dependence beyond the
block length, so trend *continuation* is destroyed even though the realized drift
is matched. An agent cannot ride a twin's trend by predicting it; it can only take
exposure — the same thing it can do on the real window.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..config import Config
from ..hashing import hash_frame
from .masking import mask_series, scored_returns


@dataclass(frozen=True)
class TwinSpec:
    window_id: str  # same id as its source window: that is what pairs them
    source_ticker: str
    regime: str  # matched to the source by construction
    twin_seed: int
    attempts: int  # rejection-sampling draws needed; a diagnostic, and reproducible
    total_return: float
    realized_vol_ann: float
    bh_daily_vol: float
    series_sha256: str

    def as_dict(self) -> dict:
        return asdict(self)


class TwinMatchError(RuntimeError):
    """No draw satisfied the matching bands within the attempt budget."""


def twin_seed_for(master_seed: int, window_id: str) -> int:
    """A per-window seed derived from the master seed, so twins are independent of
    each other while the whole set is reproducible from one number."""
    digest = hashlib.sha256(f"{master_seed}:{window_id}:twin".encode()).hexdigest()
    return int(digest[:8], 16)


def bootstrap_index_batch(
    n_source: int, n_out: int, expected_block: int, rng: np.random.Generator, batch: int
) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices, `batch` paths at a time.

    Restart at a uniform draw with probability 1/L, otherwise step forward one bar,
    wrapping. Block lengths are geometric with mean L. Vectorized across the batch
    because rejection sampling needs thousands of draws per twin.
    """
    if n_source < 2:
        raise ValueError("need at least 2 source bars")
    p = 1.0 / max(expected_block, 1)

    restart = rng.random((batch, n_out)) < p
    restart[:, 0] = True
    starts = rng.integers(n_source, size=(batch, n_out))

    pos = np.arange(n_out)[None, :]
    last_restart = np.maximum.accumulate(np.where(restart, pos, 0), axis=1)
    offset = pos - last_restart
    base = np.take_along_axis(starts, last_restart, axis=1)
    return (base + offset) % n_source


def bar_relatives(raw: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Raw bars -> tuples of (O, H, L, C over the previous close, raw volume).

    Volume is carried through **raw**, not pre-normalized. That looks like a detail
    and is not. `mask_series` already divides volume by its own trailing average;
    normalizing here as well would put the twin's volume through that transform
    twice, which compresses spikes — the first build measured twin volume spikes at
    3.1x average against the real track's 3.7x, a difference a leakage probe could
    learn in one shot. Both tracks must reach `mask_series` as a *raw* OHLCV series
    and be normalized exactly once, by the same code.

    Drawing the whole bar as one tuple also preserves the joint structure: a violent
    day arrives with its volume spike attached.
    """
    o, h, l, c = (raw[k].to_numpy(dtype=np.float64) for k in ("open", "high", "low", "close"))
    v = raw["volume"].to_numpy(dtype=np.float64)
    prev_c = c[:-1]
    return np.column_stack(
        [o[1:] / prev_c, h[1:] / prev_c, l[1:] / prev_c, c[1:] / prev_c, v[1:]]
    )


def _segment_stats(close: np.ndarray, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    """Total return and daily vol of close[:, lo:hi] for a batch of paths."""
    seg = close[:, lo:hi]
    total = seg[:, -1] / seg[:, 0] - 1.0
    r = seg[:, 1:] / seg[:, :-1] - 1.0
    return total, np.std(r, axis=1, ddof=1)


def make_twin(
    source_raw: pd.DataFrame, window_id: str, source_regime: str, cfg: Config
) -> tuple[pd.DataFrame, int, int]:
    """Build one twin, matched to its source. Returns (presented, seed, attempts)."""
    from .windows import classify

    seed = twin_seed_for(cfg.master_seed, window_id)
    rng = np.random.default_rng(seed)

    rel = bar_relatives(source_raw, cfg)
    n_src = len(rel)

    # Targets are read off the source's own presented series, so twin and real are
    # compared on identically processed numbers.
    src_presented = mask_series(source_raw, cfg)
    src_close = src_presented["close"].to_numpy(dtype=np.float64)[None, :]
    (src_scored_ret,), (src_scored_vol,) = _segment_stats(src_close, cfg.n_warmup, cfg.n_bars)
    (src_warm_ret,), (src_warm_vol,) = _segment_stats(src_close, 0, cfg.n_warmup)

    attempts = 0
    while attempts < cfg.twin_max_attempts:
        batch = min(cfg.twin_batch, cfg.twin_max_attempts - attempts)
        idx = bootstrap_index_batch(n_src, cfg.n_bars, cfg.expected_block_length, rng, batch)
        c_rel = rel[:, 3][idx]  # (batch, n_bars)
        close = cfg.price_base * np.cumprod(c_rel, axis=1)

        scored_ret, scored_vol = _segment_stats(close, cfg.n_warmup, cfg.n_bars)
        warm_ret, warm_vol = _segment_stats(close, 0, cfg.n_warmup)

        ok = (
            (np.abs(scored_ret - src_scored_ret) <= cfg.twin_return_tol)
            & (np.abs(scored_vol / src_scored_vol - 1.0) <= cfg.twin_vol_rel_tol)
            & (np.abs(warm_ret - src_warm_ret) <= cfg.twin_warmup_return_tol)
            & (np.abs(warm_vol / src_warm_vol - 1.0) <= cfg.twin_warmup_vol_rel_tol)
        )

        for j in np.flatnonzero(ok):
            presented = _reconstruct(rel, idx[j], cfg)
            # The regime label is read off the masked series, which is the final
            # authority; a knife-edge draw that flips under rounding is rejected.
            r = scored_returns(presented)
            if classify(float(np.prod(1.0 + r) - 1.0), cfg) != source_regime:
                continue
            return presented, seed, attempts + int(j) + 1

        attempts += batch

    raise TwinMatchError(
        f"{window_id}: no matched twin in {cfg.twin_max_attempts} draws "
        f"(target ret={src_scored_ret:+.3f} vol={src_scored_vol:.4f})"
    )


def _reconstruct(rel: np.ndarray, idx: np.ndarray, cfg: Config) -> pd.DataFrame:
    """Chain one drawn index path back into bars, then mask it exactly as the real
    track is masked — same function, same parameters, no separate code path."""
    drawn = rel[idx]
    prev_close = cfg.price_base * np.concatenate([[1.0], np.cumprod(drawn[:, 3])[:-1]])
    raw = pd.DataFrame(
        {
            "open": prev_close * drawn[:, 0],
            "high": prev_close * drawn[:, 1],
            "low": prev_close * drawn[:, 2],
            "close": prev_close * drawn[:, 3],
            "volume": drawn[:, 4],
        }
    )
    return mask_series(raw, cfg)


def twin_spec(
    presented: pd.DataFrame,
    window_id: str,
    source_ticker: str,
    regime: str,
    seed: int,
    attempts: int,
    cfg: Config,
) -> TwinSpec:
    r = scored_returns(presented)
    bh_daily_vol = float(np.std(r, ddof=1))
    return TwinSpec(
        window_id=window_id,
        source_ticker=source_ticker,
        regime=regime,
        twin_seed=seed,
        attempts=attempts,
        total_return=float(np.prod(1.0 + r) - 1.0),
        realized_vol_ann=float(bh_daily_vol * np.sqrt(cfg.trading_days_per_year)),
        bh_daily_vol=bh_daily_vol,
        series_sha256=hash_frame(
            presented, ["bar_idx", "open", "high", "low", "close", "volume"]
        ),
    )


# --- diagnostics used by the twin-sanity tests -------------------------------


def variance_ratio(returns: np.ndarray, q: int) -> float:
    """VR(q) = Var(q-bar return) / (q * Var(1-bar return)).

    ~1 under a random walk. Trend persistence pushes it above 1, mean reversion
    below. The bootstrap should pull it toward 1 for q well beyond the block length
    while leaving short horizons alone — that is the "trend continuation is
    destroyed, short-range structure survives" claim, made measurable.
    """
    x = np.asarray(returns, dtype=np.float64)
    if len(x) < 2 * q:
        raise ValueError("series too short for this q")
    log_r = np.log1p(x)
    var1 = np.var(log_r, ddof=1)
    if var1 <= 0:
        return 1.0
    agg = np.convolve(log_r, np.ones(q), mode="valid")
    return float(np.var(agg, ddof=1) / (q * var1))


def abs_autocorr(returns: np.ndarray, lag: int) -> float:
    """Autocorrelation of |returns| — the standard volatility-clustering probe."""
    a = np.abs(np.asarray(returns, dtype=np.float64))
    if lag >= len(a) - 1:
        raise ValueError("lag too large")
    x, y = a[:-lag], a[lag:]
    xc, yc = x - x.mean(), y - y.mean()
    denom = np.sqrt((xc @ xc) * (yc @ yc))
    return float(xc @ yc / denom) if denom > 0 else 0.0
