"""Twin sanity — the claims the leakage experiment rests on.

These are not generic sanity checks. Each one is a load-bearing sentence from the
design, turned into something that can fail:

  D9  a twin is matched to its source on everything a non-memorizing agent could
      exploit, and differs only in path identity
  D9  no surface feature separates the tracks, so the probe measures the model's
      memory rather than our formatting
  D7  volatility clustering survives the bootstrap, which is what makes vol timing
      a real edge on the synthetic track
  D6  fat tails and bar texture survive
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pdtbench.data.bootstrap import (
    abs_autocorr,
    bootstrap_index_batch,
    make_twin,
    twin_seed_for,
)
from pdtbench.data.fetch import load_dataset
from pdtbench.data.windows import cut_raw, load_episode
from pdtbench.hashing import hash_frame

SERIES_COLS = ["bar_idx", "open", "high", "low", "close", "volume"]


@pytest.fixture(scope="module")
def pairs(windows_dir, manifest):
    """Every (real, twin) pair, loaded once."""
    out = []
    for r in manifest["real"]:
        t = next(x for x in manifest["twin"] if x["window_id"] == r["window_id"])
        real, _ = load_episode(windows_dir, r["window_id"], "real")
        twin, _ = load_episode(windows_dir, r["window_id"], "twin")
        out.append((r, t, real, twin))
    return out


def rets(df: pd.DataFrame) -> np.ndarray:
    c = df["close"].to_numpy(float)
    return c[1:] / c[:-1] - 1.0


# ============================================== the twin is a control (D9)


def test_every_real_window_has_exactly_one_twin(manifest):
    real_ids = [s["window_id"] for s in manifest["real"]]
    twin_ids = [s["window_id"] for s in manifest["twin"]]
    assert real_ids == twin_ids, "the shared window_id IS the pairing"


def test_a_twin_realizes_its_source_s_return_and_volatility(pairs, cfg):
    """The bug that nearly sank the design: a block bootstrap draws *with
    replacement*, so an unmatched twin's realized 90-bar path is a fresh draw with
    a standard deviation of ~16% on total return. META's +54% bull window produced
    a -19% bear twin. Sharpe_real - Sharpe_twin would then have measured which path
    happened to trend, not which path the model remembers."""
    for r, t, _, _ in pairs:
        gap = abs(t["total_return"] - r["total_return"])
        vol_gap = abs(t["bh_daily_vol"] / r["bh_daily_vol"] - 1)

        assert gap <= cfg.twin_return_tol + 1e-9, (
            f"{r['window_id']} ({r['ticker']}): real {r['total_return']:+.1%} vs "
            f"twin {t['total_return']:+.1%}"
        )
        assert vol_gap <= cfg.twin_vol_rel_tol + 1e-9, f"{r['window_id']}: vol gap {vol_gap:.1%}"
        assert t["regime"] == r["regime"]


def test_a_twin_is_indistinguishable_from_its_source_on_marginals(pairs):
    """KS over the full 289 returns. Non-rejection is the point: if a twin's return
    distribution were distinguishable, the two tracks would differ in opportunity
    and the paired comparison would be measuring that instead of memory."""
    pvals = np.array([stats.ks_2samp(rets(real), rets(twin)).pvalue for _, _, real, twin in pairs])
    assert (pvals > 0.05).all(), (
        f"{(pvals <= 0.05).sum()}/30 twins are distinguishable from their source "
        f"(min p = {pvals.min():.3f})"
    )


def test_no_surface_feature_separates_the_tracks(pairs):
    """A leakage probe must have nothing to grab onto but the path itself.

    The first build failed exactly here: twin volume was normalized twice and real
    volume once, which compressed twin volume spikes to 3.1x average against the
    real track's 3.7x — at p < 0.001. A model could learn "big volume spike => real
    series" and score above chance without remembering anything at all.
    """

    def surface(df):
        o, h, l, c, v = (df[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
        r = rets(df)
        return {
            "bar_range": np.mean((h - l) / c),
            "overnight_gap": np.mean(np.abs(o[1:] / c[:-1] - 1)),
            "volume_mean": v.mean(),
            "volume_std": v.std(),
            "volume_max": v.max(),
            "kurtosis": stats.kurtosis(r),
            "skew": stats.skew(r),
        }

    real = pd.DataFrame([surface(d) for _, _, d, _ in pairs])
    twin = pd.DataFrame([surface(d) for _, _, _, d in pairs])

    offenders = {
        col: p
        for col in real.columns
        if (p := stats.ttest_rel(real[col], twin[col]).pvalue) < 0.01
    }
    assert not offenders, f"these features give the track away: {offenders}"


# ================================ what survives the bootstrap, and what does not


def test_volatility_clustering_survives(pairs):
    """D7's whole edge. Direction is unpredictable on the synthetic track, but risk
    is not — and a Sharpe-based metric rewards exactly that. If clustering did not
    survive the resampling, the synthetic track would offer no ex-ante edge at all
    and the benchmark would be measuring luck."""
    lags = range(1, 6)
    real = np.array([[abs_autocorr(rets(d), k) for k in lags] for _, _, d, _ in pairs])
    twin = np.array([[abs_autocorr(rets(d), k) for k in lags] for _, _, _, d in pairs])

    assert twin.mean() > 0.03, f"clustering did not survive: mean |r| autocorr {twin.mean():.3f}"
    assert (twin.mean(axis=1) > 0).sum() >= 22, "clustering vanished in too many twins"
    # attenuated, and we say so out loud rather than pretending otherwise
    assert twin.mean() < real.mean(), "expected the bootstrap to attenuate clustering"


def test_fat_tails_survive(pairs):
    """Gaussian noise would make the environment a toy. Excess kurtosis has to live."""
    kurt = np.array([stats.kurtosis(rets(d)) for _, _, _, d in pairs])
    assert np.median(kurt) > 1.0, f"twin returns are too Gaussian: median kurtosis {np.median(kurt)}"


# ================================================== reproducibility of the twins


def test_a_twin_rebuilds_bit_for_bit(windows_dir, processed_dir, manifest, cfg):
    """Seeded from the master seed and the window id, so the whole synthetic track is
    reproducible from one number in the config."""
    prices, _ = load_dataset(processed_dir)
    spec = manifest["real"][0]
    twin_spec = manifest["twin"][0]

    raw = cut_raw(prices, {"ticker": spec["ticker"], "start_idx": spec["start_idx"]}, cfg)
    rebuilt, seed, attempts = make_twin(raw, spec["window_id"], spec["regime"], cfg)

    assert seed == twin_seed_for(cfg.master_seed, spec["window_id"]) == twin_spec["twin_seed"]
    assert attempts == twin_spec["attempts"]
    assert hash_frame(rebuilt, SERIES_COLS) == twin_spec["series_sha256"]


def test_the_resampler_produces_geometric_blocks(cfg):
    """Politis-Romano: a new block starts with probability 1/L, so the mean block
    length is L and dependence beyond it is destroyed."""
    rng = np.random.default_rng(0)
    idx = bootstrap_index_batch(289, 290, cfg.expected_block_length, rng, batch=200)

    # a "break" is any position that is not one step on from its predecessor
    steps = (idx[:, 1:] - idx[:, :-1]) % 289
    breaks = (steps != 1).mean()
    assert breaks == pytest.approx(1 / cfg.expected_block_length, abs=0.01)
    assert idx.min() >= 0 and idx.max() < 289
