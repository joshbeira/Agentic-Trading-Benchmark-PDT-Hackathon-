#!/usr/bin/env python
"""Build the whole dataset: fetch -> validate -> cache -> windows -> twins.

Run once. Everything downstream reads the committed Parquet, so no run — and no
demo — ever touches the network.

    python scripts/build_dataset.py            # full build
    python scripts/build_dataset.py --no-fetch # rebuild windows from the cache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdtbench.config import DEFAULT, PROCESSED_DIR, WINDOWS_DIR, Config  # noqa: E402
from pdtbench.data import bootstrap, fetch, windows  # noqa: E402
from pdtbench.data.universe import UNIVERSE  # noqa: E402
from pdtbench.hashing import hash_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true", help="reuse the cached Parquet")
    args = ap.parse_args()
    cfg: Config = DEFAULT

    # 1. price data ------------------------------------------------------------
    if args.no_fetch and (PROCESSED_DIR / "prices.parquet").exists():
        print("[1/4] loading cached prices ...")
        prices, meta = fetch.load_dataset(PROCESSED_DIR)
    else:
        print(f"[1/4] fetching {len(UNIVERSE)} tickers from {cfg.data_start} ...")
        raw = fetch.fetch_raw(list(UNIVERSE), start=cfg.data_start, end=cfg.data_end)
        print(f"      {len(raw):,} raw bars")
        prices, report = fetch.validate_and_clean(raw, min_bars=cfg.n_bars + 50)
        meta = fetch.save_dataset(prices, cfg, PROCESSED_DIR, report)
        r = report.as_dict()
        print(
            f"      cleaned: {r['n_final']:,} bars, {r['tickers_kept']} tickers "
            f"({meta['date_min']} .. {meta['date_max']})"
        )
        print(
            f"      dropped: nan={r['dropped_nan']} nonpos={r['dropped_nonpositive']} "
            f"zerovol={r['dropped_zero_volume']} ohlc={r['dropped_ohlc_violation']} | "
            f"repaired={r['repaired_ohlc']} | flagged extreme moves={r['n_extreme_moves']}"
        )
        if r["tickers_dropped"]:
            print(f"      tickers dropped (too little history): {r['tickers_dropped']}")
    print(f"      dataset_sha256 = {meta['dataset_sha256'][:16]}...")

    # 2. candidate windows -----------------------------------------------------
    print("[2/4] scanning candidate windows ...")
    extreme = meta.get("validation", {}).get("extreme_moves", [])
    cands = windows.build_candidates(prices, cfg, extreme_moves=extreme)
    by_regime = cands["regime"].value_counts().to_dict()
    print(f"      {len(cands):,} candidates: {by_regime}")
    for regime in windows.REGIMES:
        pool = cands[cands["regime"] == regime]
        if pool["ticker"].nunique() < cfg.n_windows_per_regime:
            print(
                f"      [warn] only {pool['ticker'].nunique()} distinct tickers "
                f"offer a '{regime}' window"
            )

    selected = windows.select_windows(cands, cfg)

    # 3. materialize the real track -------------------------------------------
    print("[3/4] materializing 30 masked-real windows ...")
    real_dir, twin_dir = WINDOWS_DIR / "real", WINDOWS_DIR / "twin"
    real_dir.mkdir(parents=True, exist_ok=True)
    twin_dir.mkdir(parents=True, exist_ok=True)

    real_specs, twin_specs = [], []
    for cand in selected:
        presented, spec = windows.materialize(prices, cand, cfg)
        presented.to_parquet(real_dir / f"{spec.window_id}.parquet", index=False)
        real_specs.append(spec.as_dict())

        # 4. and its twin, resampled from the same raw bars --------------------
        raw = windows.cut_raw(prices, cand, cfg)
        twin_presented, seed, attempts = bootstrap.make_twin(
            raw, spec.window_id, spec.regime, cfg
        )
        twin_presented.to_parquet(twin_dir / f"{spec.window_id}.parquet", index=False)
        twin_specs.append(
            bootstrap.twin_spec(
                twin_presented, spec.window_id, spec.ticker, spec.regime, seed, attempts, cfg
            ).as_dict()
        )
        print(f"      {spec.window_id} {spec.ticker:<6} twin matched in {attempts:,} draws")

    print("[4/4] writing manifest ...")
    manifest = {
        "config_sha256": cfg.sha256(),
        "config": cfg.log_block(),
        "master_seed": cfg.master_seed,
        "dataset_sha256": meta["dataset_sha256"],
        "dataset_as_of": meta["as_of"],
        "n_windows": len(real_specs),
        "presentation_order": [s["window_id"] for s in real_specs],
        "real": real_specs,
        "twin": twin_specs,
    }
    manifest["manifest_sha256"] = hash_json(manifest)
    WINDOWS_DIR.mkdir(parents=True, exist_ok=True)
    (WINDOWS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    _summary(real_specs, twin_specs)
    print(f"\n  manifest_sha256 = {manifest['manifest_sha256'][:16]}...")
    print(f"  wrote {WINDOWS_DIR}")
    return 0


def _summary(real: list[dict], twin: list[dict]) -> None:
    print("\n  id   ticker  regime  window                     ret     vol |  twin ret    vol   draws")
    print("  " + "-" * 96)
    for r, t in zip(real, twin):
        print(
            f"  {r['window_id']}  {r['ticker']:<6}  {r['regime']:<6}  "
            f"{r['date_scored_start']}..{r['date_scored_end']}  "
            f"{r['total_return']:+7.1%}  {r['realized_vol_ann']:5.1%} | "
            f"{t['total_return']:+8.1%}  {t['realized_vol_ann']:5.1%}  {t['attempts']:>7,}"
        )

    rr = np.array([r["total_return"] for r in real])
    tr = np.array([t["total_return"] for t in twin])
    rv = np.array([r["realized_vol_ann"] for r in real])
    tv = np.array([t["realized_vol_ann"] for t in twin])
    draws = np.array([t["attempts"] for t in twin])

    print("  " + "-" * 96)
    print(
        f"  return  |  real median |ret| {np.median(np.abs(rr)):.1%}   "
        f"twin {np.median(np.abs(tr)):.1%}   "
        f"max paired gap {np.max(np.abs(rr - tr)):.2%}"
    )
    print(
        f"  vol     |  real median {np.median(rv):.1%}   twin {np.median(tv):.1%}   "
        f"max paired rel gap {np.max(np.abs(tv / rv - 1)):.1%}"
    )
    print(f"  regime  |  matched by construction in {len(real)}/{len(real)} windows")
    print(
        f"  draws   |  median {int(np.median(draws)):,}  max {int(np.max(draws)):,} "
        f"(rejection sampling to match realized moments)"
    )
    print(
        "\n  Each twin is matched to its source on realized return, volatility, and\n"
        "  regime — so a Sharpe gap across the pair cannot be explained by one path\n"
        "  having trended and the other not. Only path identity differs."
    )


if __name__ == "__main__":
    raise SystemExit(main())
