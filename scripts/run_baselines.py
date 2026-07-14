#!/usr/bin/env python
"""Produce the baseline arm of a run: 4 baselines x 2 tracks x 30 windows = 240 episodes.

    python scripts/run_baselines.py                  # a fresh runs/{run_id}/
    python scripts/run_baselines.py --only flat sma_10_50
    python scripts/run_baselines.py --out runs/dev --validate

The baselines are the load-bearing part of the analysis, not a warm-up act. They cannot
learn, so their per-episode Sharpe *is* the window-difficulty signal, and an LLM's
learning curve is only ever measured as divergence from that trace (D11). They also run
through the same `EpisodeSession` an LLM does, so fee parity is structural (D14).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdtbench.config import DEFAULT, WINDOWS_DIR  # noqa: E402
from pdtbench.engine.replay import replay  # noqa: E402
from pdtbench.mcp import BASELINES, run_baselines  # noqa: E402
from pdtbench.schema import validate_episode  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None, help="run directory (default runs/{run_id})")
    ap.add_argument("--only", nargs="+", choices=list(BASELINES), default=None)
    ap.add_argument("--validate", action="store_true",
                    help="schema-validate and replay every episode written")
    args = ap.parse_args()

    print(f"baselines: {', '.join(args.only or BASELINES)}\n")
    run_dir = run_baselines(run_dir=args.out, only=args.only)

    if not args.validate:
        return 0

    logs = sorted((run_dir / "episodes").glob("*.jsonl"))
    bad_schema = [p.name for p in logs if not validate_episode(p).ok]
    bad_replay = [p.name for p in logs if not replay(p, WINDOWS_DIR).ok]

    print(f"\nvalidating {len(logs)} episodes")
    print(f"  schema : {len(logs) - len(bad_schema)}/{len(logs)} "
          f"{'PASS' if not bad_schema else 'FAIL ' + str(bad_schema[:3])}")
    print(f"  replay : {len(logs) - len(bad_replay)}/{len(logs)} "
          f"{'PASS' if not bad_replay else 'FAIL ' + str(bad_replay[:3])}")
    return 1 if (bad_schema or bad_replay) else 0


if __name__ == "__main__":
    raise SystemExit(main())
