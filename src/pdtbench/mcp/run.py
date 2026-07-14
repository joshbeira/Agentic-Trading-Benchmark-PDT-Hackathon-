"""Assemble a run: the manifest, then the episodes.

Layout is fixed by `schemas/tick_log.md` and `schemas/analysis_artifacts.md`:

    runs/{run_id}/
      run_manifest.json
      episodes/{agent_id}__{track}__{window_id}.jsonl
      probe/{agent_id}__{track}__{window_id}.json

`run_manifest.json` is written **before the first episode**, so a run that dies halfway
is still interpretable rather than a directory of orphan logs.

`episode_index` is the index within the `(agent, track)` **memory lane**, 0..29 — the
x-axis of the learning curve. Every agent, baseline and LLM alike, walks the same
`presentation_order`, which is what makes index *i* the same window for everyone and the
baseline difficulty trace subtractable (D11).
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path

from ..config import DEFAULT, RUNS_DIR, WINDOWS_DIR, Config
from ..data.windows import load_manifest
from .baselines import BASELINES, Baseline, drive
from .session import EpisodeSession

TRACKS = ("real", "twin")

#: The leakage probe's shape (D9). Recorded in the manifest so the analytics reads the
#: chance level from the run rather than assuming it.
PROBE = {"n_options": 8, "n_reps": 5, "period_granularity": "year", "chance_level": 0.125}


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    return f"{stamp}_{secrets.token_hex(4)}"


def write_run_manifest(
    run_dir: Path,
    agents: list[dict],
    cfg: Config = DEFAULT,
    windows_dir: Path = WINDOWS_DIR,
    run_id: str | None = None,
) -> dict:
    man = load_manifest(windows_dir)
    run_dir = Path(run_dir)
    (run_dir / "episodes").mkdir(parents=True, exist_ok=True)
    (run_dir / "probe").mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": "1.1.0",
        "run_id": run_id or run_dir.name,
        "created_at": datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
        "config": cfg.log_block(),
        "dataset": {
            "dataset_sha256": man["dataset_sha256"],
            "as_of": man["dataset_as_of"],
        },
        "windows_manifest_sha256": man["manifest_sha256"],
        "agents": agents,
        "tracks": list(TRACKS),
        "presentation_order": man["presentation_order"],
        "windows": [
            {
                "window_id": w["window_id"],
                "regime": w["regime"],
                "bh_daily_vol": w["bh_daily_vol"],
            }
            for w in man["real"]
        ],
        "probe": dict(PROBE),
        "seeds": {"master_seed": man["master_seed"]},
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def run_baseline_episode(
    baseline: Baseline,
    window_id: str,
    track: str,
    episode_index: int,
    run_dir: Path,
    run_id: str,
    cfg: Config = DEFAULT,
    windows_dir: Path = WINDOWS_DIR,
) -> dict:
    session = EpisodeSession.start(
        window_id=window_id,
        track=track,
        agent=baseline.agent(),
        episode_index=episode_index,
        run_dir=run_dir,
        run_id=run_id,
        cfg=cfg,
        windows_dir=windows_dir,
    )
    drive(session, baseline)
    session.finish()
    return session.summary or {}


def run_baselines(
    run_dir: Path | None = None,
    cfg: Config = DEFAULT,
    windows_dir: Path = WINDOWS_DIR,
    only: list[str] | None = None,
    quiet: bool = False,
) -> Path:
    """Every baseline x every track x every window. 4 x 2 x 30 = 240 episodes."""
    run_id = new_run_id()
    run_dir = Path(run_dir) if run_dir else RUNS_DIR / run_id
    names = only or list(BASELINES)

    agents = [BASELINES[n]().agent() for n in names]
    man = write_run_manifest(run_dir, agents, cfg, windows_dir, run_id=run_id)
    order = man["presentation_order"]

    n = 0
    for name in names:
        for track in TRACKS:
            for i, wid in enumerate(order):
                # A fresh instance per *episode*, not per lane. A baseline carries state
                # — the seeded RNG, the buy-and-hold latch — and reusing one instance
                # across a 30-window lane would let w00 poison the other 29.
                baseline = BASELINES[name]()
                baseline.on_episode(track, wid)
                run_baseline_episode(
                    baseline, wid, track, i, run_dir, run_id, cfg, windows_dir
                )
                n += 1
            if not quiet:
                print(f"  {name:<14} {track:<5} {len(order)} episodes")

    if not quiet:
        print(f"\n{n} episodes -> {run_dir}")
    return run_dir
