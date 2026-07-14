"""Read a run. This is the only module in the analytics layer that touches disk.

Everything downstream — scoreboard, reliability, learning curves, leakage — is a pure
function of what comes out of here, which is what lets the whole layer be tested against
hand-written fixture logs with no engine anywhere in sight.

**Metrics are recomputed, not trusted.** The schema is explicit that the `metrics` block
in `episode_end` is a cache. The scoreboard therefore rebuilds every number from
`equity_series_cents` + `fill` records + `bh_daily_vol`, using the `vol_floor_multiple`
and `trading_days_per_year` *stamped in that episode's own config block* — never from a
default that might have drifted since the run. Disagreement with the cached block is
recorded on `Episode.metric_drift` and surfaced loudly, because it means the engine and
the specification have diverged and one of them is wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .. import metrics as M

TRACKS = ("real", "twin")
TERMINAL_STATUSES = ("ok", "wallclock_capped", "agent_error")

#: Excluded from the trading scoreboard: a provider or harness failure is not a trading
#: result, and scoring a crash as "flat" would reward crashing on a bad window. Counted
#: and reported on the reliability board instead (see schemas/analysis_artifacts.md Q5).
SCOREBOARD_EXCLUDED = ("agent_error",)


@dataclass
class Episode:
    episode_id: str
    run_id: str | None
    episode_index: int
    agent_id: str
    agent_kind: str  # "llm" | "baseline"
    agent_memory: str  # "rolling_note" | "none"
    model: str | None
    track: str
    window_id: str
    regime: str
    source_ticker: str | None
    bh_daily_vol: float
    status: str

    equity_cents: list[int]
    shares_by_tick: list[float]
    fills: list[dict]
    calls: list[dict]
    reliability: dict
    config: dict
    cached_metrics: dict
    metrics: dict  # recomputed from the log, per the schema
    metric_drift: dict = field(default_factory=dict)
    memory_note_in: str | None = None
    memory_note_out: str | None = None
    tokens: dict = field(default_factory=dict)

    @property
    def is_baseline(self) -> bool:
        return self.agent_kind == "baseline"

    @property
    def scorable(self) -> bool:
        return self.status not in SCOREBOARD_EXCLUDED

    @property
    def sharpe(self) -> float:
        return self.metrics["sharpe_floored"]


@dataclass
class Run:
    run_id: str
    episodes: list[Episode]
    manifest: dict | None = None

    @property
    def agents(self) -> list[str]:
        seen: dict[str, None] = {}
        for e in self.episodes:
            seen.setdefault(e.agent_id, None)
        return list(seen)

    @property
    def baselines(self) -> list[str]:
        return [a for a in self.agents if self.agent_kind(a) == "baseline"]

    @property
    def llms(self) -> list[str]:
        return [a for a in self.agents if self.agent_kind(a) == "llm"]

    def agent_kind(self, agent_id: str) -> str:
        return next(e.agent_kind for e in self.episodes if e.agent_id == agent_id)

    def agent_memory(self, agent_id: str) -> str:
        return next(e.agent_memory for e in self.episodes if e.agent_id == agent_id)

    def select(self, agent_id=None, track=None, regime=None, scorable_only=True) -> list[Episode]:
        out = self.episodes
        if scorable_only:
            out = [e for e in out if e.scorable]
        if agent_id is not None:
            out = [e for e in out if e.agent_id == agent_id]
        if track is not None:
            out = [e for e in out if e.track == track]
        if regime is not None:
            out = [e for e in out if e.regime == regime]
        return sorted(out, key=lambda e: e.episode_index)

    def lane(self, agent_id: str, track: str) -> list[Episode]:
        """One memory lane, ordered by episode_index — the learning curve's x-axis."""
        return self.select(agent_id=agent_id, track=track)

    def paired(self, agent_id: str) -> list[tuple[Episode, Episode]]:
        """(real, twin) pairs for one agent, matched on window_id. The shared window_id
        IS the pairing (D9); a pair with a missing half is dropped."""
        real = {e.window_id: e for e in self.select(agent_id=agent_id, track="real")}
        twin = {e.window_id: e for e in self.select(agent_id=agent_id, track="twin")}
        return [(real[w], twin[w]) for w in real if w in twin]

    @property
    def excluded(self) -> list[Episode]:
        return [e for e in self.episodes if not e.scorable]

    @property
    def drifted(self) -> list[Episode]:
        return [e for e in self.episodes if e.metric_drift]


# --------------------------------------------------------------------------- reading


def load_episode(path: Path) -> Episode:
    recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not recs or recs[0].get("type") != "meta":
        raise ValueError(f"{path.name}: first record must be `meta`")
    if recs[-1].get("type") != "episode_end":
        raise ValueError(f"{path.name}: last record must be `episode_end`")

    meta, end = recs[0], recs[-1]
    ticks = [r for r in recs[1:-1] if r.get("type") == "tick"]
    if [t["t"] for t in ticks] != list(range(len(ticks))):
        raise ValueError(f"{path.name}: tick indices are not 0..n-1 in order")

    cfg = meta["config"]
    win = meta["window"]
    agent = meta["agent"]

    equity = [t["equity_cents"] for t in ticks]
    shares = [t["obs"]["portfolio"]["shares"] for t in ticks]
    fills = [t["fill"] for t in ticks if t.get("fill")]
    calls = [c for t in ticks for c in t.get("calls", [])]

    if equity != end["equity_series_cents"]:
        raise ValueError(
            f"{path.name}: episode_end.equity_series_cents disagrees with the per-tick "
            "marks -- the log contradicts itself"
        )

    # Recompute from the log, using the config the episode actually ran under.
    recomputed = M.compute(
        equity,
        fills,
        shares,
        win["bh_daily_vol"],
        vol_floor_multiple=cfg.get("vol_floor_multiple", M.DEFAULT_VOL_FLOOR_MULTIPLE),
        trading_days=cfg.get("trading_days_per_year", M.DEFAULT_TRADING_DAYS),
    )
    cached = end.get("metrics", {})
    drift = {
        k: {"cached": cached[k], "recomputed": v}
        for k, v in recomputed.items()
        if k in cached and not _close(cached[k], v)
    }

    return Episode(
        episode_id=meta.get("episode_id", path.stem),
        run_id=meta.get("run_id"),
        episode_index=meta.get("episode_index", 0),
        agent_id=agent["id"],
        agent_kind=agent.get("kind", "llm"),
        agent_memory=agent.get("memory", "none"),
        model=agent.get("model"),
        track=meta["track"],
        window_id=win["window_id"],
        regime=win["regime"],
        source_ticker=win.get("source_ticker"),
        bh_daily_vol=win["bh_daily_vol"],
        status=end.get("status", "ok"),
        equity_cents=equity,
        shares_by_tick=shares,
        fills=fills,
        calls=calls,
        reliability=end.get("reliability", {}),
        config=cfg,
        cached_metrics=cached,
        metrics=recomputed,
        metric_drift=drift,
        memory_note_in=end.get("memory_note_in"),
        memory_note_out=end.get("memory_note_out"),
        tokens=end.get("cost", {}),
    )


def load_run(run_dir: Path) -> Run:
    run_dir = Path(run_dir)
    ep_dir = run_dir / "episodes"
    if not ep_dir.is_dir():
        raise FileNotFoundError(f"no episodes/ under {run_dir}")

    episodes = [load_episode(p) for p in sorted(ep_dir.glob("*.jsonl"))]
    if not episodes:
        raise ValueError(f"no episode logs in {ep_dir}")

    manifest = None
    mpath = run_dir / "run_manifest.json"
    if mpath.exists():
        manifest = json.loads(mpath.read_text())

    run_id = (manifest or {}).get("run_id") or episodes[0].run_id or run_dir.name
    return Run(run_id=run_id, episodes=episodes, manifest=manifest)


def _close(a, b, tol: float = 1e-9) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    return a == b
