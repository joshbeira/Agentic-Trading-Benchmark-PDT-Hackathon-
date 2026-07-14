"""The episode session — the one door into the engine.

Every agent reaches the engine through `EpisodeSession.call()`: the MCP server hands
an LLM's tool call to it, and the four baselines call it directly in-process. There is
no second path to a fill.

That is what makes D14's fee parity **structural rather than asserted**. A baseline
cannot accidentally get a cheaper execution than a model, or skip the read cap, or see a
bar the model could not — not because we checked, but because there is nowhere else to
go. `tests/test_mcp.py` pins this down by driving the same baseline through the MCP
transport and through the direct path and demanding the two logs agree to the byte,
excepting only the two timestamps and the duration that cannot possibly match.

The session also owns the two D13 rules the engine structurally cannot enforce, because
the engine has no clock and never sees anything but tool calls:

  * the **wall-clock cap** — at ~20 minutes the remaining ticks auto-Wait and the
    episode is flagged `wallclock_capped`. It still reaches the trading scoreboard
    (`analysis_artifacts.md` Q5): dithering until the clock runs out is a real, bad
    trading outcome, not a harness failure.
  * the **prose nudge** — a reply with no tool call at all.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

from ..config import DEFAULT, WINDOWS_DIR, Config
from ..data.windows import load_episode, load_manifest
from ..engine import TradingEnv
from ..engine.types import TIME_ADVANCING, Tool

#: The tool contract, in one place. The MCP server writes its own surface out longhand
#: (each tool needs a signature and a description), so `tests/test_mcp.py` asserts what it
#: serves equals this — a tool cannot be added to one and forgotten in the other.
TOOLS: tuple[str, ...] = tuple(str(t) for t in Tool)
ADVANCING: frozenset[str] = frozenset(str(t) for t in TIME_ADVANCING)


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - a log without a commit is still a log
        return None


class EpisodeSession:
    """One episode of one agent on one window."""

    def __init__(
        self,
        env: TradingEnv,
        window: dict,
        track: str,
        agent: dict,
        cfg: Config,
    ):
        self.env = env
        self.window = window
        self.track = track
        self.agent = agent
        self.cfg = cfg
        self._status = "ok"
        self._t0 = time.monotonic()
        self._obs: dict = env.reset()

    # ------------------------------------------------------------------ start

    @classmethod
    def start(
        cls,
        *,
        window_id: str,
        track: str,
        agent: dict,
        episode_index: int,
        run_dir: Path | None = None,
        run_id: str | None = None,
        cfg: Config = DEFAULT,
        windows_dir: Path = WINDOWS_DIR,
    ) -> EpisodeSession:
        """Load a window and open an episode. The log path is fixed by the run layout
        (`schemas/tick_log.md`), so no caller can invent its own filename."""
        series, spec = load_episode(windows_dir, window_id, track)
        man = load_manifest(windows_dir)

        log_path = None
        if run_dir is not None:
            log_path = (
                Path(run_dir) / "episodes" / f"{agent['id']}__{track}__{window_id}.jsonl"
            )

        env = TradingEnv(
            series,
            spec,
            cfg,
            log_path=log_path,
            meta_extra={
                "episode_id": f"{agent['id']}__{track}__{window_id}",
                "episode_index": episode_index,
                "run_id": run_id,
                "track": track,
                "agent": dict(agent),
                "dataset": {
                    "dataset_sha256": man["dataset_sha256"],
                    "parquet_path": "data/processed/prices.parquet",
                    "as_of": man["dataset_as_of"],
                },
                "seeds": {
                    "master_seed": man["master_seed"],
                    "window_seed": None,
                    # A real window has no twin seed. Saying so explicitly beats
                    # omitting the key and letting a reader guess.
                    "twin_seed": spec.get("twin_seed") if track == "twin" else None,
                    "agent_seed": agent.get("seed"),
                },
                "git_commit": git_commit(),
            },
        )
        return cls(env, spec, track, dict(agent), cfg)

    # ------------------------------------------------------------------- state

    @property
    def obs(self) -> dict:
        return self._obs

    @property
    def done(self) -> bool:
        return self.env.done

    @property
    def t(self) -> int:
        return self.env.t

    @property
    def status(self) -> str:
        return self._status

    @property
    def summary(self) -> dict | None:
        return self.env.summary

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self._t0

    # -------------------------------------------------------------- the door

    def call(
        self,
        tool: str,
        args: dict | None = None,
        latency_ms: float | None = None,
        tokens: dict | None = None,
    ) -> dict:
        """Dispatch one tool call. Never raises on a bad call — an invalid call is a
        *result*, not an exception: the agent has to see the structured error to correct
        itself, and the reliability scoreboard has to count it."""
        if not self.env.done and self.elapsed_s > self.cfg.episode_wallclock_cap_s:
            self._cap_out()

        res = self.env.step(tool, args or {}, latency_ms=latency_ms, tokens=tokens)
        if res.obs is not None:
            self._obs = res.obs
        return res.as_agent_payload()

    def record_prose_nudge(self) -> None:
        self.env.record_prose_nudge()

    def _cap_out(self) -> None:
        """The clock ran out. Auto-Wait the remaining ticks and flag the episode (D13)."""
        self._status = "wallclock_capped"
        guard = self.cfg.n_scored + 2
        while not self.env.done and guard > 0:
            guard -= 1
            remaining = (self.cfg.n_scored - 1) - self.env.t
            self.env.step("Wait", {"n": max(1, min(self.cfg.max_wait, remaining))})

    # ------------------------------------------------------------------ finish

    def finish(
        self,
        memory_note_in: str | None = None,
        memory_note_out: str | None = None,
        cost: dict | None = None,
        status: str | None = None,
    ) -> dict | None:
        """Seal the log. An episode that never reached the terminal bar is auto-Waited
        out first: a half-written log is not a shorter episode, it is an unreadable one,
        and the schema requires all `n_scored` ticks."""
        if not self.env.done:
            self._cap_out()
        return self.env.close_log(
            memory_note_in=memory_note_in,
            memory_note_out=memory_note_out,
            status=status or self._status,
            cost=cost,
        )
