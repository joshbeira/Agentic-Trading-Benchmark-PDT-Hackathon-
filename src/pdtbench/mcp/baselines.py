"""The four benchmark baselines (D14).

They drive `EpisodeSession.call()` — the same door the MCP server puts an LLM through.
Not a private fast path, and emphatically not the engine's internals: `sma_10_50` pulls
its history with `fetchData` and computes its own moving averages, exactly as a model
would have to. (Contrast `tests/policies.py`, which reaches straight into `env.bars`.
That is fine for a test policy stressing the engine; it would be cheating here.)

This is what makes "baselines pay the same fees" a fact about the code rather than a
claim on a slide: there is one execution path, so there is one fee schedule.

The baselines carry the whole weight of the learning curve. They cannot learn, so their
per-episode Sharpe *is* the window-difficulty signal, and an LLM's learning is only ever
measured as divergence from that trace (D11). If a baseline were quietly cheaper to
execute, every learning curve in the deck would be wrong.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .session import EpisodeSession


class Baseline:
    """A baseline makes any number of reads and exactly one time-advancing call."""

    id: str = ""
    seed: int | None = None

    def agent(self) -> dict:
        spec = {"id": self.id, "kind": "baseline", "memory": "none"}
        if self.seed is not None:
            spec["seed"] = self.seed
        return spec

    def on_episode(self, track: str, window_id: str) -> None:
        """Reset per-episode state. Called once, before the first tick.

        A baseline carries state — a seeded RNG, a latch — and an instance reused across a
        30-window lane carries it *between* windows. Buy-and-hold would then buy on w00
        and never trade again, and the difficulty trace that every learning curve is
        subtracted from would be 29/30 flat. The engine cannot catch that; only this can.
        """

    def act(self, s: EpisodeSession) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    @staticmethod
    def _hold(s: EpisodeSession) -> None:
        """Wait as far as the interface allows. `Wait(n)` returns all n elapsed bars in
        one response, so holding for ten bars costs one call, not ten (D4)."""
        s.call("Wait", {"n": max(1, min(10, s.obs["ticks_remaining"]))})


class Flat(Baseline):
    """Never trades. Scores exactly 0 — the fixed point the vol-floored Sharpe is built
    to have (D2), and the reason a never-trading agent cannot post a Sharpe of +30."""

    id = "flat"

    def act(self, s: EpisodeSession) -> None:
        self._hold(s)


class BuyAndHold(Baseline):
    """All in at the first opportunity, then ride to the end.

    Exits only through the terminal liquidation, so it pays exactly two frictions — the
    cheapest any strategy can be. It also sits at the 88/90 `time_in_market` ceiling,
    because tick 0 is necessarily flat (the first fill lands at `open_1`) and tick 89 is
    necessarily flat (post-liquidation).
    """

    id = "buy_and_hold"

    def __init__(self) -> None:
        self.bought = False

    def on_episode(self, track: str, window_id: str) -> None:
        self.bought = False

    def act(self, s: EpisodeSession) -> None:
        if not self.bought:
            self.bought = True
            s.call("Buy", {"fraction": 1.0})
            return
        self._hold(s)


class Random5pct(Baseline):
    """Trades on a 5% coin per bar; seeded, so it is identical for every model's run.

    Present as a control for turnover, not as a strategy: it shows what paying friction
    at a given trade rate costs you when the trades carry no information.
    """

    id = "random_5pct"
    seed = 11

    def __init__(self, seed: int = 11, p: float = 0.05) -> None:
        self.seed = seed
        self.p = p
        self.rng = np.random.default_rng(seed)

    def on_episode(self, track: str, window_id: str) -> None:
        """An independent but reproducible stream per episode, derived from the agent's
        declared seed. Carrying one stream across a lane would make w29's coin flips
        depend on how many draws w00..w28 happened to consume — reproducible, but coupled
        to the traversal order for no reason. Same derivation as the twin seeds."""
        digest = hashlib.sha256(f"{self.seed}:{track}:{window_id}".encode()).hexdigest()
        self.rng = np.random.default_rng(int(digest[:8], 16))

    def act(self, s: EpisodeSession) -> None:
        p = s.obs["portfolio"]
        if self.rng.random() < self.p:
            if p["shares"] > 0 and self.rng.random() < 0.5:
                s.call("Sell", {"fraction": 1.0})
                return
            # Only buy with cash actually worth deploying. A Buy against an empty wallet
            # would be a NONPOSITIVE_QTY error, and a baseline has no business generating
            # invalid calls — it would pollute the reliability board it is the control for.
            if p["cash_cents"] >= 100:
                s.call("Buy", {"fraction": 0.5})
                return
        s.call("Wait", {"n": 1})


class SmaCrossover(Baseline):
    """10/50 moving-average crossover, all-in / all-out.

    Pulls its own history through `fetchData` and computes both averages itself, because
    `getStats` publishes SMA-20 and SMA-50 but not SMA-10 — so this baseline exercises the
    same read path an LLM would need, and pays the same read budget.

    On this window set it *loses money* (−0.54 median Sharpe, −3.6% median return). That
    is reported, not fixed. D7 struck the original claim that friction was "calibrated so
    the momentum baseline is roughly breakeven" precisely because calibrating a cost
    constant against ex-post-selected windows is circular, and this audience would say so.
    Friction is 10 bps because 10 bps is realistic.
    """

    id = "sma_10_50"

    def __init__(self, fast: int = 10, slow: int = 50) -> None:
        self.fast, self.slow = fast, slow

    def act(self, s: EpisodeSession) -> None:
        res = s.call("fetchData", {"lookback": self.slow})
        closes = np.array([b["close"] for b in res["bars"]], dtype=np.float64)
        fast = float(closes[-self.fast :].mean())
        slow = float(closes.mean())

        long = s.obs["portfolio"]["shares"] > 0
        if fast > slow and not long:
            s.call("Buy", {"fraction": 1.0})
        elif fast <= slow and long:
            s.call("Sell", {"fraction": 1.0})
        else:
            s.call("Wait", {"n": 1})


#: The four, in the order they appear on the scoreboard.
BASELINES: dict[str, type[Baseline]] = {
    "buy_and_hold": BuyAndHold,
    "flat": Flat,
    "random_5pct": Random5pct,
    "sma_10_50": SmaCrossover,
}


def drive(session: EpisodeSession, baseline: Baseline) -> None:
    """Run a baseline to the terminal bar. Guards against a policy that never advances —
    an episode that cannot end would otherwise hang the overnight batch."""
    guard = 20_000
    while not session.done:
        guard -= 1
        if guard <= 0:
            raise RuntimeError(f"{baseline.id} never terminated the episode")
        baseline.act(session)
